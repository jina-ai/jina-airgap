"""
Air-gapped license-key gate for Jina SM.

WHAT THIS IS (and is not)
-------------------------
A lightweight, offline, time-bound *entitlement signal*. It gives sales and
audit a visible "the deployment carries a key that expires" control, without a
license server, without phone-home, and without rebuilding the image to issue
or renew a key. It is deliberately 防君子不防小人 ("keep honest people honest,
not a real lock"): the signing secret ships inside the image, so anyone who
reads this file can mint or bypass a key. That is intentional. It is NOT DRM
and must not be sold as such - the customer holds the model weights, so there
is nothing to truly lock.

THE ONE RULE THAT OVERRIDES EVERYTHING
--------------------------------------
A paying, already-deployed customer must NEVER be blocked by this mechanism.
Not by a missing key, an expired key, a corrupted key, or a wrong system
clock. Therefore the DEFAULT MODE IS FAIL-OPEN ("warn"): the server always
answers; a bad/missing/expired key only produces a log line and a /health
status field. Hard blocking (HTTP 403) is strictly opt-in and is meant for
time-boxed trials / POCs where you *want* access to lapse.

MODES  (env JINA_LICENSE_MODE, default "warn")
----------------------------------------------
  warn     Default. Fail-open. Always serve. Log + /health report key state.
           Ship SOLD customers in this mode: they can never be blocked.
  enforce  Fail-closed. Return 403 on inference endpoints when the key is
           missing / expired (past grace) / invalid. For trials and POCs.
  off      No checking, no logging. Fully transparent.

  Back-compat: JINA_LICENSE_ENFORCE=0 forces "off". JINA_LICENSE_ENFORCE=1 is
  ignored unless JINA_LICENSE_MODE is unset, in which case it selects
  "enforce" (legacy behaviour for anyone who already wired that flag).

GRACE (enforce mode only)  (env JINA_LICENSE_GRACE_DAYS, default 14)
-------------------------------------------------------------------
Even in enforce mode, an expired key keeps working for this many days, logging
loudly. This absorbs clock skew and renewal lag so a genuine customer is never
cut off by a day-boundary or a wrong RTC. Set 0 for a hard cutoff at expiry.

CRYPTO NOTES (for the curious - why this shape)
-----------------------------------------------
  * The key is a signed token carrying an ``exp`` timestamp. This is the
    standard primitive for *offline* license validation (same idea as a JWT
    with an ``exp`` claim, signed HS256). Verification is a local HMAC-SHA256
    compare - no network, no clock server.
  * TOTP / Google Authenticator (RFC 6238) is a different tool: rotating
    30-second one-time codes for interactive 2FA against a live verifier. It
    is NOT suitable for a durable, offline, air-gapped license window, so we
    do not use it here.
  * We use SYMMETRIC HMAC with a public secret on purpose (防君子不防小人). If
    one ever wanted a real lock, the minimal upgrade is ASYMMETRIC signing
    (e.g. Ed25519): ship only the PUBLIC key in the image so it can verify but
    not mint keys, and keep the private key on the issuing side. That is a
    deliberate non-goal today - it would not change the fact that the customer
    holds the weights, and it adds key-management overhead for no real gain.

KEY FORMAT (compact, single line, copy-paste safe)
--------------------------------------------------
    JINA-<base64url(payload_json)>.<base64url(hmac_sha256(payload_json))>

    payload_json = {"sub": "<customer>", "iat": <unix>, "exp": <unix>,
                    "model": "<model-id or *>", "v": 1}
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Public default secret. Overridable at build time (--build-arg
# LICENSE_SECRET=...) or run time (-e JINA_LICENSE_SECRET=...). Being public is
# a feature here, not a leak - see module docstring.
DEFAULT_SECRET = "jina-on-prem-symbolic-license-v1"

PREFIX = "JINA-"

VALID_MODES = ("warn", "enforce", "off")


def _secret() -> bytes:
    # Empty env (e.g. an unset --build-arg baked as "") falls back to the
    # public default so the CPU/GPU images stay consistent with keygen.
    return (os.environ.get("JINA_LICENSE_SECRET") or DEFAULT_SECRET).encode("utf-8")


def mode() -> str:
    """Resolve the effective enforcement mode. Default: warn (fail-open)."""
    m = (os.environ.get("JINA_LICENSE_MODE") or "").strip().lower()
    if m in VALID_MODES:
        return m
    # Back-compat with the earlier JINA_LICENSE_ENFORCE flag.
    legacy = os.environ.get("JINA_LICENSE_ENFORCE")
    if legacy == "0":
        return "off"
    if legacy == "1":
        return "enforce"
    return "warn"


def grace_days() -> float:
    try:
        return max(0.0, float(os.environ.get("JINA_LICENSE_GRACE_DAYS", "14")))
    except (TypeError, ValueError):
        return 14.0


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign_with(secret: bytes, payload_bytes: bytes) -> str:
    return _b64e(hmac.new(secret, payload_bytes, hashlib.sha256).digest())


def issue(sub: str, days: int, model: str = "*", secret: Optional[str] = None) -> str:
    """Mint a license key valid for ``days`` from now.

    Dependency-free and importable so ``jina-on-prem.py keygen`` can call it
    directly without importing the server stack.
    """
    now = int(time.time())
    payload = {
        "sub": sub,
        "iat": now,
        "exp": now + int(days) * 86400,
        "model": model,
        "v": 1,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sec = secret.encode("utf-8") if secret is not None else _secret()
    sig = _sign_with(sec, payload_bytes)
    return f"{PREFIX}{_b64e(payload_bytes)}.{sig}"


def inspect(key: str) -> dict:
    """Decode a key WITHOUT verifying the signature (for humans / logging)."""
    body = key[len(PREFIX):] if key.startswith(PREFIX) else key
    token, _, _sig = body.partition(".")
    return json.loads(_b64d(token))


def validate(key: Optional[str], model_id: str = "") -> tuple[bool, str, dict]:
    """Validate a key against secret, expiry, and (optionally) model scope.

    Returns (ok, reason, payload). Never raises - any unexpected error maps to
    a non-fatal reason so callers can fail open. ``payload`` is the decoded
    claims dict (possibly empty).
    """
    try:
        if not key:
            return False, "no_license", {}

        body = key[len(PREFIX):] if key.startswith(PREFIX) else key
        token, sep, sig = body.partition(".")
        if not sep or not token or not sig:
            return False, "malformed_license", {}

        try:
            payload_bytes = _b64d(token)
            payload = json.loads(payload_bytes)
        except Exception:
            return False, "malformed_license", {}

        expected = _sign_with(_secret(), payload_bytes)
        if not hmac.compare_digest(expected, sig):
            return False, "bad_signature", payload

        exp = int(payload.get("exp", 0))
        if int(time.time()) >= exp:
            return False, "license_expired", payload

        scope = payload.get("model", "*")
        # An empty scope means a buggy issuer, not "unrestricted" - use "*" for
        # that. Checked separately because the tuple below holds model_id, which
        # is itself "" when JINA_MODEL_ID is unset, so an empty scope would slip
        # through there. A *missing* model claim still means any model (default
        # above); only an explicitly empty one is rejected.
        if not scope:
            return False, "empty_scope", payload

        short = model_id.split("/")[-1] if model_id else ""
        if scope not in ("*", model_id, short):
            return False, "model_not_licensed", payload

        return True, "ok", payload
    except Exception as e:  # defensive: never let validation crash the gate
        logger.warning("license validate() unexpected error: %s", e)
        return False, "validation_error", {}


def days_until_expiry(payload: dict) -> Optional[float]:
    exp = payload.get("exp")
    if not exp:
        return None
    return round((int(exp) - time.time()) / 86400, 1)


def decide(key: Optional[str], model_id: str = "") -> dict:
    """Single source of truth for the middleware.

    Returns a dict:
      allow:  bool   - whether the request should be served
      block:  bool   - whether to return 403 (only ever true in enforce mode)
      reason: str    - machine-ish reason code
      mode:   str    - resolved mode
      payload: dict  - decoded claims (may be empty)

    Fail-open guarantee: allow is False ONLY in enforce mode with a genuinely
    missing/expired(past grace)/invalid key. warn and off always allow.
    """
    m = mode()
    ok, reason, payload = validate(key, model_id)

    if m == "off":
        return {"allow": True, "block": False, "reason": "off", "mode": m, "payload": payload}

    if m == "warn":
        # Fail-open. Never block. Surface state for logs / health only.
        return {"allow": True, "block": False, "reason": reason, "mode": m, "payload": payload}

    # enforce
    if ok:
        return {"allow": True, "block": False, "reason": "ok", "mode": m, "payload": payload}

    # In enforce mode, an *expired* key still gets a grace window (absorbs
    # clock skew + renewal lag). Signature/scope failures do not get grace.
    if reason == "license_expired":
        d = days_until_expiry(payload)  # negative once expired
        g = grace_days()
        if d is not None and d > -g:
            return {"allow": True, "block": False, "reason": "expired_in_grace",
                    "mode": m, "payload": payload}

    return {"allow": False, "block": True, "reason": reason, "mode": m, "payload": payload}


def status(key: Optional[str], model_id: str = "") -> dict:
    """Compact status block for /health and startup logs."""
    m = mode()
    ok, reason, payload = validate(key, model_id)
    out = {"mode": m, "valid": ok, "reason": reason, "fail_open": m != "enforce"}
    if payload:
        out["licensed_to"] = payload.get("sub")
        exp = payload.get("exp")
        if exp:
            out["expires"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(exp)))
            out["days_left"] = days_until_expiry(payload)
    if m == "enforce":
        out["grace_days"] = grace_days()
    return out
