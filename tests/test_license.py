"""
Unit tests for the license-key gate. No server, no Docker, no network.

Run:
  python tests/test_license.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
import license as L  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results = []


def check(name, cond):
    _results.append(cond)
    print(f"  [{PASS if cond else FAIL}] {name}")


def clear_env():
    for k in ("JINA_LICENSE_MODE", "JINA_LICENSE_ENFORCE",
              "JINA_LICENSE_GRACE_DAYS", "JINA_LICENSE_SECRET"):
        os.environ.pop(k, None)


def main():
    print("License gate unit tests\n")
    clear_env()

    # --- signing / validation primitives ---
    k = L.issue("acme", 30, "*")
    ok, reason, payload = L.validate(k)
    check("valid key -> ok", ok and reason == "ok")
    check("valid key claims sub", payload.get("sub") == "acme")

    ke = L.issue("acme", 0, "*")
    check("expired key -> license_expired", L.validate(ke)[1] == "license_expired")
    check("tampered sig -> bad_signature", L.validate(k[:-3] + "AAA")[1] == "bad_signature")
    check("garbage -> not valid", not L.validate("JINA-notbase64.sig")[0])
    check("no dot -> malformed", L.validate("JINA-abc")[1] == "malformed_license")
    check("empty -> no_license", L.validate("")[1] == "no_license")

    km = L.issue("acme", 30, "jina-embeddings-v5-text-nano")
    check("scope match (short)", L.validate(km, "jinaai/jina-embeddings-v5-text-nano")[0])
    check("scope mismatch", L.validate(km, "jinaai/jina-reranker-v3")[1] == "model_not_licensed")
    check("wildcard any model", L.validate(k, "jinaai/anything")[0])

    # --- an empty scope must never read as unrestricted ---
    # Only a buggy issuer can produce one (the CLI maps a falsy --model to "*"),
    # and in warn mode it would otherwise be an invisible unlimited-scope key.
    kz = L.issue("acme", 30, "")
    check("empty scope rejected", L.validate(kz, "jinaai/anything")[1] == "empty_scope")
    check("empty scope rejected when model_id unknown",
          L.validate(kz, "")[1] == "empty_scope")
    check("empty scope still fail-open in warn",
          L.decide(kz, "jinaai/anything")["block"] is False)
    os.environ["JINA_LICENSE_MODE"] = "enforce"
    check("empty scope blocks under enforce", L.decide(kz, "jinaai/anything")["block"])
    clear_env()
    # A *missing* model claim still means any model, unlike an empty one.
    claims = {"sub": "acme", "iat": 0, "exp": 2 ** 31 - 1, "v": 1}
    raw = json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8")
    kn = f"{L.PREFIX}{L._b64e(raw)}.{L._sign_with(L._secret(), raw)}"
    check("missing model claim -> unrestricted", L.validate(kn, "jinaai/anything")[0])

    # --- secret rotation ---
    kc = L.issue("acme", 30, "*", secret="custom-secret")
    check("custom-secret key fails default", not L.validate(kc)[0])
    os.environ["JINA_LICENSE_SECRET"] = "custom-secret"
    check("custom-secret key passes matching env", L.validate(kc)[0])
    os.environ["JINA_LICENSE_SECRET"] = ""
    check("empty env secret uses default", L.validate(k)[0])
    clear_env()

    # --- mode resolution ---
    check("default mode is warn (fail-open)", L.mode() == "warn")
    os.environ["JINA_LICENSE_ENFORCE"] = "0"
    check("legacy ENFORCE=0 -> off", L.mode() == "off")
    os.environ["JINA_LICENSE_ENFORCE"] = "1"
    check("legacy ENFORCE=1 -> enforce", L.mode() == "enforce")
    os.environ["JINA_LICENSE_MODE"] = "warn"
    check("explicit MODE beats legacy flag", L.mode() == "warn")
    clear_env()

    # --- decide(): the fail-open guarantee ---
    # warn mode: never blocks, whatever the key state
    for key, label in [(None, "no key"), (ke, "expired"), ("JINA-x.y", "garbage"), (k, "valid")]:
        d = L.decide(key)
        check(f"warn: {label} -> allow, no block", d["allow"] and not d["block"])

    # off mode: never blocks
    os.environ["JINA_LICENSE_MODE"] = "off"
    check("off: no key -> allow", L.decide(None)["allow"] and not L.decide(None)["block"])
    clear_env()

    # enforce mode: blocks bad keys, allows valid
    os.environ["JINA_LICENSE_MODE"] = "enforce"
    check("enforce: valid -> allow", L.decide(k)["allow"])
    check("enforce: no key -> block", L.decide(None)["block"])
    check("enforce: bad sig -> block", L.decide(k[:-3] + "AAA")["block"])

    # enforce grace: key expired 5 days ago, 14-day grace -> still allowed
    k_exp5 = L.issue("acme", -5, "*")  # exp 5 days in the past
    d = L.decide(k_exp5)
    check("enforce: expired within grace -> allow", d["allow"] and d["reason"] == "expired_in_grace")

    # enforce, grace=0: expired -> block
    os.environ["JINA_LICENSE_GRACE_DAYS"] = "0"
    check("enforce grace=0: expired -> block", L.decide(k_exp5)["block"])
    clear_env()

    # --- status() shape ---
    st = L.status(k, "jinaai/jina-embeddings-v5-text-nano")
    check("status warn: fail_open true", st["fail_open"] and st["valid"])
    os.environ["JINA_LICENSE_MODE"] = "enforce"
    st2 = L.status(k)
    check("status enforce: fail_open false + grace_days", (not st2["fail_open"]) and "grace_days" in st2)
    clear_env()

    check("inspect decodes sub", L.inspect(k)["sub"] == "acme")

    print()
    total, passed = len(_results), sum(_results)
    print(f"{PASS if passed == total else FAIL}: {passed}/{total}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
