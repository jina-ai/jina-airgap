"""Jina AI On-Prem Inference Server.

One model per image, five wire contracts over it.

  POST /v1/embeddings                            Jina  (also OpenAI / Voyage / Gemini fields)
  POST /v1/rerank                                Jina  (also Voyage top_k, Cohere max_tokens_per_doc)
  POST /v2/embed, /v2/rerank                     Cohere
  POST /v1/models/{model}:embedContent           Google Gemini
  POST /v1/models/{model}:batchEmbedContents     Google Gemini
  POST /v1/multimodalembeddings                  Voyage AI
  POST /v1/chat/completions                      OpenAI
  GET  /health                                   health, license, throughput

The routes live in ``adapters``; the model lives behind ``engine``. What is
left here is the process: the transformers patches that must run before
anything imports transformers, the license gate, startup, health, and the
exception handlers that hand each error to the adapter that owns the route.
"""

import os
import sys
from pathlib import Path


def _register_dynamic_module_dirs() -> None:
    """Register dynamic model module directories so transformers check_imports
    can find sibling modules (e.g. configuration_eurobert imported by
    modeling_jina_embeddings_v5). Without this, air-gapped containers fail
    because importlib.util.find_spec cannot locate these modules."""
    root = Path(os.environ.get("HF_HOME", "")) / "modules" / "transformers_modules"
    if not root.is_dir():
        return
    for module in root.rglob("*.py"):
        parent = str(module.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)


def _patch_transformers() -> None:
    """Patch transformers for air-gapped Jina model loading.

    1. resolve_trust_remote_code: force True so custom_st.py AutoConfig/AutoModel
       calls don't prompt for interactive confirmation in offline containers.
    2. add_generation_mixin_to_remote_model: guard against models without
       prepare_inputs_for_generation (embedding-only models like JinaEmbeddingsV5).

    A third patch used to live here, bypassing ``_patch_mistral_regex`` -- a
    private classmethod that only transformers 4.57.3 has, and that hits the hub
    whenever a tokenizer is loaded by repo id. jina-reranker-v3.5 was the only
    model pinned to 4.57.3; it now pins 4.57.1, where the classmethod does not
    exist, so the bypass is gone rather than dormant.
    """
    try:
        from transformers import dynamic_module_utils

        dynamic_module_utils.resolve_trust_remote_code = lambda *a, **kw: True
    except Exception:
        pass

    try:
        from transformers.models.auto import auto_factory

        add_generation_mixin = getattr(
            auto_factory, "add_generation_mixin_to_remote_model", None
        )
        if add_generation_mixin:

            def safe_add_generation_mixin(model_class):
                if not hasattr(model_class, "prepare_inputs_for_generation"):
                    return model_class
                return add_generation_mixin(model_class)

            auto_factory.add_generation_mixin_to_remote_model = (
                safe_add_generation_mixin
            )
    except Exception:
        pass


# Both must run before transformers is imported anywhere else in the process.
_register_dynamic_module_dirs()
_patch_transformers()

import logging  # noqa: E402

from fastapi import FastAPI, Response  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

import adapters  # noqa: E402
import engine  # noqa: E402
import telemetry  # noqa: E402
from config import settings  # noqa: E402
from errors import JinaError  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Jina AI On-Prem Server",
    version="4.0.0",
    description="One model, five schemas: Jina, OpenAI, Cohere, Google Gemini, Voyage AI",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.include_router(adapters.router)


@app.exception_handler(JinaError)
async def _jina_error_handler(request, exc: JinaError):
    return JSONResponse(
        status_code=exc.status, content=adapters.renderer_for(request).error(exc)
    )


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=adapters.renderer_for(request).validation(exc.errors()),
    )


@app.middleware("http")
async def _timing_headers(request, call_next):
    """Throughput belongs in headers, not the body: api.jina.ai reports none,
    and a `tok_per_s` key inside `usage` is exactly the kind of drift this
    rewrite removes."""
    box = telemetry.begin()
    response = await call_next(request)
    if "elapsed_ms" in box:
        response.headers["X-Jina-Elapsed-Ms"] = f"{box['elapsed_ms']:.1f}"
        response.headers["X-Jina-Tok-Per-S"] = f"{box['tok_per_s']:.1f}"
    return response


# --- License gate (time-sensitive, offline, runtime-injected) ---
# Symbolic entitlement signal for sales/audit, NOT DRM. The signing secret
# ships in the image (see server/license.py): honest users get a visible
# expiry knob, determined users can trivially bypass. 防君子不防小人.
#
# THE OVERRIDING RULE: a paying, already-deployed customer must never be
# blocked by this. The DEFAULT mode is fail-open ("warn") - the server always
# answers; a missing/expired/bad key only logs and shows up in /health. Hard
# 403 blocking happens ONLY in opt-in "enforce" mode (for trials/POCs), and
# even then an expired key keeps working through a grace window. /health and
# docs are always open so Docker healthchecks and "is my key ok?" probes work.
import license as _license  # noqa: E402

_LICENSE_OPEN_PATHS = {
    "/health",
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}
_license_warned = False  # log the warn-mode notice once, not per request


@app.middleware("http")
async def _license_gate(request, call_next):
    # Only inference-type POSTs are ever gated; reads/probes always pass.
    if (
        request.method in ("GET", "OPTIONS", "HEAD")
        or request.url.path in _LICENSE_OPEN_PATHS
    ):
        return await call_next(request)

    d = _license.decide(settings.license_key, settings.model_id)

    # Fail-open path (warn/off, or enforce-within-grace): serve, but surface a
    # single warning line so operators notice a lapsed/absent key.
    if d["allow"]:
        if d["mode"] == "warn" and d["reason"] not in ("ok",):
            global _license_warned
            if not _license_warned:
                logger.warning(
                    "License check (warn mode, fail-open): reason=%s - serving anyway. "
                    "Set JINA_LICENSE_MODE=enforce only for trials/POCs.",
                    d["reason"],
                )
                _license_warned = True
        elif d["reason"] == "expired_in_grace":
            logger.warning(
                "License expired but within grace window - still serving. Renew soon."
            )
        return await call_next(request)

    # Blocking path: enforce mode only.
    hint = {
        "no_license": "Set a license key: docker run -e JINA_LICENSE_KEY=<key> ... (or switch to warn mode).",
        "license_expired": "License expired past its grace window. Request a renewed key (no rebuild needed).",
        "model_not_licensed": f"Key not valid for model '{settings.short_model_id}'.",
        "bad_signature": "License key signature invalid.",
        "malformed_license": "License key is malformed.",
        # Its own message, because this is almost always a category mistyped at
        # issuance rather than the purchasing problem the generic one implies.
        "unknown_category": (
            "License key names a category this image does not recognise: "
            f"{d['payload'].get('category')!r}. Expected one of: "
            f"{', '.join(_license.CATEGORIES)}. Ask for the key to be reissued."
        ),
        "conflicting_scope": (
            "License key is scoped to both a category and a model id, which "
            "contradict each other. Ask for the key to be reissued."
        ),
    }.get(d["reason"], "License validation failed.")
    return JSONResponse(
        status_code=403,
        content={
            "error": {"code": d["reason"], "message": hint, "type": "license_error"}
        },
    )


@app.on_event("startup")
async def startup():
    engine.load()
    _lic = _license.status(settings.license_key, settings.model_id)
    _m = _lic.get("mode")
    if _m == "off":
        logger.info("License checking OFF (JINA_LICENSE_MODE=off) - fully transparent.")
    elif _lic.get("valid"):
        logger.info(
            f"License OK (mode={_m}): sub={_lic.get('licensed_to')} "
            f"expires={_lic.get('expires')} days_left={_lic.get('days_left')}"
        )
    elif _m == "enforce":
        logger.warning(
            f"License not valid ({_lic.get('reason')}) and mode=enforce: inference endpoints "
            f"will 403 after the {_lic.get('grace_days')}-day grace window. This mode is for "
            f"trials/POCs only - do NOT use it for sold, deployed customers."
        )
    else:
        logger.warning(
            f"License not valid ({_lic.get('reason')}) - mode=warn (fail-open), serving normally. "
            f"This never blocks a deployed customer; the key is only a visible expiry signal."
        )


@app.get("/health")
async def health(response: Response):
    ready = engine.is_ready()
    if not ready:
        # 503, not a 200 carrying `ready: false`. The Dockerfile's HEALTHCHECK
        # is `curl -sf /health`, which passes on any 2xx -- so a body-only
        # signal is one nothing reads. An orchestrator replaces the container
        # on this; plain `docker run` at least marks it unhealthy.
        response.status_code = 503
    resp = {
        "status": "ok" if ready else "unavailable",
        "model": settings.short_model_id,
        # A fact about the model, from the catalog baked in here. What a key
        # covers is under "license"; this is what one would have to cover.
        "category": _license.category_of(settings.model_id),
        "device": settings.device,
        "ready": ready,
        "multimodal": engine.is_multimodal(),
        "endpoint": engine.SPEC.api_endpoint,
        "schemas": ["jina", "openai", "voyage", "gemini", "cohere"],
        "license": _license.status(settings.license_key, settings.model_id),
    }
    throughput = telemetry.throughput()
    if throughput:
        resp["throughput"] = throughput
    return resp


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
