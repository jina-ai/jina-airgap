"""
Jina AI On-Prem Inference Server
Multi-schema embedding API: OpenAI, Voyage AI, Google Gemini, Cohere.
Real tok/s throughput measurement.

Endpoints:
  POST /v1/embeddings                          - OpenAI + Voyage AI compatible (text + multimodal)
  POST /v1/embed                               - Cohere compatible (text + multimodal)
  POST /v1/multimodalembeddings                - Voyage AI multimodal endpoint
  POST /v1/models/{model_id}:embedContent      - Gemini single-content (text + multimodal)
  POST /v1/models/{model_id}:batchEmbedContents - Gemini batch
  GET  /health                                 - health + throughput stats
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
import time  # noqa: E402
import uuid  # noqa: E402
from typing import Optional, Union  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import engine  # noqa: E402
import telemetry  # noqa: E402
from config import settings  # noqa: E402
from errors import BadRequest, JinaError  # noqa: E402
from media import (  # noqa: E402
    _bytes_to_st_input as bytes_to_st_input,
    _decode_b64 as decode_b64,
    _parse_content_part as parse_content_part,
    _parse_openai_item as parse_openai_item,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Jina AI On-Prem Server",
    version="4.0.0",
    description="Multi-schema embedding server: OpenAI, Voyage AI, Gemini, Cohere (text + multimodal)",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.exception_handler(JinaError)
async def _jina_error_handler(request, exc: JinaError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


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


# =============================================================================
# Health
# =============================================================================


@app.get("/health")
async def health():
    resp = {
        "status": "ok",
        "model": settings.short_model_id,
        "device": settings.device,
        "ready": engine.is_ready(),
        "multimodal": engine.is_multimodal(),
        "schemas": ["openai", "voyage", "gemini", "cohere"],
        "license": _license.status(settings.license_key, settings.model_id),
    }
    throughput = telemetry.throughput()
    if throughput:
        resp["throughput"] = throughput
    return resp


# =============================================================================
# Schema 1: OpenAI + Voyage AI  (POST /v1/embeddings)
#
# Multimodal: `input` items may be structured dicts in addition to plain strings.
# Formats:
#   {"type": "image", "format": "base64", "value": "<b64>"}            (Elastic)
#   {"type": "image_base64", "image_base64": {"base64":"...", "mime_type":"image/png"}}
#   {"type": "audio_base64", "audio_base64": {"base64":"...", "mime_type":"audio/wav"}}
#   {"type": "video_base64", "video_base64": {"base64":"...", "mime_type":"video/mp4"}}
#   {"content": [part, ...]}  fused multimodal -> ONE embedding per block
# =============================================================================


class OpenAIEmbeddingRequest(BaseModel):
    input: Union[str, list]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    # Default to None (not "retrieval") so the family's task resolution can
    # apply a per-family default that matches prod (omni/v4 -> text-matching).
    task: Optional[str] = None
    # Voyage AI extensions
    input_type: Optional[str] = None
    output_dimension: Optional[int] = None
    output_dtype: Optional[str] = None


class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: list
    model: str
    usage: dict


VOYAGE_TASK_MAP = {
    "query": "retrieval.query",
    "document": "retrieval.passage",
    "classification": "classification",
    "clustering": "clustering",
}


@app.post(
    "/v1/embeddings",
    response_model=OpenAIEmbeddingResponse,
    tags=["OpenAI / Voyage AI"],
)
async def create_embeddings_openai(request: OpenAIEmbeddingRequest):
    """
    OpenAI-compatible embedding endpoint. Also accepts Voyage AI fields.
    Supports multimodal inputs for omni models via structured input items.

    Compatible with:
    - OpenAI client: openai.embeddings.create(model=..., input=...)
    - Voyage AI client: vo.embed(texts, model=..., input_type="query")
    - Elasticsearch inference service type: openai
    """
    raw_inputs = request.input if isinstance(request.input, list) else [request.input]
    task = request.task
    if request.input_type:
        task = VOYAGE_TASK_MAP.get(request.input_type, "retrieval")

    if any(not isinstance(item, str) for item in raw_inputs):
        engine.require_multimodal()
        items = []
        for raw in raw_inputs:
            parts = parse_openai_item(raw)
            items.append(parts if len(parts) > 1 else parts[0])
    else:
        items = raw_inputs

    embeddings, n_tokens, tok_per_s = engine.embed(
        items, task=task, dimensions=request.dimensions or request.output_dimension
    )

    return OpenAIEmbeddingResponse(
        data=[
            {"object": "embedding", "embedding": embedding.tolist(), "index": i}
            for i, embedding in enumerate(embeddings)
        ],
        model=settings.short_model_id,
        usage={
            "prompt_tokens": n_tokens,
            "total_tokens": n_tokens,
            "tok_per_s": round(tok_per_s, 1),
        },
    )


# =============================================================================
# Schema 2: Cohere  (POST /v1/embed)
#
# Multimodal extensions:
# - Legacy: {"texts": [...], "images": ["data:image/png;base64,..."]}
# - V2:     {"inputs": [{"content": [{"type":"image_url","image_url":{"url":"data:..."}}, ...]}]}
# =============================================================================


class CohereEmbedRequest(BaseModel):
    texts: Optional[list] = None
    images: Optional[list] = None  # legacy: list of data-URL strings
    inputs: Optional[list] = None  # V2: content-block array
    model: Optional[str] = None
    input_type: Optional[str] = "search_document"
    truncate: Optional[str] = "END"
    embedding_types: Optional[list] = Field(default_factory=lambda: ["float"])


COHERE_TASK_MAP = {
    "search_query": "retrieval.query",
    "search_document": "retrieval.passage",
    "classification": "classification",
    "clustering": "clustering",
}


@app.post("/v1/embed", tags=["Cohere"])
async def create_embeddings_cohere(request: CohereEmbedRequest):
    """
    Cohere-compatible embedding endpoint.

    Text-only:
      {"texts": [...], "input_type": "search_document"}

    Legacy multimodal:
      {"images": ["data:image/png;base64,..."], "input_type": "search_document"}

    V2 multimodal:
      {"inputs": [{"content": [{"type": "image_url", "image_url": {"url": "data:..."}}, {"type": "text", "text": "..."}]}]}
    """
    task = COHERE_TASK_MAP.get(request.input_type or "search_document", "retrieval")

    if request.inputs is not None:
        engine.require_multimodal()
        items = [_fuse_content(inp.get("content", [])) for inp in request.inputs]
        texts_out = []
        images_count = len(items)
    elif request.images is not None:
        engine.require_multimodal()
        items = [bytes_to_st_input(*decode_b64(url)) for url in request.images]
        texts_out = request.texts or []
        images_count = len(request.images)
    else:
        items = request.texts or []
        texts_out = items
        images_count = 0

    embeddings, n_tokens, tok_per_s = engine.embed(items, task=task)

    return {
        "id": str(uuid.uuid4()),
        "texts": texts_out,
        "embeddings": {
            "float": [embedding.tolist() for embedding in embeddings],
        },
        "meta": {
            "api_version": {"version": "2"},
            "billed_units": {
                "input_tokens": n_tokens,
                "images": images_count,
            },
            "tok_per_s": round(tok_per_s, 1),
        },
        "response_type": "embeddings_floats",
    }


# =============================================================================
# Schema 3: Voyage AI Multimodal  (POST /v1/multimodalembeddings)
#
# Format:
# {
#   "inputs": [
#     {"content": [
#       {"type": "text", "text": "..."},
#       {"type": "image_base64", "image_base64": "data:image/jpeg;base64,..."},
#       {"type": "video_base64", "video_base64": "data:video/mp4;base64,..."}
#     ]}
#   ],
#   "model": "voyage-multimodal-3.5",
#   "input_type": "document"
# }
# =============================================================================


class VoyageMultimodalRequest(BaseModel):
    inputs: list
    model: Optional[str] = None
    input_type: Optional[str] = None
    truncation: Optional[bool] = True


@app.post("/v1/multimodalembeddings", tags=["Voyage AI Multimodal"])
async def create_multimodal_embeddings_voyage(request: VoyageMultimodalRequest):
    """
    Voyage AI-compatible multimodal embedding endpoint.

    Each input is {"content": [text/image/video/audio parts]}.
    All parts within one input are fused into a single embedding.

    Compatible with the Voyage AI multimodalembeddings REST API.
    """
    engine.require_multimodal()

    task = {"query": "retrieval.query", "document": "retrieval.passage"}.get(
        request.input_type or "", "retrieval"
    )
    items = [_fuse_content(inp.get("content", [])) for inp in request.inputs]
    embeddings, n_tokens, _ = engine.embed(items, task=task)

    return {
        "embeddings": [embedding.tolist() for embedding in embeddings],
        "text_tokens": n_tokens,
        "image_pixels": 0,  # not tracked
        "total_tokens": n_tokens,
        "model": settings.short_model_id,
    }


# =============================================================================
# Schema 4: Google Gemini  (POST /v1/models/{model}:embedContent)
#
# Multimodal: parts may include inlineData in addition to text:
#   {"inlineData": {"mimeType": "image/png", "data": "base64..."}}
# =============================================================================


class GeminiPart(BaseModel):
    text: Optional[str] = None
    inlineData: Optional[dict] = None  # {"mimeType": "image/png", "data": "base64..."}


class GeminiContent(BaseModel):
    parts: list  # list of GeminiPart-like dicts
    role: Optional[str] = None


class GeminiEmbedContentRequest(BaseModel):
    content: GeminiContent
    taskType: Optional[str] = None
    title: Optional[str] = None
    outputDimensionality: Optional[int] = None


class GeminiBatchEmbedRequest(BaseModel):
    requests: list  # list of GeminiEmbedContentRequest-like dicts


GEMINI_TASK_MAP = {
    "RETRIEVAL_QUERY": "retrieval.query",
    "RETRIEVAL_DOCUMENT": "retrieval.passage",
    "SEMANTIC_SIMILARITY": "text-matching",
    "CLASSIFICATION": "classification",
    "CLUSTERING": "clustering",
    "QUESTION_ANSWERING": "retrieval.query",
    "FACT_VERIFICATION": "retrieval.query",
    "CODE_RETRIEVAL_QUERY": "retrieval.query",
}


def _fuse_content(content: list) -> Union[str, list]:
    """Flatten a content-part array into one fused embedding input."""
    parts = []
    for part in content:
        parts.extend(parse_content_part(part))
    if len(parts) > 1:
        return parts
    return parts[0] if parts else ""


def _parse_gemini_content(content_obj) -> list:
    """
    Parse a Gemini content object into ST-compatible inputs.
    Returns a list of parts (fused if multiple).
    """
    parts_raw = (
        content_obj if isinstance(content_obj, list) else content_obj.get("parts", [])
    )
    result = []
    for part in parts_raw:
        if isinstance(part, dict):
            if "text" in part and "inlineData" not in part:
                result.append(part["text"])
            else:
                result.extend(parse_content_part(part))
        elif hasattr(part, "text") and part.text is not None:
            result.append(part.text)
        elif hasattr(part, "inlineData") and part.inlineData is not None:
            result.extend(parse_content_part({"inlineData": part.inlineData}))
    return result


@app.post("/v1/models/{model_id}:embedContent", tags=["Google Gemini"])
async def embed_content_gemini(
    model_id: str,
    request: GeminiEmbedContentRequest,
):
    """
    Google Gemini-compatible single embedding endpoint.
    Supports text parts and inlineData (image/audio/video) parts.

    Compatible with:
    - genai.embed_content(model=..., content=..., task_type="RETRIEVAL_QUERY")
    """
    task = GEMINI_TASK_MAP.get(request.taskType or "", "retrieval")
    parts = _parse_gemini_content(request.content.dict())

    if any(not isinstance(part, str) for part in parts):
        engine.require_multimodal()
        item = parts if len(parts) > 1 else parts[0]
    else:
        item = " ".join(str(part) for part in parts)

    embeddings, n_tokens, tok_per_s = engine.embed(
        [item], task=task, dimensions=request.outputDimensionality
    )

    return {
        "embedding": {
            "values": embeddings[0].tolist(),
        },
        "metadata": {
            "tokenCount": n_tokens,
            "tok_per_s": round(tok_per_s, 1),
        },
    }


@app.post("/v1/models/{model_id}:batchEmbedContents", tags=["Google Gemini"])
async def batch_embed_contents_gemini(
    model_id: str,
    request: GeminiBatchEmbedRequest,
):
    """
    Google Gemini-compatible batch embedding endpoint.
    Supports text and inlineData (multimodal) parts.

    Compatible with:
    - genai.batch_embed_contents(requests=[...])
    """
    items = []
    dimensions = None
    has_multimodal = False
    first_task_type = ""

    for sub_request in request.requests:
        if not dimensions and sub_request.get("outputDimensionality"):
            dimensions = sub_request["outputDimensionality"]
        if not first_task_type:
            first_task_type = sub_request.get("taskType", "")

        parts = _parse_gemini_content(sub_request.get("content", {}))
        if any(not isinstance(part, str) for part in parts):
            has_multimodal = True
        items.append(parts if len(parts) > 1 else (parts[0] if parts else ""))

    task = GEMINI_TASK_MAP.get(first_task_type, "retrieval")

    if has_multimodal:
        engine.require_multimodal()
    else:
        # Text-only: a multi-part content is joined rather than fused.
        items = [
            item if isinstance(item, str) else " ".join(str(p) for p in item)
            for item in items
        ]

    embeddings, n_tokens, tok_per_s = engine.embed(
        items, task=task, dimensions=dimensions
    )

    return {
        "embeddings": [{"values": embedding.tolist()} for embedding in embeddings],
        "metadata": {
            "tokenCount": n_tokens,
            "tok_per_s": round(tok_per_s, 1),
        },
    }


# =============================================================================
# Reranker endpoint (bonus - for reranker models)
# =============================================================================


class RerankRequest(BaseModel):
    query: str
    documents: list
    model: Optional[str] = None
    top_n: Optional[int] = None
    return_documents: Optional[bool] = True


@app.post("/v1/rerank", tags=["Reranker"])
async def rerank(request: RerankRequest):
    """OpenAI-style rerank endpoint (for reranker models)."""
    results, _n_tokens, elapsed = engine.rerank(
        request.query,
        request.documents,
        top_n=request.top_n,
        return_documents=request.return_documents,
    )
    return {
        "model": settings.short_model_id,
        "results": results,
        "meta": {"elapsed_ms": round(elapsed * 1000, 1)},
    }


# =============================================================================
# Schema 6: OpenAI Chat Completions (POST /v1/chat/completions)
#
# For VLM / reader models. Supports OpenAI message format with optional images:
#   messages = [{"role": "user", "content": "hello"}]                          (text-only)
#   messages = [{"role": "user", "content": [                                  (vision)
#       {"type": "text", "text": "What is in this image?"},
#       {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
#   ]}]
# =============================================================================


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


def _parse_chat_message(msg: dict) -> tuple:
    """Convert one OpenAI-style message to (role, content_parts, images).

    content_parts: list of {"type": "text"|"image", ...} for processor.apply_chat_template
    images: list of PIL.Image for processor(images=...)
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        return msg.get("role", "user"), [{"type": "text", "text": content}], []
    if not isinstance(content, list):
        raise BadRequest("message.content must be a string or list of parts")

    parts, images = [], []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type", "")
        if kind == "text":
            parts.append({"type": "text", "text": part.get("text", "")})
            continue
        if kind == "image_url":
            url = part.get("image_url", {})
            if isinstance(url, dict):
                url = url.get("url", "")
            if not isinstance(url, str):
                raise BadRequest("image_url.url must be a string")
            if not url.startswith("data:"):
                raise BadRequest(
                    "image_url.url must be a base64 data URL "
                    "(offline server cannot fetch http(s) URLs)"
                )
        elif kind == "image":
            # Elastic-style {"type":"image","format":"base64","value":"..."}
            if part.get("format") != "base64":
                raise BadRequest("image part requires format=base64")
            url = part.get("value", "")
        else:
            raise BadRequest(f"Unsupported content part type: {kind!r}")
        image = bytes_to_st_input(*decode_b64(url))
        parts.append({"type": "image", "image": image})
        images.append(image)
    return msg.get("role", "user"), parts, images


@app.post("/v1/chat/completions", tags=["OpenAI Chat"])
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint (VLM / text-chat / reader models)."""
    engine.require_chat_model()
    if request.stream:
        raise BadRequest("stream=true is not supported in this build")
    if not request.messages:
        raise BadRequest("messages must be a non-empty list")

    conversation, images = [], []
    for msg in request.messages:
        if not isinstance(msg, dict):
            raise BadRequest("each message must be a dict")
        role, parts, message_images = _parse_chat_message(msg)
        conversation.append({"role": role, "content": parts})
        images.extend(message_images)

    text, prompt_tokens, completion_tokens = engine.generate(
        conversation,
        images,
        request.max_tokens,
        request.temperature,
        request.top_p,
    )

    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": settings.short_model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
