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

import base64
import io
import os
import json
import sys
import time
import logging
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Union

# Register dynamic model module directories so transformers check_imports
# can find sibling modules (e.g. configuration_eurobert imported by
# modeling_jina_embeddings_v5). Without this, air-gapped containers fail
# because importlib.util.find_spec cannot locate these modules.
_hf_modules = Path(os.environ.get("HF_HOME", "")) / "modules" / "transformers_modules"
if _hf_modules.is_dir():
    for _commit_dir in _hf_modules.rglob("*.py"):
        _parent = str(_commit_dir.parent)
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
    del _commit_dir, _parent
del _hf_modules

import torch
import numpy as np

# Patch transformers for air-gapped Jina model loading.
# 1. resolve_trust_remote_code: force True so custom_st.py AutoConfig/AutoModel
#    calls don't prompt for interactive confirmation in offline containers.
# 2. add_generation_mixin_to_remote_model: guard against models without
#    prepare_inputs_for_generation (embedding-only models like JinaEmbeddingsV5).
# 3. _patch_mistral_regex (transformers >=4.57.0): unconditionally calls
#    huggingface_hub.model_info(repo_id) to sniff base-mistral tokenizers by tag,
#    which raises OfflineModeIsEnabled in an air-gapped container loading by repo
#    id (e.g. jina-reranker-v3.5, whose custom rerank() re-loads its tokenizer by
#    name). No Jina model is a base-mistral model, so when offline we skip the
#    probe and return the tokenizer unchanged — the regex fix is a no-op for
#    non-mistral tokenizers anyway.
try:
    from transformers import dynamic_module_utils as _dmu
    _orig_resolve = _dmu.resolve_trust_remote_code
    def _always_trust(*args, **kwargs):
        return True
    _dmu.resolve_trust_remote_code = _always_trust
except Exception:
    pass
try:
    from transformers.models.auto import auto_factory as _af
    _orig_add_gen = getattr(_af, 'add_generation_mixin_to_remote_model', None)
    if _orig_add_gen:
        def _safe_add_gen(model_class):
            if not hasattr(model_class, 'prepare_inputs_for_generation'):
                return model_class
            return _orig_add_gen(model_class)
        _af.add_generation_mixin_to_remote_model = _safe_add_gen
except Exception:
    pass
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTTB
    _orig_patch_mistral = getattr(_PTTB, "_patch_mistral_regex", None)
    if _orig_patch_mistral is not None:
        def _safe_patch_mistral(tokenizer, *args, **kwargs):
            if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
                return tokenizer
            return _orig_patch_mistral(tokenizer, *args, **kwargs)
        _PTTB._patch_mistral_regex = staticmethod(_safe_patch_mistral)
except Exception:
    pass

# Boost matmul precision: uses TF32 on Ampere/Ada/Hopper GPUs, ~1.2x faster on L4
torch.set_float32_matmul_precision('high')
from fastapi import FastAPI, HTTPException, Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config from env ---
MODEL_ID = os.environ.get("JINA_MODEL_ID", "")
# Short form used in API responses (matches catalog id, what users typed in `model` field).
# MODEL_ID stays as the full HF path because it's the on-disk model directory under
# the HF cache, used by from_pretrained() during load_model().
SHORT_MODEL_ID = MODEL_ID.split("/")[-1] if MODEL_ID else ""
if os.environ.get("JINA_OFFLINE", "0") == "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Detect device
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    DEVICE = "cpu"
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
logger.info(f"Using device: {DEVICE}")

# --- Runtime optimizations (applied at load time) ---
# CPU: use all physical cores (0 = auto-detect from env or os.cpu_count)
_omp_env = os.environ.get("OMP_NUM_THREADS", "")
_n_cpu_threads = int(_omp_env) if _omp_env and _omp_env != "0" else (os.cpu_count() or 4)
torch.set_num_threads(_n_cpu_threads)
torch.set_num_interop_threads(max(1, _n_cpu_threads // 2))


def _cpu_supports_bf16() -> bool:
    """True if the CPU has native bf16 matmul (AMX on Sapphire Rapids+, or
    AVX512_BF16 on Cooper Lake / Zen4). On these, bf16 autocast is a 1.8-2.5x
    throughput win on encoders at batch with cos-sim ~0.9999 vs fp32 (measured
    on jina-embeddings-v5-text-nano). Without the flag, bf16 matmul is emulated
    and can be slower, so we keep fp32 there."""
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
        return ("amx_bf16" in flags) or ("avx512_bf16" in flags)
    except Exception:
        return False


def _finalize_cpu_bf16(model):
    """Decide whether to use CPU bf16 autocast, called once after the model
    loads. Sets the module-level _CPU_BF16. In "auto" mode (default), enables
    bf16 only when the CPU has native bf16 support AND a probe encode succeeds
    with cos-sim >= 0.99 vs fp32 on this specific model -- so models that error
    or degrade under bf16 (e.g. xlm-roberta-flash rotary) transparently keep
    fp32. JINA_CPU_AUTOCAST=bf16/off bypasses the probe."""
    global _CPU_BF16
    if DEVICE != "cpu":
        return
    if _cpu_autocast_env in ("off", "0", "fp32", "float32"):
        _CPU_BF16 = False
        logger.info("CPU bf16 autocast: disabled (JINA_CPU_AUTOCAST=off)")
        return
    forced = _cpu_autocast_env in ("bf16", "bfloat16", "on", "1")
    if not forced and not _cpu_supports_bf16():
        _CPU_BF16 = False
        logger.info("CPU bf16 autocast: disabled (no native bf16: needs AMX or AVX512_BF16)")
        return
    if forced:
        _CPU_BF16 = True
        logger.info("CPU bf16 autocast: ENABLED (JINA_CPU_AUTOCAST=bf16, probe skipped)")
        return
    # auto + capable hardware: probe this model for correctness + accuracy.
    if model is None or not hasattr(model, "encode"):
        _CPU_BF16 = False
        logger.info("CPU bf16 autocast: disabled (model not probe-compatible; force with JINA_CPU_AUTOCAST=bf16)")
        return
    probe = ["The quick brown fox jumps over the lazy dog.",
             "Semantic search finds relevant content by meaning."]
    base_kw = dict(convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    # Find an encode signature that works in fp32. Task names differ across the
    # fleet (v5: "retrieval", v3: "retrieval.passage", omni: default_task set at
    # load so no kwarg). Only "task must be specified"-type errors advance the
    # fallback; a RuntimeError here is a real failure and aborts the probe.
    a, chosen_kw = None, None
    for _task in ("retrieval", "retrieval.passage", None):
        kw = dict(base_kw)
        if _task is not None and MODEL_ACCEPTS_TASK_KWARG:
            kw["task"] = _task
        try:
            with torch.inference_mode():
                a = model.encode(probe, **kw)
            chosen_kw = kw
            break
        except (ValueError, TypeError):
            continue
        except Exception as e:
            logger.info(f"CPU bf16 autocast: disabled (fp32 probe encode failed: {type(e).__name__})")
            _CPU_BF16 = False
            return
    if chosen_kw is None:
        _CPU_BF16 = False
        logger.info("CPU bf16 autocast: disabled (could not find a working probe encode signature)")
        return
    # Same signature under bf16 autocast: must not error and must stay accurate.
    try:
        import numpy as _np
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            b = model.encode(probe, **chosen_kw)
        cos = float(_np.mean(_np.sum(a * b, axis=1) /
                             (_np.linalg.norm(a, axis=1) * _np.linalg.norm(b, axis=1) + 1e-9)))
        if cos >= 0.99:
            _CPU_BF16 = True
            logger.info(f"CPU bf16 autocast: ENABLED (probe cos-sim={cos:.5f})")
        else:
            _CPU_BF16 = False
            logger.info(f"CPU bf16 autocast: disabled (probe cos-sim={cos:.5f} < 0.99, keeping fp32)")
    except Exception as e:
        _CPU_BF16 = False
        logger.info(f"CPU bf16 autocast: disabled (model errors under bf16: {type(e).__name__})")


# CPU bf16 autocast. Default "auto": a load-time probe (see _finalize_cpu_bf16)
# enables it only when the CPU has native bf16 AND the loaded model is both
# error-free and accuracy-preserving under bf16. This is necessary because the
# trick is not universal: EuroBERT (v5) gets ~2-2.5x lossless, but the
# xlm-roberta-flash family (v3, clip text tower) raises an index_put dtype
# mismatch in its rotary path under autocast. JINA_CPU_AUTOCAST=bf16 forces it
# on (skip probe), =off forces it off. No effect on GPU/MPS.
_cpu_autocast_env = os.environ.get("JINA_CPU_AUTOCAST", "auto").lower()
# Final value is decided in _finalize_cpu_bf16() once the model is loaded.
_CPU_BF16 = False

# GPU: enable cuDNN auto-tuner and TF32 matmul (no effect on CPU/MPS)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


def _encode_autocast_ctx():
    """Autocast context for embedding/reranker encode().

    Honours JINA_DTYPE: float32 → no autocast (some models — notably jinaai/jina-bert
    variants with ALiBi — produce NaN under fp16 due to attention-score overflow).
    """
    if DEVICE == "cpu":
        # bf16 has fp32's exponent range, so no ALiBi/fp16-style overflow NaN risk.
        return torch.autocast("cpu", dtype=torch.bfloat16) if _CPU_BF16 else nullcontext()
    if DEVICE != "cuda":
        return nullcontext()
    _dtype = os.environ.get("JINA_DTYPE", "float16").lower()
    if _dtype in ("float16", "fp16", "half"):
        return torch.autocast("cuda", dtype=torch.float16)
    if _dtype in ("bfloat16", "bf16"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()

app = FastAPI(
    title="Jina AI On-Prem Server",
    version="4.0.0",
    description="Multi-schema embedding server: OpenAI, Voyage AI, Gemini, Cohere (text + multimodal)",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
import license as _license

LICENSE_KEY = os.environ.get("JINA_LICENSE_KEY", "")
_LICENSE_OPEN_PATHS = {"/health", "/", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}
_license_warned = False  # log the warn-mode notice once, not per request


@app.middleware("http")
async def _license_gate(request, call_next):
    # Only inference-type POSTs are ever gated; reads/probes always pass.
    if request.method in ("GET", "OPTIONS", "HEAD") or request.url.path in _LICENSE_OPEN_PATHS:
        return await call_next(request)

    d = _license.decide(LICENSE_KEY, MODEL_ID)

    # Fail-open path (warn/off, or enforce-within-grace): serve, but surface a
    # single warning line so operators notice a lapsed/absent key.
    if d["allow"]:
        if d["mode"] == "warn" and d["reason"] not in ("ok",):
            global _license_warned
            if not _license_warned:
                logger.warning(
                    "License check (warn mode, fail-open): reason=%s - serving anyway. "
                    "Set JINA_LICENSE_MODE=enforce only for trials/POCs.", d["reason"]
                )
                _license_warned = True
        elif d["reason"] == "expired_in_grace":
            logger.warning("License expired but within grace window - still serving. Renew soon.")
        return await call_next(request)

    # Blocking path: enforce mode only.
    from fastapi.responses import JSONResponse
    hint = {
        "no_license": "Set a license key: docker run -e JINA_LICENSE_KEY=<key> ... (or switch to warn mode).",
        "license_expired": "License expired past its grace window. Request a renewed key (no rebuild needed).",
        "model_not_licensed": f"Key not valid for model '{SHORT_MODEL_ID}'.",
        "bad_signature": "License key signature invalid.",
        "malformed_license": "License key is malformed.",
    }.get(d["reason"], "License validation failed.")
    return JSONResponse(
        status_code=403,
        content={"error": {"code": d["reason"], "message": hint, "type": "license_error"}},
    )

# --- Global model holder ---
MODEL = None
TOKENIZER = None
PROCESSOR = None  # VLM processor (AutoProcessor); None for embedding/reranker models
MODEL_INFO = {}
# True if MODEL.encode() accepts a `task` kwarg (v3/v4/v5/code-embeddings/clip have
# custom_st.py that defines it; v1/v2 use stock SentenceTransformer.encode which does not).
MODEL_ACCEPTS_TASK_KWARG = False

# --- Throughput stats (thread-safe) ---
_stats_lock = threading.Lock()
_stats = {
    "total_requests": 0,
    "total_tokens": 0,
    "total_latency_s": 0.0,
    "last_tok_per_s": 0.0,
    "peak_tok_per_s": 0.0,
}

# --- Multimodal config ---
# Models that support image / audio / video inputs
MULTIMODAL_MODEL_IDS = {
    "jina-embeddings-v5-omni-small",
    "jina-embeddings-v5-omni-nano",
    "jina-embeddings-v4",
    "jina-clip-v2",
    "jina-clip-v1",
    "jina-reranker-m0",
    "jina-vlm",
}

# VLM (vision-language) models served via /v1/chat/completions.
# These bypass SentenceTransformer and load AutoProcessor + AutoModelForCausalLM directly.
VLM_MODEL_IDS = {
    "jina-vlm",
}

# Text-only chat models served via /v1/chat/completions. No vision tower, no
# AutoProcessor — load AutoTokenizer + AutoModelForCausalLM directly.
TEXT_CHAT_MODEL_IDS = {
    "reader-lm-0.5b",
    "reader-lm-1.5b",
    "ReaderLM-v2",
    "readerlm-v2",
}

MAX_MEDIA_BYTES = 10 * 1024 * 1024  # 10 MB per input

IMAGE_MIMES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif",
    "image/bmp", "image/tiff", "image/avif", "image/heic", "image/svg+xml",
}
AUDIO_MIMES = {
    "audio/wav", "audio/x-wav", "audio/mp3", "audio/mpeg", "audio/flac",
    "audio/ogg", "audio/m4a", "audio/x-m4a", "audio/opus",
}
VIDEO_MIMES = {
    "video/mp4", "video/avi", "video/x-msvideo", "video/quicktime",
    "video/x-matroska", "video/webm", "video/x-flv", "video/x-ms-wmv",
}


# =============================================================================
# Multimodal helpers
# =============================================================================

def _is_omni_model() -> bool:
    """True if the loaded model supports multimodal (image/audio/video) inputs."""
    short_id = MODEL_ID.split("/")[-1] if MODEL_ID else ""
    return short_id in MULTIMODAL_MODEL_IDS


def _is_v5_omni_text_model() -> bool:
    """True only for v5-omni-nano / v5-omni-small.

    These two models share a custom_st module that picks the LoRA adapter via
    a ``default_task`` attribute on the module rather than accepting ``task=``
    via encode kwargs. Other multimodal models (v4, clip-v1, clip-v2,
    reranker-m0) route task differently and must not take this branch.
    """
    short_id = MODEL_ID.split("/")[-1] if MODEL_ID else ""
    return short_id in {"jina-embeddings-v5-omni-nano", "jina-embeddings-v5-omni-small"}


def _default_task() -> str:
    """Default task when the caller omits ``task`` from the request.

    Matches prod ``/v1/embeddings`` behaviour (probed against api.jina.ai):

      - v5-text-nano, v5-text-small, omni-nano, omni-small, v4 ->
        ``text-matching`` (prod no-task == prod task=text-matching at cos
        1.0000 for v5-text/omni-nano, 0.9998 for omni-small/v4).
      - code-embeddings (0.5b / 1.5b) -> ``nl2code.query`` (prod no-task
        == prod task=nl2code.query at cos 1.0000).
      - everything else -> ``retrieval`` (back-compat for v3 / v1 / v2 / b-en-v1).

    Without this alignment, no-task local vs prod cos for the affected
    families sits at 0.70 – 0.90 instead of >0.99. (v5-text was previously
    missing here and defaulted to ``retrieval``, giving cos ~0.69 vs prod.)
    """
    short_id = MODEL_ID.split("/")[-1] if MODEL_ID else ""
    if short_id in {
        "jina-embeddings-v5-text-nano",
        "jina-embeddings-v5-text-small",
        "jina-embeddings-v5-omni-nano",
        "jina-embeddings-v5-omni-small",
        "jina-embeddings-v4",
    }:
        return "text-matching"
    if "code-embeddings" in short_id:
        return "nl2code.query"
    return "retrieval"


def _map_prompt_name(task: str, prompts: Optional[dict]) -> Optional[str]:
    """Pick the matching ST prompt_name for ``task`` from the model's prompts dict.

    Different model families use different key conventions:
      - v4: ``{"query": ..., "passage": ...}``
      - v5-omni: ``{"query": ..., "document": ...}``
      - code-embeddings: ``{"nl2code_query": ..., "nl2code_document": ...,
        "qa_query": ..., "code2code_query": ..., ...}``

    Mapping rules (data-driven against the model's actual prompts dict so we
    never pass a key that would raise
    ``ValueError: Prompt name 'X' not found in...``):

      - ``{base}.query`` -> first hit of ``{base}_query``, ``query``
      - ``{base}.passage`` -> first hit of ``{base}_document``, ``document``,
        ``passage``
      - no suffix (bare task like ``text-matching``, ``classification``,
        ``retrieval``): prod uses each model's own canonical encode default,
        which differs by family — v5-omni's ``JinaEmbeddingsV5OmniModel.encode``
        defaults ``prompt_name="document"`` (prepends ``"Document: "``);
        v4's ``JinaEmbeddingsV4Model.encode`` defaults to ``"query"``
        (prepends ``"Query: "``). We mirror that fallback when the model
        prompts dict has the matching key. Code-embeddings has neither key,
        return None.
    """
    if not prompts or not task:
        return None
    # v3 keys its prompts dict by the task name itself (e.g.
    # ``"retrieval.passage": "Represent the document for retrieval: "``) rather
    # than the standard ST ``query``/``passage`` keys. Try the literal task
    # first; for other families this is a no-op because their task names — e.g.
    # ``"retrieval.passage"`` for v4, ``"nl2code.query"`` for code-embeddings —
    # are not present as prompts keys, so the suffix logic below still applies.
    if task in prompts:
        return task
    base, _, suffix = task.partition(".")
    if suffix == "query":
        for cand in (f"{base}_query", "query"):
            if cand in prompts:
                return cand
        return None
    if suffix == "passage":
        for cand in (f"{base}_document", "document", "passage"):
            if cand in prompts:
                return cand
        return None
    if "document" in prompts:
        return "document"
    if "query" in prompts:
        return "query"
    return None


def _is_vlm_model() -> bool:
    """True if the loaded model is a vision-language model served via /v1/chat/completions."""
    short_id = MODEL_ID.split("/")[-1] if MODEL_ID else ""
    return short_id in VLM_MODEL_IDS


def _is_text_chat_model() -> bool:
    """True if the loaded model is a text-only chat/reader model served via /v1/chat/completions."""
    short_id = MODEL_ID.split("/")[-1] if MODEL_ID else ""
    return short_id in TEXT_CHAT_MODEL_IDS


def _is_chat_model() -> bool:
    """True for any model served via /v1/chat/completions (VLM or text-only chat)."""
    return _is_vlm_model() or _is_text_chat_model()


def _require_omni():
    """Raise HTTP 400 if the current model does not support multimodal inputs."""
    if not _is_omni_model():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{SHORT_MODEL_ID}' is text-only and does not accept image/audio/video inputs. "
                f"Use one of: {sorted(MULTIMODAL_MODEL_IDS)}"
            ),
        )


def _decode_b64(b64_str: str) -> tuple:
    """
    Decode a base64 string in raw or data-URL format.
    Returns (raw_bytes, mime_type).

    Accepts:
    - Raw base64: "iVBORw0KGgo..."
    - Data URL:   "data:image/png;base64,iVBORw0KGgo..."
    """
    mime_type = ""
    data = b64_str.strip()
    if data.startswith("data:"):
        try:
            header, data = data.split(",", 1)
            mime_type = header.split(";")[0][5:]  # strip "data:"
        except ValueError:
            raise HTTPException(status_code=400, detail="Malformed data URL")
    try:
        raw = base64.b64decode(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")
    if len(raw) > MAX_MEDIA_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Media too large: {len(raw):,} bytes exceeds {MAX_MEDIA_BYTES:,} byte limit",
        )
    return raw, mime_type


def _detect_mime(raw: bytes, hint: str = "") -> str:
    """Detect MIME type from magic bytes; fall back to hint."""
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:2] == b"\xff\xd8":
        return "image/jpeg"
    if len(raw) >= 6 and raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "audio/wav"
    if len(raw) >= 3 and raw[:3] == b"ID3":
        return "audio/mp3"
    if len(raw) >= 4 and raw[:4] == b"fLaC":
        return "audio/flac"
    if len(raw) >= 12 and b"ftyp" in raw[4:12]:
        return "video/mp4"
    if len(raw) >= 4 and raw[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    return hint


def _bytes_to_st_input(raw: bytes, mime_hint: str = ""):
    """
    Convert raw bytes to the input type expected by sentence-transformers encode().
    - image/* -> PIL.Image.Image
    - audio/* or video/* -> io.BytesIO (sentence-transformers accepts bytes/BytesIO)
    """
    mime = (_detect_mime(raw, mime_hint) or mime_hint).lower()

    if mime in IMAGE_MIMES:
        from PIL import Image
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
            return img
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    if mime in AUDIO_MIMES or mime in VIDEO_MIMES:
        return io.BytesIO(raw)

    # Unknown MIME: attempt image decode, fall back to BytesIO
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img
    except Exception:
        return io.BytesIO(raw)


def _parse_typed_base64(item: dict, key: str, default_mime: str) -> list:
    """Parse image_base64 / audio_base64 / video_base64 typed fields."""
    inner = item.get(key, "")
    if isinstance(inner, str):
        raw, mime = _decode_b64(inner)
        mime = mime or default_mime
    elif isinstance(inner, dict):
        raw, mime = _decode_b64(inner.get("base64", inner.get("data", "")))
        mime = mime or inner.get("mime_type", inner.get("mimeType", default_mime))
    else:
        raise HTTPException(status_code=400, detail=f"Invalid value for '{key}'")
    return [_bytes_to_st_input(raw, mime)]


def _parse_content_part(part: dict) -> list:
    """
    Parse a single content-array part into a list of ST-compatible inputs.

    Handles:
    - {"type": "text", "text": "..."}
    - {"type": "image", "format": "base64", "value": "..."}         (Elastic format)
    - {"type": "image_url", "image_url": {"url": "data:..."}}       (Cohere/Voyage)
    - {"type": "image_base64", "image_base64": "data:..." | {...}}
    - {"type": "audio_base64", "audio_base64": "data:..." | {...}}
    - {"type": "video_base64", "video_base64": "data:..." | {...}}
    - {"inlineData": {"mimeType": "...", "data": "..."}}             (Gemini format)
    """
    if not isinstance(part, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Content part must be a dict, got {type(part).__name__}",
        )

    t = part.get("type", "")

    if t == "text":
        return [part.get("text", "")]

    # Elastic Inference Service: {"type": "image", "format": "base64", "value": "..."}
    if t == "image" and part.get("format") == "base64":
        raw, mime = _decode_b64(part["value"])
        return [_bytes_to_st_input(raw, mime or "image/jpeg")]

    # Cohere/Voyage: {"type": "image_url", "image_url": {"url": "data:..."}}
    if t == "image_url":
        url_val = part.get("image_url", {})
        url = url_val.get("url", url_val) if isinstance(url_val, dict) else str(url_val)
        if not url.startswith("data:"):
            raise HTTPException(
                status_code=400,
                detail="image_url: only data: URLs (base64) are supported in air-gapped mode",
            )
        raw, mime = _decode_b64(url)
        return [_bytes_to_st_input(raw, mime)]

    # Gemini: {"inlineData": {"mimeType": "...", "data": "..."}}
    if "inlineData" in part:
        inline = part["inlineData"]
        raw, _ = _decode_b64(inline.get("data", ""))
        return [_bytes_to_st_input(raw, inline.get("mimeType", ""))]

    # Typed base64 formats
    if t == "image_base64" or "image_base64" in part:
        return _parse_typed_base64(part, "image_base64", "image/jpeg")
    if t == "audio_base64" or "audio_base64" in part:
        return _parse_typed_base64(part, "audio_base64", "audio/wav")
    if t == "video_base64" or "video_base64" in part:
        return _parse_typed_base64(part, "video_base64", "video/mp4")

    raise HTTPException(status_code=400, detail=f"Unknown content part type: '{t}'")


def _parse_openai_item(item) -> list:
    """
    Parse one element of the OpenAI `input` array into a list of ST-compatible inputs.

    Returns a list:
    - [str]          -> plain text (len 1)
    - [PIL.Image]    -> single image (len 1)
    - [BytesIO]      -> single audio/video (len 1)
    - [x, y, ...]    -> fused multimodal parts (len > 1, caller wraps in tuple)

    Supported formats:
    1. "plain text"
    2. {"type": "text", "text": "..."}
    3. {"type": "image", "format": "base64", "value": "..."}                    (Elastic)
    4. {"type": "image_base64", "image_base64": {"base64": "...", "mime_type": "..."}}
    5. {"type": "audio_base64", "audio_base64": {"base64": "...", "mime_type": "..."}}
    6. {"type": "video_base64", "video_base64": {"base64": "...", "mime_type": "..."}}
    7. {"content": [...]}  fused multimodal block -> parts merged into ONE embedding
    """
    if isinstance(item, str):
        return [item]

    if not isinstance(item, dict):
        raise HTTPException(
            status_code=400,
            detail=f"Input item must be str or dict, got {type(item).__name__}",
        )

    # Fused content block: {"content": [...]}
    if "content" in item and isinstance(item["content"], list):
        parts = []
        for p in item["content"]:
            parts.extend(_parse_content_part(p))
        return parts

    # Single-part item: delegate to content-part parser
    return _parse_content_part(item)


# =============================================================================
# Core embedding (text-only and multimodal)
# =============================================================================

def _update_stats(n_tokens: int, elapsed_s: float) -> float:
    tok_per_s = n_tokens / elapsed_s if elapsed_s > 0 else 0.0
    with _stats_lock:
        _stats["total_requests"] += 1
        _stats["total_tokens"] += n_tokens
        _stats["total_latency_s"] += elapsed_s
        _stats["last_tok_per_s"] = tok_per_s
        if tok_per_s > _stats["peak_tok_per_s"]:
            _stats["peak_tok_per_s"] = tok_per_s
    return tok_per_s


def _count_tokens(texts: list) -> int:
    if TOKENIZER is not None:
        try:
            enc = TOKENIZER(texts, add_special_tokens=True)
            return sum(len(ids) for ids in enc["input_ids"])
        except Exception:
            pass
    return sum(len(t.split()) for t in texts)


def _resolve_task(task: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Return (task_for_encode, prompt_name) for the loaded model.

    The API-level task string carries two pieces of info that different model
    families consume in different ways:

    1. The base task name (``retrieval``, ``text-matching``, ``code``, ...)
       picks the LoRA adapter (v3, v4) or the omni model's ``default_task``.
       Code-embeddings has no LoRA — it ignores base task entirely.
    2. The ``.query`` / ``.passage`` suffix picks the prompt prefix that ST's
       ``encode()`` prepends, looked up in the model's ``prompts`` dict from
       ``config_sentence_transformers.json``. The actual prompt KEY differs
       per model (v4 uses ``query``/``passage``; v5-omni uses ``query``/
       ``document``; code-embeddings uses ``{base}_query``/``{base}_document``),
       so we resolve it data-driven via ``_map_prompt_name``.

    Returns:
        task_for_encode: passed via ``encode(task=...)`` (for ST 3.4+ models
            whose custom_st declares ``task`` in its kwargs list — v3, v4) or
            assigned on ``default_task`` for v5-omni. For v4 the suffix is
            stripped to the base (v4's ``config.task_names`` lists
            ``retrieval``/``text-matching``/``code`` only, with no
            ``.query``/``.passage`` form). For v3 the suffix is preserved
            because v3's task vocabulary IS ``retrieval.query`` /
            ``retrieval.passage``. For v5 the suffix is collapsed because
            v5's custom encode only accepts the bare base task. ``None``
            signals "do not pass ``task=`` to encode" — currently only used
            by v3 + no-task to match prod's raw-base behaviour.
        prompt_name: passed via ``encode(prompt_name=...)`` when set, ``None``
            otherwise. Mapped from the suffix against the model's actual
            prompts dict, so we never pass an unknown key (which would raise
            ``ValueError: Prompt name 'X' not found...``).
    """
    short_id = MODEL_ID.split("/")[-1] if MODEL_ID else ""
    is_v3 = "v3" in short_id and "v5" not in short_id
    is_v5 = "v5" in short_id
    is_v4 = short_id == "jina-embeddings-v4"

    # v3 no-task: prod runs the raw base xlm-roberta with no LoRA and no
    # prefix when ``task`` is omitted (verified against api.jina.ai —
    # prod_no_task vs prod_task=retrieval.passage cos≈0.63). Skip the
    # generic _default_task() fallback so we mirror that behaviour;
    # otherwise the default ``retrieval`` would route into v3_map below and
    # encode with the retrieval LoRA + "Represent the document..." prefix.
    if is_v3 and not task:
        return None, None

    if not task:
        task = _default_task()

    prompts = getattr(MODEL, "prompts", None) if MODEL is not None else None
    prompt_name = _map_prompt_name(task, prompts)

    if is_v3:
        v3_map = {
            "retrieval": "retrieval.passage",
            "retrieval.query": "retrieval.query",
            "retrieval.passage": "retrieval.passage",
            "text-matching": "text-matching",
            "classification": "classification",
            "clustering": "text-matching",
            "separation": "separation",
        }
        mapped_task = v3_map.get(task, "retrieval.passage")
        # Re-resolve prompt against the MAPPED task because v3's prompts are
        # keyed by task name. Without this, the suffix-less ``retrieval``
        # alias would skip the prompt lookup and encode without v3's
        # required ``"Represent the document for retrieval: "`` prefix —
        # previously observed as cos≈0.92 vs prod for retrieval.passage.
        return mapped_task, _map_prompt_name(mapped_task, prompts)
    if is_v5:
        # v5 (omni and text): custom encode only accepts the bare base task.
        # For omni the .query/.passage suffix is forwarded via prompt_name
        # (resolved above). v5-text has no prompts map so prompt_name stays
        # None.
        v5_map = {
            "retrieval": "retrieval",
            "retrieval.query": "retrieval",
            "retrieval.passage": "retrieval",
            "text-matching": "text-matching",
            "classification": "classification",
            "clustering": "clustering",
        }
        return v5_map.get(task, "retrieval"), prompt_name
    if is_v4:
        # v4 task_names = ["retrieval", "text-matching", "code"]; the .query /
        # .passage suffix would fail v4's task validator, so strip to base.
        # The suffix is carried into prompt_name (v4 prompts dict has
        # "query"/"passage" keys).
        base, _, _ = task.partition(".")
        return base, prompt_name
    # Plain models (v1/v2/b-en-v1) and code-embeddings: pass task through.
    # code-embeddings has no LoRA — its standard sentence_transformers
    # Transformer ignores unknown kwargs, so task= is a no-op; the prompt
    # routing happens entirely via prompts/prompt_name. v1/v2 have
    # MODEL_ACCEPTS_TASK_KWARG=False so the caller never forwards task=.
    return task, prompt_name


def _embed(inputs: list, task: Optional[str] = None, dimensions: Optional[int] = None):
    """Text-only embedding. Returns (embeddings_np, n_tokens, tok_per_s)."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _is_chat_model():
        raise HTTPException(
            status_code=400,
            detail=f"Model '{SHORT_MODEL_ID}' is a chat model. Use POST /v1/chat/completions instead.",
        )
    if _is_reranker_model() or _is_colbert_model():
        kind = "ColBERT (late-interaction)" if _is_colbert_model() else "reranker"
        raise HTTPException(
            status_code=400,
            detail=f"Model '{SHORT_MODEL_ID}' is a {kind} model. Use POST /v1/rerank instead.",
        )

    task, prompt_name = _resolve_task(task)
    n_tokens = _count_tokens(inputs)
    t0 = time.perf_counter()
    with torch.inference_mode(), _encode_autocast_ctx():
        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        # Task routing:
        #   v5-omni: custom_st module reads default_task; mutate it before encode
        #            (ST doesn't forward task= to its forward()).
        #   v3/v4/code-embeddings (ST 3.4+ with **kwargs): pass via task=.
        #   v1/v2 (older ST): MODEL_ACCEPTS_TASK_KWARG is False, skip task=.
        #   task=None (v3 + no-task): skip the kwarg so MODEL.encode falls
        #            through to its native "no task" path (raw base, no LoRA).
        # Prompt routing (always when the model exposes a matching prompts entry):
        #   v5-omni: "Query: "/"Document: " prefix; without prompt_name, last-token
        #            pooling on the LLaMA/Qwen text tower drifts (cos ~0.16 on nano).
        #   v4: "Query: "/"Passage: " prefix on retrieval/code LoRAs.
        #   code-embeddings: "Find the most relevant ..."/"Candidate ..." per task family.
        if _is_v5_omni_text_model():
            for mod in MODEL.modules():
                if hasattr(mod, 'default_task'):
                    mod.default_task = task
                    break
        elif MODEL_ACCEPTS_TASK_KWARG and task is not None:
            encode_kwargs["task"] = task
        if prompt_name is not None:
            encode_kwargs["prompt_name"] = prompt_name
        embeddings = MODEL.encode(
            inputs,
            **encode_kwargs,
        )
    elapsed = time.perf_counter() - t0

    tok_per_s = _update_stats(n_tokens, elapsed)
    logger.info(f"Embedded {len(inputs)} texts | {n_tokens} tokens | {elapsed*1000:.0f}ms | {tok_per_s:.0f} tok/s")

    if dimensions and dimensions < embeddings.shape[-1]:
        embeddings = embeddings[..., :dimensions]
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1.0)

    return embeddings, n_tokens, tok_per_s


def _embed_mixed(items: list, task: Optional[str] = None, dimensions: Optional[int] = None):
    """
    Embed a mixed list of inputs (text strings, PIL.Images, BytesIO, or lists for fused multimodal).

    Each items[i] is:
    - str / PIL.Image / BytesIO  -> one embedding
    - list of the above         -> fused multimodal tuple -> one embedding

    Returns (embeddings_np, n_tokens, tok_per_s).
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _is_chat_model():
        raise HTTPException(
            status_code=400,
            detail=f"Model '{SHORT_MODEL_ID}' is a chat model. Use POST /v1/chat/completions instead.",
        )
    if _is_reranker_model() or _is_colbert_model():
        kind = "ColBERT (late-interaction)" if _is_colbert_model() else "reranker"
        raise HTTPException(
            status_code=400,
            detail=f"Model '{SHORT_MODEL_ID}' is a {kind} model. Use POST /v1/rerank instead.",
        )

    task, prompt_name = _resolve_task(task)

    encode_inputs = []
    text_parts = []

    for item in items:
        if isinstance(item, list):
            # Sort: put non-text items (PIL.Image, BytesIO) first in the tuple.
            # The model's _encode_single_image has a bug when called via the
            # shortcut at line 903 (text-first ordering). Keeping non-text first
            # forces _encode_composite_parts which handles all orderings correctly.
            sorted_parts = sorted(item, key=lambda x: isinstance(x, str))
            encode_inputs.append(tuple(sorted_parts))
            for p in item:
                if isinstance(p, str):
                    text_parts.append(p)
        elif isinstance(item, str):
            encode_inputs.append(item)
            text_parts.append(item)
        else:
            # ST 3.4.1's _text_length() raises on PIL/BytesIO inside a tuple.
            # Pass standalone media items as bare entries so ST routes through
            # custom_st's _encode_single_image; for the multimodal models we target,
            # that path handles pure-image input correctly.
            encode_inputs.append(item)

    n_tokens = _count_tokens(text_parts) if text_parts else len(items)

    t0 = time.perf_counter()
    with torch.inference_mode(), _encode_autocast_ctx():
        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        if _is_v5_omni_text_model():
            for mod in MODEL.modules():
                if hasattr(mod, 'default_task'):
                    mod.default_task = task
                    break
        elif MODEL_ACCEPTS_TASK_KWARG and task is not None:
            encode_kwargs["task"] = task
        # Only forward prompt_name when every encode input is a string: ST
        # prepends ``prompts[prompt_name]`` to each input verbatim and would
        # raise on PIL.Image / BytesIO / fused tuple items. Pure-multimodal
        # calls keep working via the model's internal default.
        if prompt_name is not None and all(isinstance(x, str) for x in encode_inputs):
            encode_kwargs["prompt_name"] = prompt_name
        embeddings = MODEL.encode(
            encode_inputs,
            **encode_kwargs,
        )
    elapsed = time.perf_counter() - t0

    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
        embeddings = embeddings[None, :]

    tok_per_s = _update_stats(n_tokens, elapsed)
    logger.info(
        f"Embedded {len(items)} multimodal inputs | {n_tokens} text-tokens | "
        f"{elapsed*1000:.0f}ms | {tok_per_s:.0f} tok/s"
    )

    if dimensions and dimensions < embeddings.shape[-1]:
        embeddings = embeddings[..., :dimensions]
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        embeddings = embeddings / np.where(norms > 0, norms, 1.0)

    return embeddings, n_tokens, tok_per_s


RERANKER_MODEL_IDS = {
    "jina-reranker-v3",
    "jina-reranker-m0",
    "jina-reranker-v2-base-multilingual",
    "jina-reranker-v1-base-en",
    "jina-reranker-v1-turbo-en",
    "jina-reranker-v1-tiny-en",
    "jina-colbert-v2",
    "jina-colbert-v1-en",
}

COLBERT_MODEL_IDS = {
    "jina-colbert-v1-en",
    "jina-colbert-v2",
}


def _is_reranker_model() -> bool:
    short = MODEL_ID.split("/")[-1] if MODEL_ID else ""
    return any(r in short for r in RERANKER_MODEL_IDS)


def _is_colbert_model() -> bool:
    short = MODEL_ID.split("/")[-1] if MODEL_ID else ""
    return short in COLBERT_MODEL_IDS


def _is_reranker_v3() -> bool:
    """jina-reranker-v3 / v3.5 are Qwen3-based listwise rerankers with a custom
    JinaForRanking class (auto_map.AutoModel -> modeling.JinaForRanking) that
    exposes its own ``.rerank(query, documents)``. They are NOT
    sentence-transformers CrossEncoders, so they need a dedicated load + rerank
    branch separate from the generic reranker path. v3.5 is a drop-in upgrade:
    same class, same interface, per-item truncation baked into its own rerank().
    """
    short = MODEL_ID.split("/")[-1] if MODEL_ID else ""
    return short in ("jina-reranker-v3", "jina-reranker-v3.5")


def load_model():
    global MODEL, TOKENIZER, PROCESSOR, MODEL_INFO, MODEL_ACCEPTS_TASK_KWARG

    model_id = MODEL_ID
    if not model_id:
        raise RuntimeError("JINA_MODEL_ID not set")

    logger.info(f"Loading model: {model_id}")

    if _is_vlm_model():
        from transformers import AutoProcessor, AutoModelForCausalLM
        # VLM target dtype: fp16 on cuda (no bf16 native on L4), fp32 on cpu/mps
        if DEVICE == "cuda":
            dtype_env = os.environ.get("JINA_DTYPE", "float16").lower()
            vlm_dtype = torch.bfloat16 if dtype_env in ("bfloat16", "bf16") else torch.float16
        else:
            vlm_dtype = torch.float32
        # Flash-attn is intentionally not installed (no nvcc/git in runtime image);
        # sdpa is the supported fallback and matches HF's documented CPU/non-fa path.
        PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=vlm_dtype,
            low_cpu_mem_usage=True,
            device_map=DEVICE if DEVICE != "mps" else None,
            attn_implementation="sdpa",
        )
        if DEVICE == "mps":
            MODEL = MODEL.to(DEVICE)
        MODEL.eval()
        MODEL_INFO = {"model": model_id, "type": "vlm"}
        logger.info(f"Loaded as VLM (AutoModelForCausalLM): {model_id} dtype={vlm_dtype}")
    elif _is_text_chat_model():
        from transformers import AutoTokenizer, AutoModelForCausalLM
        # Text-only chat target dtype: fp16 on cuda (no bf16 native on L4), fp32 on cpu/mps.
        if DEVICE == "cuda":
            dtype_env = os.environ.get("JINA_DTYPE", "float16").lower()
            chat_dtype = torch.bfloat16 if dtype_env in ("bfloat16", "bf16") else torch.float16
        else:
            chat_dtype = torch.float32
        TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=chat_dtype,
            low_cpu_mem_usage=True,
            device_map=DEVICE if DEVICE != "mps" else None,
            attn_implementation="sdpa",
        )
        if DEVICE == "mps":
            MODEL = MODEL.to(DEVICE)
        MODEL.eval()
        MODEL_INFO = {"model": model_id, "type": "chat"}
        logger.info(f"Loaded as text chat (AutoModelForCausalLM): {model_id} dtype={chat_dtype}")
    elif _is_colbert_model():
        from pylate import models as pylate_models
        # PyLate wraps sentence-transformers and handles ColBERT-specific Q/D
        # markers, query expansion, and the 128-dim projection head.
        MODEL = pylate_models.ColBERT(
            model_name_or_path=model_id,
            trust_remote_code=True,
            device=DEVICE,
        )
        MODEL_INFO = {"model": model_id, "type": "colbert"}
        logger.info(f"Loaded as pylate ColBERT (late-interaction): {model_id}")
    elif _is_reranker_v3():
        from transformers import AutoModel
        # jina-reranker-v3: AutoModel + trust_remote_code loads the custom
        # JinaForRanking (Qwen3ForCausalLM subclass with a 1024->512->256 MLP
        # projector). Native dtype is bf16; keep it on cuda, use fp32 on cpu
        # because generic x86_64 has no bf16 SIMD.
        v3_dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        MODEL = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=v3_dtype,
            low_cpu_mem_usage=True,
        )
        MODEL = MODEL.to(DEVICE).eval()
        MODEL_INFO = {"model": model_id, "type": "reranker"}
        logger.info(f"Loaded as JinaForRanking (reranker-v3): {model_id} dtype={v3_dtype}")
    elif _is_reranker_model():
        from sentence_transformers import CrossEncoder
        MODEL = CrossEncoder(model_id, trust_remote_code=True, device=DEVICE)
        # Set pad_token if missing (qwen3-based rerankers need this for batched inference)
        if MODEL.tokenizer.pad_token is None:
            MODEL.tokenizer.pad_token = MODEL.tokenizer.eos_token
            if hasattr(MODEL.model, 'config'):
                MODEL.model.config.pad_token_id = MODEL.tokenizer.eos_token_id
            logger.info("Set pad_token = eos_token for reranker")
        MODEL_INFO = {"model": model_id, "type": "reranker"}
        logger.info(f"Loaded as CrossEncoder (reranker): {model_id}")
    else:
        from sentence_transformers import SentenceTransformer
        # Omni models need default_task set at load time because their custom_st.py
        # forward() receives task from SentenceTransformer internals, and st 3.4.1
        # doesn't pass task= through to forward(). Setting default_task ensures the
        # model always has a valid task.
        st_kwargs = {}
        # v5-omni models pass model_kwargs={"default_task": ...} to their underlying
        # transformer; other multimodal models (clip-v1/v2, v4, reranker-m0, vlm)
        # use older custom_st code that forwards model_kwargs straight to the model
        # ctor, which rejects unknown kwargs.
        _short = model_id.split("/")[-1]
        if _short in {"jina-embeddings-v5-omni-small", "jina-embeddings-v5-omni-nano"}:
            st_kwargs["model_kwargs"] = {"default_task": "retrieval"}
        MODEL = SentenceTransformer(model_id, trust_remote_code=True, device=DEVICE, **st_kwargs)
        if _short in {"jina-embeddings-v5-omni-small", "jina-embeddings-v5-omni-nano"}:
            # Force custom_st._build_eval_image_prompt into its bare-prompt fallback so
            # image inputs emit `<|vision_start|><|image_pad|><|vision_end|>` directly;
            # chat-template wrapping shifts last-token pooling and drops image cos
            # vs api.jina.ai from ~1.0 to ~0.90 (issue #23).
            _inner = MODEL[0]
            if getattr(_inner, "processor", None) is not None:
                _inner.processor.chat_template = None
                _tok = getattr(_inner.processor, "tokenizer", None)
                if _tok is not None:
                    _tok.chat_template = None
        MODEL_INFO = {"model": model_id, "type": "embedding"}

    if MODEL is not None and hasattr(MODEL, "encode"):
        try:
            import inspect as _inspect
            _sig = _inspect.signature(MODEL.encode)
            _params = _sig.parameters
            _has_var_kw = any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in _params.values())
            # ST 3.4+ exposes encode(..., **kwargs) and routes them to module forward()
            # filtered by each module's declared kwargs list (custom_st.Transformer for
            # v3 / code-embeddings declares ["task"], so passing task= is required for
            # LoRA adapter routing). ST 2.7 has no **kwargs and rejects unknown kwargs.
            MODEL_ACCEPTS_TASK_KWARG = "task" in _params or _has_var_kw
        except (ValueError, TypeError):
            MODEL_ACCEPTS_TASK_KWARG = False
        logger.info(f"MODEL.encode() accepts task= kwarg: {MODEL_ACCEPTS_TASK_KWARG}")

    try:
        from transformers import AutoTokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        logger.info("Tokenizer loaded for real tok/s measurement")
    except Exception as e:
        logger.warning(f"Could not load tokenizer ({e}), will use word-split approximation")

    # --- Apply dtype optimization (GPU only, embedding/reranker models) ---
    # VLM / text-chat dtype is set at load_pretrained time; skip the ST-style
    # .half() / torch.compile path for those.
    if DEVICE == "cuda" and not _is_reranker_model() and not _is_chat_model():
        dtype_env = os.environ.get("JINA_DTYPE", "float16").lower()
        if dtype_env in ("float16", "fp16", "half"):
            MODEL.half()
            logger.info("Model converted to FP16 (JINA_DTYPE=float16)")
        elif dtype_env in ("bfloat16", "bf16"):
            MODEL.bfloat16()
            logger.info("Model converted to BF16 (JINA_DTYPE=bfloat16)")
        else:
            logger.info(f"Running in FP32 (JINA_DTYPE={dtype_env})")

        # torch.compile: fuses ops, ~10-30% additional speedup. xlm-roberta-flash
        # models (jina-embeddings-v3 family, jina-clip-* text tower) mutate an
        # internal rotary _cos_cached tensor inside the forward pass; CUDA
        # Graphs (reduce-overhead) sees it as constant and raises "accessing
        # tensor output of CUDAGraphs that has been overwritten by a subsequent
        # run" the first time the captured-shape changes (e.g. different task
        # prompt length). Detect xlm-roberta-flash and skip compile for those.
        try:
            first_module = MODEL._first_module()
            if hasattr(first_module, "auto_model"):
                _auto = first_module.auto_model
                _is_flash_rotary = "xlm-roberta-flash-implementation" in (
                    getattr(type(_auto), "__module__", "") or ""
                )
                if _is_flash_rotary:
                    logger.info("torch.compile skipped: xlm-roberta-flash rotary cache is incompatible with CUDA Graphs")
                else:
                    first_module.auto_model = torch.compile(
                        _auto,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    logger.info("torch.compile(reduce-overhead) applied to encoder")
        except Exception as e:
            logger.warning(f"torch.compile skipped: {e}")

    # Decide CPU bf16 autocast now that the model is loaded (probe-gated in auto mode).
    _finalize_cpu_bf16(MODEL)

    multimodal = _is_omni_model()
    logger.info(f"Model loaded: {model_id} on {DEVICE} | multimodal={multimodal} | threads={_n_cpu_threads}")


@app.on_event("startup")
async def startup():
    load_model()
    _lic = _license.status(LICENSE_KEY, MODEL_ID)
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
    with _stats_lock:
        snap = dict(_stats)

    avg_tok_per_s = (
        snap["total_tokens"] / snap["total_latency_s"]
        if snap["total_latency_s"] > 0
        else None
    )

    resp = {
        "status": "ok",
        "model": SHORT_MODEL_ID,
        "device": DEVICE,
        "ready": MODEL is not None,
        "multimodal": _is_omni_model(),
        "schemas": ["openai", "voyage", "gemini", "cohere"],
        "license": _license.status(LICENSE_KEY, MODEL_ID),
    }

    if snap["total_requests"] > 0:
        resp["throughput"] = {
            "total_requests": snap["total_requests"],
            "total_tokens": snap["total_tokens"],
            "last_tok_per_s": round(snap["last_tok_per_s"], 1),
            "avg_tok_per_s": round(avg_tok_per_s, 1) if avg_tok_per_s else None,
            "peak_tok_per_s": round(snap["peak_tok_per_s"], 1),
        }

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
    # Default to None (not "retrieval") so _resolve_task can apply a
    # per-family default that matches prod (omni/v4 -> text-matching).
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


@app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse, tags=["OpenAI / Voyage AI"])
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

    dims = request.dimensions or request.output_dimension
    task = request.task
    if request.input_type:
        voyage_task_map = {
            "query": "retrieval.query",
            "document": "retrieval.passage",
            "classification": "classification",
            "clustering": "clustering",
        }
        task = voyage_task_map.get(request.input_type, "retrieval")

    has_multimodal = any(not isinstance(x, str) for x in raw_inputs)

    if has_multimodal:
        _require_omni()
        parsed = []
        for item in raw_inputs:
            parts = _parse_openai_item(item)
            parsed.append(parts if len(parts) > 1 else parts[0])
        embeddings, n_tokens, tok_per_s = _embed_mixed(parsed, task=task, dimensions=dims)
    else:
        embeddings, n_tokens, tok_per_s = _embed(raw_inputs, task=task, dimensions=dims)

    data = [
        {"object": "embedding", "embedding": emb.tolist(), "index": i}
        for i, emb in enumerate(embeddings)
    ]

    return OpenAIEmbeddingResponse(
        data=data,
        model=SHORT_MODEL_ID,
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
    images: Optional[list] = None       # legacy: list of data-URL strings
    inputs: Optional[list] = None       # V2: content-block array
    model: Optional[str] = None
    input_type: Optional[str] = "search_document"
    truncate: Optional[str] = "END"
    embedding_types: Optional[list] = Field(default_factory=lambda: ["float"])


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
    cohere_task_map = {
        "search_query": "retrieval.query",
        "search_document": "retrieval.passage",
        "classification": "classification",
        "clustering": "clustering",
    }
    task = cohere_task_map.get(request.input_type or "search_document", "retrieval")

    import uuid

    if request.inputs is not None:
        # V2 inputs format
        _require_omni()
        parsed = []
        for inp in request.inputs:
            content = inp.get("content", [])
            parts = []
            for p in content:
                parts.extend(_parse_content_part(p))
            parsed.append(parts if len(parts) > 1 else (parts[0] if parts else ""))
        embeddings, n_tokens, tok_per_s = _embed_mixed(parsed, task=task)
        texts_out = []
        images_count = len(parsed)
    elif request.images is not None:
        # Legacy: images as data-URL list
        _require_omni()
        parsed = []
        for img_data_url in request.images:
            raw, mime = _decode_b64(img_data_url)
            parsed.append(_bytes_to_st_input(raw, mime))
        embeddings, n_tokens, tok_per_s = _embed_mixed(parsed, task=task)
        texts_out = request.texts or []
        images_count = len(request.images)
    else:
        # Text-only
        texts = request.texts or []
        embeddings, n_tokens, tok_per_s = _embed(texts, task=task)
        texts_out = texts
        images_count = 0

    embedding_list = [emb.tolist() for emb in embeddings]

    return {
        "id": str(uuid.uuid4()),
        "texts": texts_out,
        "embeddings": {
            "float": embedding_list,
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
    _require_omni()

    voyage_task_map = {
        "query": "retrieval.query",
        "document": "retrieval.passage",
    }
    task = voyage_task_map.get(request.input_type or "", "retrieval")

    parsed = []
    for inp in request.inputs:
        content = inp.get("content", [])
        parts = []
        for p in content:
            parts.extend(_parse_content_part(p))
        parsed.append(parts if len(parts) > 1 else (parts[0] if parts else ""))

    embeddings, n_tokens, tok_per_s = _embed_mixed(parsed, task=task)

    return {
        "embeddings": [emb.tolist() for emb in embeddings],
        "text_tokens": n_tokens,
        "image_pixels": 0,  # not tracked
        "total_tokens": n_tokens,
        "model": SHORT_MODEL_ID,
    }


# =============================================================================
# Schema 4: Google Gemini  (POST /v1/models/{model}:embedContent)
#
# Multimodal: parts may include inlineData in addition to text:
#   {"inlineData": {"mimeType": "image/png", "data": "base64..."}}
# =============================================================================

class GeminiPart(BaseModel):
    text: Optional[str] = None
    inlineData: Optional[dict] = None   # {"mimeType": "image/png", "data": "base64..."}


class GeminiContent(BaseModel):
    parts: list                          # list of GeminiPart-like dicts
    role: Optional[str] = None


class GeminiEmbedContentRequest(BaseModel):
    content: GeminiContent
    taskType: Optional[str] = None
    title: Optional[str] = None
    outputDimensionality: Optional[int] = None


class GeminiBatchEmbedRequest(BaseModel):
    requests: list   # list of GeminiEmbedContentRequest-like dicts


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


def _parse_gemini_content(content_obj) -> list:
    """
    Parse a Gemini content object into ST-compatible inputs.
    Returns a list of parts (fused if multiple).
    """
    parts_raw = content_obj if isinstance(content_obj, list) else content_obj.get("parts", [])
    result = []
    for part in parts_raw:
        if isinstance(part, dict):
            if "inlineData" in part:
                result.extend(_parse_content_part(part))
            elif "text" in part:
                result.append(part["text"])
            else:
                result.extend(_parse_content_part(part))
        elif hasattr(part, "text") and part.text is not None:
            result.append(part.text)
        elif hasattr(part, "inlineData") and part.inlineData is not None:
            result.extend(_parse_content_part({"inlineData": part.inlineData}))
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
    has_multimodal = any(not isinstance(p, str) for p in parts)

    if has_multimodal:
        _require_omni()
        item = parts if len(parts) > 1 else parts[0]
        embeddings, n_tokens, tok_per_s = _embed_mixed([item], task=task, dimensions=request.outputDimensionality)
    else:
        text = " ".join(str(p) for p in parts)
        embeddings, n_tokens, tok_per_s = _embed([text], task=task, dimensions=request.outputDimensionality)

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
    all_items = []
    dims = None
    has_multimodal = False
    first_task_type = ""

    for req in request.requests:
        if not dims and req.get("outputDimensionality"):
            dims = req["outputDimensionality"]
        if not first_task_type:
            first_task_type = req.get("taskType", "")

        content = req.get("content", {})
        parts = _parse_gemini_content(content)

        if any(not isinstance(p, str) for p in parts):
            has_multimodal = True

        all_items.append(parts if len(parts) > 1 else (parts[0] if parts else ""))

    task = GEMINI_TASK_MAP.get(first_task_type, "retrieval")

    if has_multimodal:
        _require_omni()
        embeddings, n_tokens, tok_per_s = _embed_mixed(all_items, task=task, dimensions=dims)
    else:
        texts = [item if isinstance(item, str) else " ".join(str(p) for p in item) for item in all_items]
        embeddings, n_tokens, tok_per_s = _embed(texts, task=task, dimensions=dims)

    return {
        "embeddings": [{"values": emb.tolist()} for emb in embeddings],
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
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    docs = []
    for d in request.documents:
        docs.append(d if isinstance(d, str) else d.get("text", ""))

    t0 = time.perf_counter()
    with torch.inference_mode():
        if _is_colbert_model():
            # ColBERT late-interaction: encode query and docs as token-level
            # multi-vectors and score via MaxSim. pylate.rank.rerank takes a
            # nested-by-query layout (one inner list per query); since we have
            # one query we pass a 1-element outer list and pop result[0] back
            # out. NOTE: encode() with a nested-list documents arg internally
            # torch.stack()s per-doc embeddings without padding and crashes on
            # variable-length docs (jina-colbert returns (n_tokens, 128) per
            # doc, n_tokens varies). So encode docs as a flat list and wrap the
            # returned list-of-tensors for rerank.
            from pylate import rank as pylate_rank
            queries_emb = MODEL.encode([request.query], is_query=True, convert_to_tensor=True)
            docs_emb_flat = MODEL.encode(docs, is_query=False, convert_to_tensor=True)
            reranked = pylate_rank.rerank(
                documents_ids=[list(range(len(docs)))],
                queries_embeddings=queries_emb,
                documents_embeddings=[docs_emb_flat],
            )
            colbert_results = reranked[0]
        elif _is_reranker_v3():
            # jina-reranker-v3 ships its own block-wise listwise .rerank() that
            # already returns docs sorted by relevance_score descending, with
            # top_n applied. No need for a manual sort pass.
            v3_results = MODEL.rerank(request.query, docs, top_n=request.top_n)
        else:
            pairs = [[request.query, doc] for doc in docs]
            try:
                # convert_to_tensor=True keeps the model's native dtype (e.g. bf16 for
                # jina-reranker-v2-base-multilingual); cast to fp32 before numpy because
                # numpy has no bf16 dtype.
                scores = MODEL.predict(pairs, convert_to_numpy=False, convert_to_tensor=True)
            except AttributeError:
                raise HTTPException(status_code=400, detail="Loaded model does not support reranking")
            if hasattr(scores, "float"):
                scores = scores.float().detach().cpu().numpy()
    elapsed = time.perf_counter() - t0

    if _is_colbert_model():
        results = [
            {
                "index": int(item["id"]),
                "relevance_score": float(item["score"]),
                "document": {"text": docs[int(item["id"])]} if request.return_documents else None,
            }
            for item in colbert_results
        ]
        if request.top_n:
            results = results[: request.top_n]
        return {
            "model": SHORT_MODEL_ID,
            "results": results,
            "meta": {"elapsed_ms": round(elapsed * 1000, 1)},
        }

    if _is_reranker_v3():
        results = [
            {
                "index": int(item["index"]),
                "relevance_score": float(item["relevance_score"]),
                "document": {"text": docs[int(item["index"])]} if request.return_documents else None,
            }
            for item in v3_results
        ]
        return {
            "model": SHORT_MODEL_ID,
            "results": results,
            "meta": {"elapsed_ms": round(elapsed * 1000, 1)},
        }

    results = [
        {"index": i, "relevance_score": float(s), "document": {"text": docs[i]} if request.return_documents else None}
        for i, s in enumerate(scores)
    ]
    results.sort(key=lambda x: x["relevance_score"], reverse=True)

    if request.top_n:
        results = results[:request.top_n]

    return {
        "model": SHORT_MODEL_ID,
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


def _openai_msg_to_vlm_content(msg: dict) -> tuple:
    """Convert one OpenAI-style message to (role, content_parts, images).

    content_parts: list of {"type": "text"|"image", ...} for processor.apply_chat_template
    images: list of PIL.Image for processor(images=...)
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")
    parts = []
    images = []
    if isinstance(content, str):
        parts.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type", "")
            if t == "text":
                parts.append({"type": "text", "text": p.get("text", "")})
            elif t == "image_url":
                url = p.get("image_url", {})
                if isinstance(url, dict):
                    url = url.get("url", "")
                if not isinstance(url, str):
                    raise HTTPException(status_code=400, detail="image_url.url must be a string")
                if url.startswith("data:"):
                    raw, mime = _decode_b64(url)
                    img = _bytes_to_st_input(raw, mime)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="image_url.url must be a base64 data URL (offline server cannot fetch http(s) URLs)",
                    )
                parts.append({"type": "image", "image": img})
                images.append(img)
            elif t == "image":
                # Elastic-style {"type":"image","format":"base64","value":"..."}
                if p.get("format") == "base64":
                    raw, mime = _decode_b64(p.get("value", ""))
                    img = _bytes_to_st_input(raw, mime)
                    parts.append({"type": "image", "image": img})
                    images.append(img)
                else:
                    raise HTTPException(status_code=400, detail="image part requires format=base64")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported content part type: {t!r}")
    else:
        raise HTTPException(status_code=400, detail="message.content must be a string or list of parts")
    return role, parts, images


@app.post("/v1/chat/completions", tags=["OpenAI Chat"])
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint (VLM / text-chat / reader models)."""
    if MODEL is None or (PROCESSOR is None and not _is_text_chat_model()):
        raise HTTPException(
            status_code=503,
            detail="Chat completions endpoint requires a chat/VLM model. Loaded model: "
                   f"{MODEL_ID}",
        )
    if request.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported in this build")
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    conversation = []
    all_images = []
    for msg in request.messages:
        if not isinstance(msg, dict):
            raise HTTPException(status_code=400, detail="each message must be a dict")
        role, parts, imgs = _openai_msg_to_vlm_content(msg)
        if imgs and _is_text_chat_model():
            raise HTTPException(
                status_code=400,
                detail=f"Model '{SHORT_MODEL_ID}' is text-only and does not accept image inputs.",
            )
        conversation.append({"role": role, "content": parts})
        all_images.extend(imgs)

    max_sequence_length = (
        getattr(MODEL.config, "max_sequence_length", None)
        or getattr(MODEL.config, "max_position_embeddings", None)
        or 32768
    )

    t0 = time.perf_counter()
    if _is_text_chat_model():
        # Text-only path: AutoTokenizer's apply_chat_template only needs role+text.
        text_messages = [
            {
                "role": m["role"],
                "content": "".join(
                    p.get("text", "") for p in m["content"] if p.get("type") == "text"
                ),
            }
            for m in conversation
        ]
        templated = TOKENIZER.apply_chat_template(
            text_messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        proc_inputs = TOKENIZER(
            templated,
            return_tensors="pt",
            truncation=True,
            max_length=max_sequence_length,
        )
        decoder_tokenizer = TOKENIZER
    else:
        # apply_chat_template returns the templated string for ONE conversation when given a single list
        texts = PROCESSOR.apply_chat_template([conversation], add_generation_prompt=True)
        proc_inputs = PROCESSOR(
            text=texts,
            images=[all_images] if all_images else None,
            padding="longest",
            max_length=max_sequence_length,
            return_tensors="pt",
        )
        decoder_tokenizer = PROCESSOR.tokenizer

    device_inputs = {}
    dtype = next(MODEL.parameters()).dtype
    for k, v in proc_inputs.items():
        if k == "labels":
            continue
        if isinstance(v, torch.Tensor):
            if v.is_floating_point():
                device_inputs[k] = v.to(DEVICE, dtype=dtype, non_blocking=True)
            else:
                device_inputs[k] = v.to(DEVICE, non_blocking=True)
        else:
            device_inputs[k] = v

    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=request.max_tokens or 256,
        do_sample=(request.temperature is not None and request.temperature > 0.0),
        temperature=request.temperature if (request.temperature and request.temperature > 0.0) else None,
        top_p=request.top_p,
    )

    autocast_ctx = (
        torch.autocast(DEVICE, dtype=dtype)
        if DEVICE in ("cuda",) and dtype != torch.float32
        else nullcontext()
    )
    generate_kwargs = {
        "generation_config": gen_config,
        "return_dict_in_generate": True,
    }
    # use_model_defaults was added in transformers >=4.51; reader-lm pins 4.48.3.
    if not _is_text_chat_model():
        generate_kwargs["use_model_defaults"] = True

    with torch.inference_mode(), autocast_ctx:
        output = MODEL.generate(
            **device_inputs,
            **generate_kwargs,
        )

    input_ids = device_inputs["input_ids"]
    input_len = input_ids.shape[1]
    out_tokens = output.sequences[0][input_len:]
    n_completion_tokens = int(out_tokens.shape[0])
    response_text = decoder_tokenizer.decode(out_tokens, skip_special_tokens=True)
    elapsed = time.perf_counter() - t0

    prompt_tokens = int(input_len)
    _update_stats(n_completion_tokens, elapsed)
    logger.info(
        f"Chat: prompt={prompt_tokens} tok | gen={n_completion_tokens} tok | "
        f"{elapsed*1000:.0f}ms | images={len(all_images)}"
    )

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SHORT_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": n_completion_tokens,
            "total_tokens": prompt_tokens + n_completion_tokens,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
