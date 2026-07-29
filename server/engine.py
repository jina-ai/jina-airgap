"""The three inference verbs, and the lifecycle of the one loaded model.

Routes call in here; ``families`` decides what actually happens per model.
"""

import itertools
import logging
import threading
import time
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch

import serialize
import telemetry
from batching import Batcher, autotune_budget
from catalog import spec_for
from config import settings
from errors import BadRequest, ModelNotLoaded
from families import Family, family_for
from families.reranking import document_text

logger = logging.getLogger(__name__)

# Resolved at import so an unknown JINA_MODEL_ID fails before the server binds.
SPEC = spec_for(settings.model_id)

# Rendered into the text-only rejection message below. The capability test
# itself is spec.modality, not membership of this list.
MULTIMODAL_MODEL_IDS = {
    "jina-embeddings-v5-omni-small",
    "jina-embeddings-v5-omni-nano",
    "jina-embeddings-v4",
    "jina-clip-v2",
    "jina-clip-v1",
    "jina-reranker-m0",
    "jina-vlm",
}

_family: Optional[Family] = None
_batcher: Optional[Batcher] = None
_solo = itertools.count()

# One thread in the model at a time. The routes are synchronous handlers, so
# Starlette dispatches them to its threadpool and several can arrive at once --
# and none of these models is thread-safe. Without this, two concurrent
# requests to a compiled CUDA model corrupt each other's state and surface as
# `index out of bounds` from inside an Inductor kernel, which reads like a bad
# input rather than a race.
#
# It does not make anything slower than it was: a single GPU runs one forward
# at a time regardless, and before the handlers were synchronous the event loop
# was serialising them anyway. What changed is that the serialisation now
# happens here instead of in the event loop, so /health still answers while a
# forward is running.
_MODEL_LOCK = threading.Lock()


def load() -> None:
    global _family, _batcher
    logger.info(f"Loading model: {settings.model_id}")
    family = family_for(SPEC)
    family.load()
    _family = family
    if family.verb == "embed":
        # Always, not only when batching is on. The worker exists first to keep
        # a single thread on the model -- Starlette dispatches the synchronous
        # handlers to a pool, and neither these models nor torch's CUDA-graph
        # cache tolerate two threads. `merge` is what the throughput variant
        # turns on; with it off the worker runs exactly one job per forward and
        # the vectors are bit-identical to calling encode inline.
        _batcher = Batcher(
            encode=lambda key, inputs, merge: _encode(
                family, key, inputs, len(inputs) if merge else None
            ),
            wait_ms=settings.batch_wait_ms,
            merge=settings.batching,
            startup=(lambda: _initial_budget(family)) if settings.batching else None,
        )
    logger.info(
        f"Model loaded: {settings.model_id} on {settings.device} | "
        f"multimodal={is_multimodal()} | threads={settings.cpu_threads} | "
        f"batching={_batcher is not None}"
    )


def _initial_budget(family: Family) -> int:
    if settings.batch_tokens > 0:
        logger.info(f"Token budget fixed by JINA_BATCH_TOKENS: {settings.batch_tokens}")
        return settings.batch_tokens
    try:
        task, prompt_name = family.resolve_task(None)
        # ~512 tokens of ordinary prose. A synthetic repeat is fine here: the
        # probe is measuring how much VRAM a shape costs, not what it means.
        sample = "semantic search over a document collection " * 64

        def probe(rows: int) -> None:
            # batch_size must match the row count or sentence-transformers
            # chunks the probe at 32 and the measurement is of 32 rows however
            # many were asked for -- which reads as "memory does not grow".
            with torch.inference_mode(), family.autocast():
                family.encode(
                    [sample] * rows,
                    task,
                    prompt_name,
                    normalized=True,
                    extra=None,
                    batch_size=rows,
                )

        return autotune_budget(probe, SPEC.context)
    except Exception as exc:  # noqa: BLE001 - a failed probe must not block startup
        logger.warning(f"Token-budget autotune failed ({type(exc).__name__}: {exc}); using 8192")
        return 8192


def is_ready() -> bool:
    return _family is not None


def is_multimodal() -> bool:
    return SPEC.modality == "multimodal"


def require_multimodal() -> None:
    if not is_multimodal():
        raise BadRequest(
            f"Model '{settings.short_model_id}' is text-only and does not accept "
            f"image/audio/video inputs. Use one of: {sorted(MULTIMODAL_MODEL_IDS)}"
        )


def embedding_family() -> Family:
    """Public because a route has to ask the loaded model what it supports --
    which encode kwargs it takes, which task names it knows -- before it can
    refuse a request truthfully."""
    return _require_embed()


def _require_embed() -> Family:
    if _family is None:
        raise ModelNotLoaded("Model not loaded")
    if _family.verb != "embed":
        raise BadRequest(
            f"Model '{settings.short_model_id}' is a {_family.kind} model. "
            f"Use POST {_family.endpoint} instead."
        )
    return _family


def _require_rerank() -> Family:
    if _family is None:
        raise ModelNotLoaded("Model not loaded")
    if _family.verb != "rerank":
        # The old code reached this by calling .predict() on something that is
        # not a CrossEncoder and catching the AttributeError.
        raise BadRequest("Loaded model does not support reranking")
    return _family


def require_chat_model() -> Family:
    """Public because /v1/chat/completions rejects a non-chat image before it
    validates the request body."""
    if _family is None:
        raise ModelNotLoaded("Model not loaded")
    if _family.verb != "generate":
        # 400, not 503: one image serves one model, so this never becomes true
        # by retrying, and a 503 would tell every SDK to back off and try again
        # forever. Same reasoning as the embed and rerank guards.
        raise BadRequest(
            f"Model '{settings.short_model_id}' is a {_family.kind} model. "
            f"Use POST {_family.endpoint} instead."
        )
    return _family


def tokenizer():
    """The loaded model's tokenizer, or None if it would not load standalone."""
    return _family.tokenizer if _family else None


def token_lengths(texts: list[str], cap: Optional[int] = None) -> list[int]:
    """Tokens the model will actually process.

    ``cap`` is the per-input context limit. It matters for ``truncate=true``:
    the model only ever sees the first ``cap`` tokens, so counting the whole
    submission would report work that never happened. Measured on
    api.jina.ai -- a 60k-character input to v3 with ``truncate=true`` comes
    back as ``total_tokens: 8194`` against an 8192-token context, not as the
    submitted count.
    """
    tokenizer = _family.tokenizer if _family else None
    if tokenizer is not None:
        try:
            encoded = tokenizer(texts, add_special_tokens=True)
            lengths = [len(ids) for ids in encoded["input_ids"]]
        except Exception:
            lengths = [len(text.split()) for text in texts]
    else:
        lengths = [len(text.split()) for text in texts]
    if cap:
        lengths = [min(length, cap) for length in lengths]
    return lengths


def count_tokens(texts: list[str], cap: Optional[int] = None) -> int:
    return sum(token_lengths(texts, cap))


def embed(
    items: list,
    task: Optional[str] = None,
    dimensions: Optional[int] = None,
    normalized: bool = True,
    extra: Optional[dict] = None,
) -> tuple[np.ndarray, int, float]:
    """Embed text, media, and fused multimodal blocks in one call.

    Each element of ``items`` is a string, a PIL.Image / BytesIO, or a list of
    those -- a fused block that produces one embedding.

    ``extra`` carries the native parameters that change the forward pass
    (``late_chunking``, ``return_multivector``, ``return_tokenized_input``).
    The route has already checked them against ``family.encode_kwargs``, so
    anything arriving here is a kwarg the model genuinely consumes.
    """
    family = _require_embed()
    task, prompt_name = family.resolve_task(task)
    inputs, texts = _split_inputs(items)
    n_tokens = count_tokens(texts, cap=SPEC.context) if texts else len(items)

    start = time.perf_counter()
    try:
        embeddings = _dispatch(family, inputs, task, prompt_name, normalized, extra)
    except ValueError as exc:
        # The model is the authority on its own task names, and some -- v4,
        # v5 -- expose none to check against beforehand. When one rejects an
        # argument it is the caller's mistake, so it answers 400 carrying the
        # model's own message rather than a 500 that reads like a crash.
        raise BadRequest(str(exc), field="task") from exc
    elapsed = time.perf_counter() - start

    if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1 and len(items) == 1:
        # A lone media item comes back as a flat vector, not a 1-row matrix.
        embeddings = embeddings[None, :]

    tok_per_s = telemetry.record(n_tokens, elapsed)
    logger.info(
        f"Embedded {len(items)} inputs | {n_tokens} tokens | "
        f"{elapsed * 1000:.0f}ms | {tok_per_s:.0f} tok/s"
    )

    if dimensions:
        embeddings = _truncate(embeddings, dimensions, normalized)

    return embeddings, n_tokens, tok_per_s


def _encode(family: Family, key: tuple, inputs: list, batch_size: Optional[int] = None):
    """One forward. The only place ``family.encode`` is called."""
    task, prompt_name, normalized, extra_items, _ = key
    with _MODEL_LOCK, torch.inference_mode(), family.autocast():
        return family.encode(
            inputs,
            task,
            prompt_name,
            normalized=normalized,
            extra=dict(extra_items) or None,
            batch_size=batch_size,
        )


def _dispatch(family: Family, inputs, task, prompt_name, normalized, extra):
    """Encode directly, or hand the rows to the batcher and wait for them.

    Both paths reach the same ``family.encode``; batching only changes which
    rows share a forward. Everything that would change the *result* -- task,
    prompt prefix, normalisation, the native extras -- is in the grouping key,
    so a merged batch computes exactly what the rows would have computed alone.

    Anything that is not plain text gets a key nothing else can share. Media
    has no token length to sort by and its cost per row is unrelated to text's,
    so it is served alone -- still on the worker thread, because the invariant
    that matters is one thread on the model, not that every batch is full.
    """
    extra_items = tuple(sorted((extra or {}).items()))
    key = (task, prompt_name, normalized, extra_items, None)
    if _batcher is None:
        return _encode(family, key, inputs)
    if _batcher.merges and not all(isinstance(item, str) for item in inputs):
        key = (task, prompt_name, normalized, extra_items, next(_solo))
    rows = _batcher.submit(inputs, key, _row_lengths(inputs))
    return _restack(rows)


def _row_lengths(inputs: list) -> list[int]:
    """Token length per row, for length-sorted packing. One tokenizer call."""
    lengths = token_lengths(
        [item if isinstance(item, str) else "" for item in inputs], cap=SPEC.context
    )
    # A non-text row is charged the full context so packing never puts it in a
    # chunk it would blow up; it is alone in its group anyway.
    return [
        length if isinstance(item, str) else (SPEC.context or 512)
        for item, length in zip(inputs, lengths)
    ]


def _restack(rows: list):
    """Undo the per-row split so both paths return the same shape.

    Plain embeddings come back as an ``(n, dim)`` matrix; ``late_chunking`` and
    ``return_multivector`` come back as a list of per-token matrices, which
    ``_truncate`` and the serializer both already expect to stay a list.
    """
    if rows and all(isinstance(row, np.ndarray) and row.ndim == 1 for row in rows):
        return np.stack(rows)
    return rows


def _truncate(embeddings, dimensions: int, normalized: bool):
    """Matryoshka truncation, over either a matrix or a list of per-token
    matrices. Re-normalises only when the caller asked for normalised output:
    slicing a unit vector leaves it short of unit length, but re-scaling raw
    magnitudes would contradict ``normalized=false``."""

    def one(matrix: np.ndarray) -> np.ndarray:
        if dimensions >= matrix.shape[-1]:
            return matrix
        matrix = matrix[..., :dimensions]
        if not normalized:
            return matrix
        norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
        return matrix / np.where(norms > 0, norms, 1.0)

    if isinstance(embeddings, list):
        return [one(np.asarray(matrix)) for matrix in embeddings]
    return one(embeddings)


def rerank(
    query: str,
    documents: list,
    top_n: Optional[int] = None,
    return_documents: bool = True,
    max_doc_length: Optional[int] = None,
) -> tuple[list[dict], int, float]:
    """Score ``documents`` against ``query``.

    ``max_doc_length`` (Jina) / ``max_tokens_per_doc`` (Cohere) shortens what
    the model *scores*. The echoed ``document`` is always the caller's original
    text: truncation is a scoring directive, and round-tripping a detokenised
    fragment back to the client would corrupt it -- lower-cased, respaced, or
    with CJK split at subword boundaries.
    """
    family = _require_rerank()
    texts = [document_text(document) for document in documents]
    if max_doc_length:
        texts = [_clip(text, max_doc_length) for text in texts]
    n_tokens = count_tokens([query] + texts, cap=SPEC.context)

    start = time.perf_counter()
    with _MODEL_LOCK, torch.inference_mode():
        ranked = family.rank(query, texts, top_n)
    elapsed = time.perf_counter() - start

    telemetry.record(n_tokens, elapsed)
    logger.info(
        f"Reranked {len(documents)} documents | {n_tokens} tokens | "
        f"{elapsed * 1000:.0f}ms"
    )

    results = []
    for index, score in ranked:
        # `document` is omitted, never null, when return_documents is false --
        # measured on api.jina.ai for every reranker.
        result = {"index": index, "relevance_score": serialize.relevance_score(score)}
        if return_documents:
            result["document"] = family.render_document(documents[index])
        results.append(result)
    return results, n_tokens, elapsed


def _clip(text: str, max_tokens: int) -> str:
    encoder = tokenizer()
    if encoder is None:
        raise BadRequest(
            "Per-document truncation needs a tokenizer, which did not load for "
            f"{settings.short_model_id}."
        )
    ids = encoder(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return encoder.decode(ids[:max_tokens], skip_special_tokens=True)


def generate(
    conversation: list,
    images: list,
    max_tokens: int,
    temperature: Optional[float],
    top_p: Optional[float],
) -> tuple[str, int, int]:
    """Run one chat completion. Returns (text, prompt_tokens, completion_tokens)."""
    family = require_chat_model()
    if images and not family.accepts_images:
        raise BadRequest(
            f"Model '{settings.short_model_id}' is text-only and does not "
            f"accept image inputs."
        )

    config = family.model.config
    max_length = (
        getattr(config, "max_sequence_length", None)
        or getattr(config, "max_position_embeddings", None)
        or 32768
    )

    start = time.perf_counter()
    dtype = next(family.model.parameters()).dtype
    inputs = _to_device(family.encode_prompt(conversation, images, max_length), dtype)

    from transformers import GenerationConfig

    sampling = temperature is not None and temperature > 0.0
    generation_config = GenerationConfig(
        max_new_tokens=max_tokens or 256,
        do_sample=sampling,
        temperature=temperature if sampling else None,
        top_p=top_p,
    )
    autocast = (
        torch.autocast(settings.device, dtype=dtype)
        if settings.device == "cuda" and dtype != torch.float32
        else nullcontext()
    )
    with _MODEL_LOCK, torch.inference_mode(), autocast:
        output = family.model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            **family.generate_kwargs,
        )

    prompt_tokens = int(inputs["input_ids"].shape[1])
    generated = output.sequences[0][prompt_tokens:]
    completion_tokens = int(generated.shape[0])
    text = family.decoder_tokenizer.decode(generated, skip_special_tokens=True)
    elapsed = time.perf_counter() - start

    telemetry.record(completion_tokens, elapsed)
    logger.info(
        f"Chat: prompt={prompt_tokens} tok | gen={completion_tokens} tok | "
        f"{elapsed * 1000:.0f}ms | images={len(images)}"
    )
    return text, prompt_tokens, completion_tokens


def _split_inputs(items: list) -> tuple[list, list[str]]:
    """Return (encode inputs, the text parts that token counting sees)."""
    inputs, texts = [], []
    for item in items:
        if isinstance(item, list):
            # Put non-text parts first in the tuple. The model's
            # _encode_single_image has a bug when reached via the text-first
            # shortcut; non-text-first forces _encode_composite_parts, which
            # handles every ordering correctly.
            inputs.append(tuple(sorted(item, key=lambda part: isinstance(part, str))))
            texts.extend(part for part in item if isinstance(part, str))
        elif isinstance(item, str):
            inputs.append(item)
            texts.append(item)
        else:
            # ST 3.4.1's _text_length() raises on PIL/BytesIO inside a tuple, so
            # standalone media goes in bare and ST routes it through custom_st's
            # _encode_single_image, which handles pure-image input correctly.
            inputs.append(item)
    return inputs, texts


def _to_device(inputs: Any, dtype: torch.dtype) -> dict:
    moved = {}
    for key, value in inputs.items():
        if key == "labels":
            continue
        if not isinstance(value, torch.Tensor):
            moved[key] = value
        elif value.is_floating_point():
            moved[key] = value.to(settings.device, dtype=dtype, non_blocking=True)
        else:
            moved[key] = value.to(settings.device, non_blocking=True)
    return moved
