"""The three inference verbs, and the lifecycle of the one loaded model.

Routes call in here; ``families`` decides what actually happens per model.
"""

import itertools
import logging
import re
import threading
import time
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch

import serialize
import tasks
import telemetry
from batching import Batcher, autotune_budget
from catalog import spec_for
from config import settings
from errors import (
    BadRequest,
    ModelNotLoaded,
    PayloadTooLarge,
    UnprocessableEntity,
)
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
        # Every embed image has the worker; only the throughput variant merges
        # across requests. Reporting the worker's existence as "batching" made
        # a plain :gpu image claim it was batching when it was not.
        f"worker={_batcher is not None} | batching={settings.batching}"
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
    """Loaded, and still able to answer.

    ``_family is not None`` alone stays true forever once the weights are in
    memory, including after the model worker thread has gone. A container in
    that state reported ``ready: true`` while every request queued behind a
    thread that would never take it -- healthy to anything watching, dead to
    anything asking.
    """
    return _family is not None and (_batcher is None or _batcher.alive)


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


_approximation_warned = False


def _warn_approximating(reason: str) -> None:
    """Say once that token counts are estimates, not the tokenizer's answer.

    Once rather than per call: this sits on the request path.
    """
    global _approximation_warned
    if not _approximation_warned:
        _approximation_warned = True
        logger.warning(
            "Token counts are approximate for this model (%s); this affects the "
            "over-length refusal as well as reported usage.",
            reason,
        )


# Escapes rather than the characters: three of these bounds are unassigned
# codepoints with no glyph, so written literally the class renders as tofu and
# a wrong bound is invisible.
_DENSE_SCRIPT = re.compile(
    "["
    "\u3040-\u30ff"  # hiragana and katakana
    "\u3400-\u4dbf"  # CJK unified ideographs extension A
    "\u4e00-\u9fff"  # CJK unified ideographs
    "\uac00-\ud7af"  # hangul syllables
    "\uf900-\ufaff"  # CJK compatibility ideographs
    "]"
)


def _approximate_tokens(text: str) -> int:
    """Token estimate for when the tokenizer cannot answer.

    This number decides ``_refuse_over_length``, so it errs high: counting low
    admits input the model has no room for. Four characters per token is an
    assumed floor for Latin rather than a measured ratio -- real ratios run
    looser, and a model whose tokenizer loads is never counted from here. Dense
    scripts are weighted near one per character, where a word split can return 1
    for a whole sentence.
    """
    latin = len(_DENSE_SCRIPT.sub("", text))
    dense = len(text) - latin
    return max(len(text.split()), dense + latin // 4)


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
        except Exception as exc:
            _warn_approximating(f"tokenizer raised ({type(exc).__name__}: {exc})")
            lengths = [_approximate_tokens(text) for text in texts]
    else:
        _warn_approximating("tokenizer did not load")
        lengths = [_approximate_tokens(text) for text in texts]
    if cap:
        lengths = [min(length, cap) for length in lengths]
    return lengths


def count_tokens(texts: list[str], cap: Optional[int] = None) -> int:
    return sum(token_lengths(texts, cap))


def _check_task(task: Optional[str]) -> None:
    """Reject a task the public API rejects, so a typo cannot silently select a
    different LoRA adapter.

    The vocabulary is ``SPEC.task_enum``, matched whole. Matching the base name
    instead let through a whole family of values the public API refuses, and
    two of them were not merely permissive:

    - bare ``retrieval``, on every model whose enum lists only
      ``retrieval.query`` and ``retrieval.passage``;
    - suffixed forms like ``classification.query``, whose base is known but
      whose full name no family recognises -- on v3 those reached the
      translation table, missed, and came back as ``retrieval.passage``
      vectors with a 200 on them.

    An empty enum means the public API declares no ``task`` field for this
    model at all, so there is nothing here to contradict: the value is passed
    through and the model ignores it, which is what api.jina.ai does too.
    """
    if not task or not SPEC.task_enum or task in SPEC.task_enum:
        return
    raise BadRequest(
        f"Unknown task '{task}' for {settings.short_model_id}. "
        f"This model accepts: {', '.join(SPEC.task_enum)}.",
        field="task",
    )


def fit_task(
    candidates: tuple, role: Optional[str], *, field: str, value: str
) -> Optional[str]:
    """Turn another vendor's ``(preferred families, role)`` into a task THIS
    model has -- or refuse, when there is nothing honest to turn it into.

    A vendor's ``input_type`` / ``taskType`` is a closed single-choice list.
    ``search_query`` is the *only* thing a Cohere client can say for a query,
    and code-embeddings has no ``retrieval`` task, so refusing that pair would
    reject every request that client is capable of making. The role is the
    analogy that survives: a query is still a query, so it lands on the
    model's own default family as ``nl2code.query``.

    ``classification`` is the other case, and it must not be treated the same
    way. Code-embeddings has no classification task and no role to fall back
    on, so there is no reading of the request this model can answer -- and
    ``task=classification`` is already a 400 on the native route. Returning
    ``nl2code`` vectors instead would be the exact defect this server exists
    to remove, in a new place: a wrong answer with a 200 on it.

    Returning ``None`` means "no task", which is a real answer for three cases:
    the vendor's own "unspecified" value, a model that publishes no vocabulary
    at all, and a model that has the concept but names no form of it for this
    role -- see the clip-v2 note below.

    The answer is picked out of ``SPEC.task_enum`` rather than assembled from
    base names, because the enum is the only thing that knows which forms
    exist. v3 is the case that forces it: it has a clustering adapter and calls
    it ``separation``, so a Cohere ``clustering`` must skip the word it shares
    and land on the word this model actually takes.
    """
    family = _require_embed()
    allowed = SPEC.task_enum
    if not allowed:
        return _fit_without_enum(family, candidates, role)

    for name in candidates:
        if role and f"{name}.{role}" in allowed:
            return f"{name}.{role}"
        if name in allowed:
            return name

    # The model has the concept and names no form of it for this role. clip-v2
    # is the whole of this case: its enum is `retrieval.query` alone, because
    # only the query side carries a prefix -- `config_sentence_transformers`
    # has that one prompt and a null default. Its document side is therefore
    # the unprefixed encoding, reached by sending no task, which is a real
    # value of the native field. Refusing instead would leave a Cohere client
    # unable to index against this model at all: `input_type` is required
    # there, with no neutral member, so `search_document` is the only thing it
    # can say for a document.
    bases = {name.partition(".")[0] for name in allowed}
    if any(name in bases for name in candidates):
        return None
    if not candidates:
        return None
    if role:
        fallback = f"{tasks.default_task(SPEC.family).partition('.')[0]}.{role}"
        if fallback in allowed:
            return fallback
    raise BadRequest(
        f"'{value}' has no counterpart in {settings.short_model_id}, which "
        f"serves: {', '.join(allowed)}. Nothing in the request says whether "
        f"the text is a query or a document either, so there is no reading of "
        f"it this model can answer.",
        field=field,
    )


def _fit_without_enum(
    family: Family, candidates: tuple, role: Optional[str]
) -> Optional[str]:
    """``fit_task`` for a model the public API declares no ``task`` field on.

    Nothing to match against and nothing to refuse: the v2 family and clip-v1
    take any task and ignore it, so the only question left is whether a role
    survives the translation.
    """
    known = family.known_tasks
    base = next((name for name in candidates if name in known), None)
    if base is not None:
        return f"{base}.{role}" if role else base
    if role and candidates:
        return f"{tasks.default_task(SPEC.family).partition('.')[0]}.{role}"
    return None


def named_task(name: str, *, field: str, vendor_values) -> str:
    """Accept a task named outright in a vendor's field.

    Cohere has no word for ``nl2code.query`` and never will, so a caller who
    knows this image is a Jina model has no way to ask for it through Cohere's
    envelope. The envelope should not be what stops them: the field is
    Cohere's, the vocabulary can be ours. It is the same move as writing
    "jiaozi" in English rather than pretending it is a dumpling.

    Only names this model actually serves; anything else is still a typo. A
    model that declares no vocabulary at all serves none of them and
    contradicts none of them either -- ``/v1/embeddings`` takes any task name
    on those images and the model ignores it, which is also what the public
    API does. Refusing here would have made the envelope decide the answer,
    which is the one thing it must never do.
    """
    _require_embed()
    allowed = SPEC.task_enum
    if not allowed or name in allowed:
        return name
    raise BadRequest(
        f"Unknown {field} '{name}'. Accepted: "
        f"{', '.join(sorted(vendor_values))}, or a task "
        f"{settings.short_model_id} serves ({', '.join(allowed)}).",
        field=field,
    )


def _check_dimensions(dimensions: Optional[int]) -> None:
    """The public API caps `dimensions` at the model's own output dimension and
    422s above it; without the check a too-large value silently no-ops."""
    limit = SPEC.output_dim
    if dimensions and limit and dimensions > limit:
        raise UnprocessableEntity(
            f"Input should be less than or equal to {limit}",
            field="body -> dimensions",
            code="less_than_equal",
        )


def _check_extras(family: Family, extra: Optional[dict]) -> Optional[dict]:
    """sentence-transformers drops undeclared encode kwargs silently, so a
    `late_chunking` request against a model without it would return ordinary
    embeddings and look like a success."""
    if not extra:
        return extra
    accepted = family.encode_kwargs
    for name in extra:
        if name not in accepted:
            raise BadRequest(
                f"'{name}' is not supported by {settings.short_model_id}.",
                field=name,
            )
    return extra


def _refuse_over_length(lengths: list) -> None:
    """`truncate=false` means "error rather than silently shorten". Measured on
    api.jina.ai: over-length input returns HTTP 400 `INPUT_TOKEN_LIMIT_EXCEEDED`,
    and only `truncate=true` succeeds. Left to sentence-transformers the input
    would be clipped with nothing on the wire to say so, which changes the
    embedding invisibly."""
    limit = SPEC.context
    if not limit:
        return
    if any(length > limit for length in lengths):
        raise BadRequest(
            f"Input text exceeds the model's maximum of {limit} tokens. "
            f"Use 'truncate: true' to automatically truncate, or split "
            f"into smaller chunks.",
            field="input",
            code="INPUT_TOKEN_LIMIT_EXCEEDED",
        )


def embed(
    items: list,
    task: Optional[str] = None,
    dimensions: Optional[int] = None,
    normalized: bool = True,
    extra: Optional[dict] = None,
    truncate: bool = True,
) -> tuple[np.ndarray, int, float]:
    """Embed text, media, and fused multimodal blocks in one call.

    Each element of ``items`` is a string, a PIL.Image / BytesIO, or a list of
    those -- a fused block that produces one embedding.

    ``extra`` carries the native parameters that change the forward pass
    (``late_chunking``, ``return_multivector``, ``return_tokenized_input``).

    The contract checks live here rather than in an adapter. They used to sit
    in ``adapters/jina.py``, which made the Jina route the only one that
    enforced them: the same over-limit ``dimensions`` was a 422 through
    ``/v1/embeddings`` and a silent full-width 200 through Cohere's, because
    ``_truncate`` no-ops when the request is wider than the model. An adapter
    can only be trusted to translate its own dialect; what a request is allowed
    to *say* is the model's business, and this is where the model is.
    """
    family = _require_embed()
    _check_task(task)
    _check_dimensions(dimensions)
    extra = _check_extras(family, extra)
    task, prompt_name = family.resolve_task(task)
    inputs, texts, per_row = _split_inputs(items)
    # One tokenizer pass, used three times: to refuse over-length input, for
    # the usage figure, and for the packing weights. It runs on a threadpool
    # thread, so a second pass over the same text competes with the model
    # worker for the CPU it needs to feed the GPU -- which is what the old
    # per-item `count_tokens` call in the route was doing.
    raw_lengths = token_lengths(texts) if texts else []
    if not truncate:
        _refuse_over_length(raw_lengths)
    lengths = (
        [min(length, SPEC.context) for length in raw_lengths]
        if SPEC.context
        else raw_lengths
    )
    n_tokens = sum(lengths) if texts else len(items)

    start = time.perf_counter()
    try:
        embeddings = _dispatch(
            family,
            inputs,
            _row_lengths(inputs, lengths, per_row),
            task,
            prompt_name,
            normalized,
            extra,
        )
    except ValueError as exc:
        # The model is the authority on its own task names, and some -- v4,
        # v5 -- expose none to check against beforehand. When one rejects an
        # argument it is the caller's mistake, so it answers 400 carrying the
        # model's own message rather than a 500 that reads like a crash.
        raise BadRequest(str(exc), field="task") from exc
    except torch.OutOfMemoryError as exc:
        # Reached only after the batcher has already halved the batch down to a
        # single row, so the input itself does not fit -- retrying or sending
        # less of it is the caller's only remedy, and "Internal Server Error"
        # tells them neither. Attention over N tokens costs O(N^2) on the
        # models that materialise a score matrix, so a max-context input can
        # exceed a 23 GB card on its own.
        raise PayloadTooLarge(
            f"Input too large for '{settings.short_model_id}' on this hardware: "
            f"{n_tokens} tokens exhausted GPU memory. Send fewer tokens per "
            f"input, or set 'dimensions'/'truncate' to reduce the work."
        ) from exc
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


def _dispatch(family: Family, inputs, row_lengths, task, prompt_name, normalized, extra):
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
    rows = _batcher.submit(inputs, key, row_lengths)
    return _restack(rows)


def _row_lengths(inputs: list, lengths: list[int], per_row: list[int]) -> list[int]:
    """Packing weight per row, read off the tokenizer pass ``embed`` made.

    A non-text row is charged the full context so packing never puts it in a
    chunk it would blow up; it is alone in its group anyway.
    """
    weights, cursor = [], 0
    for item, parts in zip(inputs, per_row):
        weights.append(
            lengths[cursor] if isinstance(item, str) else (SPEC.context or 512)
        )
        cursor += parts
    return weights


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
    return_embeddings: bool = False,
    max_doc_length: Optional[int] = None,
    truncate: bool = True,
) -> tuple[list[dict], int, float]:
    """Score ``documents`` against ``query``.

    ``max_doc_length`` (Jina) / ``max_tokens_per_doc`` (Cohere) shortens what
    the model *scores*. The echoed ``document`` is always the caller's original
    text: truncation is a scoring directive, and round-tripping a detokenised
    fragment back to the client would corrupt it -- lower-cased, respaced, or
    with CJK split at subword boundaries.

    ``truncate`` is Voyage's ``truncation``, and false means raise rather than
    trim. Silently scoring the first N tokens of a document the caller believes
    was read in full is exactly what they turned it off to avoid.

    ``return_embeddings`` is refused rather than ignored on a model that has no
    document vector. Accepting it and answering without one would be a 200 that
    is missing the thing the caller asked for, with nothing in the response to
    say so.
    """
    family = _require_rerank()
    if return_embeddings and not family.embeds_documents:
        raise BadRequest(
            f"{settings.short_model_id} scores each query-document pair "
            "directly and never forms a document vector, so there is no "
            "embedding for it to return.",
            field="return_embeddings",
        )
    texts = [document_text(document) for document in documents]
    if max_doc_length:
        texts = [_clip(text, max_doc_length) for text in texts]
    if not truncate:
        _refuse_over_length(token_lengths([query] + texts))
    n_tokens = count_tokens([query] + texts, cap=SPEC.context)

    start = time.perf_counter()
    with _MODEL_LOCK, torch.inference_mode():
        ranked = family.rank(query, texts, top_n, return_embeddings=return_embeddings)
    elapsed = time.perf_counter() - start

    telemetry.record(n_tokens, elapsed)
    logger.info(
        f"Reranked {len(documents)} documents | {n_tokens} tokens | "
        f"{elapsed * 1000:.0f}ms"
    )

    results = []
    for item in ranked:
        # `document` and `embedding` are omitted, never null, when not asked
        # for -- measured on api.jina.ai for every reranker.
        result = {
            "index": item.index,
            "relevance_score": serialize.relevance_score(item.score),
        }
        if return_documents:
            result["document"] = family.render_document(documents[item.index])
        if item.embedding is not None:
            result["embedding"] = item.embedding
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


def _split_inputs(items: list) -> tuple[list, list[str], list[int]]:
    """Return (encode inputs, the text parts token counting sees, and how many
    of those parts each row contributed).

    The last one exists so the caller can redistribute one batched tokenizer
    pass back over the rows: a fused block contributes several texts, bare
    media none, so the two lists do not line up index for index.
    """
    inputs, texts, per_row = [], [], []
    for item in items:
        if isinstance(item, list):
            # Put non-text parts first in the tuple. The model's
            # _encode_single_image has a bug when reached via the text-first
            # shortcut; non-text-first forces _encode_composite_parts, which
            # handles every ordering correctly.
            inputs.append(tuple(sorted(item, key=lambda part: isinstance(part, str))))
            parts = [part for part in item if isinstance(part, str)]
            texts.extend(parts)
            per_row.append(len(parts))
        elif isinstance(item, str):
            inputs.append(item)
            texts.append(item)
            per_row.append(1)
        else:
            # ST 3.4.1's _text_length() raises on PIL/BytesIO inside a tuple, so
            # standalone media goes in bare and ST routes it through custom_st's
            # _encode_single_image, which handles pure-image input correctly.
            inputs.append(item)
            per_row.append(0)
    return inputs, texts, per_row


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
