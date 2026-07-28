"""Wire rendering for the Jina-native envelope.

Everything here is pure: it turns what ``engine`` produces into the exact JSON
``api.jina.ai`` returns. It lives apart from the route handlers because the
provider adapters re-use the same encoders, and because the response shape is
the one thing this rewrite exists to define in a single place.

The envelope, as ``api.jina.ai`` returns it:

    {"model": ..., "object": "list", "usage": {...}, "data": [...]}
    {"model": ..., "object": "list", "usage": {...}, "results": [...]}

Key order matches the wire. It does not affect JSON equality, but it makes a
diff against a captured response readable. ``tests/test_contracts.py`` pins
every value here against one.
"""

import base64
from typing import Any, Optional, Union

import numpy as np

# Jina's own value set. `encoding_format` (OpenAI: float|base64) and
# `output_dtype` (Voyage: adds int8|uint8) alias onto this field, so a Voyage
# client asking for int8 gets a 422 naming these four rather than a silent
# float array. Widening the set here would drift the native contract away from
# the public API, which is the one thing the golden gate protects.
EMBEDDING_TYPES = ("float", "base64", "binary", "ubinary")


def encode_vector(vector: np.ndarray, embedding_type: str) -> Union[list, str]:
    """Render one embedding in the requested wire encoding."""
    if embedding_type == "float":
        return vector.tolist()
    if embedding_type == "base64":
        return base64.b64encode(vector.astype("<f4").tobytes()).decode("ascii")
    # binary / ubinary: one sign bit per dimension, packed MSB-first.
    #
    # `ubinary` is the packed byte as-is. `binary` is that byte minus 128 --
    # offset binary, NOT a two's-complement reinterpretation. Measured against
    # api.jina.ai on jina-embeddings-v2-base-en with identical input: prod
    # returns ubinary [50, 193, 128, 255, ...] and binary [-78, 65, 0, 127,
    # ...], which is exactly ubinary-128 and not the int8 view (which would
    # give [50, -63, -128, -1, ...]).
    #
    # Both come back as JSON floats in range rather than integers, so the cast
    # is to float32 and not to int.
    packed = np.packbits(np.asarray(vector) > 0).astype(np.float32)
    if embedding_type == "binary":
        packed -= 128.0
    return packed.tolist()


def usage(total_tokens: int, *, prompt_tokens: bool) -> dict:
    """The `usage` block. `prompt_tokens`, when present, is the same number.

    Embeddings always carry it; reranking never does, which is what
    `api.jina.ai` returns for every reranker.

    On the embeddings side the public API is inconsistent -- v2 and CLIP always
    emit it, v3 emits it only under `late_chunking` or when `input` is a dict,
    v4 never does -- and reproducing that faithfully would mean a response
    envelope that changes with the request's syntax. It is emitted
    unconditionally instead: it is a duplicate of `total_tokens`, so a client
    loses nothing, and OpenAI's own schema declares it required, so a client
    reading `usage.prompt_tokens` through an OpenAI SDK gets an int rather than
    None.
    """
    block = {"total_tokens": total_tokens}
    if prompt_tokens:
        block["prompt_tokens"] = total_tokens
    return block


def embedding_item(index: int, vector: np.ndarray, embedding_type: str) -> dict:
    return {
        "object": "embedding",
        "index": index,
        "embedding": encode_vector(vector, embedding_type),
    }


def multivector_item(
    index: int,
    vectors: np.ndarray,
    embedding_type: str,
    tokens: Optional[list] = None,
) -> dict:
    """Per-token embeddings: the key turns plural and so does `object`."""
    item: dict[str, Any] = {
        "object": "embeddings",
        "index": index,
        "embeddings": [encode_vector(vector, embedding_type) for vector in vectors],
    }
    if tokens is not None:
        item["tokenized_input"] = tokens
    return item


def embeddings_response(model: str, usage_block: dict, data: list) -> dict:
    return {"model": model, "object": "list", "usage": usage_block, "data": data}


def rerank_response(model: str, usage_block: dict, results: list) -> dict:
    return {"model": model, "object": "list", "usage": usage_block, "results": results}


def relevance_score(score: float) -> float:
    """8 decimal places, matching `RerankingResult.round_relevance_score` in the
    public schema -- without it the last digits of a float32 score leak into the
    response and no two runs render identically."""
    return round(float(score), 8)
