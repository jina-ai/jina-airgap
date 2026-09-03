"""Wire rendering for the Jina-native envelope.

Everything here is pure: it turns what ``engine`` produces into the exact JSON
``api.jina.ai`` returns. It lives apart from the route handlers because the
provider adapters re-use the same encoders, and because the response shape is
the one thing this server exists to define in a single place.

The envelope, as ``api.jina.ai`` returns it:

    {"model": ..., "object": "list", "usage": {...}, "data": [...]}
    {"model": ..., "object": "list", "usage": {...}, "results": [...]}

Key order matches the wire. It does not affect JSON equality, but it makes a
diff against a captured response readable, and every value here is pinned
against a real one.
"""

import base64
from typing import Any, Optional, Union

import numpy as np

# What an embedding can be made of, and how it can be written down. These are
# two independent choices, and only Voyage's schema says so out loud:
# `output_dtype` picks the numbers, `encoding_format` picks the wire form. Jina
# and OpenAI have one field for both, whose four values are four of the pairs.
#
# Collapsing them into one field is what made `{"output_dtype": "binary",
# "encoding_format": "base64"}` -- a legal Voyage request meaning "packed bits,
# base64" -- return base64 floats, because whichever alias pydantic matched
# first won and the other was dropped.
DTYPES = ("float", "int8", "uint8", "binary", "ubinary")

# Jina's own `embedding_type`, as the (dtype, base64) pair each value means.
NATIVE_TYPES = {
    "float": ("float", False),
    "base64": ("float", True),
    "binary": ("binary", False),
    "ubinary": ("ubinary", False),
}

# Buffer type for the base64 form. Voyage documents base64 as "a Base64-encoded
# NumPy array ... dtype follows output_dtype", so the bytes have to be the
# natural width, not float32 of an integer value.
_BUFFERS = {
    "float": "<f4",
    "int8": "int8",
    "binary": "int8",
    "uint8": "uint8",
    "ubinary": "uint8",
}


def quantize(vector: np.ndarray, dtype: str) -> np.ndarray:
    """The numbers, before any decision about how to write them down.

    int8 / uint8 are the symmetric mapping for unit-norm vectors. Cohere and
    Voyage both publish the value set but neither publishes its quantiser, so
    these are correctly *ranged* and not bit-comparable with their output.

    binary / ubinary are one sign bit per dimension, packed MSB-first.
    `ubinary` is the packed byte as-is; `binary` is that byte minus 128 --
    offset binary, NOT a two's-complement reinterpretation. Measured against
    api.jina.ai on jina-embeddings-v2-base-en with identical input: the hosted API
    returns ubinary [50, 193, 128, 255, ...] and binary [-78, 65, 0, 127, ...],
    which is exactly ubinary-128 and not the int8 view (which would give
    [50, -63, -128, -1, ...]).

    Everything stays float32 so the JSON form is floats in range rather than
    integers, which is what the hosted API emits.
    """
    values = np.asarray(vector)
    if dtype == "float":
        return values
    if dtype == "int8":
        return np.clip(np.rint(values * 127), -128, 127)
    if dtype == "uint8":
        return np.clip(np.rint((values + 1) * 127.5), 0, 255)
    if dtype in ("binary", "ubinary"):
        packed = np.packbits(values > 0).astype(np.float32)
        return packed - 128.0 if dtype == "binary" else packed
    # Not defensive: `base64` used to be a member of this value set, and
    # falling through to packbits would answer it with bits and a 200.
    raise ValueError(f"Unknown dtype {dtype!r}; expected one of {', '.join(DTYPES)}")


def encode_vector(
    vector: np.ndarray, dtype: str = "float", *, as_base64: bool = False
) -> Union[list, str]:
    """Render one embedding: the numbers, then the wire form."""
    values = quantize(vector, dtype)
    if as_base64:
        return base64.b64encode(values.astype(_BUFFERS[dtype]).tobytes()).decode(
            "ascii"
        )
    return values.tolist()


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


def embedding_item(
    index: int, vector: np.ndarray, dtype: str, *, as_base64: bool = False
) -> dict:
    return {
        "object": "embedding",
        "index": index,
        "embedding": encode_vector(vector, dtype, as_base64=as_base64),
    }


def multivector_item(
    index: int,
    vectors: np.ndarray,
    dtype: str,
    tokens: Optional[list] = None,
    *,
    as_base64: bool = False,
) -> dict:
    """Per-token embeddings: the key turns plural and so does `object`."""
    item: dict[str, Any] = {
        "object": "embeddings",
        "index": index,
        "embeddings": [
            encode_vector(vector, dtype, as_base64=as_base64) for vector in vectors
        ],
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
