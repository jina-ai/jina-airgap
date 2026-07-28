"""Cohere, on Cohere's real paths.

Cohere's current API is v2-only -- `POST /v2/embed`, `POST /v2/rerank` -- and
the container already returned v2-shaped bodies from `/v1/embed`, a path
Cohere does not have. Serving them at `/v2/*` is honest implementation rather
than an alias, and it gives Elasticsearch's `cohere` service the `/v2/rerank`
it insists on. `/v1/embed` is deleted; backwards compatibility is waived.

Contract fetched from docs.cohere.com/reference/{embed,rerank}, 2026-07-27.
"""

import uuid
from typing import Literal, Optional

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

import engine
import serialize
from errors import BadRequest, JinaError, request_id
from media import _bytes_to_st_input as bytes_to_st_input, _decode_b64 as decode_b64
from media import fuse_content

from .jina import INPUT_TYPE_TASKS, reject_foreign_model

router = APIRouter(tags=["Cohere"])

EmbeddingType = Literal["float", "int8", "uint8", "binary", "ubinary", "base64"]

# Cohere's rerank default. Documented as silent per-document truncation, so it
# is applied rather than ignored: an unapplied truncation directive changes
# every relevance score with nothing on the wire to say so.
DEFAULT_MAX_TOKENS_PER_DOC = 4096


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    input_type: str
    texts: Optional[list[str]] = None
    images: Optional[list[str]] = None
    inputs: Optional[list[dict]] = None
    embedding_types: list[EmbeddingType] = Field(default_factory=lambda: ["float"])
    truncate: Literal["NONE", "START", "END"] = "END"
    output_dimension: Optional[int] = Field(default=None, gt=0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    # Cohere's cloud queue priority. No queue here, so it is accepted and has
    # no effect -- stated rather than silently dropped.
    priority: Optional[int] = None


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    query: str
    documents: list[str] = Field(min_length=1)
    top_n: Optional[int] = Field(default=None, ge=1)
    max_tokens_per_doc: int = DEFAULT_MAX_TOKENS_PER_DOC
    priority: Optional[int] = None


@router.post("/v2/embed")
async def embed(request: EmbedRequest):
    reject_foreign_model(request.model)
    task = INPUT_TYPE_TASKS.get(request.input_type, "retrieval")

    if request.inputs is not None:
        engine.require_multimodal()
        items = [fuse_content(item.get("content", [])) for item in request.inputs]
        texts, image_count = [], len(items)
    elif request.images is not None:
        engine.require_multimodal()
        items = [bytes_to_st_input(*decode_b64(url)) for url in request.images]
        texts, image_count = request.texts or [], len(request.images)
    else:
        items = request.texts or []
        texts, image_count = items, 0
        truncate_texts(items, request.truncate, request.max_tokens)

    vectors, n_tokens, _ = engine.embed(
        items, task=task, dimensions=request.output_dimension
    )
    return {
        "id": str(uuid.uuid4()),
        "embeddings": {
            kind: [encode(vector, kind) for vector in vectors]
            for kind in request.embedding_types
        },
        "texts": texts,
        "images": [],
        "meta": {
            "api_version": {
                "version": "2",
                "is_deprecated": False,
                "is_experimental": False,
            },
            "billed_units": {
                "input_tokens": n_tokens,
                "image_tokens": 0,
                "images": image_count,
                "search_units": 0,
                "classifications": 0,
            },
            "tokens": {"input_tokens": n_tokens, "output_tokens": 0},
            "cached_tokens": 0,
            "warnings": [],
        },
    }


@router.post("/v2/rerank")
async def rerank(request: RerankRequest):
    reject_foreign_model(request.model)
    results, n_tokens, _ = engine.rerank(
        request.query,
        request.documents,
        top_n=request.top_n,
        return_documents=False,
        max_doc_length=request.max_tokens_per_doc,
    )
    return {
        "id": str(uuid.uuid4()),
        "results": results,
        "meta": {
            "api_version": {
                "version": "2",
                "is_deprecated": False,
                "is_experimental": False,
            },
            "billed_units": {
                "search_units": 1,
                "input_tokens": n_tokens,
                "output_tokens": 0,
            },
            "tokens": {"input_tokens": n_tokens, "output_tokens": 0},
            "cached_tokens": 0,
            "warnings": [],
        },
    }


def encode(vector: np.ndarray, kind: str):
    """Cohere's `embedding_types` covers Jina's four plus int8 / uint8.

    The int8 / uint8 mapping is the symmetric one for unit-norm vectors --
    Cohere does not publish its quantiser, so these are correctly *ranged* but
    not bit-comparable with Cohere's own output.
    """
    if kind == "int8":
        return np.clip(np.rint(np.asarray(vector) * 127), -128, 127).tolist()
    if kind == "uint8":
        return np.clip(np.rint((np.asarray(vector) + 1) * 127.5), 0, 255).tolist()
    return serialize.encode_vector(vector, kind)


def truncate_texts(texts: list[str], direction: str, max_tokens: Optional[int]) -> None:
    """`truncate` says which end to drop, `max_tokens` how much to keep.
    `NONE` means refuse rather than shorten."""
    limit = max_tokens or engine.SPEC.context
    if not limit:
        return
    tokenizer = engine.tokenizer()
    for index, text in enumerate(texts):
        ids = (
            tokenizer(text, add_special_tokens=False)["input_ids"]
            if tokenizer
            else text.split()
        )
        if len(ids) <= limit:
            continue
        if direction == "NONE":
            raise BadRequest(
                f"Document at index {index} is {len(ids)} tokens, over the "
                f"{limit}-token limit, and truncate is NONE."
            )
        if tokenizer is None:
            continue
        kept = ids[-limit:] if direction == "START" else ids[:limit]
        texts[index] = tokenizer.decode(kept, skip_special_tokens=True)


def error(exc: JinaError) -> dict:
    """Cohere's error body is a flat `{message, id}` on every status."""
    return {"message": exc.message, "id": request_id()}


def validation(errors: list) -> dict:
    first = errors[0] if errors else {}
    field = " -> ".join(str(part) for part in first.get("loc", ()))
    return {
        "message": f"invalid request: {field}: {first.get('msg', 'invalid')}",
        "id": request_id(),
    }
