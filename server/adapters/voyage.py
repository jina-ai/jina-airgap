"""Voyage AI `/v1/multimodalembeddings`.

Voyage's text embeddings and rerank are served by the Jina routes: their
request fields (`input_type`, `truncation`, `output_dtype`, `top_k`) are
aliases on the native contract, and their response envelope is the same
object / data / model / usage shape. Only the multimodal path is Voyage-only.

Two things are deliberately *not* changed here.

`output_encoding` is a distinct field from `/v1/embeddings`'s
`encoding_format` -- aliasing them together would be wrong -- so it is read
under its own name.

The response body keeps the shape the container already returns. Voyage does
not publish a response schema for this endpoint (docs.voyageai.com/reference/
multimodal-embeddings-api documents the request only), so reshaping it would
be a breaking change made on inference. Recorded as an open question instead.

Error translation is a no-op: Voyage's own envelope is `{"detail": "..."}`,
measured against api.voyageai.com, which is exactly what this container
already returns.
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

import engine
import serialize
from config import settings
from errors import JinaError
from media import fuse_content

from .jina import input_type_task, reject_foreign_model

router = APIRouter(tags=["Voyage AI"])


class MultimodalRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    inputs: list[dict]
    model: Optional[str] = None
    input_type: Optional[str] = None
    truncation: bool = True
    output_encoding: Optional[str] = None


@router.post("/v1/multimodalembeddings")
def multimodal_embeddings(request: MultimodalRequest):
    """Each input is `{"content": [parts]}`, and all parts of one input fuse
    into a single embedding."""
    reject_foreign_model(request.model)
    engine.require_multimodal()

    items = [fuse_content(item.get("content", [])) for item in request.inputs]
    vectors, n_tokens, _ = engine.embed(
        items,
        task=input_type_task(request.input_type),
        truncate=request.truncation,
    )
    as_base64 = request.output_encoding == "base64"
    return {
        "embeddings": [
            serialize.encode_vector(vector, "float", as_base64=as_base64)
            for vector in vectors
        ],
        "text_tokens": n_tokens,
        "image_pixels": 0,  # not tracked: pixel counts need the decoded frames
        "total_tokens": n_tokens,
        "model": settings.short_model_id,
    }


def error(exc: JinaError) -> dict:
    return {"detail": exc.message}


def validation(errors: list) -> dict:
    first = errors[0] if errors else {}
    field = " -> ".join(str(part) for part in first.get("loc", ()))
    return {"detail": f"{field}: {first.get('msg', 'invalid request')}"}
