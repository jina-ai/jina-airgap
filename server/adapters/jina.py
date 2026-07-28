"""Jina's own contract, on Jina's own paths.

`/v1/embeddings` and `/v1/rerank` are what `api.jina.ai` serves, and the
responses here are byte-compatible with it -- that parity is the gate this
rewrite is measured against.

The request deliberately accepts each provider's field names as aliases,
because the public API does exactly the same thing (`AliasChoices` throughout
`sefo-gcp` `models/embedding.py` and `models/reranking.py`): OpenAI's
`encoding_format`, Voyage's `output_dtype` / `truncation` / `top_k`, Gemini's
`task_type` / `output_dimensionality`, Cohere's `texts` / `max_tokens_per_doc`.
That is a field union, not an adapter. Anything a provider has that Jina does
not -- OpenAI's `user`, Cohere's `priority` -- is accepted and ignored by
`extra="ignore"`; anything that changes semantics lives in that provider's
adapter.
"""

from typing import Literal, Optional, Union

from fastapi import APIRouter
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

import catalog
import engine
import serialize
from config import settings
from errors import (
    BadRequest,
    JinaError,
    UnprocessableEntity,
    jina_error_body,
    jina_validation_body,
)
from media import _parse_openai_item as parse_openai_item

router = APIRouter(tags=["Jina"])

# Voyage and Cohere both express "is this a query or a document" as
# `input_type`; Jina expresses it as a task suffix. Same concept, so it maps
# rather than living in an adapter.
INPUT_TYPE_TASKS = {
    "query": "retrieval.query",
    "document": "retrieval.passage",
    "search_query": "retrieval.query",
    "search_document": "retrieval.passage",
    "classification": "classification",
    "clustering": "clustering",
}

# Native parameters that change the forward pass rather than the wire format.
# Forwarded only when the loaded model's encode() genuinely takes them.
INFERENCE_PARAMS = ("late_chunking", "return_multivector")


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    input: Union[str, dict, list[Union[str, dict]]] = Field(
        validation_alias=AliasChoices("input", "content", "texts")
    )
    model: Optional[str] = None
    task: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("task", "task_type")
    )
    dimensions: Optional[int] = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices(
            "dimensions", "output_dimensionality", "output_dimension"
        ),
    )
    normalized: bool = True
    truncate: bool = Field(
        default=False, validation_alias=AliasChoices("truncate", "truncation")
    )
    embedding_type: Literal["float", "base64", "binary", "ubinary"] = Field(
        default="float",
        validation_alias=AliasChoices(
            "embedding_type",
            "encoding_type",
            "encoding_format",
            "output_dtype",
            "embedding_types",
        ),
    )
    late_chunking: bool = False
    return_multivector: bool = False
    return_tokenized_input: bool = False
    input_type: Optional[str] = None

    @field_validator("embedding_type", mode="before")
    @classmethod
    def _single_value(cls, value):
        # The public API types this as `Union[Literal, List[Literal]]` but
        # returns HTTP 500 on the list form. Reject it at validation rather
        # than reproduce the crash.
        if isinstance(value, list):
            raise ValueError(
                "embedding_type must be a single value, not a list. Accepted: "
                + ", ".join(serialize.EMBEDDING_TYPES)
            )
        return value if value is not None else "float"

    @field_validator("input")
    @classmethod
    def _non_empty(cls, value):
        if isinstance(value, list) and not value:
            raise ValueError("Input list cannot be empty. Provide at least one item.")
        return value

    @model_validator(mode="after")
    def _compatible_options(self):
        if self.return_multivector and self.dimensions is not None:
            raise ValueError(
                "Incompatible options: 'dimensions' cannot be set when "
                "'return_multivector' is True. Multi-vector embeddings use the "
                "full dimension size."
            )
        if self.return_tokenized_input and not self.return_multivector:
            raise ValueError(
                "Incompatible options: 'return_tokenized_input' requires "
                "'return_multivector' to be True."
            )
        return self


class RerankRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    query: Union[str, dict]
    documents: list[Union[str, dict]] = Field(min_length=1)
    model: Optional[str] = None
    top_n: Optional[int] = Field(
        default=None, ge=1, validation_alias=AliasChoices("top_n", "top_k")
    )
    return_documents: bool = True
    max_doc_length: Optional[int] = Field(
        default=None,
        ge=1,
        le=8192,
        validation_alias=AliasChoices("max_doc_length", "max_tokens_per_doc"),
    )


@router.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    """Jina `/v1/embeddings`. Also serves OpenAI, Voyage and Gemini-shaped
    requests, which differ from Jina's only in field naming."""
    reject_foreign_model(request.model)
    family = engine.embedding_family()
    check_task(family, request.task)
    check_dimensions(request.dimensions)

    items = embedding_items(request.input)
    enforce_truncation(items, request.truncate)

    vectors, n_tokens, _ = engine.embed(
        items,
        task=resolve_task(request),
        dimensions=request.dimensions,
        normalized=request.normalized,
        extra=inference_kwargs(family, request),
    )
    return serialize.embeddings_response(
        settings.short_model_id,
        serialize.usage(n_tokens, prompt_tokens=wants_prompt_tokens(request)),
        embedding_data(vectors, request),
    )


def wants_prompt_tokens(request: EmbeddingsRequest) -> bool:
    rule = engine.SPEC.usage_prompt_tokens
    return rule == "always" or (rule == "late_chunking" and request.late_chunking)


@router.post("/v1/rerank")
async def rerank(request: RerankRequest):
    """Jina `/v1/rerank`. Voyage's `top_k` and Cohere's `max_tokens_per_doc`
    are accepted here as aliases; the Cohere *response* shape lives in
    `/v2/rerank`."""
    reject_foreign_model(request.model)
    results, n_tokens, _ = engine.rerank(
        query_text(request.query),
        request.documents,
        top_n=request.top_n,
        return_documents=request.return_documents,
        max_doc_length=request.max_doc_length,
    )
    return serialize.rerank_response(
        settings.short_model_id,
        serialize.usage(n_tokens, prompt_tokens=False),
        results,
    )


# --- request helpers, shared with the provider adapters --------------------


def reject_foreign_model(requested: Optional[str]) -> None:
    """A Jina model id that is not this image's model is a deployment mistake
    and must be visible: one image serves one model, so the request would
    otherwise be answered by the wrong one and echo a model the caller never
    asked for.

    A name that is not a Jina model at all -- `text-embedding-3-small`,
    `rerank-v3.5` -- is another provider's, and the container advertises itself
    as a drop-in for those, so it is accepted and ignored. This is narrower
    than "400 on any mismatch": that would break the documented
    `service: openai` Elasticsearch scenario, which is the integration this
    rewrite exists to fix.
    """
    if not requested:
        return
    if requested in (settings.model_id, settings.short_model_id):
        return
    if requested.split("/")[-1] == settings.short_model_id:
        return
    if catalog.is_known(requested):
        raise BadRequest(
            f"This image serves '{settings.short_model_id}', not "
            f"'{requested}'. One model per image: deploy the "
            f"'{requested}' image, or drop the 'model' field.",
            field="model",
        )


def check_task(family, task: Optional[str]) -> None:
    """Reject a task the loaded model does not know, so a typo cannot silently
    select a different LoRA adapter."""
    if not task:
        return
    known = family.known_tasks
    if not known:
        return
    if task.partition(".")[0] in known or task in (family.prompts or ()):
        return
    raise BadRequest(
        f"Unknown task '{task}' for {settings.short_model_id}. "
        f"This model accepts: {', '.join(sorted(known))} "
        f"(optionally suffixed .query or .passage).",
        field="task",
    )


def resolve_task(request: EmbeddingsRequest) -> Optional[str]:
    if request.input_type:
        return INPUT_TYPE_TASKS.get(request.input_type, "retrieval")
    return request.task


def embedding_items(raw: Union[str, dict, list]) -> list:
    """Turn the request's `input` into what `engine.embed` consumes: strings,
    decoded media, or a list of parts meaning one fused embedding."""
    items = raw if isinstance(raw, list) else [raw]
    if all(isinstance(item, str) for item in items):
        return items
    engine.require_multimodal()
    parsed = []
    for item in items:
        parts = parse_openai_item(item)
        parsed.append(parts if len(parts) > 1 else parts[0])
    return parsed


def check_dimensions(dimensions: Optional[int]) -> None:
    """The public API caps `dimensions` at the model's own output dimension and
    422s above it; without the check a too-large value silently no-ops."""
    limit = engine.SPEC.output_dim
    if dimensions and limit and dimensions > limit:
        raise UnprocessableEntity(
            f"Input should be less than or equal to {limit}",
            field="body -> dimensions",
            code="less_than_equal",
        )


def enforce_truncation(items: list, truncate: bool) -> None:
    """`truncate=false` (the default) means "error rather than silently
    shorten". Measured on api.jina.ai: over-length input returns HTTP 400
    `INPUT_TOKEN_LIMIT_EXCEEDED`, and only `truncate=true` succeeds. Left to
    sentence-transformers the input would be clipped with nothing on the wire
    to say so, which changes the embedding invisibly."""
    if truncate:
        return
    limit = engine.SPEC.context
    if not limit:
        return
    for item in items:
        if not isinstance(item, str):
            continue
        tokens = engine.count_tokens([item])
        if tokens > limit:
            raise BadRequest(
                f"Input text exceeds the model's maximum of {limit} tokens. "
                f"Use 'truncate: true' to automatically truncate, or split "
                f"into smaller chunks.",
                field="input",
                code="INPUT_TOKEN_LIMIT_EXCEEDED",
            )


def inference_kwargs(family, request: EmbeddingsRequest) -> dict:
    """Forward the forward-pass-changing parameters, or refuse them.

    sentence-transformers drops undeclared encode kwargs silently, so a
    `late_chunking` request against a model without it would return ordinary
    embeddings and look like a success.
    """
    if request.return_tokenized_input:
        raise BadRequest(
            "return_tokenized_input is not implemented in this build.",
            field="return_tokenized_input",
        )
    accepted = family.encode_kwargs
    extra = {}
    for name in INFERENCE_PARAMS:
        if not getattr(request, name):
            continue
        if name not in accepted:
            raise BadRequest(
                f"'{name}' is not supported by {settings.short_model_id}.",
                field=name,
            )
        extra[name] = True
    return extra


def embedding_data(vectors, request: EmbeddingsRequest) -> list:
    if request.return_multivector:
        return [
            serialize.multivector_item(index, matrix, request.embedding_type)
            for index, matrix in enumerate(vectors)
        ]
    return [
        serialize.embedding_item(index, vector, request.embedding_type)
        for index, vector in enumerate(vectors)
    ]


def query_text(query: Union[str, dict]):
    """A rerank query is a string, or `{"text": ...}` / `{"image": ...}` for
    the multimodal reranker."""
    if isinstance(query, str):
        return query
    if "text" in query:
        return query["text"]
    if "image" in query:
        return parse_openai_item(query)[0]
    raise BadRequest("Query object must have a 'text' or 'image' key.", field="query")


# --- error rendering -------------------------------------------------------


def error(exc: JinaError) -> dict:
    return jina_error_body(exc)


def validation(errors: list) -> dict:
    return jina_validation_body(errors)
