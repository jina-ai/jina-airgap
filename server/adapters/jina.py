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

import engine
import serialize
import tasks
from config import settings
from errors import (
    BadRequest,
    JinaError,
    jina_error_body,
    jina_validation_body,
)
from media import _parse_openai_item as parse_openai_item

router = APIRouter(tags=["Jina"])

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
            "embedding_types",
        ),
    )
    # Voyage's, and deliberately not an alias of the field above: theirs are
    # two orthogonal choices -- which numbers, and whether to base64 them --
    # where Jina's one field encodes four of the pairs. Aliased together, the
    # first one pydantic matched silently discarded the other.
    output_dtype: Optional[Literal["float", "int8", "uint8", "binary", "ubinary"]] = (
        None
    )
    late_chunking: bool = False
    return_multivector: bool = False
    return_tokenized_input: bool = False
    input_type: Optional[str] = None

    @field_validator("embedding_type", mode="before")
    @classmethod
    def _single_value(cls, value):
        # Cohere's spelling of this field is a list, because Cohere returns
        # every type asked for at once. This envelope has one `embedding` key
        # per item and nowhere to put a second encoding, so a one-element list
        # is unwrapped and a longer one is sent to the route that can answer
        # it. (The public API types this as `Union[Literal, List[Literal]]`
        # and returns HTTP 500 on the list form; that is not worth copying.)
        if isinstance(value, list):
            if len(value) == 1:
                return value[0]
            raise ValueError(
                "embedding_type takes one value here; POST /v2/embed returns "
                "several at once. Accepted: " + ", ".join(serialize.NATIVE_TYPES)
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
def create_embeddings(request: EmbeddingsRequest):
    """Jina `/v1/embeddings`. Also serves OpenAI, Voyage and Gemini-shaped
    requests, which differ from Jina's only in field naming."""
    reject_foreign_model(request.model)
    family = engine.embedding_family()
    vectors, n_tokens, _ = engine.embed(
        embedding_items(request.input),
        task=resolve_task(request),
        dimensions=request.dimensions,
        normalized=request.normalized,
        extra=inference_kwargs(family, request),
        truncate=request.truncate,
    )
    return serialize.embeddings_response(
        settings.short_model_id,
        serialize.usage(n_tokens, prompt_tokens=True),
        embedding_data(vectors, request),
    )


@router.post("/v1/rerank")
def rerank(request: RerankRequest):
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
    """Any `model` that is not this image's is a deployment mistake.

    The provider adapters exist to share other vendors' *schemas*, not their
    catalogues -- no OpenAI, Cohere, Voyage or Gemini deployment has a Jina
    model in it. So `text-embedding-3-small` arriving here is not a drop-in
    working as intended, it is a client pointed at the wrong container, and
    answering it with Jina vectors under a name the caller did not ask for
    hides that. Omitting the field is still fine: there is nothing to disagree
    with, and one image serves one model.
    """
    if not requested:
        return
    if requested in (settings.model_id, settings.short_model_id):
        return
    if requested.split("/")[-1] == settings.short_model_id:
        return
    raise BadRequest(
        f"This image serves '{settings.short_model_id}', not '{requested}'. "
        f"One model per image: point the client at the '{requested}' image, "
        f"set 'model' to '{settings.short_model_id}', or drop the field.",
        field="model",
    )


def resolve_task(request: EmbeddingsRequest) -> Optional[str]:
    if request.input_type:
        return input_type_task(request.input_type)
    return native_task(request.task)


def native_task(task: Optional[str]) -> Optional[str]:
    """Jina's own ``task``, except when it is carrying a Gemini value.

    ``task_type`` is an accepted alias for this field, which is Gemini's
    spelling -- so accepting the name while rejecting the values it can hold
    left a Gemini-shaped client failing on a field we advertised. The two
    namespaces do not overlap (``RETRIEVAL_QUERY`` against ``retrieval.query``),
    so a value from theirs is translated and everything else is Jina's own,
    refused by the core if it is wrong.
    """
    if not task:
        return None
    roles = tasks.TASK_TYPE_ROLES.get(task)
    return engine.fit_task(*roles) if roles else task


def input_type_task(input_type: Optional[str]) -> Optional[str]:
    """Translate a vendor's ``input_type`` into a task this model has.

    Shared by every schema with this field, and it has to avoid two opposite
    mistakes. The old ``.get(value, "retrieval")`` substituted a default for a
    value outside the vendor's own enum, so ``search_documnt`` was served
    query vectors and a 200. Replacing it with a strict lookup then refused
    ``image``, which IS one of Cohere's five -- and a Cohere client sending an
    image has no other value it could use.

    So: outside the vendor's enum is refused, inside it is always answered.
    """
    if not input_type:
        return None
    roles = tasks.INPUT_TYPE_ROLES.get(input_type)
    if roles is None:
        raise BadRequest(
            f"Unknown input_type '{input_type}'. Valid values: "
            f"{', '.join(sorted(tasks.INPUT_TYPE_ROLES))}.",
            field="input_type",
        )
    return engine.fit_task(*roles)


def embedding_items(raw: Union[str, dict, list]) -> list:
    """Turn the request's `input` into what `engine.embed` consumes: strings,
    decoded media, or a list of parts meaning one fused embedding.

    The multimodal check runs on what the parts turn out to *be*, not on how
    the request was written. `{"text": "..."}` and `{"type": "text", "text":
    "..."}` are Jina's own TextDoc forms and are plain text however they are
    spelled, so a text-only model must serve them -- checking first rejected
    them with "this model is text-only", which is both wrong and confusing.
    """
    items = raw if isinstance(raw, list) else [raw]
    if all(isinstance(item, str) for item in items):
        return items
    parsed = [parse_openai_item(item) for item in items]
    if any(not isinstance(part, str) for parts in parsed for part in parts):
        engine.require_multimodal()
    return [parts[0] if len(parts) == 1 else parts for parts in parsed]


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
    dtype, as_base64 = serialize.NATIVE_TYPES[request.embedding_type]
    if request.output_dtype:
        # More specific than the shorthand: it names the numbers outright,
        # leaving the shorthand to say only whether they are base64.
        dtype = request.output_dtype
    if request.return_multivector:
        return [
            serialize.multivector_item(index, matrix, dtype, as_base64=as_base64)
            for index, matrix in enumerate(vectors)
        ]
    return [
        serialize.embedding_item(index, vector, dtype, as_base64=as_base64)
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
