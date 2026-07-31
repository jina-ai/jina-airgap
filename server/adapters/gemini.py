"""Google Gemini `embedContent` / `batchEmbedContents`.

Contract from ai.google.dev/api/embeddings, fetched 2026-07-27.

Two request forms are accepted. The flat `taskType` / `title` /
`outputDimensionality` fields are deprecated but still work, and
`embedContentConfig` is the current form; customer SDK versions are not
something an air-gapped deployment can control, so both are read and the flat
form is folded into the config.

`ContentEmbedding` also declares a `shape` field, documented as the soft-token
tensor frame shape. Plain text embeddings have no such frame and the docs give
no guarantee it is populated, so it is omitted rather than emitted as null.
"""

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

import engine
import tasks
from errors import BadRequest, JinaError, request_id
from media import _parse_content_part as parse_content_part

from .jina import reject_foreign_model

router = APIRouter(tags=["Google Gemini"])

# Canonical gRPC status names, keyed by HTTP status (google.aip.dev/193).
STATUS_NAMES = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    406: "INVALID_ARGUMENT",
    413: "INVALID_ARGUMENT",
    422: "INVALID_ARGUMENT",
    429: "RESOURCE_EXHAUSTED",
    500: "INTERNAL",
    503: "UNAVAILABLE",
}


class Content(BaseModel):
    model_config = ConfigDict(extra="ignore")
    parts: list = []
    role: Optional[str] = None


class EmbedContentRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: Content
    embedContentConfig: Optional[dict] = None
    taskType: Optional[str] = None
    title: Optional[str] = None
    outputDimensionality: Optional[int] = None

    def config(self) -> dict:
        """Flat fields folded into the config object; the config wins where
        both are set, since it is the non-deprecated form."""
        merged = {
            "taskType": self.taskType,
            "title": self.title,
            "outputDimensionality": self.outputDimensionality,
        }
        merged.update(
            {k: v for k, v in (self.embedContentConfig or {}).items() if v is not None}
        )
        return merged


class BatchEmbedRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    requests: list[dict] = []


@router.post("/v1/models/{model_id}:embedContent")
def embed_content(model_id: str, request: EmbedContentRequest):
    reject_foreign_model(model_id)
    config = request.config()
    item = content_item(request.content.model_dump())

    vectors, n_tokens, _ = engine.embed(
        [item],
        task=_task_of(config),
        dimensions=config.get("outputDimensionality"),
        truncate=_truncate_of(config),
    )
    return {
        "embedding": {"values": vectors[0].tolist()},
        "usageMetadata": {"promptTokenCount": n_tokens, "promptTokenDetails": []},
    }


@router.post("/v1/models/{model_id}:batchEmbedContents")
def batch_embed_contents(model_id: str, request: BatchEmbedRequest):
    reject_foreign_model(model_id)
    # batchEmbedContents allows a per-request model override; one image serves
    # one model, so a foreign Jina id in any sub-request is still a mistake.
    for sub in request.requests:
        reject_foreign_model(sub.get("model"))

    # `taskType` and `outputDimensionality` are per sub-request, which is the
    # only reason the field lives there rather than on the batch. Collapsing
    # them onto the first non-empty value -- what this did -- encoded an
    # index-building batch of one query and one passage entirely as queries,
    # and returned 200. Sub-requests are grouped by the pair instead, so each
    # row gets the encoding it asked for; rows sharing a pair still share one
    # forward, so the common case costs no extra call.
    groups: dict = {}
    for position, sub in enumerate(request.requests):
        config = _config_of(sub)
        key = (
            _task_of(config),
            config.get("outputDimensionality"),
            _truncate_of(config),
        )
        groups.setdefault(key, []).append(
            (position, content_item(sub.get("content", {})))
        )

    ordered: list = [None] * len(request.requests)
    n_tokens = 0
    for (task, dimensions, truncate), entries in groups.items():
        vectors, tokens, _ = engine.embed(
            [item for _, item in entries],
            task=task,
            dimensions=dimensions,
            truncate=truncate,
        )
        n_tokens += tokens
        for (position, _), vector in zip(entries, vectors):
            ordered[position] = vector

    return {
        "embeddings": [{"values": vector.tolist()} for vector in ordered],
        "usageMetadata": {"promptTokenCount": n_tokens, "promptTokenDetails": []},
    }


def _task_of(config: dict) -> Optional[str]:
    """Gemini's `taskType`, as a task this model has.

    Substituting a default for an unrecognised task type answered a typo with
    retrieval vectors and a 200; refusing everything outside Gemini's enum then
    rejected the tasks this model actually serves, which Gemini has no value
    for. Both of Gemini's nine and this model's own names are answered; a typo
    is neither.

    An absent `taskType` no longer forces ``retrieval`` either. Gemini treats
    it as unspecified, and unspecified is each model's own default -- the same
    answer `/v1/embeddings` gives when `task` is omitted, which is the point of
    serving both schemas from one core.
    """
    task_type = config.get("taskType")
    if not task_type:
        return None
    roles = tasks.TASK_TYPE_ROLES.get(task_type)
    if roles is not None:
        return engine.fit_task(*roles, field="taskType", value=task_type)
    return engine.named_task(
        task_type, field="taskType", vendor_values=tasks.TASK_TYPE_ROLES
    )


def _truncate_of(config: dict) -> bool:
    """`embedContentConfig.autoTruncate`, which was read and then dropped.

    Only an explicit false changes anything: absent means the caller expressed
    no opinion, and this route has always trimmed in that case. False is a
    request to be told, and answering it with a quietly shortened embedding is
    the failure it exists to prevent.
    """
    return config.get("autoTruncate") is not False


def _config_of(sub: dict) -> dict:
    merged = {
        key: sub.get(key) for key in ("taskType", "title", "outputDimensionality")
    }
    merged.update(
        {
            k: v
            for k, v in (sub.get("embedContentConfig") or {}).items()
            if v is not None
        }
    )
    return merged


def content_item(content):
    """A Gemini `content` is a parts array. Text-only parts are joined into one
    string; anything with `inlineData` makes the whole thing a fused
    multimodal block."""
    raw = content if isinstance(content, list) else (content or {}).get("parts", [])
    parts = []
    for part in raw:
        if isinstance(part, dict) and "text" in part and "inlineData" not in part:
            parts.append(part["text"])
        elif isinstance(part, dict):
            parts.extend(parse_content_part(part))
    if any(not isinstance(part, str) for part in parts):
        engine.require_multimodal()
        return parts if len(parts) > 1 else parts[0]
    return " ".join(parts)


def error(exc: JinaError) -> dict:
    return {
        "error": {
            "code": exc.status,
            "message": exc.message,
            "status": STATUS_NAMES.get(exc.status, "UNKNOWN"),
            "details": [],
        }
    }


def validation(errors: list) -> dict:
    return {
        "error": {
            "code": 400,
            "message": "; ".join(
                f"{' -> '.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', '')}"
                for e in errors
            )
            or "Request contains an invalid argument.",
            "status": "INVALID_ARGUMENT",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "INVALID_ARGUMENT",
                    "domain": "jina-on-prem",
                    "metadata": {"request_id": request_id()},
                }
            ],
        }
    }
