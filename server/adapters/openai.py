"""OpenAI `/v1/chat/completions`, and OpenAI's error envelope.

OpenAI embeddings are served by the Jina route: the two requests differ only
in field naming (`encoding_format`, `user`) and the response envelopes are
already identical. Chat completions has no Jina equivalent, and four shipped
models -- jina-vlm and the three reader models -- have no other entry point,
so it stays a first-class route.

The error envelope is the part a client actually depends on. From
`openai-openapi/master/openapi.yaml`, `Error` lists all four of
`{type, message, param, code}` as *required*: `param` and `code` are nullable
but must be present, so this renderer emits JSON nulls rather than dropping
them. That is the opposite of the Jina rerank `document` rule, which is why
the two renderers cannot share a serialization policy.

Only `invalid_request_error` and `server_error` appear as `type` values in the
spec. `authentication_error` and `rate_limit_error` are SDK class names, not
spec values, and an air-gapped single-model container has neither auth nor
rate limits, so they are not emitted.

`image_url` takes an http(s) URL as well as an inline data: one -- see the
fetch rules in `media`.
"""

import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict

import engine
from config import settings
from errors import BadRequest, JinaError
from media import _from_url_or_data as from_url_or_data

from .jina import reject_foreign_model

router = APIRouter(tags=["OpenAI Chat"])


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    messages: list
    model: Optional[str] = None
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    stream: bool = False
    # End-user tracking. Opaque to a local model; accepted and unused.
    user: Optional[str] = None


@router.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    engine.require_chat_model()
    reject_foreign_model(request.model)
    if request.stream:
        raise BadRequest("stream=true is not supported in this build")
    if not request.messages:
        raise BadRequest("messages must be a non-empty list")

    conversation, images = [], []
    for message in request.messages:
        if not isinstance(message, dict):
            raise BadRequest("each message must be a dict")
        role, parts, message_images = parse_message(message)
        conversation.append({"role": role, "content": parts})
        images.extend(message_images)

    text, prompt_tokens, completion_tokens = engine.generate(
        conversation, images, request.max_tokens, request.temperature, request.top_p
    )
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": settings.short_model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "refusal": None,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "service_tier": None,
        "system_fingerprint": None,
    }


def parse_message(message: dict) -> tuple:
    """One OpenAI message to (role, content_parts, images).

    `content_parts` feeds `processor.apply_chat_template`; `images` feeds
    `processor(images=...)`.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return message.get("role", "user"), [{"type": "text", "text": content}], []
    if not isinstance(content, list):
        raise BadRequest("message.content must be a string or list of parts")

    parts, images = [], []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type", "")
        if kind == "text":
            parts.append({"type": "text", "text": part.get("text", "")})
            continue
        if kind == "image_url":
            url = part.get("image_url", {})
            if isinstance(url, dict):
                url = url.get("url", "")
            if not isinstance(url, str):
                raise BadRequest("image_url.url must be a string")
        elif kind == "image":
            # Elastic-style {"type":"image","format":"base64","value":"..."}
            if part.get("format") != "base64":
                raise BadRequest("image part requires format=base64")
            url = part.get("value", "")
        else:
            raise BadRequest(f"Unsupported content part type: {kind!r}")
        image = from_url_or_data(url)[0]
        parts.append({"type": "image", "image": image})
        images.append(image)
    return message.get("role", "user"), parts, images


def error(exc: JinaError) -> dict:
    return envelope(exc.status, exc.message, param=exc.field)


def validation(errors: list) -> dict:
    first = errors[0] if errors else {}
    field = " -> ".join(str(part) for part in first.get("loc", ())) or None
    return envelope(422, first.get("msg", "Invalid request"), param=field)


def envelope(status: int, message: str, param: Optional[str] = None) -> dict:
    return {
        "error": {
            "message": message,
            "type": "server_error" if status >= 500 else "invalid_request_error",
            "param": param,
            "code": None,
        }
    }
