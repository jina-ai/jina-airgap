"""Error types, and the Jina-native rendering of them.

The exception classes carry only what happened. How it reaches the wire is the
adapter's business: an OpenAI client parses ``error.type``, a Cohere client
parses ``message``, a Gemini client parses ``error.status`` -- an adapter that
translates only the happy path is not a drop-in. The renderers for the other
four providers live in ``adapters/``; this module owns Jina's own.
"""

import uuid
from typing import Any, Optional


class JinaError(Exception):
    status: int
    code: str

    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        code: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.field = field
        if code:
            # Some conditions have a published code string on api.jina.ai --
            # INPUT_TOKEN_LIMIT_EXCEEDED, for one -- and a client switching on
            # it should see the same value from a container.
            self.code = code


class BadRequest(JinaError):
    status = 400
    code = "bad_request"


class UnprocessableEntity(JinaError):
    """A request that parses but asks for something the contract forbids.

    Separate from ``BadRequest`` because the public API answers this class of
    mistake with 422, and reproducing the status is the point of the rewrite.
    """

    status = 422
    code = "validation_error"


class NotAcceptable(JinaError):
    status = 406
    code = "not_acceptable"


class PayloadTooLarge(JinaError):
    status = 413
    code = "payload_too_large"


class ModelNotLoaded(JinaError):
    status = 503
    code = "model_not_loaded"


class LicenseBlocked(JinaError):
    status = 403
    code = "license_blocked"


def request_id() -> str:
    """32-char hex, the shape ``api.jina.ai`` returns in every error body."""
    return uuid.uuid4().hex


def jina_error_body(exc: JinaError) -> dict:
    """Errors raised in a handler, in the public API's nested application-error
    shape: ``{"detail": {"message", "request_id", "code"}}``.

    A 422 raised in code takes the validation envelope instead -- the public
    API answers every 422 that way regardless of where it came from, so a
    client parsing `errors[]` must find it there.

    Measured, not assumed -- an over-length input to
    ``api.jina.ai/v1/embeddings`` returns exactly this with
    ``code: INPUT_TOKEN_LIMIT_EXCEEDED``, and the captured 500 uses the same
    nesting. (Its 401 is flat instead, but that one is produced by the auth
    layer in front of the app, which a container has no equivalent of.)
    """
    if exc.status == 422:
        return jina_validation_body(
            [
                {
                    "loc": tuple((exc.field or "body").split(" -> ")),
                    "msg": exc.message,
                    "type": exc.code,
                    "input": None,
                }
            ]
        )
    return {
        "detail": {
            "message": exc.message,
            "request_id": request_id(),
            "code": exc.code,
        }
    }


def jina_validation_body(errors: list[dict[str, Any]]) -> dict:
    """422 bodies, matching the public API's
    ``{detail, request_id, errors: [{field, message, type, input}]}``.

    ``detail`` renders the first error for a human. The public API appends a
    literal ``"Field errors: : Invalid value"`` to it -- an empty-field
    f-string artifact on their side, reported rather than reproduced.
    """
    rendered = [
        {
            "field": " -> ".join(str(part) for part in error.get("loc", ())),
            "message": error.get("msg", ""),
            "type": error.get("type", ""),
            "input": error.get("input"),
        }
        for error in errors
    ]
    head = rendered[0] if rendered else {"field": "body", "message": "Invalid request"}
    return {
        "detail": f"Validation error: '{head['field']}' {head['message']}",
        "request_id": request_id(),
        "errors": rendered,
    }
