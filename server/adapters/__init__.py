"""One module per wire contract.

Each module owns a router plus two renderers, ``error`` and ``validation``.
The renderers are not decoration: provider SDKs parse error bodies to decide
whether to retry, so an adapter that translates only the happy path is not a
drop-in. Cohere reads ``message``, Gemini reads ``error.status``, OpenAI reads
``error.type``, and Voyage reads ``detail`` -- which is what FastAPI already
emits, so that one adapter correctly needs no translation at all.

Adding a schema means adding a module here and touching no model code; adding
a model means adding a family and touching no HTTP code.

Every handler is a plain ``def``, not ``async def``, and that is deliberate.
Inference is synchronous, blocking, CPU/GPU-bound work; running it inside a
coroutine pins the event loop for the whole forward pass, so concurrent callers
do not merely queue -- ``/health`` stops answering too, and a Docker healthcheck
can fail while the server is perfectly healthy. A synchronous handler is
dispatched to Starlette's threadpool instead, which frees the loop and lets the
batcher see more than one request at a time. None of these handlers await
anything, so there is nothing to lose by it.
"""

from fastapi import APIRouter, Request

from . import cohere, gemini, jina, openai, voyage

MODULES = (jina, cohere, gemini, voyage, openai)

router = APIRouter()
for _module in MODULES:
    router.include_router(_module.router)

# Keyed by route template, so `/v1/models/{model_id}:embedContent` resolves as
# reliably as a literal path.
_BY_PATH = {route.path: module for module in MODULES for route in module.router.routes}


def renderer_for(request: Request):
    """The adapter that owns this request. Jina's is the default: it owns the
    native paths, and it is the right answer for anything unrouted."""
    route = request.scope.get("route")
    return _BY_PATH.get(getattr(route, "path", None), jina)
