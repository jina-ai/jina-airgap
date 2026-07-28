"""Wire-contract tests: the response envelope, the encoders, the error bodies.

Asserted against captures from the live `api.jina.ai`, embedded below as
expected values rather than re-fetched, so the test runs offline and in CI.
Every constant here traces to a recorded response; the provenance is in the
comment above it.

No server, no network, no model weights -- numpy is the only dependency. The
adapter *response* bodies are covered end to end by
`scripts/validate_matrix.py`, which exercises real images; what this file
pins down is the pure rendering, where a regression is silent.

Run:
  python tests/test_contracts.py
"""

import base64
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import errors  # noqa: E402
import serialize  # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

_results = []


def check(name, condition, detail=""):
    _results.append(bool(condition))
    print(f"  {PASS if condition else FAIL}  {name}{'  ' + detail if detail else ''}")


# --- the envelope ----------------------------------------------------------
# Key order is taken from the golden captures. It does not affect JSON
# equality, but matching it keeps a diff against a golden file readable.

print("\nResponse envelope")

embeddings = serialize.embeddings_response(
    "jina-embeddings-v2-base-en",
    serialize.usage(190, prompt_tokens=True),
    [serialize.embedding_item(0, np.array([0.1, 0.2], dtype=np.float32), "float")],
)
check(
    "embeddings top-level keys and order",
    list(embeddings) == ["model", "object", "usage", "data"],
    str(list(embeddings)),
)
check("embeddings object is 'list'", embeddings["object"] == "list")
check(
    "embeddings data item keys and order",
    list(embeddings["data"][0]) == ["object", "index", "embedding"],
    str(list(embeddings["data"][0])),
)
check(
    "usage puts total_tokens first",
    list(embeddings["usage"]) == ["total_tokens", "prompt_tokens"],
    str(embeddings["usage"]),
)
check(
    "prompt_tokens equals total_tokens when present",
    embeddings["usage"]["prompt_tokens"] == embeddings["usage"]["total_tokens"],
)
check(
    "usage omits prompt_tokens when the model does not emit it",
    serialize.usage(185, prompt_tokens=False) == {"total_tokens": 185},
)

rerank = serialize.rerank_response(
    "jina-reranker-v2-base-multilingual",
    serialize.usage(226, prompt_tokens=False),
    [{"index": 0, "relevance_score": 0.49922183, "document": "..."}],
)
check(
    "rerank top-level keys and order",
    list(rerank) == ["model", "object", "usage", "results"],
    str(list(rerank)),
)
check("rerank usage is total_tokens only", rerank["usage"] == {"total_tokens": 226})

multivector = serialize.multivector_item(
    0, np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32), "float"
)
check(
    "multivector item turns object and key plural",
    multivector["object"] == "embeddings" and "embeddings" in multivector,
    str(list(multivector)),
)
check(
    "tokenized_input only when supplied",
    "tokenized_input" not in multivector
    and "tokenized_input"
    in serialize.multivector_item(
        0, np.array([[0.1]], dtype=np.float32), "float", tokens=["a"]
    ),
)

# api.jina.ai rounds relevance_score to 8 places (RerankingResult in the public
# schema); golden captures show values like 0.49922183.
check(
    "relevance_score rounds to 8 places",
    serialize.relevance_score(0.4992218300104141) == 0.49922183,
    str(serialize.relevance_score(0.4992218300104141)),
)

# --- embedding_type encoders ----------------------------------------------
# Measured on api.jina.ai/v1/embeddings, jina-embeddings-v2-base-en, one input,
# 2026-07-27: for the same vector the service returns
#   ubinary [50, 193, 128, 255, ...]
#   binary  [-78,  65,   0, 127, ...]
# binary is the packed byte minus 128 (offset binary), NOT the two's-complement
# int8 view, which would give [50, -63, -128, -1, ...].

print("\nembedding_type encoders")

# A 32-dim vector whose signs pack, MSB first, into exactly those bytes.
PROD_UBINARY = [50, 193, 128, 255]
vector = np.array(
    [
        1.0 if (byte >> (7 - bit)) & 1 else -1.0
        for byte in PROD_UBINARY
        for bit in range(8)
    ],
    dtype=np.float32,
)

check(
    "ubinary is the packed byte",
    serialize.encode_vector(vector, "ubinary") == [50.0, 193.0, 128.0, 255.0],
    str(serialize.encode_vector(vector, "ubinary")),
)
check(
    "binary is offset binary (byte - 128), not the int8 view",
    serialize.encode_vector(vector, "binary") == [-78.0, 65.0, 0.0, 127.0],
    str(serialize.encode_vector(vector, "binary")),
)
check(
    "binary and ubinary differ by exactly 128 per byte",
    all(
        b == u - 128
        for b, u in zip(
            serialize.encode_vector(vector, "binary"),
            serialize.encode_vector(vector, "ubinary"),
        )
    ),
)
check(
    "packed length is dim / 8",
    len(serialize.encode_vector(vector, "binary")) == len(vector) // 8,
)
check(
    "binary values are JSON floats, not ints",
    all(isinstance(v, float) for v in serialize.encode_vector(vector, "binary")),
)
check(
    "sign threshold is > 0",
    serialize.encode_vector(np.array([0.0] * 8, dtype=np.float32), "ubinary") == [0.0],
)

small = np.array([0.25, -0.5, 1.0], dtype=np.float32)
decoded = np.frombuffer(
    base64.b64decode(serialize.encode_vector(small, "base64")), dtype="<f4"
)
check(
    "base64 is little-endian float32, round-trips exactly",
    decoded.tolist() == small.tolist(),
    str(decoded.tolist()),
)
check(
    "float returns a plain list",
    serialize.encode_vector(small, "float") == small.tolist(),
)

# --- error bodies ----------------------------------------------------------
# api.jina.ai shapes, 2026-07-27:
#   422 -> {detail: str, request_id, errors: [{field, message, type, input}]}
#   400 -> {detail: {message, request_id, code}}
#          e.g. code INPUT_TOKEN_LIMIT_EXCEEDED for over-length input

print("\nJina error bodies")

validation = errors.jina_validation_body(
    [
        {
            "loc": ("body", "top_n"),
            "msg": "Input should be greater than or equal to 1",
            "type": "greater_than_equal",
            "input": 0,
        }
    ]
)
check(
    "422 keys",
    list(validation) == ["detail", "request_id", "errors"],
    str(list(validation)),
)
check(
    "422 error entry keys",
    list(validation["errors"][0]) == ["field", "message", "type", "input"],
)
check(
    "422 field renders loc with ' -> '",
    validation["errors"][0]["field"] == "body -> top_n",
)
check("request_id is 32 hex chars", len(validation["request_id"]) == 32)

over_length = errors.BadRequest(
    "Input text exceeds the model's maximum of 8192 tokens.",
    field="input",
    code="INPUT_TOKEN_LIMIT_EXCEEDED",
)
body = errors.jina_error_body(over_length)
check(
    "400 nests detail as an object",
    isinstance(body["detail"], dict)
    and list(body["detail"]) == ["message", "request_id", "code"],
    str(body["detail"].keys() if isinstance(body["detail"], dict) else body["detail"]),
)
check(
    "a published code string survives to the wire",
    body["detail"]["code"] == "INPUT_TOKEN_LIMIT_EXCEEDED",
)
check(
    "a 422 raised in code still uses the validation envelope",
    "errors"
    in errors.jina_error_body(
        errors.UnprocessableEntity("x", field="body -> dimensions")
    ),
)
check("BadRequest is 400", errors.BadRequest("x").status == 400)
check("UnprocessableEntity is 422", errors.UnprocessableEntity("x").status == 422)

# --- provider error bodies -------------------------------------------------
# Cohere: {message, id}. Gemini: {error: {code, message, status, details}}.
# OpenAI: {error: {message, type, param, code}} with ALL FOUR present -- the
# spec lists them in `required`, so exclude_none would break a schema-checking
# client. Voyage: {detail: str}, measured against api.voyageai.com, which is
# what FastAPI already emits.

print("\nProvider error bodies")


def _renderer(name):
    """Imported lazily: the adapter modules pull in engine, and therefore
    torch, which a contract test should not require."""
    try:
        import importlib

        return importlib.import_module(f"adapters.{name}")
    except Exception as exc:  # torch absent in a bare checkout
        print(f"  \033[33mSKIP\033[0m adapters.{name} ({type(exc).__name__})")
        return None


exc = errors.BadRequest("bad input", field="documents")
for name, assertion in (
    ("cohere", lambda m: sorted(m.error(exc)) == ["id", "message"]),
    (
        "gemini",
        lambda m: sorted(m.error(exc)["error"])
        == ["code", "details", "message", "status"]
        and m.error(exc)["error"]["status"] == "INVALID_ARGUMENT",
    ),
    ("voyage", lambda m: m.error(exc) == {"detail": "bad input"}),
    (
        "openai",
        lambda m: sorted(m.error(exc)["error"]) == ["code", "message", "param", "type"]
        and m.error(exc)["error"]["type"] == "invalid_request_error",
    ),
):
    module = _renderer(name)
    if module is not None:
        check(f"{name} error envelope", assertion(module))

openai_module = _renderer("openai")
if openai_module is not None:
    rendered = openai_module.error(errors.ModelNotLoaded("not loaded"))
    check(
        "OpenAI keeps param and code present as JSON null",
        rendered["error"]["param"] is None and "code" in rendered["error"],
    )
    check(
        "OpenAI 5xx uses server_error, the other spec-evidenced type",
        rendered["error"]["type"] == "server_error",
    )

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
