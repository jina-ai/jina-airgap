"""
End-to-end tests for jina-on-prem inference server.
Run against a live server: TEST_URL=http://localhost:8080 python tests/test_e2e.py

For multimodal tests (omni models), set: JINA_MODEL_ID=jinaai/jina-embeddings-v5-omni-nano
"""

import os
import sys
import json
import time
import base64
import io
import urllib.request
import urllib.error

BASE_URL = os.environ.get("TEST_URL", "http://localhost:8080")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


def request(method, path, data=None, headers=None):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode() if data else None
    h = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, {"error": str(e)}


def is_omni_model() -> bool:
    """Returns True if the running server uses an omni (multimodal) model."""
    status, body = request("GET", "/health")
    return status == 200 and body.get("multimodal", False)


def make_tiny_png_b64() -> str:
    """Generate a minimal valid 1x1 red PNG and return as base64."""
    import struct
    import zlib

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)    # 1x1 RGB
    raw_row = b"\x00\xFF\x00\x00"                            # filter=0, R=255, G=0, B=0
    idat = zlib.compress(raw_row)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", idat)
        + png_chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


# Every test below needs a running container. Under pytest, skip instead of
# failing when there is none: a suite that is red by default trains people to
# ignore it, which defeats the point of having it. Direct invocation
# (`python tests/test_e2e.py`) is unaffected and still reports failures.
if "pytest" in sys.modules:
    import pytest

    if request("GET", "/health")[0] != 200:
        pytest.skip(
            f"no server at {BASE_URL}; run a container or set TEST_URL",
            allow_module_level=True,
        )


# =============================================================================
# Text tests (all models)
# =============================================================================

def test_health():
    status, body = request("GET", "/health")
    assert status == 200, f"Expected 200, got {status}"
    assert body.get("status") == "ok", f"Expected ok, got {body}"
    assert body.get("ready"), "Model not ready"
    mm = body.get("multimodal", False)
    print(f"  {PASS} /health -> model={body.get('model', '?')} device={body.get('device', '?')} multimodal={mm}")


def test_embeddings_basic():
    status, body = request("POST", "/v1/embeddings", {
        "input": ["Hello world", "Jina AI embeddings"],
        "model": "test",
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    data = body["data"]
    assert len(data) == 2, f"Expected 2 embeddings, got {len(data)}"
    emb = data[0]["embedding"]
    assert isinstance(emb, list) and len(emb) > 0, "Empty embedding"
    assert abs(sum(x**2 for x in emb) - 1.0) < 0.01, "Embedding not normalized"
    print(f"  {PASS} /v1/embeddings -> dim={len(emb)}, normalized")


def test_embeddings_matryoshka():
    target_dim = 128
    status, body = request("POST", "/v1/embeddings", {
        "input": ["Test matryoshka truncation"],
        "dimensions": target_dim,
    })
    assert status == 200, f"Expected 200, got {status}"
    emb = body["data"][0]["embedding"]
    assert len(emb) == target_dim, f"Expected dim {target_dim}, got {len(emb)}"
    print(f"  {PASS} /v1/embeddings Matryoshka -> dim={len(emb)}")


def test_embeddings_tasks():
    for task in ["retrieval", "text-matching", "clustering", "classification"]:
        status, body = request("POST", "/v1/embeddings", {
            "input": ["Test task parameter"],
            "task": task,
        })
        assert status == 200, f"Task {task} failed: {status}: {body}"
    print(f"  {PASS} /v1/embeddings task parameter works (all 4 text tasks)")


def test_throughput_reported():
    texts = [f"This is sentence number {i} for throughput testing." for i in range(10)]
    status, body = request("POST", "/v1/embeddings", {"input": texts})
    assert status == 200, f"Expected 200, got {status}"

    usage = body.get("usage", {})
    assert "tok_per_s" in usage, f"tok_per_s missing from usage: {usage}"
    tok_per_s = usage["tok_per_s"]
    assert isinstance(tok_per_s, (int, float)) and tok_per_s > 0, \
        f"tok_per_s should be positive, got {tok_per_s}"
    assert "prompt_tokens" in usage, "prompt_tokens missing"
    assert usage["prompt_tokens"] > 0, "prompt_tokens should be > 0"

    print(f"  {PASS} tok/s reported in usage -> {tok_per_s:.1f} tok/s | {usage['prompt_tokens']} tokens")

    status2, health = request("GET", "/health")
    assert status2 == 200
    assert "throughput" in health, f"throughput missing from /health: {health}"
    tp = health["throughput"]
    assert "avg_tok_per_s" in tp and tp["avg_tok_per_s"] > 0, \
        f"avg_tok_per_s missing or zero: {tp}"
    print(f"  {PASS} /health throughput -> avg={tp['avg_tok_per_s']:.1f} tok/s peak={tp['peak_tok_per_s']:.1f} tok/s")


def test_throughput_gpu_vs_cpu():
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 5 + f" Sentence {i}."
        for i in range(100)
    ]
    t0 = time.time()
    status, body = request("POST", "/v1/embeddings", {"input": texts})
    wall_time = time.time() - t0
    assert status == 200, f"Expected 200, got {status}"

    tok_per_s = body["usage"].get("tok_per_s", 0)
    n_tokens = body["usage"].get("prompt_tokens", 0)
    print(f"  {PASS} Batch 100 texts: {n_tokens} tokens in {wall_time:.2f}s wall -> {tok_per_s:.1f} tok/s (encode only)")


# =============================================================================
# Multimodal tests (omni models only)
# =============================================================================

def test_openai_image_elastic_format():
    """OpenAI endpoint with Elastic Inference Service image format."""
    if not is_omni_model():
        print(f"  {SKIP} test_openai_image_elastic_format (text-only model)")
        return

    png_b64 = make_tiny_png_b64()
    status, body = request("POST", "/v1/embeddings", {
        "input": [
            {"type": "image", "format": "base64", "value": png_b64}
        ],
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    data = body["data"]
    assert len(data) == 1, f"Expected 1 embedding, got {len(data)}"
    emb = data[0]["embedding"]
    assert len(emb) > 0, "Empty embedding"
    assert any(v != 0 for v in emb), "All-zero embedding"
    print(f"  {PASS} OpenAI image (Elastic format) -> dim={len(emb)}")


def test_openai_image_base64_format():
    """OpenAI endpoint with image_base64 typed format."""
    if not is_omni_model():
        print(f"  {SKIP} test_openai_image_base64_format (text-only model)")
        return

    png_b64 = make_tiny_png_b64()
    status, body = request("POST", "/v1/embeddings", {
        "input": [
            {"type": "image_base64", "image_base64": {"base64": png_b64, "mime_type": "image/png"}}
        ],
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    emb = body["data"][0]["embedding"]
    assert len(emb) > 0 and any(v != 0 for v in emb)
    print(f"  {PASS} OpenAI image (image_base64 format) -> dim={len(emb)}")


def test_openai_data_url_format():
    """OpenAI endpoint with data URL inside image_base64."""
    if not is_omni_model():
        print(f"  {SKIP} test_openai_data_url_format (text-only model)")
        return

    png_b64 = make_tiny_png_b64()
    data_url = f"data:image/png;base64,{png_b64}"
    status, body = request("POST", "/v1/embeddings", {
        "input": [
            {"type": "image_base64", "image_base64": {"base64": data_url, "mime_type": "image/png"}}
        ],
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    emb = body["data"][0]["embedding"]
    assert len(emb) > 0
    print(f"  {PASS} OpenAI image (data URL inside image_base64) -> dim={len(emb)}")


def test_openai_fused_multimodal():
    """OpenAI content block: text + image -> ONE fused embedding."""
    if not is_omni_model():
        print(f"  {SKIP} test_openai_fused_multimodal (text-only model)")
        return

    png_b64 = make_tiny_png_b64()
    status, body = request("POST", "/v1/embeddings", {
        "input": [
            {
                "content": [
                    {"type": "text", "text": "A small red square"},
                    {"type": "image", "format": "base64", "value": png_b64},
                ]
            }
        ],
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    data = body["data"]
    assert len(data) == 1, "Content block should produce exactly ONE embedding"
    emb = data[0]["embedding"]
    assert len(emb) > 0
    print(f"  {PASS} OpenAI fused multimodal (text+image content block) -> 1 embedding, dim={len(emb)}")


def test_openai_mixed_batch():
    """OpenAI endpoint: mixed batch of text and images."""
    if not is_omni_model():
        print(f"  {SKIP} test_openai_mixed_batch (text-only model)")
        return

    png_b64 = make_tiny_png_b64()
    status, body = request("POST", "/v1/embeddings", {
        "input": [
            "plain text input",
            {"type": "image", "format": "base64", "value": png_b64},
            "another text",
        ],
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    data = body["data"]
    assert len(data) == 3, f"Expected 3 embeddings, got {len(data)}"
    for i, item in enumerate(data):
        emb = item["embedding"]
        assert len(emb) > 0, f"Empty embedding at index {i}"
    print(f"  {PASS} OpenAI mixed batch (text+image+text) -> 3 embeddings, dim={len(data[0]['embedding'])}")


def test_gemini_inline_data():
    """Gemini endpoint with inlineData image part."""
    if not is_omni_model():
        print(f"  {SKIP} test_gemini_inline_data (text-only model)")
        return

    png_b64 = make_tiny_png_b64()
    status, body = request(
        "POST",
        "/v1/models/jina-embeddings-v5-omni-nano:embedContent",
        {
            "content": {
                "parts": [
                    {"text": "A small red square"},
                    {"inlineData": {"mimeType": "image/png", "data": png_b64}},
                ]
            },
            "taskType": "RETRIEVAL_DOCUMENT",
        },
    )
    assert status == 200, f"Expected 200, got {status}: {body}"
    emb = body["embedding"]["values"]
    assert len(emb) > 0
    print(f"  {PASS} Gemini inlineData image -> dim={len(emb)}")


def test_gemini_text_only():
    """Gemini endpoint text-only still works."""
    status, body = request(
        "POST",
        "/v1/models/jina-embeddings-v5-text-nano:embedContent",
        {
            "content": {"parts": [{"text": "Hello from Gemini API"}]},
            "taskType": "RETRIEVAL_QUERY",
        },
    )
    assert status == 200, f"Expected 200, got {status}: {body}"
    emb = body["embedding"]["values"]
    assert len(emb) > 0
    print(f"  {PASS} Gemini text-only -> dim={len(emb)}")


def test_cohere_text_only():
    """Cohere text-only embedding still works."""
    status, body = request("POST", "/v1/embed", {
        "texts": ["hello cohere", "jina ai"],
        "input_type": "search_document",
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    embs = body["embeddings"]["float"]
    assert len(embs) == 2
    print(f"  {PASS} Cohere text-only -> 2 embeddings, dim={len(embs[0])}")


def test_cohere_legacy_images():
    """Cohere legacy images field with data URL."""
    if not is_omni_model():
        print(f"  {SKIP} test_cohere_legacy_images (text-only model)")
        return

    png_b64 = make_tiny_png_b64()
    data_url = f"data:image/png;base64,{png_b64}"
    status, body = request("POST", "/v1/embed", {
        "images": [data_url],
        "input_type": "search_document",
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    embs = body["embeddings"]["float"]
    assert len(embs) == 1
    emb = embs[0]
    assert len(emb) > 0
    print(f"  {PASS} Cohere legacy images (data URL) -> dim={len(emb)}")


def test_cohere_v2_inputs():
    """Cohere V2 inputs format with content blocks."""
    if not is_omni_model():
        print(f"  {SKIP} test_cohere_v2_inputs (text-only model)")
        return

    png_b64 = make_tiny_png_b64()
    data_url = f"data:image/png;base64,{png_b64}"
    status, body = request("POST", "/v1/embed", {
        "inputs": [
            {"content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "A small red square"},
            ]}
        ],
        "input_type": "search_document",
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    embs = body["embeddings"]["float"]
    assert len(embs) == 1
    print(f"  {PASS} Cohere V2 inputs (image_url + text) -> dim={len(embs[0])}")


def test_voyage_multimodal():
    """Voyage multimodalembeddings endpoint."""
    if not is_omni_model():
        print(f"  {SKIP} test_voyage_multimodal (text-only model)")
        return

    png_b64 = make_tiny_png_b64()
    data_url = f"data:image/png;base64,{png_b64}"
    status, body = request("POST", "/v1/multimodalembeddings", {
        "inputs": [
            {"content": [
                {"type": "text", "text": "A small red square"},
                {"type": "image_base64", "image_base64": data_url},
            ]}
        ],
        "model": "voyage-multimodal-3.5",
        "input_type": "document",
    })
    assert status == 200, f"Expected 200, got {status}: {body}"
    embs = body["embeddings"]
    assert len(embs) == 1
    emb = embs[0]
    assert len(emb) > 0
    print(f"  {PASS} Voyage multimodalembeddings (text+image) -> dim={len(emb)}")


def test_text_model_rejects_multimodal():
    """Text-only model returns 400 for multimodal input."""
    if is_omni_model():
        print(f"  {SKIP} test_text_model_rejects_multimodal (this is an omni model)")
        return

    png_b64 = make_tiny_png_b64()
    status, body = request("POST", "/v1/embeddings", {
        "input": [{"type": "image", "format": "base64", "value": png_b64}],
    })
    assert status == 400, f"Expected 400 for multimodal on text model, got {status}: {body}"
    assert "text-only" in body.get("detail", "").lower() or "multimodal" in body.get("detail", "").lower(), \
        f"Error message not clear: {body}"
    print(f"  {PASS} Text-only model correctly rejects multimodal input (400)")


# =============================================================================
# Runner
# =============================================================================

def wait_for_ready(timeout=300):
    print(f"  Waiting for server at {BASE_URL}...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status, body = request("GET", "/health")
            if status == 200 and body.get("ready"):
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main():
    print(f"\n=== Jina AI v5 On-Prem E2E Tests ===")
    print(f"URL: {BASE_URL}\n")

    if not wait_for_ready():
        print(f"{FAIL} Server not ready after 300s")
        sys.exit(1)

    omni = is_omni_model()
    print(f"Model type: {'multimodal (omni)' if omni else 'text-only'}\n")

    tests = [
        # Text tests (all models)
        test_health,
        test_embeddings_basic,
        test_embeddings_matryoshka,
        test_embeddings_tasks,
        test_throughput_reported,
        test_throughput_gpu_vs_cpu,
        test_gemini_text_only,
        test_cohere_text_only,
        # Multimodal tests (omni models)
        test_openai_image_elastic_format,
        test_openai_image_base64_format,
        test_openai_data_url_format,
        test_openai_fused_multimodal,
        test_openai_mixed_batch,
        test_gemini_inline_data,
        test_cohere_legacy_images,
        test_cohere_v2_inputs,
        test_voyage_multimodal,
        # Validation
        test_text_model_rejects_multimodal,
    ]

    print("Running tests...\n")
    failed = 0
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  {FAIL} {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  {FAIL} {test.__name__}: unexpected error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    total = len(tests)
    passed = total - failed
    if failed == 0:
        print(f"\033[32mAll {total} tests passed!\033[0m")
    else:
        print(f"\033[31m{failed}/{total} tests failed\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
