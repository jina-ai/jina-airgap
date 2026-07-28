#!/usr/bin/env python3
"""Every probe for one image, executed inside the container.

Copied in and run by `validate_matrix.py`. It runs inside rather than over a
published port for one reason: the container is started with `--network none`,
so there is no port to publish. That makes air-gap the default condition of
every functional probe instead of a separate run -- if the image needed the
network for anything, none of this would answer.

Prints one JSON object on stdout. Errors are results: a probe that raises is
recorded and the run continues.
"""

import json
import socket
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8080"

EN = (
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife. However little known the "
    "feelings or views of such a man may be on his first entering a "
    "neighbourhood, this truth is so well fixed in the minds of the "
    "surrounding families, that he is considered the rightful property of "
    "some one or other of their daughters."
)
ZH = "机器学习模型的生产部署需要考虑多个维度：推理延迟、吞吐量、硬件成本以及模型更新的持续集成与部署流程。"
CODE = (
    "def encode_batch(texts, model, batch_size=32):\n"
    "    vectors = []\n"
    "    for start in range(0, len(texts), batch_size):\n"
    "        chunk = texts[start : start + batch_size]\n"
    "        vectors.extend(model.encode(chunk, normalize_embeddings=True))\n"
    "    return vectors\n"
)
UNICODE = "Ünïcödé ☂ 漢字 かな 한글 emoji 🌊🚀 mixed with ASCII and math ∑∫≈"
QUERY = "How do teams deploy large language models efficiently in production?"
DOCS = [
    "Deploying large language models at scale presents significant "
    "infrastructure challenges, including GPU memory management, batching "
    "strategies, and latency requirements for real-time inference.",
    ZH,
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and "
    "established the principle of state sovereignty in European diplomacy.",
    CODE,
]


def call(path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw, headers = (
                response.status,
                response.read(),
                dict(response.headers),
            )
    except urllib.error.HTTPError as exc:
        status, raw, headers = exc.code, exc.read(), dict(exc.headers)
    except Exception as exc:
        return {"status": None, "error": f"{type(exc).__name__}: {exc}"}
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"_raw": raw.decode(errors="replace")[:400]}
    return {
        "status": status,
        "client_ms": elapsed,
        "server_ms": headers.get("X-Jina-Elapsed-Ms"),
        "tok_per_s": headers.get("X-Jina-Tok-Per-S"),
        "body": parsed,
    }


def shrink(node, keep=8):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "embedding" and isinstance(value, list) and len(value) > keep:
                out[key], out["_full_dim"] = value[:keep], len(value)
            elif key in ("embeddings", "values") and isinstance(value, list) and value:
                if isinstance(value[0], list):
                    out[key] = [row[:keep] for row in value[:2]]
                    out["_num_vectors"], out["_full_dim"] = len(value), len(value[0])
                elif isinstance(value[0], float):
                    out[key], out["_full_dim"] = value[:keep], len(value)
                else:
                    out[key] = shrink(value, keep)
            elif key == "tokenized_input" and isinstance(value, list):
                out[key], out["_num_tokens"] = value[:keep], len(value)
            elif key == "document" and isinstance(value, str) and len(value) > 90:
                out[key], out["_document_chars"] = value[:90] + "...", len(value)
            else:
                out[key] = shrink(value, keep)
        return out
    if isinstance(node, list):
        return [shrink(item, keep) for item in node]
    return node


COMMON = [
    ("health", "/health", None),
    ("wrong_verb_embed", "/v1/embeddings", {"input": [EN]}),
    ("wrong_verb_rerank", "/v1/rerank", {"query": QUERY, "documents": DOCS}),
    (
        "wrong_verb_chat",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": EN}]},
    ),
    ("cohere_v1_embed_deleted", "/v1/embed", {"texts": [EN]}),
]

EMBEDDING = [
    ("embed_default", "/v1/embeddings", {"input": [EN, ZH]}),
    ("embed_single_string", "/v1/embeddings", {"input": EN}),
    ("embed_code", "/v1/embeddings", {"input": [CODE]}),
    ("embed_unicode", "/v1/embeddings", {"input": [UNICODE]}),
    ("embed_batch8", "/v1/embeddings", {"input": [EN, ZH, CODE, UNICODE] * 2}),
    ("embed_dims128", "/v1/embeddings", {"input": [EN], "dimensions": 128}),
    (
        "embed_dims_over_max_422",
        "/v1/embeddings",
        {"input": [EN], "dimensions": 999999},
    ),
    ("embed_dims_zero_422", "/v1/embeddings", {"input": [EN], "dimensions": 0}),
    ("embed_binary", "/v1/embeddings", {"input": [EN], "embedding_type": "binary"}),
    ("embed_ubinary", "/v1/embeddings", {"input": [EN], "embedding_type": "ubinary"}),
    ("embed_base64", "/v1/embeddings", {"input": [EN], "embedding_type": "base64"}),
    (
        "embed_type_list_422",
        "/v1/embeddings",
        {"input": [EN], "embedding_type": ["float", "binary"]},
    ),
    (
        "embed_type_int8_422",
        "/v1/embeddings",
        {"input": [EN], "embedding_type": "int8"},
    ),
    ("embed_normalized_false", "/v1/embeddings", {"input": [EN], "normalized": False}),
    (
        "embed_norm_false_dims",
        "/v1/embeddings",
        {"input": [EN], "normalized": False, "dimensions": 128},
    ),
    ("embed_task_query", "/v1/embeddings", {"input": [EN], "task": "retrieval.query"}),
    (
        "embed_task_passage",
        "/v1/embeddings",
        {"input": [EN], "task": "retrieval.passage"},
    ),
    (
        "embed_task_text_matching",
        "/v1/embeddings",
        {"input": [EN], "task": "text-matching"},
    ),
    (
        "embed_task_invalid",
        "/v1/embeddings",
        {"input": [EN], "task": "not-a-real-task"},
    ),
    (
        "embed_late_chunking",
        "/v1/embeddings",
        {"input": [EN, ZH], "late_chunking": True},
    ),
    (
        "embed_multivector",
        "/v1/embeddings",
        {"input": [EN], "return_multivector": True},
    ),
    (
        "embed_multivector_dims_422",
        "/v1/embeddings",
        {"input": [EN], "return_multivector": True, "dimensions": 128},
    ),
    ("embed_empty_list_422", "/v1/embeddings", {"input": []}),
    ("embed_missing_input_422", "/v1/embeddings", {"model": "x"}),
    ("embed_pretokenized_422", "/v1/embeddings", {"input": [[1, 2, 3]]}),
    ("embed_overlong_400", "/v1/embeddings", {"input": [EN * 400]}),
    (
        "embed_overlong_truncate_ok",
        "/v1/embeddings",
        {"input": [EN * 400], "truncate": True},
    ),
    (
        "embed_foreign_jina_model_400",
        "/v1/embeddings",
        {"input": [EN], "model": "jina-embeddings-v4"},
    ),
    (
        "embed_openai_drop_in",
        "/v1/embeddings",
        {
            "input": [EN],
            "model": "text-embedding-3-small",
            "encoding_format": "float",
            "user": "analytics-1",
        },
    ),
    (
        "embed_voyage_fields",
        "/v1/embeddings",
        {
            "input": [EN],
            "input_type": "query",
            "output_dtype": "float",
            "truncation": True,
        },
    ),
    (
        "cohere_embed",
        "/v2/embed",
        {"model": "embed-v4.0", "input_type": "search_document", "texts": [EN, ZH]},
    ),
    (
        "cohere_embed_types",
        "/v2/embed",
        {
            "model": "embed-v4.0",
            "input_type": "search_query",
            "texts": [EN],
            "embedding_types": ["float", "int8", "uint8", "ubinary", "base64"],
        },
    ),
    (
        "cohere_embed_no_input_type_422",
        "/v2/embed",
        {"model": "embed-v4.0", "texts": [EN]},
    ),
    (
        "gemini_flat",
        "/v1/models/text-embedding-004:embedContent",
        {"content": {"parts": [{"text": EN}]}, "taskType": "RETRIEVAL_QUERY"},
    ),
    (
        "gemini_nested_config",
        "/v1/models/text-embedding-004:embedContent",
        {
            "content": {"parts": [{"text": EN}]},
            "embedContentConfig": {
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": 128,
            },
        },
    ),
    (
        "gemini_batch",
        "/v1/models/text-embedding-004:batchEmbedContents",
        {
            "requests": [
                {"content": {"parts": [{"text": EN}]}},
                {"content": {"parts": [{"text": ZH}]}},
            ]
        },
    ),
    (
        "voyage_multimodal_text",
        "/v1/multimodalembeddings",
        {
            "inputs": [{"content": [{"type": "text", "text": EN}]}],
            "model": "voyage-multimodal-3",
            "input_type": "document",
        },
    ),
    (
        "voyage_remote_url_400",
        "/v1/multimodalembeddings",
        {
            "inputs": [
                {
                    "content": [
                        {"type": "image_url", "image_url": "https://example.com/a.png"}
                    ]
                }
            ],
            "model": "voyage-multimodal-3",
        },
    ),
]

RERANK = [
    (
        "rerank_strings_top2",
        "/v1/rerank",
        {"query": QUERY, "documents": DOCS, "top_n": 2},
    ),
    (
        "rerank_return_documents_false",
        "/v1/rerank",
        {"query": QUERY, "documents": DOCS, "top_n": 2, "return_documents": False},
    ),
    ("rerank_no_top_n", "/v1/rerank", {"query": QUERY, "documents": DOCS}),
    (
        "rerank_docs_as_objects",
        "/v1/rerank",
        {"query": QUERY, "documents": [{"text": d} for d in DOCS], "top_n": 2},
    ),
    (
        "rerank_top_n_gt_docs",
        "/v1/rerank",
        {"query": QUERY, "documents": DOCS, "top_n": 999},
    ),
    (
        "rerank_top_k_alias",
        "/v1/rerank",
        {"query": QUERY, "documents": DOCS, "top_k": 2},
    ),
    (
        "rerank_max_tokens_per_doc",
        "/v1/rerank",
        {"query": QUERY, "documents": DOCS, "max_tokens_per_doc": 24},
    ),
    (
        "rerank_16_docs",
        "/v1/rerank",
        {"query": QUERY, "documents": DOCS * 4, "top_n": 5},
    ),
    (
        "rerank_long_docs",
        "/v1/rerank",
        {"query": QUERY, "documents": [d * 20 for d in DOCS], "top_n": 2},
    ),
    ("rerank_empty_docs_422", "/v1/rerank", {"query": QUERY, "documents": []}),
    (
        "rerank_top_n_zero_422",
        "/v1/rerank",
        {"query": QUERY, "documents": DOCS, "top_n": 0},
    ),
    ("rerank_missing_query_422", "/v1/rerank", {"documents": DOCS}),
    (
        "rerank_foreign_jina_model_400",
        "/v1/rerank",
        {"query": QUERY, "documents": DOCS, "model": "jina-reranker-v3"},
    ),
    (
        "cohere_rerank",
        "/v2/rerank",
        {"model": "rerank-v3.5", "query": QUERY, "documents": DOCS, "top_n": 2},
    ),
    (
        "cohere_rerank_max_tokens",
        "/v2/rerank",
        {
            "model": "rerank-v3.5",
            "query": QUERY,
            "documents": DOCS,
            "max_tokens_per_doc": 24,
        },
    ),
    (
        "cohere_rerank_empty_422",
        "/v2/rerank",
        {"model": "rerank-v3.5", "query": QUERY, "documents": []},
    ),
]

CHAT = [
    (
        "chat_basic",
        "/v1/chat/completions",
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Summarise in two sentences why batching matters "
                    "for large language model inference.",
                }
            ],
            "max_tokens": 96,
        },
    ),
    (
        "chat_html_extraction",
        "/v1/chat/completions",
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<html><body><h1>Quarterly Report</h1><p>Revenue grew "
                    "12% year over year, driven by inference volume.</p>"
                    "<ul><li>EMEA up 8%</li><li>APAC up 21%</li></ul>"
                    "</body></html>",
                }
            ],
            "max_tokens": 160,
        },
    ),
    (
        "chat_cjk",
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "用一句话说明什么是向量检索。"}],
            "max_tokens": 96,
        },
    ),
    (
        "chat_stream_400",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": EN}], "stream": True},
    ),
    ("chat_empty_messages_400", "/v1/chat/completions", {"messages": []}),
    (
        "chat_foreign_jina_model_400",
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": EN}], "model": "jina-vlm"},
    ),
]


def egress_probe():
    """Prove the container cannot reach out. HF_HUB_OFFLINE states intent;
    a refused connection is evidence."""
    results = {}
    for name, address in (
        ("huggingface.co", ("huggingface.co", 443)),
        ("api.jina.ai", ("api.jina.ai", 443)),
        ("dns_8.8.8.8", ("8.8.8.8", 53)),
        ("pypi.org", ("pypi.org", 443)),
    ):
        try:
            connection = socket.create_connection(address, timeout=4)
            connection.close()
            results[name] = "CONNECTED"
        except Exception as exc:
            results[name] = type(exc).__name__
    return results


def main() -> int:
    health = call("/health")
    endpoint = (health.get("body") or {}).get("endpoint")
    probes = COMMON + {
        "/v1/embeddings": EMBEDDING,
        "/v1/rerank": RERANK,
        "/v1/chat/completions": CHAT,
    }.get(endpoint, [])

    # Warm before anything is timed; the first call pays for lazy CUDA kernels
    # and allocator growth, and averaging that in would misreport every model.
    warm = {
        "/v1/embeddings": ("/v1/embeddings", {"input": [EN]}),
        "/v1/rerank": ("/v1/rerank", {"query": QUERY, "documents": DOCS[:2]}),
        "/v1/chat/completions": (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Warm up."}], "max_tokens": 8},
        ),
    }.get(endpoint)
    cold = call(*warm) if warm else None

    results = {}
    for probe_id, path, body in probes:
        outcome = call(path, body)
        results[probe_id] = {
            "endpoint": path,
            "request": body,
            **{k: v for k, v in outcome.items() if k != "body"},
            "body": shrink(outcome.get("body")),
        }

    print(
        "PROBE_JSON:"
        + json.dumps(
            {
                "endpoint": endpoint,
                "health": health.get("body"),
                "cold_call_ms": (cold or {}).get("client_ms"),
                "egress": egress_probe(),
                "results": results,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
