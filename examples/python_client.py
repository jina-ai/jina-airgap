"""Minimal Python client for a jina-on-prem deployment.

Demonstrates the four common patterns:
  1. OpenAI SDK against /v1/embeddings (drop-in for OpenAI text-embedding-3-*)
  2. Cohere-style /v1/embed via raw HTTP
  3. /v1/rerank with a separate reranker container
  4. Matryoshka dimensions

Run after starting a container, e.g.:
  docker run -d -p 8080:8080 jina/jina-embeddings-v5-text-nano:cpu
  docker run -d -p 8081:8080 jina/jina-reranker-v3.5:gpu  # optional

Then:
  uv pip install openai requests
  python examples/python_client.py
"""
from __future__ import annotations

import requests
from openai import OpenAI

EMBED_URL = "http://localhost:8080"
RERANK_URL = "http://localhost:8081"


def via_openai_sdk():
    """OpenAI SDK works as a drop-in. `api_key` is required by the SDK but unused server-side."""
    client = OpenAI(base_url=f"{EMBED_URL}/v1", api_key="not-needed")
    resp = client.embeddings.create(
        model="jina-embeddings-v5-text-nano",
        input=["Hello world", "OpenAI-compatible API"],
    )
    print(f"[OpenAI SDK] model={resp.model} count={len(resp.data)} dim={len(resp.data[0].embedding)}")


def via_cohere_shape():
    """Cohere /v1/embed shape. Use input_type to control task routing."""
    r = requests.post(
        f"{EMBED_URL}/v1/embed",
        json={"texts": ["query side"], "input_type": "search_query"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    vec = data["embeddings"]["float"][0]
    print(f"[Cohere shape] dim={len(vec)} first3={vec[:3]}")


def reranker():
    """Top-N reranking via a separate reranker container."""
    try:
        r = requests.post(
            f"{RERANK_URL}/v1/rerank",
            json={
                "query": "best programming language for AI",
                "documents": [
                    "Python is the most popular language for ML",
                    "Bananas are yellow",
                    "PyTorch and TensorFlow use Python",
                ],
                "top_n": 2,
            },
            timeout=30,
        )
        r.raise_for_status()
        for hit in r.json()["results"]:
            print(f"[Rerank] index={hit['index']} score={hit['relevance_score']:.3f}")
    except requests.exceptions.ConnectionError:
        print(f"[Rerank] skipped - no container at {RERANK_URL}")


def matryoshka():
    """Matryoshka truncation: pass `dimensions` to get any supported smaller dim."""
    for dim in (64, 128, 256, 512):
        r = requests.post(
            f"{EMBED_URL}/v1/embeddings",
            json={"input": ["matryoshka demo"], "dimensions": dim},
            timeout=30,
        )
        r.raise_for_status()
        actual = len(r.json()["data"][0]["embedding"])
        print(f"[Matryoshka] requested={dim} actual={actual}")


if __name__ == "__main__":
    via_openai_sdk()
    via_cohere_shape()
    matryoshka()
    reranker()
