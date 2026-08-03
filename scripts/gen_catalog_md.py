#!/usr/bin/env python3
"""Generate the Model Catalog wiki page from models/catalog.json.

Usage:
  python3 scripts/gen_catalog_md.py > /tmp/Model-Catalog.md

Designed to be re-run any time models/catalog.json changes so the wiki page
stays in sync with the source of truth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "models" / "catalog.json"
GHCR = "https://github.com/orgs/jina-ai/packages/container/package/jina-on-prem%2F"



def prebuilt_link(model_id: str, runtime: str) -> str:
    # Link the package page, not a pinned version ID: version IDs churn on every
    # re-push and their old URLs 404, whereas the package page is permanent.
    return f"[{runtime}]({GHCR}{model_id})"


def fmt_ctx(n: int | None) -> str:
    if not n:
        return "-"
    if n >= 1024:
        return f"{n // 1024}K"
    return str(n)


def fmt_dim(m: dict) -> str:
    d = m.get("output_dim")
    if not d:
        return "-"
    matryoshka = m.get("matryoshka_dims") or []
    if matryoshka and len(matryoshka) > 1:
        return f"{d} (matryoshka: {min(matryoshka)}-{max(matryoshka)})"
    return str(d)


def fmt_tasks(m: dict) -> str:
    tasks = m.get("tasks") or []
    return ", ".join(tasks) if tasks else "-"


def render() -> str:
    data = json.loads(CATALOG.read_text())
    models = data["models"]

    out: list[str] = []
    # Page title comes from wiki UI, not H1 in body
    out.append(
        "All 28 models supported by jina-on-prem. Auto-generated from "
        "[`models/catalog.json`](https://github.com/jina-ai/jina-on-prem/blob/main/models/catalog.json) - "
        "re-run `python3 scripts/gen_catalog_md.py` to refresh."
    )
    out.append("")
    out.append(
        "**License note**: Models tagged `CC-BY-NC-4.0` need a commercial "
        "license for production use. Contact [Elastic sales](https://www.elastic.co/contact)."
    )
    out.append("")

    by_type: dict[str, list[dict]] = {}
    for m in models:
        by_type.setdefault(m["type"], []).append(m)

    order = ["embedding", "reranker", "colbert", "reader", "vlm"]
    headings = {
        "embedding": "Embeddings",
        "reranker": "Rerankers",
        "colbert": "ColBERT",
        "reader": "Readers",
        "vlm": "Vision-Language",
    }

    for t in order:
        if t not in by_type:
            continue
        out.append(f"## {headings[t]}")
        out.append("")
        out.append("| Model | Prebuilt | Params | VRAM | Context | Output | Modality | License |")
        out.append("|---|---|---|---|---|---|---|---|")
        for m in by_type[t]:
            prebuilt = (
                f"{prebuilt_link(m['id'], 'cpu')} / {prebuilt_link(m['id'], 'gpu')}"
                if m.get("prebuilt")
                else "-"
            )
            row = [
                f"`{m['id']}`",
                prebuilt,
                m.get("parameters", "-"),
                f"~{m.get('vram_gb', '?')}GB",
                fmt_ctx(m.get("context")),
                fmt_dim(m),
                m.get("modality", "-"),
                m.get("license", "-"),
            ]
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    out.append("## Picking a model")
    out.append("")
    out.append("Quick rules of thumb:")
    out.append("")
    out.append("- **First-time test / latency-critical**: `jina-embeddings-v5-text-nano` (239M, ~2GB, CPU-friendly).")
    out.append("- **Multilingual production embeddings**: `jina-embeddings-v5-text-small` or `jina-embeddings-v4`.")
    out.append("- **Multimodal (text + image)**: `jina-embeddings-v5-omni-small` or `jina-clip-v2`.")
    out.append("- **Code search**: `jina-code-embeddings-1.5b` (or 0.5b for smaller deploys).")
    out.append("- **Reranking after retrieval**: `jina-reranker-v3` (best quality) or `jina-reranker-v2-base-multilingual` (faster).")
    out.append("- **HTML/document cleanup**: `ReaderLM-v2` (largest context) or `reader-lm-0.5b` (lightweight).")
    out.append("")
    out.append("See [API Reference](API-Reference) for the request shapes each model expects.")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.write(render())
