import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_CATALOG_PATHS = [
    Path("/app/catalog.json"),
    Path(__file__).parent.parent / "models" / "catalog.json",
]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    hf_repo: str
    runtime: str
    type: str
    family: str
    modality: str
    tasks: tuple
    api_endpoint: str
    context: int
    output_dim: Optional[int] = None
    matryoshka_dims: Optional[tuple] = None
    vram_gb: Optional[int] = None
    gpu_dtype: Optional[str] = None
    extra_repos: Optional[tuple] = None
    # When this model's public /v1/embeddings response carries
    # `usage.prompt_tokens` alongside `total_tokens`: "always", "late_chunking"
    # or "never". Measured per model against api.jina.ai, because it follows
    # no structural rule -- every v2-base-* and both CLIP models emit it, v3
    # emits it only under late_chunking, and v4/v5/code/colbert never do (v4
    # under late_chunking still does not). It is upstream executor drift, so
    # the honest way to carry it is as measured data.
    usage_prompt_tokens: str = "never"


def _load() -> dict:
    for path in _CATALOG_PATHS:
        if path.is_file():
            with open(path) as f:
                return {m["id"]: m for m in json.load(f)["models"]}
    raise RuntimeError(
        f"catalog.json not found; tried: {[str(p) for p in _CATALOG_PATHS]}"
    )


_catalog: dict = _load()


def is_known(model_id: str) -> bool:
    """True if this names a Jina model. Lets a route tell a deployment mistake
    (a Jina id that is not this image's model) from another provider's model
    name, which a drop-in endpoint must accept and ignore."""
    return model_id.split("/")[-1] in _catalog or model_id in _catalog


def spec_for(model_id: str) -> ModelSpec:
    short_id = model_id.split("/")[-1] if "/" in model_id else model_id
    entry = _catalog.get(short_id) or _catalog.get(model_id)
    if entry is None:
        sys.exit(
            f"Model '{short_id}' not found in catalog. "
            f"Known models: {sorted(_catalog)}"
        )
    return ModelSpec(
        id=entry["id"],
        hf_repo=entry["hf_repo"],
        runtime=entry["runtime"],
        type=entry["type"],
        family=entry["family"],
        modality=entry["modality"],
        tasks=tuple(entry.get("tasks", [])),
        api_endpoint=entry["api_endpoint"],
        context=entry.get("context", 0),
        output_dim=entry.get("output_dim"),
        matryoshka_dims=(
            tuple(entry["matryoshka_dims"]) if entry.get("matryoshka_dims") else None
        ),
        vram_gb=entry.get("vram_gb"),
        gpu_dtype=entry.get("gpu_dtype"),
        extra_repos=tuple(entry["extra_repos"]) if entry.get("extra_repos") else None,
        usage_prompt_tokens=entry.get("usage_prompt_tokens", "never"),
    )
