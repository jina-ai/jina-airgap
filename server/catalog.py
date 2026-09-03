import json
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
    # What the model does, for humans and the generated catalog table.
    tasks: tuple
    # What `task` accepts on the wire, verbatim: the exact strings, not base
    # names with suffix algebra. These differ in kind, not just in accuracy --
    # v3 *does* retrieval, and the string `retrieval` is not something it takes.
    # Empty means the public API declares no `task` field for this model, so
    # there is nothing to check a value against and it is passed through.
    task_enum: tuple
    api_endpoint: str
    context: int
    # The license category this model is sold under, and the authority the gate
    # compares a category-scoped key against. It is a stored field rather than
    # something derived from ``type`` and ``modality`` at read time because the
    # mapping is a commercial decision that must not drift: a container answers
    # from the catalog baked into its own image, so a rule edited later would
    # silently re-scope keys already in the field.
    #
    # Defaulted rather than required because this is read at import time, and a
    # KeyError here is a container that will not boot. An entry without the
    # field simply matches no category, which refuses rather than guesses.
    license_category: str = ""
    output_dim: Optional[int] = None
    matryoshka_dims: Optional[tuple] = None
    vram_gb: Optional[int] = None
    gpu_dtype: Optional[str] = None
    extra_repos: Optional[tuple] = None


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
    """Does this project ship that model at all?

    The difference decides what a wrong `model` field is told to do: pull a
    different image of ours, or stop naming another vendor's model.
    """
    return model_id.split("/")[-1] in _catalog or model_id in _catalog


def spec_for(model_id: str) -> ModelSpec:
    short_id = model_id.split("/")[-1] if "/" in model_id else model_id
    entry = _catalog.get(short_id) or _catalog.get(model_id)
    if entry is None:
        raise ValueError(
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
        task_enum=tuple(entry.get("task_enum", [])),
        api_endpoint=entry["api_endpoint"],
        context=entry.get("context", 0),
        license_category=entry.get("license_category", ""),
        output_dim=entry.get("output_dim"),
        matryoshka_dims=(
            tuple(entry["matryoshka_dims"]) if entry.get("matryoshka_dims") else None
        ),
        vram_gb=entry.get("vram_gb"),
        gpu_dtype=entry.get("gpu_dtype"),
        extra_repos=tuple(entry["extra_repos"]) if entry.get("extra_repos") else None,
    )
