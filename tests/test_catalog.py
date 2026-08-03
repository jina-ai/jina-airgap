"""
Catalog invariants. No server, no Docker, no network.

Bundles pin every repo to an exact commit so a rebuild of the same image tag
ships identical code and weights. A model added without a revision would build
fine and silently track main, so these assertions are the guard.

Run:
  python -m pytest tests/test_catalog.py
"""

import importlib.util
import json
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
CATALOG_PATH = os.path.join(ROOT, "models", "catalog.json")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SPEC_RE = re.compile(r"^[A-Za-z0-9._-]+(==|>=|<=|~=|!=|>|<)\S+$")

with open(CATALOG_PATH) as f:
    CATALOG = json.load(f)
MODELS = CATALOG["models"]

# jina-on-prem.py is not importable by name (hyphens), load it by path.
_spec = importlib.util.spec_from_file_location("cli", os.path.join(ROOT, "jina-on-prem.py"))
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def test_catalog_is_not_empty():
    assert len(MODELS) > 0


def test_every_model_has_id_and_repo():
    for model in MODELS:
        assert model.get("id"), f"model missing id: {model}"
        assert model.get("hf_repo"), f"{model['id']} missing hf_repo"


def test_model_ids_are_unique():
    ids = [m["id"] for m in MODELS]
    assert len(ids) == len(set(ids)), f"duplicate ids: {sorted(set(i for i in ids if ids.count(i) > 1))}"


def test_hf_repos_are_unique():
    repos = [m["hf_repo"] for m in MODELS]
    assert len(repos) == len(set(repos))


def test_every_model_pins_a_revision():
    missing = [m["id"] for m in MODELS if not m.get("revision")]
    assert not missing, f"models tracking main instead of a pinned commit: {missing}"


def test_revisions_are_full_commit_shas():
    bad = [(m["id"], m["revision"]) for m in MODELS if not SHA_RE.match(m["revision"])]
    assert not bad, f"revision must be a 40-char lowercase commit sha: {bad}"


def test_extra_repos_are_pinned():
    """Implementation repos are shared: xlm-roberta-flash-implementation backs
    three models, so one unpinned upstream push changes three bundles at once."""
    bad = []
    for model in MODELS:
        for entry in model.get("extra_repos", []):
            repo, separator, sha = entry.partition("@")
            if not separator or not SHA_RE.match(sha) or "/" not in repo:
                bad.append((model["id"], entry))
    assert not bad, f"extra_repos entries must be 'owner/repo@<40-char sha>': {bad}"


def test_prebuilt_models_declare_deps():
    for model in MODELS:
        if model.get("prebuilt"):
            assert model.get("deps"), f"{model['id']} is prebuilt but declares no deps"


def test_every_dep_line_is_a_valid_requirement():
    """build_model_requirements() output is written to model-requirements.txt and
    pip-installed verbatim during the build."""
    bad = []
    for model in MODELS:
        for line in cli.build_model_requirements(model).splitlines():
            if not SPEC_RE.match(line):
                bad.append((model["id"], line))
    assert not bad, f"not installable by pip: {bad}"


def test_torch_is_pinned_with_double_equals():
    """Dockerfile.cpu and Dockerfile.gpu install torch separately from the CPU or
    CUDA wheel index, extracting the version with `grep -i '^torch=='`. A dep
    written as torch>=2.5 satisfies pip but returns nothing from that grep, so the
    build silently installs an unpinned torch from the wrong index."""
    bad = []
    for model in MODELS:
        requirements = cli.build_model_requirements(model)
        for package in ("torch", "torchvision"):
            declared = model.get("deps", {}).get(package)
            if declared is None:
                continue
            if not re.search(rf"(?im)^{package}==", requirements):
                bad.append((model["id"], package, declared))
    assert not bad, f"Dockerfile greps '^{{pkg}}==' and would find nothing: {bad}"


def test_catalog_round_trips_with_two_space_indent():
    """refresh_revisions.py rewrites the file with indent=2 and a trailing
    newline. Keep the committed formatting identical so a revision bump is a
    minimal diff instead of a whole-file reformat."""
    with open(CATALOG_PATH) as f:
        on_disk = f.read()
    assert on_disk == json.dumps(CATALOG, indent=2) + "\n"
