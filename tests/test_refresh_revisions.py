"""
Unit tests for scripts/refresh_revisions.py. No network.

Run:
  python -m pytest tests/test_refresh_revisions.py
"""

import copy
import importlib.util
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
_spec = importlib.util.spec_from_file_location(
    "refresh_revisions", os.path.join(ROOT, "scripts", "refresh_revisions.py")
)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

OLD, NEW = "a" * 40, "b" * 40
IMPL_OLD, IMPL_NEW = "c" * 40, "d" * 40


def catalog(revision=None, extra=None):
    model = {"id": "m", "hf_repo": "org/m"}
    if revision:
        model["revision"] = revision
    model["type"] = "embedding"
    if extra:
        model["extra_repos"] = extra
    return {"models": [model]}


def test_pins_a_model_that_has_no_revision_yet():
    c = catalog()
    changed = R.apply_revisions(c, {"org/m": NEW})
    assert c["models"][0]["revision"] == NEW
    assert changed == [("org/m", None, NEW)]


def test_bumps_a_model_that_is_already_pinned():
    """Regression: rebuilding the dict used to copy the existing revision key
    back over the value just written, so every run after the first reported a
    change and then wrote the old sha."""
    c = catalog(revision=OLD)
    changed = R.apply_revisions(c, {"org/m": NEW})
    assert c["models"][0]["revision"] == NEW, "bump was silently reverted"
    assert changed == [("org/m", OLD, NEW)]


def test_is_idempotent_and_reports_nothing_when_current():
    c = catalog(revision=NEW)
    assert R.apply_revisions(c, {"org/m": NEW}) == []
    assert c["models"][0]["revision"] == NEW


def test_revision_sits_directly_after_hf_repo():
    c = catalog(revision=OLD)
    R.apply_revisions(c, {"org/m": NEW})
    keys = list(c["models"][0])
    assert keys[keys.index("hf_repo") + 1] == "revision"
    assert keys.count("revision") == 1


def test_bumps_extra_repos():
    c = catalog(revision=OLD, extra=[f"org/impl@{IMPL_OLD}"])
    changed = R.apply_revisions(c, {"org/m": NEW, "org/impl": IMPL_NEW})
    assert c["models"][0]["extra_repos"] == [f"org/impl@{IMPL_NEW}"]
    assert ("org/impl", IMPL_OLD, IMPL_NEW) in changed


def test_preserves_every_other_field():
    c = catalog(revision=OLD)
    before = copy.deepcopy(c["models"][0])
    R.apply_revisions(c, {"org/m": NEW})
    after = c["models"][0]
    assert {k: v for k, v in after.items() if k != "revision"} == {
        k: v for k, v in before.items() if k != "revision"
    }
