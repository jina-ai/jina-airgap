"""
Unit tests for docker/verify_staging.py auto_map parsing. No server, no Docker,
no network, no torch.

verify_staging.py is the build gate that fails a bundle when the model's
trust_remote_code files were not pre-staged. If its auto_map parsing misses a
reference, the gate silently checks nothing and the bug it exists to catch ships.

Run:
  python -m pytest tests/test_verify_staging.py
"""

import json
import os
import sys

DOCKER_DIR = os.path.join(os.path.dirname(__file__), "..", "docker")
sys.path.insert(0, DOCKER_DIR)
import verify_staging as V  # noqa: E402

MODEL = "jinaai/jina-embeddings-v5-text-nano"


def test_same_repo_reference():
    assert V.parse_auto_map_entry("modeling_eurobert.EuroBertModel", MODEL) == (
        MODEL,
        "modeling_eurobert.py",
    )


def test_cross_repo_reference():
    """jina-embeddings-v3 points every auto_map entry at an implementation repo.
    Treating these as same-repo names would check the wrong repo and pass."""
    value = "jinaai/xlm-roberta-flash-implementation--modeling_lora.XLMRobertaLoRA"
    assert V.parse_auto_map_entry(value, MODEL) == (
        "jinaai/xlm-roberta-flash-implementation",
        "modeling_lora.py",
    )


def test_nested_class_reference():
    assert V.parse_auto_map_entry("modeling_x.Outer.Inner", MODEL) == (MODEL, "modeling_x.py")


def test_modules_from_config_collects_every_entry():
    blob = {
        "auto_map": {
            "AutoConfig": "configuration_jina.JinaConfig",
            "AutoModel": "modeling_jina.JinaModel",
        }
    }
    assert V.modules_from_config(blob, MODEL) == {
        (MODEL, "configuration_jina.py"),
        (MODEL, "modeling_jina.py"),
    }


def test_modules_from_config_handles_list_values():
    """Processor auto_map entries are lists, e.g. [slow_class, fast_class]."""
    blob = {"auto_map": {"AutoProcessor": ["processing_x.SlowProc", "processing_y.FastProc"]}}
    assert V.modules_from_config(blob, MODEL) == {
        (MODEL, "processing_x.py"),
        (MODEL, "processing_y.py"),
    }


def test_modules_from_config_tolerates_missing_and_null_auto_map():
    assert V.modules_from_config({}, MODEL) == set()
    assert V.modules_from_config({"auto_map": None}, MODEL) == set()
    assert V.modules_from_config({"auto_map": {"AutoModel": None}}, MODEL) == set()


def test_modules_for_snapshot_reads_every_config_file(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"auto_map": {"AutoModel": "modeling_a.A"}})
    )
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"auto_map": {"AutoProcessor": "processing_b.B"}})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"auto_map": {"AutoTokenizer": ["tokenization_c.C", None]}})
    )
    assert V.modules_for_snapshot(str(tmp_path), MODEL) == {
        (MODEL, "modeling_a.py"),
        (MODEL, "processing_b.py"),
        (MODEL, "tokenization_c.py"),
    }


def test_modules_for_snapshot_skips_unparseable_config(tmp_path):
    (tmp_path / "config.json").write_text("{not json")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"auto_map": {"AutoTokenizer": "tokenization_c.C"}})
    )
    assert V.modules_for_snapshot(str(tmp_path), MODEL) == {(MODEL, "tokenization_c.py")}


def test_modules_for_snapshot_empty_when_model_has_no_remote_code(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))
    assert V.modules_for_snapshot(str(tmp_path), MODEL) == set()


def test_download_model_invokes_this_script_as_a_sibling():
    """download_model.py locates this file with dirname(__file__), so the two must
    be copied into the same directory. Nothing else enforces that pairing."""
    source = open(os.path.join(DOCKER_DIR, "download_model.py")).read()
    assert 'os.path.dirname(os.path.abspath(__file__)), "verify_staging.py"' in source


def test_sibling_import_path_covers_snapshot_and_staged_dirs(tmp_path, monkeypatch):
    """Jina model code imports siblings absolutely, and transformers resolves those
    through sys.path. Without the snapshot on it, check_imports raises ImportError
    and the module is never staged, which is the reported bug."""
    home = tmp_path / "cache"
    staged = home / "modules" / "transformers_modules" / "jinaai" / "m" / "sha"
    staged.mkdir(parents=True)
    (staged / "configuration_x.py").write_text("")
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    monkeypatch.setenv("HF_HOME", str(home))
    monkeypatch.setattr(sys, "path", list(sys.path))

    V.add_sibling_imports_to_path(str(snapshot))
    assert str(snapshot) in sys.path
    assert str(staged) in sys.path


def test_build_puts_the_snapshot_on_sys_path_before_preloading():
    source = open(os.path.join(DOCKER_DIR, "download_model.py")).read()
    assert source.index("sys.path.insert(0, snapshot_dir)") < source.index(
        "# Pre-load to stage the trust_remote_code"
    )


def test_preload_runs_after_the_patch_block():
    """The pre-load is what stages the .py files, so it must run after the patches
    that make those files importable. Loading first meant loading code we know does
    not import cleanly, and the swallowed failure shipped a half-written cache.
    Nothing but ordering enforces this."""
    source = open(os.path.join(DOCKER_DIR, "download_model.py")).read()
    patches = source.index("# Patch model code in all cached locations")
    preload = source.index("# Pre-load to stage the trust_remote_code")
    verify = source.index("Verifying model code is pre-staged")
    assert patches < preload, "pre-load must not run before the patch block"
    assert preload < verify, "the staging check must run after the pre-load"


def test_dockerfiles_copy_both_scripts_together():
    """A COPY line that forgets verify_staging.py breaks the build gate silently,
    and only at build time."""
    for name in ("Dockerfile.cpu", "Dockerfile.gpu"):
        dockerfile = open(os.path.join(DOCKER_DIR, name)).read()
        copy_lines = [l for l in dockerfile.splitlines() if l.startswith("COPY") and "download_model.py" in l]
        assert len(copy_lines) == 1, f"{name}: expected one COPY of download_model.py"
        assert "verify_staging.py" in copy_lines[0], f"{name}: {copy_lines[0]!r} drops verify_staging.py"
        assert copy_lines[0].rstrip().endswith("/tmp/"), (
            f"{name}: both scripts must land in one directory, got {copy_lines[0]!r}"
        )
        assert "RUN python /tmp/download_model.py" in dockerfile, f"{name}: entry point moved"


def test_v3_style_snapshot_resolves_to_the_implementation_repo(tmp_path):
    """Regression guard for the real jina-embeddings-v3 config: every entry is
    cross-repo, so a parser that skips '--' values finds nothing to verify."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "auto_map": {
                    "AutoConfig": "jinaai/xlm-roberta-flash-implementation--configuration_xlm_roberta.XLMRobertaFlashConfig",
                    "AutoModel": "jinaai/xlm-roberta-flash-implementation--modeling_lora.XLMRobertaLoRA",
                }
            }
        )
    )
    found = V.modules_for_snapshot(str(tmp_path), "jinaai/jina-embeddings-v3")
    assert found == {
        ("jinaai/xlm-roberta-flash-implementation", "configuration_xlm_roberta.py"),
        ("jinaai/xlm-roberta-flash-implementation", "modeling_lora.py"),
    }
    assert all(repo != "jinaai/jina-embeddings-v3" for repo, _ in found)
