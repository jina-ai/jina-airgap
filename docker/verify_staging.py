"""Assert the model's trust_remote_code files are already staged.

Runs at the end of the Docker build, in its own process so it can turn on HF
offline mode before transformers is imported.

transformers imports custom model code from $HF_HOME/modules/transformers_modules
rather than straight out of the hub cache. Anything not already there gets copied
on first load, and transformers logs that copy as "A new version of the following
files was downloaded from <hub url>". Nothing is fetched (HF_HUB_OFFLINE=1 makes
that impossible), but the copy fails outright when the container runs with a
read-only filesystem, and the log line reads as an air-gap violation.

Resolves the module files instead of instantiating the model, so it covers every
model type including the colbert (pylate) and reranker paths that the build's
pre-load step does not reach.
"""
import glob
import json
import os
import sys

# auto_map values name a class, either "module.Class" in this repo or
# "other/repo--module.Class" in an implementation repo (the extra_repos entries,
# e.g. jina-embeddings-v3 -> xlm-roberta-flash-implementation).
AUTO_MAP_FILES = ("config.json", "preprocessor_config.json", "tokenizer_config.json")


def parse_auto_map_entry(value, model_id):
    """Map one auto_map value to (repo_id, module_filename)."""
    repo, separator, reference = value.rpartition("--")
    return (repo if separator else model_id), reference.split(".")[0] + ".py"


def modules_from_config(blob, model_id):
    """Collect (repo_id, module_filename) pairs from one config blob's auto_map."""
    found = set()
    for value in (blob.get("auto_map") or {}).values():
        for one in value if isinstance(value, (list, tuple)) else [value]:
            if one:
                found.add(parse_auto_map_entry(one, model_id))
    return found


def modules_for_snapshot(snapshot, model_id):
    """Collect every remote-code module referenced by a downloaded snapshot."""
    found = set()
    for name in AUTO_MAP_FILES:
        path = os.path.join(snapshot, name)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            try:
                blob = json.load(f)
            except ValueError:
                continue
        found |= modules_from_config(blob, model_id)
    return found


def add_sibling_imports_to_path(snapshot):
    """Make a model's own .py files importable by bare name.

    Jina model code imports its siblings absolutely (`from configuration_eurobert
    import ...`, not `from .configuration_eurobert`). transformers resolves those
    through importlib.util.find_spec, which searches sys.path, so without the
    model's directory on it every such module raises ImportError and never gets
    staged. server/app.py does the same thing at runtime for the staged copies.
    """
    staged = os.path.join(os.environ["HF_HOME"], "modules", "transformers_modules")
    directories = [snapshot] + [
        os.path.dirname(p) for p in glob.glob(os.path.join(staged, "**", "*.py"), recursive=True)
    ]
    for directory in directories:
        if directory not in sys.path:
            sys.path.insert(0, directory)


def main(model_id, snapshot):
    add_sibling_imports_to_path(snapshot)
    modules = modules_for_snapshot(snapshot, model_id)
    if not modules:
        print("NO_REMOTE_CODE")
        return 0

    from transformers.dynamic_module_utils import get_cached_module_file

    # Cross-repo failures are fatal too. Downgrading them to a note let a build
    # pass with an implementation repo missing from extra_repos, shipping an
    # image whose remote code cannot load offline.
    for repo, module_file in sorted(modules):
        if repo == model_id and not os.path.isfile(os.path.join(snapshot, module_file)):
            continue
        get_cached_module_file(repo, module_file)
    print("VERIFY_DONE")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/model_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.exit(main(sys.argv[1], sys.argv[2]))
