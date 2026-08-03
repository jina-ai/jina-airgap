#!/usr/bin/env python3
"""Refresh the pinned HF commit for every repo in models/catalog.json.

Bundles pin each model to an exact commit so rebuilding the same image tag ships
identical code and weights. Bumping a model is therefore a deliberate act: run
this, review the diff, commit it.

Usage:
    python scripts/refresh_revisions.py            # show what would change
    python scripts/refresh_revisions.py --write    # apply
    HF_TOKEN=... python scripts/refresh_revisions.py --write   # for gated repos
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

CATALOG = Path(__file__).resolve().parent.parent / "models" / "catalog.json"


def apply_revisions(catalog, shas):
    """Pin every repo in the catalog to shas, in place. Returns what changed.

    Rebuilds each model dict so `revision` sits right after `hf_repo`. The
    existing `revision` key is dropped rather than copied, or the copy would
    overwrite the value just written and the rewrite would be a no-op on every
    run after the first.
    """
    changed = []
    for model in catalog["models"]:
        updated = collections.OrderedDict()
        for key, value in model.items():
            if key == "revision":
                continue
            updated[key] = value
            if key == "hf_repo":
                new = shas[value]
                if model.get("revision") != new:
                    changed.append((value, model.get("revision"), new))
                updated["revision"] = new
            elif key == "extra_repos":
                pinned = []
                for entry in value:
                    repo = entry.split("@")[0]
                    if entry != f"{repo}@{shas[repo]}":
                        changed.append((repo, entry.partition("@")[2] or None, shas[repo]))
                    pinned.append(f"{repo}@{shas[repo]}")
                updated[key] = pinned
        model.clear()
        model.update(updated)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write changes to catalog.json")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub not installed: pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    catalog = json.loads(CATALOG.read_text(), object_pairs_hook=collections.OrderedDict)

    repos = set()
    for model in catalog["models"]:
        repos.add(model["hf_repo"])
        for entry in model.get("extra_repos", []):
            repos.add(entry.split("@")[0])

    shas = {}
    for repo in sorted(repos):
        try:
            shas[repo] = api.model_info(repo).sha
        except Exception as e:
            print(f"ERROR resolving {repo}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    changed = apply_revisions(catalog, shas)

    if not changed:
        print(f"Up to date: {len(repos)} repos already pinned to their current commit.")
        return 0

    for repo, old, new in changed:
        print(f"{repo}\n  {old or '(unpinned)'} -> {new}")

    if not args.write:
        print(f"\n{len(changed)} change(s). Re-run with --write to apply.")
        return 0

    with CATALOG.open("w") as f:
        json.dump(catalog, f, indent=2)
        f.write("\n")
    print(f"\nWrote {len(changed)} change(s) to {CATALOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
