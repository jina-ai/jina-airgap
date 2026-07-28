#!/usr/bin/env python3
"""Build every catalog model x runtime, tagged by commit SHA.

`jina-on-prem.py bundle` builds one model at a time and writes
`docker/model-requirements.txt` into the shared repo tree, so two concurrent
bundles would race on that file. This driver gives every build its own context
directory instead, which makes the matrix parallelisable and leaves the CLI --
the thing customers run -- untouched.

Tagging: `<registry>/<model>:<runtime>-<sha>` and nothing else. `:cpu` and
`:gpu` are what customers pull today and are never written here; promoting a
validated digest onto them is a separate, deliberate act.

    ./scripts/build_matrix.py --jobs 6
    ./scripts/build_matrix.py --models jina-reranker-v1-tiny-en --runtimes cpu
    ./scripts/build_matrix.py --push          # after the image has passed validation
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / "models" / "catalog.json"
DEFAULT_REGISTRY = "ghcr.io/jina-ai/jina-on-prem"

# Refuse to start a build with less headroom than the largest known image needs
# (reranker-v3.5's GPU variant measured 22.6 GB), doubled for the intermediate
# downloader stage. Running out mid-build has corrupted layers on this project
# before, and a corrupted layer reads as a code bug for hours.
MIN_FREE_GB = 60


def catalog_models() -> list[dict]:
    return json.loads(CATALOG.read_text())["models"]


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def requirements_for(model: dict) -> str:
    lines = [f"{pkg}{spec}" for pkg, spec in model.get("deps", {}).items() if spec]
    return "\n".join(lines) + "\n" if lines else "# no model-specific deps\n"


def make_context(model: dict, runtime: str, root: Path) -> Path:
    """A self-contained build context mirroring the repo layout."""
    context = root / f"{model['id'].lower()}-{runtime}"
    if context.exists():
        shutil.rmtree(context)
    (context / "docker").mkdir(parents=True)
    (context / "models").mkdir()
    shutil.copytree(REPO / "server", context / "server")
    shutil.copy(CATALOG, context / "models" / "catalog.json")
    shutil.copy(REPO / "docker" / f"Dockerfile.{runtime}", context / "docker")
    shutil.copy(REPO / "docker" / "download_model.py", context / "docker")
    shutil.copy(REPO / ".dockerignore", context / ".dockerignore")
    (context / "docker" / "model-requirements.txt").write_text(requirements_for(model))
    return context


def build(model: dict, runtime: str, args, root: Path, log_dir: Path) -> dict:
    name = model["id"].lower()
    tag = f"{args.registry}/{name}:{runtime}-{args.sha}"
    local = f"jina/{name}:{runtime}"
    log_path = log_dir / f"{name}-{runtime}.log"
    result = {
        "model": model["id"],
        "runtime": runtime,
        "tag": tag,
        "log": str(log_path),
    }

    available = free_gb(root)
    if available < MIN_FREE_GB:
        result |= {"status": "skipped", "reason": f"only {available:.0f} GB free"}
        return result

    context = make_context(model, runtime, root)
    command = [
        "docker", "build",
        "-f", str(context / "docker" / f"Dockerfile.{runtime}"),
        "--build-arg", f"MODEL_ID={model['hf_repo']}",
        "--build-arg", f"EXTRA_REPOS={','.join(model.get('extra_repos', []))}",
        "-t", tag, "-t", local,
    ]
    if runtime == "gpu" and model.get("gpu_dtype"):
        command += ["--build-arg", f"DTYPE={model['gpu_dtype']}"]
    if args.hf_token:
        command += ["--build-arg", f"HF_TOKEN={args.hf_token}"]
    command.append(str(context))

    started = time.time()
    with open(log_path, "w") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ | {"DOCKER_BUILDKIT": "1"},
        )
    shutil.rmtree(context, ignore_errors=True)

    result |= {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "seconds": round(time.time() - started, 1),
        "free_gb_after": round(free_gb(root), 1),
    }
    if completed.returncode == 0:
        result["size_gb"] = round(int(inspect(tag, "{{.Size}}") or 0) / 1024**3, 2)
        result["image_id"] = inspect(tag, "{{.Id}}")
        if args.push:
            push = subprocess.run(["docker", "push", tag], capture_output=True, text=True)
            result["pushed"] = push.returncode == 0
            if push.returncode == 0:
                result["digest"] = inspect(tag, "{{index .RepoDigests 0}}")
            else:
                result["push_error"] = push.stderr.strip()[-400:]
    return result


def inspect(tag: str, fmt: str) -> str:
    out = subprocess.run(
        ["docker", "image", "inspect", tag, "--format", fmt],
        capture_output=True, text=True,
    )
    return out.stdout.strip() if out.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", help="comma-separated ids; default all")
    parser.add_argument("--runtimes", default="cpu,gpu")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--sha", default=os.environ.get("BUILD_SHA", ""))
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--work-dir", default="/var/tmp/jina-build")
    parser.add_argument("--log-dir", default="/var/tmp/jina-build/logs")
    parser.add_argument(
        "--push",
        action="store_true",
        help="push the SHA tag. Only for builds that have passed validation -- "
             "never pushes :cpu / :gpu.",
    )
    args = parser.parse_args()

    if not args.sha:
        args.sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
    if not args.sha:
        print("error: no --sha and not a git checkout", file=sys.stderr)
        return 2

    root = Path(args.work_dir) / "ctx"
    log_dir = Path(args.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.models.split(",")) if args.models else None
    models = [m for m in catalog_models() if wanted is None or m["id"] in wanted]
    if wanted and len(models) != len(wanted):
        missing = wanted - {m["id"] for m in models}
        print(f"error: not in catalog: {sorted(missing)}", file=sys.stderr)
        return 2
    runtimes = args.runtimes.split(",")

    # Smallest first: a pipeline mistake surfaces in minutes on a 33M model
    # instead of after a 3.8B download.
    jobs = sorted(
        ((m, r) for m in models for r in runtimes),
        key=lambda pair: (pair[0].get("vram_gb") or 0, pair[0]["id"]),
    )
    print(
        f"{len(jobs)} builds, sha={args.sha}, {args.jobs} parallel, "
        f"{free_gb(root):.0f} GB free, push={args.push}",
        flush=True,
    )

    results_path = Path(args.work_dir) / f"results-{args.sha}.jsonl"
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(build, model, runtime, args, root, log_dir): (model, runtime)
            for model, runtime in jobs
        }
        with open(results_path, "a") as sink:
            for done in as_completed(futures):
                result = done.result()
                sink.write(json.dumps(result) + "\n")
                sink.flush()
                print(
                    f"[{result['status']:7}] {result['model']}:{result['runtime']} "
                    f"{result.get('seconds', 0)}s {result.get('size_gb', '')}GB "
                    f"free={result.get('free_gb_after', '')}GB",
                    flush=True,
                )

    print(f"\nresults: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
