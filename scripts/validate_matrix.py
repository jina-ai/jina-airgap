#!/usr/bin/env python3
"""Run every built image and record what it actually does.

Each image is started with `--network none` and probed from inside via
`docker exec`, so air-gap is the condition every functional probe runs under
rather than a separate pass: if the image needed the network for anything, none
of it would answer. A socket probe from inside the same container then has to
fail to reach huggingface.co, api.jina.ai, PyPI and DNS.

Two sources are captured here, and a third is compared offline:

  A  the image, over its own loopback           (this script)
  C  the model's own library call, in the same container, no server in the path
  B  api.jina.ai                                (captured separately, from a
                                                 host that has egress)

C is what arbitrates when A and B disagree. Running it in the same container
removes "different environment" as an explanation, so a divergence is either
the server layer (A vs C) or the hosted service (B vs C).

    ./scripts/validate_matrix.py --runtimes cpu --jobs 3
    ./scripts/validate_matrix.py --runtimes gpu --jobs 1
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / "models" / "catalog.json"
PROBE_SCRIPT = REPO / "scripts" / "probe_inside.py"

# One L4. GPU work is serialised on it regardless of --jobs.
_gpu = threading.Lock()

EN = (
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife."
)
ZH = "机器学习模型的生产部署需要考虑多个维度：推理延迟、吞吐量、硬件成本以及模型更新的持续集成与部署流程。"
QUERY = "How do teams deploy large language models efficiently in production?"
DOCS = [
    "Deploying large language models at scale presents significant "
    "infrastructure challenges, including GPU memory management, batching "
    "strategies, and latency requirements for real-time inference.",
    ZH,
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and "
    "established the principle of state sovereignty in European diplomacy.",
    "def encode_batch(texts, model):\n    return model.encode(texts)\n",
]

# Source C. Deliberately the plainest call each library documents -- the point
# is to be the model's own behaviour, not the server's.
ORACLE = r"""
import json, os, numpy as np
model_id = os.environ["ORACLE_MODEL"]
runtime  = os.environ["ORACLE_RUNTIME"]
texts    = json.loads(os.environ["ORACLE_TEXTS"])
query    = os.environ.get("ORACLE_QUERY", "")
out = {"runtime": runtime}
try:
    import torch, transformers
    out["torch"], out["transformers"] = torch.__version__, transformers.__version__
    out["cuda"] = torch.cuda.is_available()
except Exception as exc:
    out["import_error"] = repr(exc)

# The same guard app.py installs. Not a Jina behaviour: transformers wraps every
# remote-code AutoModel in a generation mixin and assumes it has
# prepare_inputs_for_generation, which an embedding-only model does not. Without
# it, plain sentence-transformers cannot load a v5 model at all -- measured, and
# the reason this patch exists in the server.
try:
    from transformers.models.auto import auto_factory
    _add = getattr(auto_factory, "add_generation_mixin_to_remote_model", None)
    if _add:
        auto_factory.add_generation_mixin_to_remote_model = (
            lambda cls: cls if not hasattr(cls, "prepare_inputs_for_generation") else _add(cls)
        )
    out["applied_generation_mixin_guard"] = bool(_add)
except Exception:
    pass
try:
    if runtime in ("sentence_transformer", "embeddings_v3", "embeddings_v4",
                   "embeddings_v5_text", "embeddings_v5_omni"):
        from sentence_transformers import SentenceTransformer
        kwargs = {"model_kwargs": {"default_task": "retrieval"}} if runtime == "embeddings_v5_omni" else {}
        model = SentenceTransformer(model_id, trust_remote_code=True, **kwargs)
        vectors = np.asarray(model.encode(texts, convert_to_numpy=True,
                                          normalize_embeddings=True))
        out["dim"] = int(vectors.shape[-1])
        out["vectors_head"] = vectors[:, :8].tolist()
        out["norms"] = [float(np.linalg.norm(v)) for v in vectors]
    elif runtime == "cross_encoder":
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_id, trust_remote_code=True)
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        # v2-base-multilingual predicts in bf16, which numpy has no dtype for,
        # so go via a tensor and cast -- the same thing CrossEncoderFamily does.
        scores = model.predict([[query, t] for t in texts],
                               convert_to_numpy=False, convert_to_tensor=True)
        if hasattr(scores, "float"):
            scores = scores.float().detach().cpu().numpy()
        out["scores"] = [float(s) for s in np.asarray(scores).ravel()]
    elif runtime == "jina_ranking":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                          torch_dtype="auto", low_cpu_mem_usage=True).eval()
        ranked = model.rerank(query, texts)
        out["scores"] = [float(r["relevance_score"]) for r in sorted(ranked, key=lambda r: r["index"])]
    elif runtime == "colbert":
        from pylate import models as pm, rank as pr
        model = pm.ColBERT(model_name_or_path=model_id, trust_remote_code=True)
        qe = model.encode([query], is_query=True, convert_to_tensor=True)
        de = model.encode(texts, is_query=False, convert_to_tensor=True)
        ranked = pr.rerank(documents_ids=[list(range(len(texts)))],
                           queries_embeddings=qe, documents_embeddings=[de])
        out["scores"] = [float(r["score"]) for r in sorted(ranked[0], key=lambda r: r["id"])]
    else:
        out["skipped"] = f"no library oracle for runtime {runtime}"
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print("ORACLE_JSON:" + json.dumps(out))
"""

PROVENANCE = (
    "import json,sys,platform\n"
    "d={'python':sys.version.split()[0],'platform':platform.platform()}\n"
    "for name in ('torch','transformers','sentence_transformers','numpy','pylate','fastapi'):\n"
    "    try:\n"
    "        d[name]=__import__(name).__version__\n"
    "    except Exception: d[name]=None\n"
    "try:\n"
    "    import torch\n"
    "    d['cuda_available']=torch.cuda.is_available()\n"
    "    d['cuda_version']=torch.version.cuda\n"
    "    d['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
    "except Exception: pass\n"
    "import os\n"
    "d['env']={k:v for k,v in os.environ.items() if k.startswith(('JINA_','HF_','TRANSFORMERS_','OMP_'))}\n"
    "print('PROV_JSON:'+json.dumps(d))"
)


def run(command, timeout=1200):
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def tagged_json(output: str, tag: str):
    for line in (output or "").splitlines():
        if line.startswith(tag):
            try:
                return json.loads(line[len(tag) :])
            except Exception as exc:
                return {"parse_error": str(exc), "line": line[:400]}
    return None


def validate(spec: dict, runtime: str, args) -> dict:
    name = spec["id"].lower()
    image = f"{args.registry}/{name}:{runtime}-{args.sha}"
    record = {
        "model": spec["id"],
        "runtime": runtime,
        "image": image,
        "hf_repo": spec["hf_repo"],
        "api_endpoint": spec["api_endpoint"],
        "catalog": {
            k: spec.get(k)
            for k in (
                "runtime",
                "modality",
                "family",
                "context",
                "output_dim",
                "vram_gb",
                "usage_prompt_tokens",
                "matryoshka_dims",
            )
        },
        "deps_pinned": spec.get("deps", {}),
    }
    if run(["docker", "image", "inspect", image]).returncode != 0:
        record["status"] = "image_missing"
        return record
    record["image_id"] = run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"]
    ).stdout.strip()
    record["image_size_gb"] = round(
        int(
            run(
                ["docker", "image", "inspect", image, "--format", "{{.Size}}"]
            ).stdout.strip()
            or 0
        )
        / 1024**3,
        2,
    )

    container = f"val-{name}-{runtime}"
    run(["docker", "rm", "-f", container], timeout=180)
    command = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container,
        "--network",
        "none",
        "-e",
        "JINA_LICENSE_MODE=off",
    ]
    lock = _gpu if runtime == "gpu" else None
    if runtime == "gpu":
        command += ["--gpus", "all"]
    command += [image]

    if lock:
        lock.acquire()
    try:
        started = run(command)
        if started.returncode != 0:
            record["status"] = "start_failed"
            record["error"] = started.stderr[-600:]
            return record
        try:
            began = time.perf_counter()
            ready = False
            for _ in range(240):
                probe = run(
                    [
                        "docker",
                        "exec",
                        container,
                        "python",
                        "-c",
                        "import urllib.request,json,sys;"
                        "d=json.load(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=5));"
                        "sys.exit(0 if d.get('ready') else 1)",
                    ],
                    timeout=90,
                )
                if probe.returncode == 0:
                    ready = True
                    break
                if (
                    run(
                        ["docker", "inspect", "-f", "{{.State.Running}}", container]
                    ).stdout.strip()
                    != "true"
                ):
                    break
                time.sleep(5)
            record["load_seconds"] = round(time.perf_counter() - began, 1)
            record["ready"] = ready
            if not ready:
                logs = run(["docker", "logs", "--tail", "80", container])
                record["status"] = "never_ready"
                record["log_tail"] = (logs.stdout + logs.stderr)[-3000:]
                return record

            record["provenance"] = tagged_json(
                run(
                    ["docker", "exec", container, "python", "-c", PROVENANCE],
                    timeout=300,
                ).stdout,
                "PROV_JSON:",
            )

            run(
                [
                    "docker",
                    "cp",
                    str(PROBE_SCRIPT),
                    f"{container}:/tmp/probe_inside.py",
                ],
                timeout=180,
            )
            probed = run(
                ["docker", "exec", container, "python", "/tmp/probe_inside.py"],
                timeout=5400,
            )
            payload = tagged_json(probed.stdout, "PROBE_JSON:")
            if payload is None:
                record["status"] = "probe_failed"
                record["error"] = (probed.stdout + probed.stderr)[-2500:]
                return record
            record["probe"] = payload
            record["egress"] = payload.get("egress")
            record["no_egress"] = all(
                v != "CONNECTED"
                for v in (payload.get("egress") or {"x": "CONNECTED"}).values()
            )

            oracle_env = []
            for key, value in (
                ("ORACLE_MODEL", spec["hf_repo"]),
                ("ORACLE_RUNTIME", spec["runtime"]),
                (
                    "ORACLE_TEXTS",
                    json.dumps(
                        DOCS if spec["api_endpoint"] == "/v1/rerank" else [EN, ZH]
                    ),
                ),
                ("ORACLE_QUERY", QUERY),
            ):
                oracle_env += ["-e", f"{key}={value}"]
            oracle = run(
                ["docker", "exec", *oracle_env, container, "python", "-c", ORACLE],
                timeout=3600,
            )
            record["oracle"] = tagged_json(oracle.stdout, "ORACLE_JSON:") or {
                "error": (oracle.stdout + oracle.stderr)[-1200:]
            }
            logs = run(["docker", "logs", "--tail", "40", container])
            record["log_tail"] = (logs.stdout + logs.stderr)[-2000:]
            record["status"] = "ok"
            return record
        finally:
            run(["docker", "kill", container], timeout=180)
    finally:
        if lock:
            lock.release()


def summarise(record: dict) -> str:
    probes = (record.get("probe") or {}).get("results", {})
    unreachable = sum(1 for p in probes.values() if p.get("status") is None)
    server_errors = sum(1 for p in probes.values() if (p.get("status") or 0) >= 500)
    return (
        f"[{str(record.get('status')):13}] {record['model']}:{record['runtime']} "
        f"load={record.get('load_seconds')}s probes={len(probes)} "
        f"5xx={server_errors} unreachable={unreachable} "
        f"airgap_ok={record.get('no_egress')} "
        f"oracle={'ok' if (record.get('oracle') or {}).get('error') is None else 'ERR'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models")
    parser.add_argument("--runtimes", default="cpu,gpu")
    parser.add_argument("--sha", default=os.environ.get("BUILD_SHA", ""))
    parser.add_argument("--registry", default="ghcr.io/jina-ai/jina-on-prem")
    parser.add_argument("--out", default="/var/tmp/jina-validate")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    if not args.sha:
        args.sha = run(
            ["git", "-C", str(REPO), "rev-parse", "--short=7", "HEAD"]
        ).stdout.strip()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.models.split(",")) if args.models else None
    models = [
        m
        for m in json.loads(CATALOG.read_text())["models"]
        if wanted is None or m["id"] in wanted
    ]
    jobs = sorted(
        ((m, r) for m in models for r in args.runtimes.split(",")),
        key=lambda pair: (pair[0].get("vram_gb") or 0, pair[0]["id"], pair[1]),
    )
    sink = outdir / f"validation-{args.sha}.jsonl"
    print(
        f"{len(jobs)} validations, sha={args.sha}, {args.jobs} parallel -> {sink}",
        flush=True,
    )

    write_lock = threading.Lock()

    def one(spec, runtime):
        began = time.perf_counter()
        try:
            record = validate(spec, runtime, args)
        except Exception as exc:  # a harness fault is data, not a stop
            record = {
                "model": spec["id"],
                "runtime": runtime,
                "image": "",
                "status": "harness_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        record["wall_seconds"] = round(time.perf_counter() - began, 1)
        with write_lock, open(sink, "a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(summarise(record), flush=True)
        return record

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, spec, runtime) for spec, runtime in jobs]
        for future in as_completed(futures):
            future.result()
    print(f"\nrecords: {sink}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
