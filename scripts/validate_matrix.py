#!/usr/bin/env python3
"""Run every built image and record what it actually does.

Three sources are compared on identical input:

  A  the image, over HTTP
  B  api.jina.ai, when the model is public there
  C  the model's own library call, executed *inside the same container* -- same
     weights, same pinned deps, no server in the path

C is what arbitrates when A and B disagree. Running it inside the container is
the point: it removes "different environment" as an explanation, so a
divergence is either the server layer (A vs C) or the hosted service (B vs C).

Air-gap is asserted separately, with `--network none` plus a socket probe from
inside the container, because HF_HUB_OFFLINE only proves intent.

    ./scripts/validate_matrix.py --out /var/tmp/jina-validate
    ./scripts/validate_matrix.py --models jina-embeddings-v3 --runtimes cpu
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / "models" / "catalog.json"
PORT = 8080

# Realistic inputs. Public-domain prose, CJK, and real code -- never
# "hello world", which exercises no tokenizer path worth measuring.
EN = (
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife. However little known the "
    "feelings or views of such a man may be on his first entering a "
    "neighbourhood, this truth is so well fixed in the minds of the "
    "surrounding families, that he is considered the rightful property of "
    "some one or other of their daughters."
)
ZH = "机器学习模型的生产部署需要考虑多个维度：推理延迟、吞吐量、硬件成本以及模型更新的持续集成与部署流程。"
CODE = (
    "def encode_batch(texts, model, batch_size=32):\n"
    "    vectors = []\n"
    "    for start in range(0, len(texts), batch_size):\n"
    "        chunk = texts[start : start + batch_size]\n"
    "        vectors.extend(model.encode(chunk, normalize_embeddings=True))\n"
    "    return vectors\n"
)
QUERY = "How do teams deploy large language models efficiently in production?"
DOCS = [
    "Deploying large language models at scale presents significant "
    "infrastructure challenges, including GPU memory management, batching "
    "strategies, and latency requirements for real-time inference.",
    ZH,
    "The Treaty of Westphalia in 1648 ended the Thirty Years' War and "
    "established the principle of state sovereignty in European diplomacy.",
    CODE,
]


def run(command, timeout=900, check=False):
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=isinstance(command, str),
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"{command}: {completed.stderr[-500:]}")
    return completed


def http(url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "jina-validate"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw, headers = (
                response.status,
                response.read(),
                dict(response.headers),
            )
    except urllib.error.HTTPError as exc:
        status, raw, headers = exc.code, exc.read(), dict(exc.headers)
    except Exception as exc:
        return {"status": None, "error": f"{type(exc).__name__}: {exc}"}
    try:
        body_out = json.loads(raw)
    except Exception:
        body_out = {"_raw": raw.decode(errors="replace")[:500]}
    return {
        "status": status,
        "client_ms": round((time.perf_counter() - started) * 1000, 1),
        "x_jina_elapsed_ms": headers.get("X-Jina-Elapsed-Ms"),
        "x_jina_tok_per_s": headers.get("X-Jina-Tok-Per-S"),
        "body": body_out,
    }


def shrink(node, keep=8):
    """Keep dumps readable; record what was cut so a dim is still checkable."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "embedding" and isinstance(value, list) and len(value) > keep:
                out[key], out["_full_dim"] = value[:keep], len(value)
            elif key == "embeddings" and isinstance(value, list) and value:
                if isinstance(value[0], list):
                    out[key] = [row[:keep] for row in value[:2]]
                    out["_num_vectors"], out["_full_dim"] = len(value), len(value[0])
                else:
                    out[key] = shrink(value, keep)
            elif key == "values" and isinstance(value, list) and len(value) > keep:
                out[key], out["_full_dim"] = value[:keep], len(value)
            elif key == "tokenized_input" and isinstance(value, list):
                out[key], out["_num_tokens"] = value[:keep], len(value)
            else:
                out[key] = shrink(value, keep)
        return out
    if isinstance(node, list):
        return [shrink(item, keep) for item in node]
    return node


# --- probe sets ------------------------------------------------------------


def probes_for(endpoint: str) -> list[tuple]:
    common = [
        ("health", "GET", "/health", None),
        ("wrong_verb_embed", "POST", "/v1/embeddings", {"input": [EN]}),
        (
            "wrong_verb_rerank",
            "POST",
            "/v1/rerank",
            {"query": QUERY, "documents": DOCS},
        ),
        (
            "wrong_verb_chat",
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": EN}]},
        ),
        ("cohere_v1_embed_deleted", "POST", "/v1/embed", {"texts": [EN]}),
    ]
    if endpoint == "/v1/embeddings":
        return common + [
            ("embed_default", "POST", "/v1/embeddings", {"input": [EN, ZH]}),
            ("embed_single", "POST", "/v1/embeddings", {"input": EN}),
            ("embed_code", "POST", "/v1/embeddings", {"input": [CODE]}),
            (
                "embed_unicode",
                "POST",
                "/v1/embeddings",
                {"input": ["Ünïcödé ☂ 漢字 emoji 🌊 mixed with ASCII"]},
            ),
            (
                "embed_dims128",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "dimensions": 128},
            ),
            (
                "embed_dims_over_max",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "dimensions": 999999},
            ),
            (
                "embed_binary",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "embedding_type": "binary"},
            ),
            (
                "embed_ubinary",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "embedding_type": "ubinary"},
            ),
            (
                "embed_base64",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "embedding_type": "base64"},
            ),
            (
                "embed_type_list_422",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "embedding_type": ["float", "binary"]},
            ),
            (
                "embed_normalized_false",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "normalized": False},
            ),
            (
                "embed_task_query",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "task": "retrieval.query"},
            ),
            (
                "embed_task_passage",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "task": "retrieval.passage"},
            ),
            (
                "embed_task_text_matching",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "task": "text-matching"},
            ),
            (
                "embed_late_chunking",
                "POST",
                "/v1/embeddings",
                {"input": [EN, ZH], "late_chunking": True},
            ),
            (
                "embed_multivector",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "return_multivector": True},
            ),
            ("embed_empty_list_422", "POST", "/v1/embeddings", {"input": []}),
            (
                "embed_pretokenized_422",
                "POST",
                "/v1/embeddings",
                {"input": [[1, 2, 3]]},
            ),
            ("embed_overlong_400", "POST", "/v1/embeddings", {"input": [EN * 400]}),
            (
                "embed_overlong_truncate",
                "POST",
                "/v1/embeddings",
                {"input": [EN * 400], "truncate": True},
            ),
            (
                "embed_foreign_jina_model_400",
                "POST",
                "/v1/embeddings",
                {"input": [EN], "model": "jina-embeddings-v4"},
            ),
            (
                "embed_openai_model_ok",
                "POST",
                "/v1/embeddings",
                {
                    "input": [EN],
                    "model": "text-embedding-3-small",
                    "encoding_format": "float",
                    "user": "u",
                },
            ),
            (
                "cohere_embed",
                "POST",
                "/v2/embed",
                {
                    "model": "embed-v4.0",
                    "input_type": "search_document",
                    "texts": [EN, ZH],
                },
            ),
            (
                "cohere_embed_types",
                "POST",
                "/v2/embed",
                {
                    "model": "embed-v4.0",
                    "input_type": "search_query",
                    "texts": [EN],
                    "embedding_types": ["float", "int8", "ubinary", "base64"],
                },
            ),
            (
                "gemini_flat",
                "POST",
                "/v1/models/m:embedContent",
                {"content": {"parts": [{"text": EN}]}, "taskType": "RETRIEVAL_QUERY"},
            ),
            (
                "gemini_nested",
                "POST",
                "/v1/models/m:embedContent",
                {
                    "content": {"parts": [{"text": EN}]},
                    "embedContentConfig": {
                        "taskType": "RETRIEVAL_DOCUMENT",
                        "outputDimensionality": 128,
                    },
                },
            ),
            (
                "gemini_batch",
                "POST",
                "/v1/models/m:batchEmbedContents",
                {
                    "requests": [
                        {"content": {"parts": [{"text": EN}]}},
                        {"content": {"parts": [{"text": ZH}]}},
                    ]
                },
            ),
            (
                "voyage_multimodal_text",
                "POST",
                "/v1/multimodalembeddings",
                {
                    "inputs": [{"content": [{"type": "text", "text": EN}]}],
                    "model": "voyage-multimodal-3",
                },
            ),
        ]
    if endpoint == "/v1/rerank":
        return common + [
            (
                "rerank_strings_top2",
                "POST",
                "/v1/rerank",
                {"query": QUERY, "documents": DOCS, "top_n": 2},
            ),
            (
                "rerank_return_false",
                "POST",
                "/v1/rerank",
                {
                    "query": QUERY,
                    "documents": DOCS,
                    "top_n": 2,
                    "return_documents": False,
                },
            ),
            (
                "rerank_no_top_n",
                "POST",
                "/v1/rerank",
                {"query": QUERY, "documents": DOCS},
            ),
            (
                "rerank_docs_objects",
                "POST",
                "/v1/rerank",
                {"query": QUERY, "documents": [{"text": d} for d in DOCS], "top_n": 2},
            ),
            (
                "rerank_empty_docs_422",
                "POST",
                "/v1/rerank",
                {"query": QUERY, "documents": []},
            ),
            (
                "rerank_top_n_zero_422",
                "POST",
                "/v1/rerank",
                {"query": QUERY, "documents": DOCS, "top_n": 0},
            ),
            (
                "rerank_top_k_alias",
                "POST",
                "/v1/rerank",
                {"query": QUERY, "documents": DOCS, "top_k": 2},
            ),
            (
                "rerank_max_doc_length",
                "POST",
                "/v1/rerank",
                {"query": QUERY, "documents": DOCS, "max_tokens_per_doc": 24},
            ),
            (
                "rerank_top_n_gt_docs",
                "POST",
                "/v1/rerank",
                {"query": QUERY, "documents": DOCS, "top_n": 999},
            ),
            (
                "rerank_long_docs",
                "POST",
                "/v1/rerank",
                {"query": QUERY, "documents": [d * 30 for d in DOCS], "top_n": 2},
            ),
            (
                "cohere_rerank",
                "POST",
                "/v2/rerank",
                {"model": "rerank-v3.5", "query": QUERY, "documents": DOCS, "top_n": 2},
            ),
            (
                "cohere_rerank_max_tokens",
                "POST",
                "/v2/rerank",
                {
                    "model": "rerank-v3.5",
                    "query": QUERY,
                    "documents": DOCS,
                    "max_tokens_per_doc": 24,
                },
            ),
        ]
    return common + [
        (
            "chat_basic",
            "POST",
            "/v1/chat/completions",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Summarise, in two sentences, why batching matters for LLM inference.",
                    }
                ],
                "max_tokens": 96,
            },
        ),
        (
            "chat_html_read",
            "POST",
            "/v1/chat/completions",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "<html><body><h1>Quarterly Report</h1><p>Revenue grew 12% "
                        "year over year, driven by inference volume.</p></body></html>",
                    }
                ],
                "max_tokens": 128,
            },
        ),
        (
            "chat_stream_400",
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": EN}], "stream": True},
        ),
        ("chat_empty_400", "POST", "/v1/chat/completions", {"messages": []}),
    ]


# --- source C: the model's own library call, inside the container ----------

ORACLE = r"""
import json, os, sys, numpy as np
model_id = os.environ["JINA_MODEL_ID"]
runtime  = os.environ["JINA_RUNTIME"]
texts    = json.loads(os.environ["JINA_TEXTS"])
query    = os.environ.get("JINA_QUERY", "")
out = {"runtime": runtime, "model_id": model_id}
try:
    import torch
    out["torch"] = torch.__version__
    import transformers
    out["transformers"] = transformers.__version__
except Exception as exc:
    out["import_error"] = repr(exc)

try:
    if runtime in ("sentence_transformer", "embeddings_v3", "embeddings_v4",
                   "embeddings_v5_text", "embeddings_v5_omni"):
        from sentence_transformers import SentenceTransformer
        kwargs = {}
        if runtime == "embeddings_v5_omni":
            kwargs["model_kwargs"] = {"default_task": "retrieval"}
        model = SentenceTransformer(model_id, trust_remote_code=True, **kwargs)
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        out["dim"] = int(np.asarray(vectors).shape[-1])
        out["vectors_head"] = np.asarray(vectors)[:, :8].tolist()
    elif runtime == "cross_encoder":
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(model_id, trust_remote_code=True)
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        scores = model.predict([[query, t] for t in texts])
        out["scores"] = [float(s) for s in np.asarray(scores).ravel()]
    elif runtime == "jina_ranking":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True,
                                          torch_dtype="auto", low_cpu_mem_usage=True).eval()
        ranked = model.rerank(query, texts)
        out["scores"] = [float(r["relevance_score"]) for r in
                         sorted(ranked, key=lambda r: r["index"])]
    elif runtime == "colbert":
        from pylate import models as pm, rank as pr
        model = pm.ColBERT(model_name_or_path=model_id, trust_remote_code=True)
        qe = model.encode([query], is_query=True, convert_to_tensor=True)
        de = model.encode(texts, is_query=False, convert_to_tensor=True)
        ranked = pr.rerank(documents_ids=[list(range(len(texts)))],
                           queries_embeddings=qe, documents_embeddings=[de])
        out["scores"] = [float(r["score"]) for r in
                         sorted(ranked[0], key=lambda r: r["id"])]
    else:
        out["skipped"] = f"no library oracle defined for runtime {runtime}"
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print("ORACLE_JSON:" + json.dumps(out))
"""


def container_oracle(container: str, spec: dict) -> dict:
    env = {
        "JINA_RUNTIME": spec["runtime"],
        "JINA_TEXTS": json.dumps(
            DOCS if spec["api_endpoint"] == "/v1/rerank" else [EN, ZH]
        ),
        "JINA_QUERY": QUERY,
    }
    command = ["docker", "exec"]
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    command += [container, "python", "-c", ORACLE]
    completed = run(command, timeout=1800)
    for line in (completed.stdout or "").splitlines():
        if line.startswith("ORACLE_JSON:"):
            return json.loads(line[len("ORACLE_JSON:") :])
    return {"error": "no oracle output", "stderr": (completed.stderr or "")[-800:]}


def container_provenance(container: str) -> dict:
    probe = (
        "import json,torch,transformers,sys;"
        "d={'torch':torch.__version__,'transformers':transformers.__version__,"
        "'python':sys.version.split()[0],'cuda_available':torch.cuda.is_available(),"
        "'cuda_device':(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)};"
        "\ntry:\n import sentence_transformers as st; d['sentence_transformers']=st.__version__\n"
        "except Exception: pass\n"
        "print('PROV_JSON:'+json.dumps(d))"
    )
    completed = run(["docker", "exec", container, "python", "-c", probe], timeout=600)
    for line in (completed.stdout or "").splitlines():
        if line.startswith("PROV_JSON:"):
            return json.loads(line[len("PROV_JSON:") :])
    return {"error": (completed.stderr or "")[-400:]}


AIRGAP_PROBE = (
    "import socket,json\n"
    "results={}\n"
    "for name,addr in (('huggingface.co',('huggingface.co',443)),"
    "('dns',('8.8.8.8',53)),('api.jina.ai',('api.jina.ai',443))):\n"
    "    try:\n"
    "        s=socket.create_connection(addr,timeout=4); s.close(); results[name]='CONNECTED'\n"
    "    except Exception as e: results[name]=type(e).__name__\n"
    "print('AIRGAP_JSON:'+json.dumps(results))"
)


def airgap_check(image: str, runtime: str, model_id: str) -> dict:
    """Start the image with no network at all, prove it still serves, and prove
    it cannot reach out. HF_HUB_OFFLINE states intent; this measures it."""
    name = f"airgap-{abs(hash(image)) % 10**8}"
    command = ["docker", "run", "-d", "--rm", "--name", name, "--network", "none"]
    if runtime == "gpu":
        command += ["--gpus", "all"]
    command += [image]
    started = run(command)
    if started.returncode != 0:
        return {"status": "start_failed", "error": started.stderr[-400:]}
    try:
        ready, waited = False, 0
        while waited < 900:
            probe = run(
                [
                    "docker",
                    "exec",
                    name,
                    "python",
                    "-c",
                    "import urllib.request,sys;"
                    "sys.exit(0 if urllib.request.urlopen("
                    "'http://127.0.0.1:8080/health',timeout=5).status==200 else 1)",
                ],
                timeout=60,
            )
            if probe.returncode == 0:
                ready = True
                break
            if (
                run(
                    ["docker", "inspect", "-f", "{{.State.Running}}", name]
                ).stdout.strip()
                != "true"
            ):
                break
            time.sleep(10)
            waited += 10
        result = {"served_with_network_none": ready, "wait_s": waited}
        if ready:
            infer = run(
                [
                    "docker",
                    "exec",
                    name,
                    "python",
                    "-c",
                    "import urllib.request,json;"
                    "print(urllib.request.urlopen('http://127.0.0.1:8080/health',"
                    "timeout=10).read().decode()[:300])",
                ],
                timeout=120,
            )
            result["health"] = infer.stdout.strip()[:300]
        socket_probe = run(
            ["docker", "exec", name, "python", "-c", AIRGAP_PROBE], timeout=120
        )
        for line in (socket_probe.stdout or "").splitlines():
            if line.startswith("AIRGAP_JSON:"):
                result["egress"] = json.loads(line[len("AIRGAP_JSON:") :])
        result["no_egress"] = all(
            v != "CONNECTED" for v in result.get("egress", {"x": "CONNECTED"}).values()
        )
        logs = run(["docker", "logs", "--tail", "40", name])
        result["log_tail"] = (logs.stdout + logs.stderr)[-1200:]
        return result
    finally:
        run(["docker", "kill", name], timeout=120)


def validate(spec: dict, runtime: str, args, outdir: Path) -> dict:
    name = spec["id"].lower()
    image = f"{args.registry}/{name}:{runtime}-{args.sha}"
    record = {
        "model": spec["id"],
        "runtime": runtime,
        "image": image,
        "api_endpoint": spec["api_endpoint"],
        "catalog": {
            k: spec.get(k)
            for k in (
                "runtime",
                "modality",
                "context",
                "output_dim",
                "vram_gb",
                "family",
            )
        },
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
    run(["docker", "rm", "-f", container], timeout=120)
    command = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container,
        "-p",
        f"{args.port}:8080",
        "-e",
        "JINA_LICENSE_MODE=off",
    ]
    if runtime == "gpu":
        command += ["--gpus", "all"]
    command += [image]
    started = run(command)
    if started.returncode != 0:
        record["status"] = "start_failed"
        record["error"] = started.stderr[-600:]
        return record

    base = f"http://127.0.0.1:{args.port}"
    try:
        cold_started = time.perf_counter()
        ready = False
        for _ in range(180):
            health = http(base + "/health", timeout=10)
            if health.get("status") == 200 and health["body"].get("ready"):
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
        record["cold_start_s"] = round(time.perf_counter() - cold_started, 1)
        record["ready"] = ready
        if not ready:
            logs = run(["docker", "logs", "--tail", "60", container])
            record["status"] = "never_ready"
            record["log_tail"] = (logs.stdout + logs.stderr)[-2500:]
            return record

        record["provenance"] = container_provenance(container)

        # Warm the model before anything is timed; cold rows are never averaged in.
        http(base + "/health")
        warm_body = (
            {"input": [EN]}
            if spec["api_endpoint"] == "/v1/embeddings"
            else (
                {"query": QUERY, "documents": DOCS[:2]}
                if spec["api_endpoint"] == "/v1/rerank"
                else {
                    "messages": [{"role": "user", "content": "Warm up."}],
                    "max_tokens": 8,
                }
            )
        )
        http(base + spec["api_endpoint"], warm_body)

        results = []
        for probe_id, method, path, body in probes_for(spec["api_endpoint"]):
            outcome = http(base + path) if method == "GET" else http(base + path, body)
            dump = outdir / "responses" / f"{name}-{runtime}-{probe_id}.json"
            dump.parent.mkdir(parents=True, exist_ok=True)
            dump.write_text(
                json.dumps(
                    {"_request": body, "_endpoint": path, **shrink(outcome)},
                    ensure_ascii=False,
                    indent=1,
                )
            )
            results.append(
                {
                    "id": probe_id,
                    "endpoint": path,
                    "status": outcome.get("status"),
                    "client_ms": outcome.get("client_ms"),
                    "x_jina_elapsed_ms": outcome.get("x_jina_elapsed_ms"),
                    "x_jina_tok_per_s": outcome.get("x_jina_tok_per_s"),
                    "keys": (
                        sorted(outcome.get("body", {}))
                        if isinstance(outcome.get("body"), dict)
                        else None
                    ),
                    "dump": dump.name,
                }
            )
        record["probes"] = results
        record["oracle"] = container_oracle(container, spec)
        record["status"] = "ok"
    finally:
        run(["docker", "kill", container], timeout=180)

    if not args.skip_airgap:
        record["airgap"] = airgap_check(image, runtime, spec["hf_repo"])
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models")
    parser.add_argument("--runtimes", default="cpu,gpu")
    parser.add_argument("--sha", default=os.environ.get("BUILD_SHA", ""))
    parser.add_argument("--registry", default="ghcr.io/jina-ai/jina-on-prem")
    parser.add_argument("--out", default="/var/tmp/jina-validate")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--skip-airgap", action="store_true")
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
    # GPU shares one device, and each container loads a whole model, so the
    # sweep is deliberately serial. Smallest first, so a harness bug shows up
    # in minutes.
    jobs = sorted(
        ((m, r) for m in models for r in args.runtimes.split(",")),
        key=lambda pair: (pair[0].get("vram_gb") or 0, pair[0]["id"], pair[1]),
    )

    sink = outdir / f"validation-{args.sha}.jsonl"
    print(f"{len(jobs)} image validations, sha={args.sha} -> {sink}", flush=True)
    for spec, runtime in jobs:
        started = time.perf_counter()
        try:
            record = validate(spec, runtime, args, outdir)
        except Exception as exc:  # a harness fault is data, not a stop
            record = {
                "model": spec["id"],
                "runtime": runtime,
                "status": "harness_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        record["wall_s"] = round(time.perf_counter() - started, 1)
        with open(sink, "a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        bad = sum(1 for p in record.get("probes", []) if p.get("status") is None)
        print(
            f"[{record.get('status'):14}] {spec['id']}:{runtime} "
            f"{record['wall_s']}s probes={len(record.get('probes', []))} unreachable={bad} "
            f"airgap_ok={record.get('airgap', {}).get('no_egress')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
