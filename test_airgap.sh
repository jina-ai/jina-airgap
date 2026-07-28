#!/bin/bash
# Air-gap test: prove an image serves with no network at all, and prove it
# cannot reach out.
#
# Runs every container with `--network none`, so egress is impossible rather
# than merely discouraged by HF_HUB_OFFLINE, and probes it over the container's
# own loopback via `docker exec`. A socket probe from inside then has to FAIL
# to reach huggingface.co, api.jina.ai and DNS.
#
# The endpoint comes from the container's own /health, not from the image name:
# jina-colbert-* serves /v1/rerank and the reader/vlm models serve
# /v1/chat/completions, neither of which matches a "reranker" substring.
#
# Inputs are real prose, long enough to exercise batching and tokenization. A
# request that merely returns HTTP 200 is not a pass: the response envelope is
# checked field by field, because an image that OOMs or returns a 500 for a
# realistic request can still answer a two-word one.
#
# Usage:
#   ./test_airgap.sh                    # every local jina/ image
#   ./test_airgap.sh jina/MODEL:cpu     # one image
#   GPU=1 ./test_airgap.sh IMAGE        # pass --gpus all

set -uo pipefail
LOGFILE="${LOGFILE:-test_results.log}"
GPU_ARGS=()
[ "${GPU:-0}" = "1" ] && GPU_ARGS=(--gpus all)
echo "=== Air-gap test $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" > "$LOGFILE"

PASSED=0
FAILED=0

# Sent to the container as a file so quoting survives docker exec.
read -r -d '' PROBE <<'PYEOF'
import json, sys, urllib.request, socket

BASE = "http://127.0.0.1:8080"
EN = ("It is a truth universally acknowledged, that a single man in possession "
      "of a good fortune, must be in want of a wife. However little known the "
      "feelings or views of such a man may be on his first entering a "
      "neighbourhood, this truth is so well fixed in the minds of the "
      "surrounding families, that he is considered the rightful property of "
      "some one or other of their daughters.")
ZH = "机器学习模型的生产部署需要考虑多个维度：推理延迟、吞吐量、硬件成本以及模型更新的持续集成与部署流程。"
QUERY = "How do teams deploy large language models efficiently in production?"
DOCS = [EN, ZH, "The Treaty of Westphalia in 1648 ended the Thirty Years' War."] * 5 + [EN]

def post(path, body):
    request = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)

failures = []

health = json.load(urllib.request.urlopen(BASE + "/health", timeout=30))
endpoint = health.get("endpoint")
if not health.get("ready"):
    failures.append("health.ready is false")

if endpoint == "/v1/embeddings":
    body = post(endpoint, {"input": [EN, ZH]})
    for key in ("model", "object", "usage", "data"):
        if key not in body:
            failures.append(f"embeddings response missing {key}")
    if body.get("object") != "list":
        failures.append(f"object={body.get('object')!r}, expected 'list'")
    data = body.get("data", [])
    if len(data) != 2:
        failures.append(f"{len(data)} embeddings for 2 inputs")
    for item in data:
        vector = item.get("embedding") or item.get("embeddings")
        if not vector:
            failures.append("data item has no embedding")
        elif isinstance(vector[0], float) and all(v == 0.0 for v in vector):
            failures.append("embedding is all zeros")
    if not body.get("usage", {}).get("total_tokens"):
        failures.append("usage.total_tokens missing or zero")
elif endpoint == "/v1/rerank":
    body = post(endpoint, {"query": QUERY, "documents": DOCS, "top_n": 5})
    for key in ("model", "object", "usage", "results"):
        if key not in body:
            failures.append(f"rerank response missing {key}")
    results = body.get("results", [])
    if len(results) != 5:
        failures.append(f"{len(results)} results for top_n=5 over {len(DOCS)} docs")
    scores = [r.get("relevance_score") for r in results]
    if scores != sorted(scores, reverse=True):
        failures.append(f"results not sorted descending: {scores}")
    if len(set(scores)) == 1:
        failures.append(f"every document scored identically ({scores[0]})")
    if any("document" not in r for r in results):
        failures.append("return_documents defaults true but document key is absent")
    omitted = post(endpoint, {"query": QUERY, "documents": DOCS,
                              "top_n": 2, "return_documents": False})
    if any("document" in r for r in omitted.get("results", [])):
        failures.append("return_documents=false still returned a document key")
elif endpoint == "/v1/chat/completions":
    body = post(endpoint, {"messages": [{"role": "user", "content":
        "Summarise in one sentence why batching matters for LLM inference."}],
        "max_tokens": 64})
    text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
    if body.get("object") != "chat.completion":
        failures.append(f"object={body.get('object')!r}")
    if len(text.strip()) < 10:
        failures.append(f"generated only {len(text.strip())} characters")
    if not body.get("usage", {}).get("completion_tokens"):
        failures.append("usage.completion_tokens missing or zero")
else:
    failures.append(f"unknown endpoint from /health: {endpoint!r}")

# Egress must be impossible, not merely disabled by env var.
egress = {}
for name, address in (("huggingface.co", ("huggingface.co", 443)),
                      ("api.jina.ai", ("api.jina.ai", 443)),
                      ("dns", ("8.8.8.8", 53))):
    try:
        connection = socket.create_connection(address, timeout=4)
        connection.close()
        egress[name] = "CONNECTED"
        failures.append(f"reached {name} from inside the container")
    except Exception as exc:
        egress[name] = type(exc).__name__

print(json.dumps({"endpoint": endpoint, "model": health.get("model"),
                  "device": health.get("device"), "egress": egress,
                  "failures": failures}))
sys.exit(1 if failures else 0)
PYEOF

test_image() {
    local IMAGE="$1"
    local NAME="airgap-$(echo "$IMAGE" | tr '/:.' '---')"
    echo "--- $IMAGE ---" | tee -a "$LOGFILE"

    docker rm -f "$NAME" >/dev/null 2>&1
    if ! docker run --rm -d --name "$NAME" --network none \
            -e JINA_LICENSE_MODE=off "${GPU_ARGS[@]}" "$IMAGE" >/dev/null; then
        echo "  FAIL: container would not start" | tee -a "$LOGFILE"
        FAILED=$((FAILED + 1))
        return
    fi

    local READY=0
    for i in $(seq 1 120); do
        sleep 5
        if docker exec "$NAME" python -c \
            "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=5)" \
            >/dev/null 2>&1; then
            READY=1
            echo "  ready after $((i * 5))s (no network)" | tee -a "$LOGFILE"
            break
        fi
        if ! docker ps -q -f "name=$NAME" | grep -q .; then
            echo "  FAIL: container exited during load" | tee -a "$LOGFILE"
            docker logs "$NAME" 2>&1 | tail -25 >> "$LOGFILE"
            FAILED=$((FAILED + 1))
            return
        fi
    done
    if [ "$READY" -eq 0 ]; then
        echo "  FAIL: never became ready within 600s" | tee -a "$LOGFILE"
        docker logs "$NAME" 2>&1 | tail -25 >> "$LOGFILE"
        docker kill "$NAME" >/dev/null 2>&1
        FAILED=$((FAILED + 1))
        return
    fi

    docker exec -i "$NAME" sh -c 'cat > /tmp/airgap_probe.py' <<< "$PROBE"
    local OUTPUT
    OUTPUT=$(docker exec "$NAME" python /tmp/airgap_probe.py 2>&1)
    local RC=$?
    echo "  $OUTPUT" | tee -a "$LOGFILE"
    if [ "$RC" -eq 0 ]; then
        echo "  PASS" | tee -a "$LOGFILE"
        PASSED=$((PASSED + 1))
    else
        echo "  FAIL" | tee -a "$LOGFILE"
        docker logs "$NAME" 2>&1 | tail -25 >> "$LOGFILE"
        FAILED=$((FAILED + 1))
    fi
    docker kill "$NAME" >/dev/null 2>&1
    echo "" | tee -a "$LOGFILE"
}

if [ $# -gt 0 ]; then
    test_image "$1"
else
    for image in $(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^jina/'); do
        test_image "$image"
    done
fi

echo "=== $PASSED passed, $FAILED failed ===" | tee -a "$LOGFILE"
[ "$FAILED" -eq 0 ]
