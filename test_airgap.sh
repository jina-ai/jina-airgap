#!/bin/bash
# Air-gap test: host port-maps to container, HF_HUB_OFFLINE=1 prevents downloads.
# Usage:
#   ./test_airgap.sh                           # test all jina/ images
#   ./test_airgap.sh jina/MODEL:cpu            # test one image

LOGFILE="${LOGFILE:-test_results.log}"
echo "=== Air-gap test $(date) ===" > "$LOGFILE"

test_image() {
    local IMAGE="$1"
    local NAME="airgap-test-$(echo $IMAGE | tr '/:' '-')"
    local PORT=$((8080 + RANDOM % 1000))

    echo "--- Testing: $IMAGE (port $PORT) ---" | tee -a "$LOGFILE"

    docker rm -f "$NAME" 2>/dev/null || true
    docker run --rm -d --name "$NAME" -p "$PORT:8080" "$IMAGE"

    echo "  Waiting for model to load (up to 180s)..." | tee -a "$LOGFILE"
    local READY=0
    for i in $(seq 1 36); do
        sleep 5
        if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
            READY=1
            echo "  Health OK after $((i*5))s" | tee -a "$LOGFILE"
            break
        fi
        if ! docker ps -q -f "name=$NAME" | grep -q .; then
            echo "  CONTAINER CRASHED" | tee -a "$LOGFILE"
            docker logs "$NAME" 2>&1 | tail -20 >> "$LOGFILE"
            echo "RESULT: $IMAGE HEALTH=crashed EMBED=skip" | tee -a "$LOGFILE"
            return
        fi
    done

    if [ "$READY" -eq 0 ]; then
        echo "  Health check timeout" | tee -a "$LOGFILE"
        docker logs "$NAME" 2>&1 | tail -20 >> "$LOGFILE"
        docker stop "$NAME" 2>/dev/null
        echo "RESULT: $IMAGE HEALTH=timeout EMBED=skip" | tee -a "$LOGFILE"
        return
    fi

    # Startup logs must not mention downloading. transformers logs "A new version of
    # the following files was downloaded from <hub url>" whenever it stages a
    # trust_remote_code .py at runtime. Nothing is actually fetched (HF_HUB_OFFLINE=1
    # makes that impossible), but it reads as an air-gap violation to customers, and
    # it means the build failed to pre-stage the model code.
    local LEAK="ok"
    if docker logs "$NAME" 2>&1 | grep -q "was downloaded from"; then
        LEAK="fail"
        echo "  DOWNLOAD WARNING IN LOGS (model code not pre-staged at build time)" | tee -a "$LOGFILE"
        docker logs "$NAME" 2>&1 | grep -A 3 "was downloaded from" >> "$LOGFILE"
    fi

    # Reranker: use /v1/rerank instead of /v1/embeddings
    if echo "$IMAGE" | grep -q 'reranker'; then
        local RERANK_RESULT
        RERANK_RESULT=$(curl -sf -X POST "http://localhost:$PORT/v1/rerank" \
            -H "Content-Type: application/json" \
            -d '{"query": "hello", "documents": ["world", "foo"], "model": "test"}' 2>&1)
        if echo "$RERANK_RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); assert len(d['results']) > 0" 2>/dev/null; then
            echo "  Rerank OK" | tee -a "$LOGFILE"
            echo "RESULT: $IMAGE HEALTH=ok RERANK=ok LOGS=$LEAK" | tee -a "$LOGFILE"
        else
            echo "  Rerank FAILED" | tee -a "$LOGFILE"
            echo "  Response: $RERANK_RESULT" >> "$LOGFILE"
            docker logs "$NAME" 2>&1 | tail -20 >> "$LOGFILE"
            echo "RESULT: $IMAGE HEALTH=ok RERANK=fail LOGS=$LEAK" | tee -a "$LOGFILE"
        fi
    else
        local EMBED_RESULT
        EMBED_RESULT=$(curl -sf -X POST "http://localhost:$PORT/v1/embeddings" \
            -H "Content-Type: application/json" \
            -d '{"input": ["hello world"], "model": "test"}' 2>&1)
        if echo "$EMBED_RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); assert len(d['data'][0]['embedding']) > 0" 2>/dev/null; then
            local DIM=$(echo "$EMBED_RESULT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d['data'][0]['embedding']))")
            echo "  Embed OK, dim=$DIM" | tee -a "$LOGFILE"
            echo "RESULT: $IMAGE HEALTH=ok EMBED=ok DIM=$DIM LOGS=$LEAK" | tee -a "$LOGFILE"
        else
            echo "  Embed FAILED" | tee -a "$LOGFILE"
            echo "  Response: $EMBED_RESULT" >> "$LOGFILE"
            docker logs "$NAME" 2>&1 | tail -20 >> "$LOGFILE"
            echo "RESULT: $IMAGE HEALTH=ok EMBED=fail LOGS=$LEAK" | tee -a "$LOGFILE"
        fi
    fi

    docker stop "$NAME" 2>/dev/null || true
    echo "" | tee -a "$LOGFILE"
}

if [ -n "$1" ]; then
    test_image "$1"
else
    for img in $(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^jina/'); do
        test_image "$img"
    done
fi

echo "=== Done ===" | tee -a "$LOGFILE"
