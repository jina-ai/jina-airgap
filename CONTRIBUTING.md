# Contributing to jina-on-prem

## Build + Test Workflow

### Prerequisites

- GCP instance with Docker (e.g. `e2-standard-8`, 100GB boot disk)
- SSH access: `ssh -i ~/.ssh/google_compute_engine hanxiao@<IP>`
- GitHub token with `write:packages` scope for GHCR push

### Build a CPU image

```bash
cd ~/jina-on-prem
git pull
python3 jina-on-prem.py bundle --model jina-embeddings-v5-text-nano --cpu-only --yes
```

### Build a GPU image

```bash
python3 jina-on-prem.py bundle --model jina-embeddings-v5-text-nano --yes
# Dockerfile.gpu uses pytorch/pytorch base with CUDA - no GPU needed at build time
```

### Test an image

```bash
bash test_airgap.sh jina/jina-embeddings-v5-text-nano:cpu
```

The test script:
1. Starts the container with port mapping (`-p PORT:8080`)
2. Waits for `/health` (up to 180s)
3. For embedding models: POST to `/v1/embeddings` and checks output dimension
4. For reranker models: POST to `/v1/rerank` and checks results array
5. Reports RESULT line with HEALTH/EMBED status

### Push to GHCR

```bash
echo $GH_TOKEN | docker login ghcr.io -u jina-ai --password-stdin
docker tag jina/MODEL:cpu ghcr.io/jina-ai/jina-on-prem/MODEL:cpu
docker push ghcr.io/jina-ai/jina-on-prem/MODEL:cpu
```

### Making the pushed package PUBLIC

**Critical**: GHCR packages default to **private**, even when the source repo is public. There is **no REST or GraphQL API** to change package visibility - it must be done once via the web UI per package.

After pushing a new model to GHCR:

1. Open `https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2FMODEL/settings`
2. Scroll to **Danger Zone** → **Change package visibility** → **Public**
3. Type the package name to confirm

To verify the current state of all prebuilt packages:

```bash
./scripts/check-prebuilt-visibility.sh
```

To avoid repeating this for every new model, set the org-level default to public at https://github.com/organizations/jina-ai/settings/packages.

### Batch build all 6 priority models

```bash
for m in jina-embeddings-v5-text-nano jina-embeddings-v5-text-small \
         jina-embeddings-v5-omni-nano jina-embeddings-v5-omni-small \
         jina-embeddings-v3 jina-reranker-v3; do
    python3 jina-on-prem.py bundle --model $m --cpu-only --yes
    docker builder prune -af  # reclaim build cache between models
done
# Then same loop without --cpu-only for GPU variants
```

### Disk space management

Images are large (2-12GB each). On a 100GB disk, you can hold ~6-8 images.

- `docker builder prune -af` between builds (reclaims build cache)
- `docker system prune -f` to remove dangling images
- Delete `.tar.gz` files after pushing to GHCR (they're 1-4GB each)
- Monitor with `df -h /`

## Known Caveats (from hard-won debugging)

### Model dependency versions

Each model has pinned deps in `models/catalog.json`. These pins exist for specific reasons:

| Model family | transformers version | Why |
|---|---|---|
| v5-text-nano, v5-text-small | `==4.51.0` | Needs `Qwen3Config` (added in 4.51) |
| v5-omni-nano, v5-omni-small | `==4.57.0` | Needs `Qwen3VLVisionConfig` (added in 4.57) |
| v3 | `==4.48.3` | Works with older version, no qwen3 dependency |
| reranker-v3 | `==4.51.0` | Based on Qwen3 architecture |
| reranker-v3.5 | `==4.57.3` | Qwen3 with hybrid sliding-window attention; needs `layer_types` (added in 4.55) |

`reranker-v3.5` is also capped from above: 4.57.2 and 4.57.3 each break offline tokenizer loading in a different way (see [Troubleshooting](docs/Troubleshooting.md)), and the server carries a bypass for the 4.57.3 case.

**DO NOT loosen these pins.** HuggingFace model repos ship their own `requirements.txt` that want `transformers>=5.x`. The `download_model.py` script deletes these files from the cache after download to prevent runtime auto-upgrade.

### Reranker models

- Must be loaded as `CrossEncoder`, not `SentenceTransformer`
- Qwen3-based rerankers need `pad_token = eos_token` set after loading (otherwise batch inference crashes with "Cannot handle batch sizes > 1 if no padding token is defined")
- Test with `/v1/rerank` endpoint, not `/v1/embeddings`

### Task mapping

Different model generations use different task names:

- **v5**: `retrieval`, `text-matching`, `classification`, `clustering`
- **v4**: `retrieval`, `text-matching`, `code`
- **v3**: `retrieval.query`, `retrieval.passage`, `text-matching`, `classification`, `separation`

The server's `_resolve_task()` function handles mapping between API-level names (e.g. Cohere's `search_query` / `search_document`) and model-level names.

### Air-gap testing

- **DO NOT use `--network=none`** for testing. It removes the entire network stack so the host can't reach the container either
- The correct approach: run with normal port mapping (`-p 8080:8080`), rely on `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` (baked into the image) to prevent any download attempts
- The real air-gap guarantee is that all weights and deps are baked into the image at build time

### OCI labels for GHCR

Dockerfiles include `LABEL org.opencontainers.image.source=https://github.com/jina-ai/jina-on-prem` so pushed images auto-link to this repo. Without this label, packages appear orphaned in GHCR and won't show in the repo sidebar.

### `trust_remote_code` monkey-patch

The server patches `transformers.dynamic_module_utils.resolve_trust_remote_code` to always return `True`. The patch uses `*args, **kwargs` because the function signature differs between transformers versions (4 args in 4.51, 5+ args in 5.x).

### Spot instances

GCP spot instances get preempted. Use `nohup` for long builds. Docker images persist across reboots but `/tmp` does not. Build progress logs should go to home directory.

## README maintenance

### Available Models table

When adding a new prebuilt image:
1. Add `cpu` / `gpu` links in the Prebuilt column (second column, after Model)
2. Link format: `[cpu](https://github.com/orgs/jina-ai/packages/container/jina-on-prem%2FMODEL_NAME) / [gpu](...)`
3. Models without prebuilt images show `-`

### Sections (in order)

1. Quick Start (bundle + deploy + Elasticsearch)
2. Available Models (table with Prebuilt links)
3. API (4 schemas: OpenAI, Cohere, Gemini, Voyage multimodal)
4. Embedding Tasks (v3/v4/v5 task lists)
5. Architecture
6. Repo Structure

Keep API examples to one curl per schema. No verbose request/response schemas.
