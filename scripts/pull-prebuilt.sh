#!/usr/bin/env bash
# Pull a prebuilt jina-on-prem image from GHCR and save as .tar.gz for offline transport.
# Skip the bundle phase entirely when a prebuilt exists.
#
# Usage:
#   ./scripts/pull-prebuilt.sh MODEL [RUNTIME]
#
# Examples:
#   ./scripts/pull-prebuilt.sh jina-embeddings-v5-text-nano        # default: cpu
#   ./scripts/pull-prebuilt.sh jina-embeddings-v5-text-small gpu
#   ./scripts/pull-prebuilt.sh jina-embeddings-v3 gpu-opt          # embedding models
#
# Output: MODEL-RUNTIME.tar.gz in the current directory.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL [cpu|gpu|gpu-opt]" >&2
  echo
  echo "List available prebuilt models in the README's 'Prebuilt' column." >&2
  exit 1
fi

MODEL="$1"        # catalog id, case preserved: find_model() matches it exactly
RUNTIME="${2:-cpu}"

case "$RUNTIME" in
  cpu|gpu|gpu-opt) ;;
  *) echo "error: RUNTIME must be 'cpu', 'gpu' or 'gpu-opt' (got: $RUNTIME)" >&2; exit 1 ;;
esac

# Image names are lowercase-only and one catalog id is not (ReaderLM-v2), so it gets
# `invalid repository name` from the registry rather than a pull. Lowercase into a
# second variable: the messages below hand the reader a catalog id back, which has to
# keep its case. `tr` and not `${MODEL,,}`, which needs bash 4 while macOS ships 3.2.
IMAGE=$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')

REGISTRY="ghcr.io/jina-ai/jina-on-prem"
SRC="${REGISTRY}/${IMAGE}:${RUNTIME}"
LOCAL_TAG="jina/${IMAGE}:${RUNTIME}"
OUTPUT="${IMAGE}-${RUNTIME}.tar.gz"

# Prebuilt images are published for linux/amd64 only, so an arm64 daemon (Apple Silicon,
# arm servers) refuses the pull with "no matching manifest" until the platform is named.
# Naming it gets the amd64 image and runs it under emulation, which works and is slow.
# Ask the daemon, not `uname -m`: the shell's architecture is not docker's when the
# context points at a VM or a remote host. Unquoted below on purpose - the empty case
# has to expand to no argument at all, and the flag itself never contains a space.
PLATFORM=""
DAEMON_ARCH="$(docker version --format '{{.Server.Arch}}' 2>/dev/null || true)"
if [[ "$DAEMON_ARCH" == "arm64" || "$DAEMON_ARCH" == "aarch64" ]]; then
  PLATFORM="--platform linux/amd64"
  echo "Docker daemon is $DAEMON_ARCH; prebuilts are linux/amd64, pulling that under emulation."
fi

echo "Pulling $SRC ..."
if ! docker pull $PLATFORM "$SRC" 2>&1 | tee /tmp/pull-prebuilt.log; then
  if grep -q "unauthorized\|denied" /tmp/pull-prebuilt.log; then
    cat >&2 <<EOF

Pull failed with unauthorized/denied. Published images pull anonymously, so this
means no image exists at $SRC rather than that you need to log in.

Two things to check:
  - the Prebuilt column of the Model Catalog, for whether this model has one
  - the runtime: gpu-opt is published for the embedding models only

Bundle it yourself if there is no prebuilt:
  python jina-on-prem.py bundle --model $MODEL
EOF
  elif grep -q "no matching manifest" /tmp/pull-prebuilt.log; then
    cat >&2 <<EOF

Pull failed because this host's architecture has no published image. Prebuilts are
linux/amd64 only; retry naming the platform, which runs it under emulation:

  docker pull --platform linux/amd64 $SRC

EOF
  fi
  rm -f /tmp/pull-prebuilt.log
  exit 1
fi
rm -f /tmp/pull-prebuilt.log

echo "Retagging as $LOCAL_TAG ..."
docker tag "$SRC" "$LOCAL_TAG"

echo "Saving to $OUTPUT ..."
docker save "$LOCAL_TAG" | gzip > "$OUTPUT"

SIZE=$(du -h "$OUTPUT" | cut -f1)
echo
echo "Done. $OUTPUT ($SIZE)"
echo
echo "Transfer this file to the air-gapped machine, then:"
echo "  docker load < $OUTPUT"
echo "  docker run -p 8080:8080 $LOCAL_TAG"
# `[[ ... ]] && echo` as the last line of a `set -e` script exits 1 whenever the test is
# false, which reported a clean cpu pull as a failure. These are hints, so they end in an
# `if` and the script exits on the status of the work it actually did.
if [[ "$RUNTIME" == gpu* ]]; then
  echo "  # for GPU runtime: docker run --gpus all -p 8080:8080 $LOCAL_TAG"
fi
# The lines above are for the amd64 target host. Running the same image here needs the
# flag again, because the daemon still has to be told to emulate.
if [[ -n "$PLATFORM" ]]; then
  echo "  # to run it on this $DAEMON_ARCH host: docker run --platform linux/amd64 -p 8080:8080 $LOCAL_TAG"
fi
