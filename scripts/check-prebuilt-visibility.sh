#!/usr/bin/env bash
# Report visibility of every jina-on-prem/* package on GHCR.
#
# GHCR packages default to private even when the source repo is public, and
# GitHub has no API for changing visibility - it's a per-package UI step
# in package settings -> Danger Zone -> Change visibility.
#
# This script checks the current state and prints the UI URL for any
# private package, so the maintainer can flip them in one batch.
#
# Usage:
#   ./scripts/check-prebuilt-visibility.sh
#
# Requires: gh CLI authenticated (`gh auth status` should succeed).

set -euo pipefail

ORG="${ORG:-jina-ai}"

# Models that have prebuilt images (keep in sync with scripts/gen_catalog_md.py PREBUILT set).
MODELS=(
  jina-embeddings-v5-omni-small
  jina-embeddings-v5-omni-nano
  jina-embeddings-v5-text-small
  jina-embeddings-v5-text-nano
  jina-reranker-v3
  jina-reranker-v3.5
  jina-embeddings-v3
  jina-clip-v2
)

private_count=0
public_count=0
private_urls=()

printf "%-40s %-10s\n" "PACKAGE" "VISIBILITY"
printf "%-40s %-10s\n" "----------------------------------------" "----------"

for m in "${MODELS[@]}"; do
  enc="jina-on-prem%2F$m"
  vis=$(gh api "/orgs/$ORG/packages/container/$enc" --jq '.visibility' 2>/dev/null || echo "missing")
  printf "%-40s %-10s\n" "$m" "$vis"
  case "$vis" in
    public)  public_count=$((public_count + 1)) ;;
    private) private_count=$((private_count + 1)); private_urls+=("https://github.com/orgs/$ORG/packages/container/$enc/settings") ;;
  esac
done

echo
echo "Summary: $public_count public, $private_count private"

if (( private_count > 0 )); then
  echo
  echo "To make private packages public, open each URL and:"
  echo "  Danger Zone -> Change package visibility -> Public -> type name -> confirm"
  echo
  for u in "${private_urls[@]}"; do
    echo "  $u"
  done
  exit 1
fi
