#!/usr/bin/env bash
# Fetch the tau-bench task data at the pinned tag into ./data (gitignored).
set -euo pipefail

TAG=v1.0.1
DEST="${1:-data}"
TMP="$(mktemp -d)"

git clone --quiet --filter=blob:none --no-checkout --depth 1 --branch "$TAG" \
    https://github.com/sierra-research/tau2-bench "$TMP"
git -C "$TMP" sparse-checkout set --no-cone '/data/tau2/domains/' '/data/tau2/user_simulator/'
git -C "$TMP" checkout --quiet "$TAG"

mkdir -p "$DEST"
cp -r "$TMP/data/tau2" "$DEST/"
rm -rf "$TMP"
echo "tau2 data ($TAG) -> $DEST/tau2"
