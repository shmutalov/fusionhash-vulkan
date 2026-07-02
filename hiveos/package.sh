#!/usr/bin/env bash
# Assemble the HiveOS custom-miner tarball: vulkminer-<version>.tar.gz
# Requires ./vulkminer (build it first with hiveos/build-linux.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

VER=$(grep -m1 CUSTOM_VERSION hiveos/h-manifest.conf | cut -d= -f2)
NAME=vulkminer
# Archive name carries the "hiveos" tag so it is distinguishable from the plain
# platform binaries on the release page. The unpacked top-level dir stays $NAME
# (== CUSTOM_NAME) because HiveOS expects <CUSTOM_NAME>/h-manifest.conf.
OUT="${NAME}-hiveos-${VER}.tar.gz"

if [[ ! -f ./vulkminer ]]; then
    echo "error: ./vulkminer not found — run: bash hiveos/build-linux.sh" >&2
    exit 1
fi

stage=$(mktemp -d)
mkdir -p "$stage/$NAME"
cp hiveos/h-manifest.conf hiveos/h-config.sh hiveos/h-run.sh hiveos/h-stats.sh "$stage/$NAME/"
cp ./vulkminer "$stage/$NAME/"
chmod +x "$stage/$NAME"/h-*.sh "$stage/$NAME/vulkminer"

tar -czf "$OUT" -C "$stage" "$NAME"
rm -rf "$stage"
echo "packaged $OUT"
