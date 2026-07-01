#!/usr/bin/env bash
# Build the Linux x86_64 vulkminer binary via Docker and copy it to ./vulkminer.
# Run from the repo root:  bash hiveos/build-linux.sh
set -euo pipefail
cd "$(dirname "$0")/.."

docker build -f hiveos/Dockerfile -t vulkminer-linux-build .
cid=$(docker create vulkminer-linux-build)
docker cp "$cid:/src/target/release/vulkminer" ./vulkminer
docker rm "$cid" >/dev/null
chmod +x ./vulkminer
echo "built ./vulkminer (linux x86_64):"
file ./vulkminer 2>/dev/null || ls -la ./vulkminer
