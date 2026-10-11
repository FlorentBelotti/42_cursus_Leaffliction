#!/usr/bin/env bash
# Writes signature.txt: the sha1 of the bundle produced by train.py
# (model + modified images). Run it once after the final training and
# do not regenerate the .zip afterwards, its hash would change.
# Usage: ./make_signature.sh [learnings.zip]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

BUNDLE="${1:-learnings.zip}"

if [ ! -f "$BUNDLE" ]; then
    echo "Error: '$BUNDLE' not found, run ./train.py first" >&2
    exit 1
fi

shasum "$BUNDLE" | cut -d' ' -f1 > signature.txt
echo "signature.txt written: $(cat signature.txt)  $BUNDLE"
