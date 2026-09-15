#!/usr/bin/env bash
# Creates a local virtualenv (.venv) and installs project dependencies.
# Usage: ./setup.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
