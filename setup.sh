#!/usr/bin/env bash
# Creates the virtualenv in /goinfre (home is too small for tensorflow),
# links it as .venv and installs project dependencies.
# Usage: ./setup.sh  (override location with VENV_DIR=/path ./setup.sh)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR="${VENV_DIR:-/goinfre/$USER/leaffliction_venv}"

# python 3.12 : tensorflow n'a pas de version stable pour 3.14
if [ ! -d "$VENV_DIR" ]; then
    uv python install 3.12
    uv venv -p 3.12 "$VENV_DIR"
fi

ln -sfn "$VENV_DIR" .venv

VIRTUAL_ENV="$VENV_DIR" uv pip install -r requirements.txt

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
