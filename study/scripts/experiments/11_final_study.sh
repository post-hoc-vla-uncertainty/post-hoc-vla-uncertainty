#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../extensions_env.sh"
cd "$VLA_PROJECT_ROOT"
export PYTHONUNBUFFERED=1
exec "$VLA_PROJECT_ROOT/.venv/bin/python" -u "$VLA_PROJECT_ROOT/scripts/final_study.py" "$@"
