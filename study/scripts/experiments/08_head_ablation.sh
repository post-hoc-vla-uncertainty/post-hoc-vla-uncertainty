#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../extensions_env.sh"
cd "$VLA_PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
"$VLA_PROJECT_ROOT/.venv/bin/python" scripts/setup_extensions.py
exec "$VLA_PROJECT_ROOT/.venv/bin/python" -u -m uq_extensions.head_study "$@"
