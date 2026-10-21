#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../extensions_env.sh"
cd "$VLA_PROJECT_ROOT"
"$VLA_PROJECT_ROOT/.venv/bin/python" scripts/setup_extensions.py
exec "$VLA_PROJECT_ROOT/.venv/bin/python" -m uq_extensions.experiments 06_condition_development "$@"
