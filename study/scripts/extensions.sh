#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/extensions_env.sh"
cd "$VLA_PROJECT_ROOT"
"$VLA_PROJECT_ROOT/.venv/bin/python" "$VLA_PROJECT_ROOT/scripts/setup_extensions.py"
exec "$VLA_PROJECT_ROOT/.venv/bin/python" -m uq_extensions.cli "$@"
