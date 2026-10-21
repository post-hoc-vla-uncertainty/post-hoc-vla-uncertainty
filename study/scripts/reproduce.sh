#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$VLA_PROJECT_ROOT"
exec "$VLA_PROJECT_ROOT/.venv/bin/python" "$VLA_PROJECT_ROOT/scripts/reproduce.py" "$@"
