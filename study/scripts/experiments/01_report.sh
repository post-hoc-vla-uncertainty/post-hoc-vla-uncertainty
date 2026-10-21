#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../extensions_env.sh"
cd "$VLA_PROJECT_ROOT"
exec "$VLA_PROJECT_ROOT/.venv/bin/python" scripts/refresh_extension_reports.py "$@"
