#!/usr/bin/env bash
# CPU-only reporting from saved evidence. Safe alongside the head study.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../extensions_env.sh"
cd "$VLA_PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg
exec "$VLA_PROJECT_ROOT/.venv/bin/python" "$VLA_PROJECT_ROOT/scripts/generate_head_report.py" "$@"
