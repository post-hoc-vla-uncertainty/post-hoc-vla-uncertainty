#!/usr/bin/env bash
# Source this file from any directory; all caches and outputs stay in this project.
VLA_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VLA_PROJECT_ROOT
export STORAGE_ROOT="${STORAGE_ROOT:-$VLA_PROJECT_ROOT/storage}"
export HF_HOME="${HF_HOME:-$STORAGE_ROOT/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$STORAGE_ROOT/datasets}"
export LIBERO_CONFIG_PATH="$STORAGE_ROOT/libero_config"
export PYTHONPATH="$VLA_PROJECT_ROOT/upstream/uq_vla/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$VLA_PROJECT_ROOT/.local/ffmpeg-7.1.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$VLA_PROJECT_ROOT/.venv/bin:$PATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export MPLBACKEND=Agg
export MPLCONFIGDIR="$VLA_PROJECT_ROOT/.cache/matplotlib"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$STORAGE_ROOT" "$LIBERO_CONFIG_PATH"
