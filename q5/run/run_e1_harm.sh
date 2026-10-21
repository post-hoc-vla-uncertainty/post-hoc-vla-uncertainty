#!/bin/bash
# E1 harm rollouts only, shard $1 of 8 on GPU $2 (nvidia-smi numbering); few CPU threads per process.
cd "$(dirname "$0")/../.."  # repository root (scripts live in logs/q5/)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$2 MUJOCO_GL=egl PYTHONPATH=functional_scripts
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
.venv/bin/python functional_scripts/q5_e1.py harm --shard $1/8 && echo "HARM_DONE $1"
