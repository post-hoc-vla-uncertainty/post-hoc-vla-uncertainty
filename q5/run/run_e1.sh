#!/bin/bash
# E1: features then harm rollouts, shard $1 of 8 on GPU $2 (nvidia-smi numbering).
cd "$(dirname "$0")/../.."  # repository root (scripts live in logs/q5/)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$2 MUJOCO_GL=egl PYTHONPATH=functional_scripts
.venv/bin/python functional_scripts/q5_e1.py features --shard $1/8 && echo "FEATURES_DONE $1"
.venv/bin/python functional_scripts/q5_e1.py harm --shard $1/8 && echo "HARM_DONE $1"
