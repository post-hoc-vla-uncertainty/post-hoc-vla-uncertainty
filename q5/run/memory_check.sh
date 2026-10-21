#!/usr/bin/env zsh
# Capped measurements before the big run is restarted (NOT run automatically). Each step runs in its own
# memory cgroup with a hard limit and no swap, so it cannot affect the server; peak RAM is read from the cgroup.
#   1  loader: the episode-filtered dataset via the new memory-mapped path equals the old in-RAM path (2 episodes)
#   2  one support extraction (dense, seed 0)        limit 40 GB
#   3  one clean evaluation, one task, 5 envs        limit 25 GB
cd "$(dirname "$0")/../.."  # repository root (scripts live in logs/q5/)
L=logs/q5/memcheck; mkdir -p $L
env_for() { echo "cd $PWD && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 MUJOCO_GL=egl OMP_NUM_THREADS=3 && set -a && source .env && set +a && export HF_HUB_OFFLINE=1 PYTHONPATH=src:functional_scripts"; }
capped() {  # name limit_gb gpu command...
  local name=$1 mem=$2 g=$3; shift 3
  local out rc
  # --wait returns when the job ends and prints its "Memory peak"; the job's own output goes to its log.
  out=$(systemd-run --user --unit="q5check-$name-$(date +%s)" -p MemoryMax=${mem}G -p MemorySwapMax=0 --wait --collect \
    sg gpu_$g -c "$(env_for $g) && $* > $PWD/$L/$name.log 2>&1" 2>&1); rc=$?
  echo "$name: exit $rc, limit ${mem}G, $(echo "$out" | grep -i 'memory peak' | sed 's/^ *//')" | tee -a $L/summary.txt
}
P=outputs/q5/policies
capped loader 40 4 .venv/bin/python functional_scripts/check_episode_loader.py
capped support_dense 40 4 .venv/bin/python functional_scripts/q5_e1.py support --stride 10 --primary $P/primary_s0/checkpoints/last/pretrained_model --companion $P/companion_s0/checkpoints/last/pretrained_model --root outputs/q5/seed0/e1 --tasks 0 1 2 3 4 5 6 7 8 9
capped eval 25 5 .venv/bin/python functional_scripts/q5_policy_eval.py --policy $P/primary_s0/checkpoints/last/pretrained_model --out $L/eval_check.json --tasks 0 --episodes 5
