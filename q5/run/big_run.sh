#!/usr/bin/env zsh
# Q5 big run (E1 + E3 for the paper; E2 later), on GPUs 4-7 only (nvidia-smi numbering).
#   A  train primaries P0-2 and companions C0-2 (paper recipe, all of LIBERO-10, 14,840 updates)
#   B  clean success per policy and task (50 standard starting states)      -> policies.json
#   C  support sets per primary: first frames (paper recipe) and every 10th frame (dense)
#   D  E1 per seed: 10 tasks x 25 test states x 67 variants (+25 calibration states), VFD on states 0-9
#   E  E3 per seed: 10 tasks x states 0,1 x (clean + 13 events), onset 100, heads + VFD every 5 steps
#   F  E1 harm, seed 0: middle level of every family, starting states 0-9, 3 executions
# Starts when the current E3 set-1 recordings are finished. Every stage waits for the previous one.
# Memory safety (after the 2026-10-14 overflow): every job runs in its own cgroup with a hard RAM limit and no
# swap, a job starts only when MemAvailable leaves RESERVE_GB for other users, and a watchdog stops our
# newest job if MemAvailable falls below FLOOR_GB.
cd "$(dirname "$0")/../.."  # repository root (scripts live in logs/q5/)
L=logs/q5/big; mkdir -p $L
GPUS=(4 5 6 7)
P=outputs/q5/policies
ckpt() { echo $P/$1/checkpoints/last/pretrained_model; }
env_for() { echo "cd $PWD && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 MUJOCO_GL=egl OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 && set -a && source .env && set +a && export HF_HUB_OFFLINE=1 PYTHONPATH=src:functional_scripts"; }
RESERVE_GB=100   # left free for other users when a job starts
FLOOR_GB=40      # watchdog threshold
mem_avail_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
# Run a command on GPU $1 in its own memory cgroup of $3 GB (hard limit, no swap); log name $2.
on_gpu() {
  local g=$1 log=$2 mem=$3; shift 3
  systemd-run --user --scope --quiet --unit="q5-$(date +%s%N)-$log" -p MemoryMax=${mem}G -p MemorySwapMax=0 \
    sg gpu_$g -c "$(env_for $g) && $*" > $L/$log.log 2>&1
}
# Wait until a job of $1 GB fits next to the reserve for other users.
wait_for_memory() { while [ $(mem_avail_gb) -lt $(( $1 + RESERVE_GB )) ]; do sleep 20; done; }
stamp() { echo "$(date '+%Y-%m-%d %H:%M:%S %Z')  $*" | tee -a $L/progress.log; }

# Run every line of a job file ("<log name> <command>") on a pool of $2 workers per GPU, each job in a
# cgroup of $3 GB, then return.
pool() {
  local jobs=$1 per_gpu=$2 mem=$3 q=$L/queue_${1:t:r}
  rm -rf $q; mkdir -p $q; echo 1 > $q/next; : > $q/lock
  local w=0
  for g in $GPUS; do for k in $(seq $per_gpu); do
    (
      while true; do
        n=$(flock $q/lock sh -c "n=\$(cat $q/next); echo \$((n+1)) > $q/next; echo \$n")
        line=$(sed -n "${n}p" $jobs); [[ -z $line ]] && break
        { flock 9; wait_for_memory $mem; sleep 30; } 9>$q/admit
        on_gpu $g ${line%% *} $mem ${line#* }
      done
    ) &
    w=$((w+1))
  done; done
  wait
}

# --- wait for the running E3 set-1 recordings -------------------------------------------------------
until [ "$(cat logs/live_eval/kvctx/set1_worker*.log | grep -c 'worker done')" -ge 12 ]; do sleep 60; done
stamp "set-1 recordings finished; big run starts"

# Watchdog: if free memory falls below FLOOR_GB, stop our newest job (its cgroup) and log it.
(
  while true; do
    if [ $(mem_avail_gb) -lt $FLOOR_GB ]; then
      unit=$(systemctl --user list-units --type=scope --no-legend 'q5-*' 2>/dev/null | awk '{print $1}' | tail -1)
      [ -n "$unit" ] && systemctl --user stop "$unit" && stamp "WATCHDOG: MemAvailable $(mem_avail_gb) GB < $FLOOR_GB GB, stopped $unit"
      sleep 30
    fi
    sleep 5
  done
) &!
WATCHDOG=$!   # disowned (&!), so the plain `wait` calls below do not wait for it
trap "kill $WATCHDOG 2>/dev/null" EXIT

# --- A: train the six policies (two per 40 GB card, one per 24 GB card) ------------------------------
# Resumable: a policy whose log already says "End of training" is not trained again.
train() { grep -q "End of training" $L/train_$2.log 2>/dev/null && return 0
          wait_for_memory 40; on_gpu $1 train_$2 40 .venv/bin/python src/lerobot/scripts/lerobot_train.py --config_path=configs/smolvla/train/q5_$2.yaml; }
( train 4 primary_s0; ) & ( train 4 companion_s0; ) & ( train 5 primary_s1; ) & ( train 5 companion_s1; ) &
( train 6 primary_s2; ) & ( train 7 companion_s2; ) &
wait
for r in primary companion; do for s in 0 1 2; do
  grep -q "End of training" $L/train_${r}_s$s.log || { stamp "training ${r}_s$s FAILED"; exit 1; }
done; done
stamp "A: six policies trained"

# --- B + C: clean success of every policy, support sets of every primary ------------------------------
J=$L/jobs_bc.txt; : > $J
for r in primary companion; do for s in 0 1 2; do
  echo "eval_${r}_s$s .venv/bin/python functional_scripts/q5_policy_eval.py --policy $(ckpt ${r}_s$s) --out $P/${r}_s$s/clean_success.json" >> $J
done; done
ALL_GPUS=($GPUS); GPUS=(4 5)
pool $J 1 25                       # two evaluations at a time, 5 env subprocesses each
GPUS=($ALL_GPUS)
J=$L/jobs_c.txt; : > $J
for s in 0 1 2; do
  base="--primary $(ckpt primary_s$s) --companion $(ckpt companion_s$s) --root outputs/q5/seed$s/e1 --tasks 0 1 2 3 4 5 6 7 8 9"
  echo "support_initial_s$s .venv/bin/python functional_scripts/q5_e1.py support $base" >> $J
  echo "support_dense_s$s .venv/bin/python functional_scripts/q5_e1.py support --stride 10 $base" >> $J
done
pool $J 1 20                       # supports: memory-mapped dataset since the loader fix
stamp "B+C: clean success and support sets done"

# --- D: E1 scoring, 4 shards per seed ------------------------------------------------------------------
J=$L/jobs_d.txt; : > $J
for s in 0 1 2; do for i in 0 1 2 3; do
  echo "e1_s${s}_$i .venv/bin/python functional_scripts/q5_e1.py features --primary $(ckpt primary_s$s) --companion $(ckpt companion_s$s) --root outputs/q5/seed$s/e1 --tasks 0 1 2 3 4 5 6 7 8 9 --shard $i/4" >> $J
done; done
pool $J 3 20
stamp "D: E1 scoring done"

# --- E: E3 rollouts, 4 shards per seed (camera frames for seed 0 only, for the videos) ------------------
J=$L/jobs_e.txt; : > $J
for s in 0 1 2; do for i in 0 1 2 3; do
  frames=$([ $s = 0 ] && echo --frames)
  echo "e3_s${s}_$i .venv/bin/python functional_scripts/q5_e3_capture.py --primary $(ckpt primary_s$s) --companion $(ckpt companion_s$s) --out outputs/q5/seed$s/e3 --shard $i/4 $frames" >> $J
done; done
pool $J 3 20
stamp "E: E3 rollouts done"

# --- F: E1 harm rollouts, seed 0 -----------------------------------------------------------------------
J=$L/jobs_f.txt; : > $J
for i in $(seq 0 11); do
  echo "harm_s0_$i .venv/bin/python functional_scripts/q5_e1.py harm --primary $(ckpt primary_s0) --companion $(ckpt companion_s0) --root outputs/q5/seed0/e1 --tasks 0 1 2 3 4 5 6 7 8 9 --shard $i/12" >> $J
done
pool $J 3 20
stamp "F: E1 harm done"

grep -l Traceback $L/*.log | while read f; do grep -q -e 'Error' <(grep -v -e EGL -e OpenGL -e glCheckError $f) && stamp "check $f"; done
stamp "BIG RUN FINISHED"
