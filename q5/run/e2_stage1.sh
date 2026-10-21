#!/usr/bin/env zsh
# Q5 E2, stage 1 (seed 0 only, then stop for the supervisor check of handoff Sections 3 and 11):
#   A  train supervisors S0-S2 on GPUs 4-6; meanwhile the primary's seed-0 E2 rollouts run on GPU 7
#   B  supervisor S0: clean success (50 starting states x 10 tasks) and its seed-0 E2 rollouts on GPUs 4-6
# GPUs 4-7 only, every job in a memory cgroup, admission by free RAM, watchdog.
cd "$(dirname "$0")/../.."  # repository root (scripts live in logs/q5/)
L=logs/q5/e2; mkdir -p $L
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

stamp "E2 stage 1 starts"
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

train() { grep -q "End of training" $L/train_$2.log 2>/dev/null && return 0
          wait_for_memory 40; on_gpu $1 train_$2 40 .venv/bin/python src/lerobot/scripts/lerobot_train.py --config_path=configs/smolvla/train/q5_$2.yaml; }
J=$L/jobs_primary_s0.txt; : > $J
for i in $(seq 0 5); do
  echo "e2_primary_s0_$i .venv/bin/python functional_scripts/q5_e2.py --role primary --policy $(ckpt primary_s0) --seed-triple 0 --shard $i/6" >> $J
done
( ALL_GPUS=($GPUS); GPUS=(7); pool $J 3 20; stamp "primary seed-0 rollouts done" ) &
PRIMARY=$!
( train 4 supervisor_s0 ) & T0=$!
( train 5 supervisor_s1 ) & T1=$!
( train 6 supervisor_s2 ) & T2=$!
wait $T0 $T1 $T2
for s in 0 1 2; do grep -q "End of training" $L/train_supervisor_s$s.log || { stamp "training supervisor_s$s FAILED"; exit 1; }; done
stamp "A: supervisors trained (primary seed-0 rollouts run alongside)"

J=$L/jobs_b.txt; : > $J
echo "eval_supervisor_s0 .venv/bin/python functional_scripts/q5_policy_eval.py --policy $(ckpt supervisor_s0) --out $P/supervisor_s0/clean_success.json" >> $J
for i in $(seq 0 9); do
  echo "e2_supervisor_s0_$i .venv/bin/python functional_scripts/q5_e2.py --role supervisor --policy $(ckpt supervisor_s0) --seed-triple 0 --shard $i/10" >> $J
done
GPUS=(4 5 6)                       # GPU 7 keeps running the primary's rollouts
pool $J 3 25
wait $PRIMARY
grep -l Traceback $L/*.log | while read f; do grep -q -e 'Error' <(grep -v -e EGL -e OpenGL -e glCheckError $f) && stamp "check $f"; done
stamp "E2 STAGE 1 FINISHED: check the supervisor before seeds 1 and 2"
