#!/usr/bin/env zsh
# Q5 E3 rerun: whole groups (task x starting state: clean + 13 events) per process, so twins stay identical up to the
# onset, plus clean calibration runs (starting states 2-4). Output outputs/q5/seed{s}/e3_v2. GPUs 4-7, memory-capped.
cd "$(dirname "$0")/../.."  # repository root (scripts live in logs/q5/)
L=logs/q5/e3_rerun; mkdir -p $L
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

stamp "E3 rerun starts"
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

J=$L/jobs_e3.txt; : > $J
for s in 0 1 2; do for i in 0 1 2 3; do
  frames=$([ $s = 0 ] && echo --frames)
  echo "e3v2_s${s}_$i .venv/bin/python functional_scripts/q5_e3_capture.py --primary $(ckpt primary_s$s) --companion $(ckpt companion_s$s) --out outputs/q5/seed$s/e3_v2 --shard $i/4 $frames" >> $J
done; done
pool $J 3 20
grep -l Traceback $L/*.log | while read f; do grep -q -e 'Error' <(grep -v -e EGL -e OpenGL -e glCheckError $f) && stamp "check $f"; done
stamp "E3 RERUN FINISHED"
