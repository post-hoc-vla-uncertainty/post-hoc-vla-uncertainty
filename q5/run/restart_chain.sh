#!/usr/bin/env zsh
# Capped memory check first; the big run starts only if the loader matches and every check finished in its limit.
cd "$(dirname "$0")/../.."  # repository root (scripts live in logs/q5/)
rm -f logs/q5/memcheck/summary.txt
zsh logs/q5/memory_check.sh
S=logs/q5/memcheck/summary.txt
if grep -q "LOADER OK" logs/q5/memcheck/loader.log && [ $(grep -c "exit 0" $S) -eq 3 ]; then
  echo "$(date '+%F %T %Z') memory check passed; starting big run" | tee -a logs/q5/big/progress.log
  zsh logs/q5/big_run.sh
else
  echo "$(date '+%F %T %Z') memory check FAILED; big run not started" | tee -a logs/q5/big/progress.log
fi
