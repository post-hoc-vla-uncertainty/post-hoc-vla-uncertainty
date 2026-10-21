#!/usr/bin/env bash
# Keep the running reproduction checkout and its environment unchanged.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export PYTHONPATH="$VLA_PROJECT_ROOT/src:$VLA_PROJECT_ROOT/upstream/uq_vla_extensions/src:$VLA_PROJECT_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset VLA_REPRODUCTION_POLICY
