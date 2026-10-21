#!/usr/bin/env python
"""Q5 / E2: within-task referral under perturbations, rollouts (handoff/q5, Section 7).

Conditions per seed triple: 10 LIBERO-10 tasks x starting states 0-9 x 13 label-preserving variants (three of them
expected harmless). Every condition is executed three times by the primary P_s and three times by the supervisor
S_s, with the same environment seed per execution for both (SEED0 + 100 task + state + 10,000 (rep + 1), as the E1
harm rollouts). The perturbation is applied for the whole episode (q5_e1.rollout). The heads' scores of a
condition are its E1 scores on the same starting observation (outputs/q5/seed{s}/e1), so nothing is scored here.

    python functional_scripts/q5_e2.py --role primary --policy CKPT --seed-triple 0 --shard 0/6

Writes outputs/q5/seed{s}/e2/{role}/task{t}_state{k:02d}_{variant}.json: {"success": [rep0, rep1, rep2]}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ["handoff/q5", "functional_scripts"]
import q5_e1  # noqa: E402
from q5_e1 import Policy, Scene, rollout  # noqa: E402

VARIANTS = ["clean", "I1:P3", "I1:P5", "I2:all", "I3:please", "V1:x0.8", "V1:x0.4", "V2:sigma0.01", "V2:sigma0.08",
            "V3:2cm3deg", "V4:50", "V6:blackout", "R2:3cm"]
HARMLESS = {"I3:please", "V1:x0.8", "V2:sigma0.01"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("primary", "supervisor"), required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--seed-triple", type=int, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--states", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    q5_e1.MEMBERS = (args.policy, args.policy)
    policy = Policy(args.device, partner=False)
    out = Path(f"outputs/q5/seed{args.seed_triple}/e2/{args.role}")
    out.mkdir(parents=True, exist_ok=True)
    conditions = [(t, s, v) for t in args.tasks for s in args.states for v in VARIANTS]
    i, n = map(int, args.shard.split("/"))
    scenes: dict[int, Scene] = {}
    for task, state, variant in conditions[i::n]:
        path = out / f"task{task}_state{state:02d}_{variant.replace(':', '-')}.json"
        if path.exists():
            continue
        started = time.time()
        scene = scenes.setdefault(task, Scene(task))
        results = [rollout(scene, policy, task, state, variant, q5_e1.SEED0 + 100 * task + state + 10_000 * (rep + 1))
                   for rep in range(args.reps)]
        path.write_text(json.dumps({"seed_triple": args.seed_triple, "role": args.role, "task": task, "state": state,
                                    "variant": variant, "harmless_expected": variant in HARMLESS,
                                    "policy": str(args.policy), "success": results}) + "\n")
        print(f"{args.role} s{args.seed_triple} task {task} state {state} {variant}: {results} ({time.time() - started:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
