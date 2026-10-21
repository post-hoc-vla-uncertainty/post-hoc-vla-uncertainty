#!/usr/bin/env python
"""Clean success of one Q5 policy per LIBERO-10 task on LIBERO's 50 standard starting states (handoff Section 3).

Episode i of a task uses starting state i and seed SEED0 + i, played in batches of async vector envs
(set_attr pins each sub-env to its starting state). Writes {"task": [successes]} to --out (resumable).

    python functional_scripts/q5_policy_eval.py --policy DIR --out outputs/q5/policies/primary_s0/clean_success.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

SEED0, MAX_STEPS = 5000, 520


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--episodes", type=int, default=50)
    # Each vector env runs this many subprocesses (torch + LIBERO + MuJoCo, about 1.5 GB each).
    parser.add_argument("--batch", type=int, default=5)
    args = parser.parse_args()

    from iterative_fine_tuning.evaluation import _extract_batch_successes
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.libero import get_task_instruction
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.random_utils import set_seed

    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    policy = SmolVLAPolicy.from_pretrained(args.policy).to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.policy),
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    for task in args.tasks:
        if len(results.get(str(task), [])) >= args.episodes:
            continue
        cfg = LiberoEnv(task="libero_10", task_ids=[task])
        env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=policy.config)
        env = make_env(cfg, n_envs=args.batch, use_async_envs=True)["libero_10"][task]
        instruction = get_task_instruction("libero_10", task)
        successes = []
        try:
            for start in range(0, args.episodes, args.batch):
                env.set_attr("_init_state_id", [start + i for i in range(args.batch)])
                seeds = [SEED0 + start + i for i in range(args.batch)]
                set_seed(seeds[0])
                policy.reset()
                obs, _ = env.reset(seed=seeds)
                done = np.zeros(args.batch, bool)
                success = np.zeros(args.batch, bool)
                for _ in range(MAX_STEPS):
                    batch = preprocess_observation(obs)
                    batch["task"] = [instruction] * args.batch
                    action = policy.select_action(pre(env_pre(batch)))
                    action = env_post({"action": post(action)})["action"]
                    action = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
                    obs, _, terminated, truncated, info = env.step(action)
                    success |= np.asarray(_extract_batch_successes(info, args.batch), bool) & ~done
                    done |= np.asarray(terminated, bool) | np.asarray(truncated, bool)
                    if done.all():
                        break
                successes += success.tolist()[: args.episodes - start]
        finally:
            env.close()
        results[str(task)] = [bool(s) for s in successes]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1) + "\n")
        print(f"{args.policy} task {task}: {np.mean(successes):.2f}", flush=True)


if __name__ == "__main__":
    main()
