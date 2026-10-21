#!/usr/bin/env python
"""Score a balanced pool of disjoint LIBERO-10 initial observations with FTTGP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.envs.libero import get_task_instruction
from lerobot.policies.factory import make_flow_matching_adapter_from_policy, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import FTTGPSamplerConfig
from lerobot.uncertainty.uncertainty_samplers.fttgp_sampler import FTTGPSampler


CHECKPOINT = Path("outputs/pretrain/libero_pretrain_leak3/seed_1/checkpoints/030000/pretrained_model")
HEAD = Path("outputs/fttgp/libero_leak3_balanced/head.pt")
ROOT = Path("storage/huggingface/lerobot/HuggingFaceVLA/libero")
OUTPUT = Path("outputs/fttgp/libero_leak3_balanced/initial_pool_results.json")


def make_batch(preprocessor, dataset, indices, device):
    samples = [preprocessor(dataset[i]) for i in indices]
    result = {}
    for key, first in samples[0].items():
        if not isinstance(first, torch.Tensor):
            continue
        tensors = []
        for sample in samples:
            value = sample[key]
            already_batched = ("language.tokens" in key or "language.attention_mask" in key) and value.ndim == 2
            tensors.append(value if already_batched else value.unsqueeze(0))
        result[key] = torch.cat(tensors).to(device)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, default=HEAD)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    device = torch.device("cuda")
    policy = SmolVLAPolicy.from_pretrained(CHECKPOINT).to(device).eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=str(CHECKPOINT))
    adapter = make_flow_matching_adapter_from_policy(policy)
    sampler = FTTGPSampler(
        FTTGPSamplerConfig(checkpoint_path=str(args.head), num_action_samples=1), adapter
    )

    meta = LeRobotDatasetMetadata("HuggingFaceVLA/libero", root=ROOT)
    instructions = {task: get_task_instruction("libero_10", task) for task in range(10)}
    episodes = {task: [] for task in range(10)}
    for row in meta.episodes:
        for task, instruction in instructions.items():
            if row["tasks"][0] == instruction:
                episodes[task].append(int(row["episode_index"]))
    # Drop the first three episodes for every task. For ID tasks these are the GP-fit
    # demonstrations; doing the same for OOD tasks keeps exactly 27 observations/task.
    pool = {task: values[3:30] for task, values in episodes.items()}
    selected = sorted({episode for values in pool.values() for episode in values})
    dataset = LeRobotDataset(
        "HuggingFaceVLA/libero", root=ROOT, episodes=selected,
        delta_timestamps=resolve_delta_timestamps(policy.config, meta),
    )
    first_rows = {}
    for index, episode in enumerate(dataset.hf_dataset["episode_index"]):
        episode = int(episode)
        first_rows.setdefault(episode, index)

    per_task = []
    for task in range(10):
        scores = []
        for start in range(0, len(pool[task]), 9):
            batch_episodes = pool[task][start : start + 9]
            indices = [first_rows[episode] for episode in batch_episodes]
            batch = make_batch(preprocessor, dataset, indices, device)
            _, uncertainty = sampler.conditional_sample_with_uncertainty_batch(
                batch, generator=torch.Generator(device=device).manual_seed(3000 + task * 10 + start)
            )
            scores.extend(float(value) for value in uncertainty)
        row = {
            "task_id": task, "fit_task": task < 3, "episode_ids": pool[task],
            "scores": scores, "mean": sum(scores) / len(scores),
        }
        per_task.append(row)
        print(f"task {task}: n={len(scores)}, mean U={row['mean']:.6f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"per_task": per_task}, indent=2) + "\n")


if __name__ == "__main__":
    main()
