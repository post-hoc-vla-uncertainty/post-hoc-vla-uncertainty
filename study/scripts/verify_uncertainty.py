"""Check the upstream calibration samplers on a batch of real observations."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs=2, required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "upstream_calibration", ROOT / "upstream/uq_vla/scripts/calibration/calibration_comparison.py"
    )
    calibration = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = calibration
    spec.loader.exec_module(calibration)

    policy, adapter = calibration.load_adapter(str(args.members[0]), "cuda:0")
    ensemble = calibration.build_ensemble_models([str(args.members[1])], policy.config)
    preprocessor, _ = calibration.make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=str(args.members[0])
    )
    dataset = calibration.make_dataset(
        dataset_cfg=calibration.DatasetConfig(
            repo_id="HuggingFaceVLA/libero",
            root=str(Path(os.environ["HF_LEROBOT_HOME"]) / "HuggingFaceVLA/libero"),
            libero_tasks={"libero_10": [0]},
        ), policy_cfg=policy.config,
    )
    episodes = calibration.filter_libero_episodes(dataset, tasks_to_use={"libero_10": [0]})
    frames = calibration.get_episode_frame_indices(dataset, episodes[:args.batch_size])
    observation = calibration.collate_observations([
        calibration.prepare_observation(preprocessor(dataset[frame]), "cuda:0") for frame in frames
    ])

    results = {}
    for method in calibration.METHODS:
        if method == calibration.ENSEMBLE_TV_METHOD:
            sampler = calibration.build_ensemble_tv_sampler(adapter, ensemble, 5)
        elif method == calibration.DECU_METHOD:
            sampler = calibration.build_decu_sampler(adapter, ensemble, 5)
        elif method == "ace":
            sampler = calibration.ACESampler(calibration.ACESamplerConfig(), adapter)
        elif method in (calibration.VLM_ENTROPY_METHOD, calibration.VLM_PERPLEXITY_METHOD):
            sampler = calibration.VLMTokenSampler(adapter, method)
        else:
            sampler = calibration.build_ensemble_sampler(adapter, ensemble, method, 5)
        with torch.no_grad():
            actions, uncertainty = sampler.conditional_sample_with_uncertainty_batch(observation=observation)
        assert uncertainty.numel() == args.batch_size and torch.isfinite(uncertainty).all(), method
        assert torch.isfinite(actions).all(), method
        results[method] = uncertainty.tolist()
        print(f"{method}: {results[method]}", flush=True)
        del sampler
    report = {"purpose": "Dependency and inference smoke test, not a reproduction result",
              "checkpoints": list(map(str, args.members)), "batch_size": args.batch_size,
              "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
              "uncertainty_scores": results}
    (ROOT / "reports/uncertainty_check.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
