"""Write local configs while retaining the pinned upstream scientific settings."""
import copy
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/uq_vla"
CONFIGS = ROOT / "configs/reproduction"

def write_config(name, config):
    path = CONFIGS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(config, indent=2) + "\n"
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"Refusing to change existing config: {path}")
    path.write_text(content)

def configure():
    lock = json.loads((ROOT / "assets.lock.json").read_text())
    storage = Path(os.environ["STORAGE_ROOT"])
    vlm = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + lock["vlm"]["repo_id"].replace("/", "--")) / "snapshots" / lock["vlm"]["revision"]
    dataset = {"repo_id": lock["dataset"]["repo_id"], "revision": lock["dataset"]["revision"],
               "root": str(Path(os.environ["HF_LEROBOT_HOME"]) / lock["dataset"]["repo_id"])}
    source = yaml.safe_load((UPSTREAM / "configs/smolvla/train/libero_pretrain_leak3.yaml").read_text())
    # Paper B.3: 30 Spatial/Object/Goal tasks and target tasks 0,1,2.
    source["dataset"]["libero_tasks"].pop("libero_90", None)
    source["dataset"].update(dataset)
    source["policy"].update(push_to_hub=False, vlm_model_name=str(vlm))
    source["wandb"].update(enable=False, mode="disabled")
    source["num_workers"] = 8
    # More frequent checkpoints support interruption recovery; optimization is unchanged.
    source["save_freq"] = 5000
    for seed in range(6):
        cfg = copy.deepcopy(source)
        cfg["seed"] = seed
        cfg["output_dir"] = str(storage / "outputs/pretrain/libero_pretrain_leak3" / f"seed_{seed}")
        cfg["job_name"] = f"libero_pretrain_leak3_seed_{seed}"
        write_config(f"pretrain_seed_{seed}.json", cfg)

    templates = {
        "random": "iter_al_uniform_episode/t0_w50_eps5_step4000_uniform_episode.yaml",
        "vfd": "iter_al/t2.5_w50_eps5_step4000_lr.yaml",
        "gu": "iter_al_ensemble_terminal_variance/t1_w50_eps5_step4000_ensemble_terminal_variance.yaml",
    }
    for method, template in templates.items():
        cfg = yaml.safe_load((UPSTREAM / "configs/iterative_fine_tuning" / template).read_text())
        cfg["dataset"].update(dataset)
        cfg["policy"].update(push_to_hub=False, vlm_model_name=str(vlm))
        cfg["paths"].update(env_file=None, run_root=str(storage / "outputs/reproduction"), run_name=method)
        cfg["wandb"].update(enable=False, mode="disabled")
        for index, run in enumerate(cfg["multi_seed"]["runs"]):
            run["run_name"] = f"seed_{1000+index}"
        write_config(f"{method}.json", cfg)

    # Table 1 uses a distinct 2,000-step random trajectory in the released README.
    cfg = yaml.safe_load((UPSTREAM / "configs/iterative_fine_tuning/calibration_experiment/uniform_leak3_weighted_pools_lr_schedule.yaml").read_text())
    cfg["dataset"].update(dataset)
    cfg["policy"].update(push_to_hub=False, vlm_model_name=str(vlm))
    cfg["paths"].update(env_file=None, run_root=str(storage / "outputs/reproduction"),
                         checkpoint_root=None, run_name="calibration_random")
    cfg["iteration"]["total_rounds"] = 15
    cfg["job_name"] = "calibration_random"
    cfg["replay"]["history_tasks"].pop("libero_90", None)
    cfg["wandb"].update(enable=False, mode="disabled")
    runs = []
    for i in range(3):
        paths = [str(storage / "outputs/pretrain/libero_pretrain_leak3" / f"seed_{j}/checkpoints/last/pretrained_model") for j in (2*i, 2*i+1)]
        runs.append({"seed":1000+i, "selection_seed":2*i, "run_name":f"seed_{1000+i}",
                     "pretrained_path":paths[0], "ensemble_model_paths":paths})
    cfg["multi_seed"] = {"runs":runs}
    cfg["policy"]["pretrained_path"] = runs[0]["pretrained_path"]
    cfg["selection"]["ensemble_model_paths"] = runs[0]["ensemble_model_paths"]
    write_config("calibration_random.json", cfg)
    # The released calibration launcher varies the model pair and selection seed,
    # but keeps policy-training seed=1000 for all pairs. Multi-seed child launching
    # would overwrite that with the run ID, so launch explicit single-run configs.
    for run in runs:
        child = copy.deepcopy(cfg)
        child["multi_seed"] = {}
        child["paths"].update(run_root=str(storage / "outputs/reproduction/calibration_random"),
                              run_name=run["run_name"])
        child["policy"]["pretrained_path"] = run["pretrained_path"]
        child["selection"].update(ensemble_model_paths=run["ensemble_model_paths"],
                                   seed=run["selection_seed"])
        write_config(f"calibration_random_{run['run_name']}.json", child)
    print(f"Configs ready: {CONFIGS}")

if __name__ == "__main__":
    configure()
