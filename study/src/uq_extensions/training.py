"""SAVE/refit dispatch with explicit policy counts and immutable starting models."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import atomic_json, identity, model_identity, plain, read_json
from .config import ExperimentConfig
from .storage import check_budget, finalize_model, recover_checkpoint


def starting_models(config, base_models, previous_models, round_index):
    if len(base_models) < config.policy_count:
        raise ValueError("Missing seed-specific initial ensemble members")
    if config.training_mode == "pristine_refit" or round_index == 0:
        return list(base_models[:config.policy_count])
    if len(previous_models) != config.policy_count:
        raise ValueError("Previous ensemble does not match required policy count")
    return list(previous_models)


def make_training_config(config, manifest, start_models, run_dir, round_index):
    from iterative_fine_tuning.config import IterativeFineTuningConfig
    from iterative_fine_tuning.main import _save_config_snapshot
    root = Path(os.environ["VLA_PROJECT_ROOT"])
    cfg = IterativeFineTuningConfig.from_pretrained(str(root / "configs/reproduction/random.json"), cli_args=[])
    cfg.multi_seed.runs = []; cfg.multi_seed.seeds = []
    cfg.paths.run_root = str(Path(run_dir).parent)
    cfg.paths.run_name = Path(run_dir).name
    cfg.paths.checkpoint_root = None
    cfg.policy.pretrained_path = start_models[0]
    cfg.selection.ensemble_model_paths = list(start_models)
    # Selection is handled by the extension runner; this config only trains.
    cfg.selection.uncertainty_method = 'ace'
    cfg.ensemble_training.parallel_members = False
    # Extension round 1 is upstream round 0. Preserve the upstream member RNG
    # schedule: seed + 1000*(completed_updates-1) + member_index.
    cfg.seed = config.seed - 1000
    cfg.steps = config.steps
    cfg.batch_size = config.train_batch_size
    cfg.save_freq = config.save_frequency
    cfg.num_workers = config.num_workers
    cfg.eval_freq = 0
    cfg.policy.scheduler_decay_steps = config.steps
    cfg.policy.scheduler_warmup_steps = min(200, max(1, config.steps // 2))
    cfg.dataset.root = manifest["dataset_root"]
    cfg.replay.history_fraction = .5 if config.training_mode == "save_warm_50_50" else 0.
    cfg.replay.new_fraction = 1 - cfg.replay.history_fraction
    cfg.validate()
    path = Path(run_dir) / f"round_{round_index:03d}/training_config.json"
    _save_config_snapshot(cfg, path)
    return path


def fit_policies(config, manifest, start_models, acquired_ids, *, run_dir, round_index, gpu_ids, overlay=None):
    run_dir = Path(run_dir)
    directory = run_dir / f"round_{round_index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    acquired = sorted(set(acquired_ids))
    if not acquired:
        raise ValueError("An active update/refit requires a nonempty acquired history")
    spec = dict(training_mode=config.training_mode, seed=config.seed, round_index=round_index,
                starting_model_hashes=[model_identity(p) for p in start_models], acquired_ids=acquired,
                steps=config.steps, batch_size=config.train_batch_size,
                replay=[.5, .5] if config.training_mode == "save_warm_50_50" else [0., 1.],
                reset_optimizer_at_new_round=True, source_manifest=manifest["manifest_hash"], overlay=overlay)
    result_path = directory / "policy_fit.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["identity"] != identity(spec):
            raise RuntimeError("Policy fit cache has different data, weights or fitting settings")
        for path, digest in zip(result["models"], result["model_hashes"]):
            if model_identity(path) != digest:
                raise RuntimeError("Completed policy weights changed")
        return result["models"]
    # BEGIN RUN PROGRESS
    from run_progress import Progress, LogFollower
    progress=Progress(f'{config.arm}/{config.training_mode}/seed {config.seed}/round {round_index}')
    print(f'Training {len(start_models)} member(s), {config.steps} updates each, {len(acquired)} acquired demonstrations; starts={start_models}',flush=True)
    # END RUN PROGRESS
    check_budget(run_dir, config.budget_gb)
    cfg_path = make_training_config(config, manifest, start_models, run_dir, round_index)
    if len(start_models)>len(gpu_ids):
        raise ValueError('Provide one GPU per concurrently trained member')
    if overlay:
        allowed={str(manifest['episode_map'][id]) for id in acquired}
        if not set(overlay).issubset(allowed): raise ValueError('Condition overlay includes unrevealed episodes')
        atomic_json(directory/'training_overlay.json',overlay)
    jobs = []
    started = time.perf_counter()
    try:
        for member, start in enumerate(start_models):
            member_dir = directory / f"training/member_{member:02d}"
            request = dict(config_path=str(cfg_path), round_index=round_index, member_index=member,
                           run_dir=str(run_dir), model_reference=str(start),
                           history_episode_ids=[manifest["episode_map"][id] for id in manifest["pretraining_ids"]],
                           selected_episode_ids=[manifest["episode_map"][id] for id in acquired],
                           output=str(member_dir / "member_training_result.json"), expected_steps=config.steps)
            request_path = member_dir / "request.json"
            atomic_json(request_path, request)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[member % len(gpu_ids)])
            from lerobot.utils.gpu_utils import configure_egl_for_visible_cuda
            if gpu_ids[member] != 'cpu': configure_egl_for_visible_cuda(env)
            else: env['CUDA_VISIBLE_DEVICES']=''
            if overlay: env['UQ_EXT_TRAINING_OVERLAY']=str((directory/'training_overlay.json').resolve())
            else: env.pop('UQ_EXT_TRAINING_OVERLAY',None)
            env["UQ_EXT_RUN_ROOT"] = str(run_dir.resolve()); env["UQ_EXT_BUDGET_GB"] = str(config.budget_gb)
            # BEGIN RUN PROGRESS
            env['PYTHONUNBUFFERED']='1'
            follower=LogFollower([member_dir/'training.log'],label=f'{config.arm}/round {round_index}/member {member}')
            # END RUN PROGRESS
            with (member_dir / "training.log").open("a") as log:
                process = subprocess.Popen([sys.executable, "-m", "uq_extensions.cli", "_train-worker", str(request_path)], env=env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            # BEGIN RUN PROGRESS
            process.progress_follower=follower
            # END RUN PROGRESS
            jobs.append((process, request))
        while any(process.poll() is None for process, _ in jobs):
            for process, request in jobs:
                # BEGIN RUN PROGRESS
                process.progress_follower.poll()
                # END RUN PROGRESS
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"Training failed; inspect {Path(request['output']).parent / 'training.log'}")
            time.sleep(1)
        if any(p.returncode for p, _ in jobs):
            raise RuntimeError("Member training failed; inspect round training logs")
    finally:
        for process, _ in jobs:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
    # BEGIN RUN PROGRESS
    for process,_ in jobs:process.progress_follower.poll(heartbeat=False,final=True)
    progress.finish(f'{len(jobs)} member(s) completed; outputs={directory}')
    # END RUN PROGRESS
    outputs = [read_json(request["output"])["final_model_path"] for _, request in jobs]
    result = {**spec, "identity": identity(spec), "models": outputs,
              "model_hashes": [model_identity(p) for p in outputs], "seconds": time.perf_counter() - started,
              "requested_steps_per_member": config.steps,
              "members": [read_json(request["output"]) for _, request in jobs]}
    atomic_json(result_path, result)
    return outputs


def train_worker(request_path):
    from iterative_fine_tuning.config import IterativeFineTuningConfig
    from iterative_fine_tuning.layout import RunLayout
    from iterative_fine_tuning.training import _run_single_training_round
    from reproduction_storage import validate_checkpoint
    request = read_json(request_path)
    cfg = IterativeFineTuningConfig.from_pretrained(request["config_path"], cli_args=[])
    import torch
    cfg.policy.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.selection.device = cfg.policy.device
    cfg.validate()
    directory = Path(request["output"]).parent
    existing = recover_checkpoint(directory / "checkpoints")
    if existing is not None and validate_checkpoint(existing) == request["expected_steps"]:
        atomic_json(request["output"], dict(member_index=request["member_index"],
                    start_model_path=request["model_reference"], final_model_path=str(existing / "pretrained_model"),
                    training_dir=str(directory), resumed_completed_fit=True, steps=request["expected_steps"]))
        return
    result = _run_single_training_round(cfg, round_index=request["round_index"], member_index=request["member_index"],
                model_reference=request["model_reference"], history_episode_ids=request["history_episode_ids"],
                selected_episode_ids=request["selected_episode_ids"],
                round_layout=RunLayout(Path(request["run_dir"])).round_layout(request["round_index"]))
    record = plain(result)
    record["steps"] = request["expected_steps"]
    record["resumed_from"] = str(existing) if existing else None
    atomic_json(request["output"], record)


def retire_optimizers(models):
    for path in models:
        finalize_model(path)
