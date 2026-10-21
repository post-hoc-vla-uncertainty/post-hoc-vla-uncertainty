"""Local, restartable launch commands for the upstream reproduction (no SLURM)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from lerobot.utils.gpu_utils import configure_egl_for_visible_cuda
from patch_simulator import check_patch
from reproduction_storage import POLICY_ENV, StoragePolicy, atomic_json, sha256
from run_progress import LogFollower, emit, duration, tail_text

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/reproduction"
METHODS = ("random", "vfd", "gu", "calibration_random")
DEFAULT_METHODS = ("random", "vfd")
PYTHON = str(ROOT / ".venv/bin/python")

def command(module, *args):
    return [PYTHON, "-m", module, *map(str, args)]

def launch(cmd, log, *, env=None, dry_run=False, watch_roots=()):
    print(shlex.join(map(str, cmd)), flush=True)
    print(f"  log: {log}", flush=True)
    if dry_run:
        return None
    log.parent.mkdir(parents=True, exist_ok=True)
    follower=LogFollower([log],watch_roots,label=log.stem)
    env=dict(os.environ if env is None else env);env['PYTHONUNBUFFERED']='1'
    emit(log.stem,'START '+shlex.join(map(str,cmd)))
    handle = log.open("a")
    handle.write(f"\nCOMMAND: {shlex.join(map(str,cmd))}\n"); handle.flush()
    process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    handle.close()
    process.progress_follower=follower
    return process

def wait_all(jobs):
    try:
        while jobs:
            for process, log in list(jobs):
                follower=getattr(process,'progress_follower',None)
                if follower:follower.poll()
                status = process.poll()
                if status is None:
                    continue
                if status:
                    if follower:follower.poll(heartbeat=False,final=True)
                    emit(log.stem,f'FAILED exit={status}; log={log}')
                    raise RuntimeError(f"Command failed ({status}). See {log}\n" + tail_text(log))
                if follower:
                    follower.poll(heartbeat=False,final=True)
                    emit(log.stem,f'DONE wall={duration(time.monotonic()-follower.started)} | {follower.latest}')
                jobs.remove((process, log))
            if jobs: time.sleep(2)
    finally:
        # Only stop processes launched by this invocation, including their workers.
        for process, _ in jobs:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()

def run(cmd, log, args, *, watch_roots=()):
    process = launch(cmd, log, dry_run=args.dry_run,watch_roots=watch_roots)
    if process is not None: wait_all([(process,log)])

def check_source():
    lock = json.loads((ROOT / "assets.lock.json").read_text())
    commit = subprocess.check_output(["git", "-C", str(ROOT/"upstream/uq_vla"), "rev-parse", "HEAD"], text=True).strip()
    if commit != lock["uq_vla"]["revision"]:
        raise RuntimeError("Upstream commit differs from assets.lock.json")
    diff = subprocess.check_output(["git", "-C", str(ROOT/"upstream/uq_vla"), "diff", "--binary"])
    if diff != (ROOT/"patches/upstream-compat.patch").read_bytes():
        raise RuntimeError("Upstream changes differ from the recorded compatibility patch")

def pretrain(args, storage, logs):
    model_seeds = [j for seed in args.seeds for j in (2*(seed-1000), 2*(seed-1000)+1)]
    for offset in range(0, len(model_seeds), 2):
        jobs = []
        try:
            for slot, seed in enumerate(model_seeds[offset:offset+2]):
                cfg_path = CONFIGS / f"pretrain_seed_{seed}.json"
                cfg = json.loads(cfg_path.read_text())
                output = Path(cfg["output_dir"])
                if not args.dry_run:
                    args.storage_policy.maintain_pretrain(output, finalize_at=cfg["steps"])
                    args.storage_policy.check_budget()
                last = output / "checkpoints/last"
                step_file = last / "training_state/training_step.json"
                if step_file.is_file():
                    step = json.loads(step_file.read_text())["step"]
                    if step >= cfg["steps"]:
                        print(f"Already complete: pretraining seed {seed}", flush=True)
                        continue
                    cmd = command("lerobot.scripts.lerobot_train", f"--config_path={last/'pretrained_model/train_config.json'}", "--resume=true")
                else:
                    if output.exists() and not args.dry_run:
                        raise RuntimeError(f"Incomplete pretraining without a resumable checkpoint: {output}. Move that directory aside before restarting this seed.")
                    cmd = command("lerobot.scripts.lerobot_train", f"--config_path={cfg_path}")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids[slot]
                if not args.dry_run:
                    configure_egl_for_visible_cuda(env)
                log = logs / f"pretrain_seed_{seed}.log"
                process = launch(cmd, log, env=env, dry_run=args.dry_run)
                if process is not None: jobs.append((process,log))
            wait_all(jobs)
            if not args.dry_run:
                for seed in model_seeds[offset:offset+2]:
                    cfg = json.loads((CONFIGS / f"pretrain_seed_{seed}.json").read_text())
                    args.storage_policy.maintain_pretrain(Path(cfg["output_dir"]), finalize_at=cfg["steps"])
        finally:
            for process, _ in jobs:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM); process.wait()

def check_pretrained(args, storage):
    if args.dry_run: return
    for seed in args.seeds:
        for member in (2*(seed-1000), 2*(seed-1000)+1):
            path = storage / f"outputs/pretrain/libero_pretrain_leak3/seed_{member}/checkpoints/last/training_state/training_step.json"
            if not path.is_file() or json.loads(path.read_text())["step"] != 30000:
                raise RuntimeError(f"Missing completed 30,000-step preparatory checkpoint for seed {member}. Run the pretrain command first.")

def train_and_evaluate(args, storage, logs, train=True):
    check_pretrained(args, storage)
    for method in args.methods:
        cfg = CONFIGS / f"{method}.json"
        for seed in args.seeds:
            if args.retention == "rolling":
                # Evaluate before advancing to the next round; the worker retains
                # only milestones and the latest committed warm-start ensemble.
                cmd = [PYTHON, str(ROOT / "scripts/reproduction_rounds.py"),
                       "--config-path", str(cfg), "--seed", str(seed), "--method", method]
                if not train:
                    cmd.append("--evaluate-only")
                run(cmd, logs / f"{method}_seed_{seed}_{'train' if train else 'eval'}.log", args,
                    watch_roots=[storage/f'outputs/reproduction/{method}/seed_{seed}'])
                continue
            if not args.dry_run:
                args.storage_policy.check_budget()
            if train:
                if method == "calibration_random":
                    train_cmd = command("iterative_fine_tuning.main", f"--config_path={CONFIGS/f'calibration_random_seed_{seed}.json'}")
                else:
                    train_cmd = command("iterative_fine_tuning.run_multi_seed_child", "--config-path", cfg, "--seed", seed)
                run(train_cmd,
                    logs / f"{method}_seed_{seed}_train.log", args,watch_roots=[storage/f'outputs/reproduction/{method}/seed_{seed}'])
            run_dir = storage / f"outputs/reproduction/{method}/seed_{seed}"
            run(command("iterative_fine_tuning.evaluate_run", "--run-dir", run_dir,
                        "--members", "0", "--member-devices", "cuda:0,cuda:1", "--workers-per-device", "1",
                        "--n-rollouts", "30", "--eval-batch-size", "15", "--max-steps", "520",
                        "--seed-start", "100", "--sync-envs", "--skip-existing"),
                logs / f"{method}_seed_{seed}_eval.log", args,watch_roots=[run_dir])
        run(command("iterative_fine_tuning.summarize_multi_seed_run", "--config-path", cfg),
            logs / f"{method}_summary.log", args)

def calibrate(args, storage, logs):
    method = "calibration_random"
    run_dirs = [storage / f"outputs/reproduction/{method}/seed_{seed}" for seed in args.seeds]
    run([PYTHON, str(ROOT/"upstream/uq_vla/scripts/calibration/calibration_comparison.py"),
         "--run_dirs", *map(str,run_dirs), "--output", str(storage/"outputs/reproduction/calibration_tables"),
         "--device", "cuda:0", "--inference_batch_size", "8", "--num_action_samples", "5", "--env", "libero"],
        logs / "calibration_scores.log", args)
    run([PYTHON, str(ROOT/"upstream/uq_vla/scripts/calibration/calibration_summary.py"),
         "--run_dirs", *map(str,run_dirs), "--output", str(storage/"outputs/reproduction/calibration_tables/table1.png"),
         "--max_round", "15", "--env", "libero", "--methods", "action_l2", "ace", "decu",
         "ensemble_terminal_variance", "vfd", "vlm_token_entropy", "vlm_perplexity"],
        logs / "calibration_summary.log", args)

def report(args, storage, logs):
    run([PYTHON, str(ROOT/"scripts/summarize_reproduction.py"),
         "--root", str(storage/"outputs/reproduction"), "--methods", *args.methods,
         "--seeds", *map(str,args.seeds)], logs / "report.log", args)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["verify", "pretrain", "run", "evaluate", "calibrate", "report", "all"])
    parser.add_argument("--gpus", default="0,1", help="Two physical NVIDIA GPU indices, default: 0,1")
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--retention", choices=["rolling", "all"], default="rolling",
                        help="Default: rolling resume checkpoints and selected model milestones")
    parser.add_argument("--budget-gb", type=float, default=100,
                        help="Generated reproduction artifact guard in decimal GB (default: 100)")
    parser.add_argument("--seeds", type=int, choices=[1000,1001,1002], nargs="+", default=[1000,1001,1002])
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training or evaluation")
    parser.add_argument('--progress-steps',type=int,default=100,help='Console training interval; epoch boundaries are also reported')
    parser.add_argument('--progress-seconds',type=float,default=60,help='Progress/heartbeat interval in seconds')
    args = parser.parse_args()
    if args.progress_steps<1 or not 0<args.progress_seconds<float('inf'):parser.error('Progress intervals must be positive and finite')
    os.environ['VLA_PROGRESS_STEPS']=str(args.progress_steps)
    os.environ['VLA_PROGRESS_SECONDS']=str(args.progress_seconds)
    os.environ['PYTHONUNBUFFERED']='1'
    if not 0 < args.budget_gb < float("inf"):
        parser.error("--budget-gb must be finite and positive")
    if args.retention == "rolling" and ("calibration_random" in args.methods or args.stage == "calibrate"):
        parser.error("Whole-history calibration requires --retention all; it is deferred from the default reproduction")
    args.gpu_ids = args.gpus.split(",")
    if len(args.gpu_ids) != 2 or len(set(args.gpu_ids)) != 2 or not all(i.isdigit() for i in args.gpu_ids):
        parser.error("--gpus must contain two distinct numeric physical GPU indices")
    if len(set(args.seeds)) != len(args.seeds): parser.error("Duplicate seeds")
    if len(set(args.methods)) != len(args.methods): parser.error("Duplicate methods")
    check_source()
    check_patch()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if not args.dry_run:
        mapping = configure_egl_for_visible_cuda()
        args.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        print(f"Render GPU mapping: {mapping}", flush=True)
    # Preparation downloads everything. Reproduction must not resolve moving hub branches.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTHONPATH"] = str(ROOT / "scripts") + os.pathsep + os.environ.get("PYTHONPATH", "")
    storage = Path(os.environ["STORAGE_ROOT"])
    configured_output = json.loads((CONFIGS/"pretrain_seed_0.json").read_text())["output_dir"]
    if Path(configured_output) != storage/"outputs/pretrain/libero_pretrain_leak3/seed_0":
        raise RuntimeError("STORAGE_ROOT differs from the prepared configs. Use the storage path recorded in configs/reproduction/pretrain_seed_0.json.")
    logs = storage / "logs/reproduction"
    if not args.dry_run:
        logs.mkdir(parents=True, exist_ok=True)
        lock_handle = (logs/"launcher.lock").open("w")
        try: fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("Another reproduction launcher is already running in this storage directory")
        args.storage_policy = StoragePolicy.create(storage, budget_gb=args.budget_gb, retention=args.retention)
        os.environ[POLICY_ENV] = str(args.storage_policy.path)
        provenance_files = [ROOT / name for name in ("assets.lock.json", "uv.lock", "patches/upstream-compat.patch")]
        provenance_files += list(CONFIGS.glob("*.json")) + list((ROOT / "scripts").glob("*.py"))
        atomic_json(logs / "launch_manifests" / f"{time.time_ns()}.json", {
            "stage": args.stage, "gpus": args.gpus, "methods": args.methods, "seeds": args.seeds,
            "retention": args.retention, "budget_gb": args.budget_gb,
            "sha256": {str(path.relative_to(ROOT)): sha256(path) for path in provenance_files},
        })
    print(f"Methods: {', '.join(args.methods)}; seeds: {args.seeds}; "
          f"retention: {args.retention}; artifact budget: {args.budget_gb:g} GB", flush=True)
    if args.retention == "rolling":
        print("Keep six starting policies, Random updates 5/10/15, final VFD pairs; "
              "evaluate every round before retirement. Expected allocation ~55 GB, not a fixed size.", flush=True)
    if args.stage in ("verify", "all"):
        run([PYTHON, str(ROOT/"scripts/verify_environment.py")], logs/"environment.log", args)
    if args.stage in ("pretrain", "all"):
        pretrain(args, storage, logs)
    if args.stage in ("run", "evaluate", "all"):
        train_and_evaluate(args, storage, logs, train=args.stage != "evaluate")
    if args.stage == "calibrate" or (args.stage == "all" and "calibration_random" in args.methods):
        calibrate(args, storage, logs)
    if args.stage in ("run", "evaluate", "report", "all"):
        report(args, storage, logs)
    print(f"Finished {args.stage}. Results: {storage/'outputs/reproduction'}", flush=True)

if __name__ == "__main__":
    main()
