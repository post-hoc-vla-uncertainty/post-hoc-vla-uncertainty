"""Evaluate and commit each upstream round before applying rolling retention."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import logging
import time
from run_progress import emit,duration

from reproduction_storage import (
    GB, POLICY_ENV, StoragePolicy, atomic_json, files_under, read_json, sha256,
    validate_checkpoint,
)


def evaluation_command(run_dir):
    return [sys.executable, "-m", "iterative_fine_tuning.evaluate_run",
            "--run-dir", str(run_dir), "--members", "0", "--member-devices", "cuda:0,cuda:1",
            "--workers-per-device", "1", "--n-rollouts", "30", "--eval-batch-size", "15",
            "--max-steps", "520", "--seed-start", "100", "--sync-envs", "--skip-existing"]


def validate_evaluation(path, round_index):
    data = read_json(path)
    expected = dict(round_index=round_index, member_index=0, policy_type="smolvla", env_type="libero",
                    eval_batch_size=15, n_rollouts_per_task=30, seed_start=100, seed_step=1,
                    max_steps=520, use_async_envs=False, num_tasks=10, total_rollouts=300)
    if any(data.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Evaluation does not match the reference recipe: {path}")
    tasks = data.get("per_task", [])
    if {(t["task_group"], t["task_id"]) for t in tasks} != {("libero_10", i) for i in range(10)}:
        raise RuntimeError(f"Incomplete task evaluation: {path}")
    if len(tasks) != 10 or any(t["n_rollouts"] != 30 or len(t["per_rollout_results"]) != 30 for t in tasks):
        raise RuntimeError(f"Incomplete rollout outcomes: {path}")


class RoundRetention:
    def __init__(self, policy, run_dir, method):
        self.policy = policy
        self.root = Path(run_dir)
        self.method = method
        self.started=time.monotonic();self.round_started={}

    def round_dir(self, index):
        return self.root / f"round_{index:03d}"

    def models(self, record):
        paths = record.get("final_model_paths", [])
        if len(paths) != 2:
            raise RuntimeError("Expected two completed ensemble members")
        checkpoints = [self.policy.owned_checkpoint(Path(path).resolve().parent) for path in paths]
        for member, checkpoint in enumerate(checkpoints):
            expected = self.round_dir(record["round_index"]) / f"training/member_{member:02d}/checkpoints"
            if checkpoint.parent != expected.absolute():
                raise RuntimeError(f"Model belongs to another run or round: {checkpoint}")
        return checkpoints

    def verify_commit(self, directory):
        path = directory / "reproduction_commit.json"
        if not path.exists():
            return False
        commit = read_json(path)
        for relative, digest in commit["evidence"].items():
            artifact = directory / relative
            if not artifact.is_file() or sha256(artifact) != digest:
                raise RuntimeError(f"Committed reproduction evidence missing or changed: {artifact}")
        validate_evaluation(directory / "training/member_00/member_evaluation.json", commit["round_index"])
        return True

    def validate_resume(self):
        state_path = self.root / "run_state.json"
        if not state_path.exists():
            return
        records = read_json(state_path)["rounds"]
        trained = [r for r in records if "train" in r["completed_stages"]]
        latest = max((r["round_index"] for r in trained), default=-1)
        for record in records:
            directory = self.round_dir(record["round_index"])
            for stage, key in (("select", "selection_manifest_path"), ("train", "training_manifest_path")):
                if stage in record["completed_stages"] and (not record.get(key) or not Path(record[key]).is_file()):
                    raise RuntimeError(f"Missing {stage} manifest; refusing to redo a possibly retired round: {directory}")
            committed = self.verify_commit(directory)
            if "train" not in record["completed_stages"]:
                continue
            for checkpoint in self.models(record):
                status_path = checkpoint / "retention.json"
                retired = status_path.exists() and read_json(status_path).get("weights_retired")
                if retired:
                    if not committed or record["round_index"] == latest:
                        raise RuntimeError(f"Required starting weights were retired: {checkpoint}")
                else:
                    validate_checkpoint(checkpoint)

    def before_round(self, cfg, state, round_index):
        self.round_started[round_index]=time.monotonic()
        record = state.get_round(round_index)
        iteration=getattr(cfg,'iteration',None)
        status='CACHED TRAINING' if record and 'train' in record.completed_stages else 'START'
        emit(f'{self.method}/round {round_index+1}/{getattr(iteration,"total_rounds","?")}',
             f'{status} (artifact round_{round_index:03d}); queries={getattr(iteration,"episodes_per_round","?")}, updates/member={getattr(cfg,"steps","?")}, evaluation=300 rollouts')
        if record and "train" in record.completed_stages:
            return
        for model in record.start_model_paths:
            validate_checkpoint(Path(model).resolve().parent)
        # An interrupted round restarts from its preceding models, as upstream does.
        # Remove only that uncommitted round's old output, preserving an audit.
        directory = self.round_dir(round_index)
        for member in (0, 1):
            parent = directory / f"training/member_{member:02d}/checkpoints"
            for checkpoint in sorted(parent.glob("[0-9]" * 6)):
                self.policy.owned_checkpoint(checkpoint)
                audit = directory / f"training/member_{member:02d}/discarded_checkpoint_{checkpoint.name}.json"
                atomic_json(audit, {"reason": "restart uncommitted round from previous policy",
                                    "files": {str(p.relative_to(checkpoint)): sha256(p) for p in files_under(checkpoint)}})
                shutil.rmtree(checkpoint)
            (parent / "last").unlink(missing_ok=True)
        self.policy.check_budget(reserve=8 * GB)

    def commit_and_prune(self):
        state = read_json(self.root / "run_state.json")
        records = [r for r in state["rounds"] if "train" in r["completed_stages"]]
        for record in records:
            index = record["round_index"]
            directory = self.round_dir(index)
            evaluation = directory / "training/member_00/member_evaluation.json"
            validate_evaluation(evaluation, index)
            if not self.verify_commit(directory):
                evidence = [directory / name for name in (
                    "selection_manifest.json", "candidate_scores.json", "training_manifest.json",
                    "training/member_00/member_evaluation.json",
                )]
                for checkpoint in self.models(record):
                    validate_checkpoint(checkpoint)
                atomic_json(directory / "reproduction_commit.json", {
                    "round_index": index,
                    "evidence": {str(p.relative_to(directory)): sha256(p) for p in evidence},
                    "purpose": "SAVE reference; not Table 1 calibration or unique_demo extension",
                })
        # Only retire once this round and its successor have both been evaluated.
        indices = {r["round_index"] for r in records}
        for record in records:
            index = record["round_index"]
            milestones = [4, 9, 14] if self.method == "random" else [14]
            retire_weights = index not in milestones and index + 1 in indices
            for checkpoint in self.models(record):
                self.policy.retire(checkpoint, weights=retire_weights,
                                   reason="evaluated round; retain milestones and latest warm-start pair")
        self.policy.check_budget(reserve=0)

    def evaluate(self):
        started=time.monotonic()
        emit(self.method,'EVALUATION starting; cached completed evaluations will be reused')
        self.validate_resume()
        subprocess.run(evaluation_command(self.root), check=True)
        # Upstream skip-existing also rebuilds the complete all-round summary.
        validate_evaluation(self.root / "initial_evaluation/member_00/member_evaluation.json", -1)
        self.commit_and_prune()
        state=read_json(self.root/'run_state.json')
        trained=[r for r in state['rounds'] if 'train' in r['completed_stages']]
        latest=max((r['round_index'] for r in trained),default=-1)
        path=(self.round_dir(latest)/'training/member_00/member_evaluation.json' if latest>=0 else self.root/'initial_evaluation/member_00/member_evaluation.json')
        data=read_json(path)
        successes=sum(sum(bool(v['success']) for v in task['per_rollout_results']) for task in data['per_task'])
        emit(self.method,f'EVALUATION DONE round={latest+1}; success={successes}/300 ({100*successes/300:.1f}%) | wall={duration(time.monotonic()-started)}')

    def after_round(self, cfg, state, round_index):
        self.evaluate()
        emit(f'{self.method}/round {round_index+1}/{cfg.iteration.total_rounds}',
             f'DONE wall={duration(time.monotonic()-self.round_started.get(round_index,self.started))}; evaluated checkpoints committed; retention applied')


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=["random", "vfd", "gu"], required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    from lerobot.utils.import_utils import register_third_party_plugins
    from iterative_fine_tuning.run_multi_seed_child import _load_cfg
    from iterative_fine_tuning.main import _build_child_cfg, _run_single
    register_third_party_plugins()
    parent = _load_cfg(args.config_path)
    spec = next(r for r in parent.multi_seed.iter_run_specs() if r.seed == args.seed)
    cfg = _build_child_cfg(parent, parent_dir=parent.run_dir(), run_spec=spec)
    cfg.validate()
    manager = RoundRetention(StoragePolicy(os.environ[POLICY_ENV]), cfg.run_dir(), args.method)
    manager.validate_resume()
    if args.evaluate_only:
        manager.evaluate()
    else:
        _run_single(cfg, before_round=manager.before_round, after_round=manager.after_round)


if __name__ == "__main__":
    main()
