"""Exercise checkpoint loss prevention, retired-run resumption and disk guards."""
import json
import os
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from reproduction_storage import (
    GB, POLICY_ENV, StoragePolicy, atomic_json, checkpoint_destination,
    checkpoint_published, read_json, set_last, validate_checkpoint,
)
from reproduction_rounds import RoundRetention, validate_evaluation


def tensor_file(path):
    header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1))


def checkpoint_files(path, step=4000):
    model, state = path / "pretrained_model", path / "training_state"
    model.mkdir(parents=True)
    state.mkdir()
    for name in ("config.json", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        atomic_json(model / name, {"steps": []})
    tensor_file(model / "model.safetensors")
    for name in ("optimizer_state.safetensors", "rng_state.safetensors"):
        tensor_file(state / name)
    for name in ("optimizer_param_groups.json", "scheduler_state.json"):
        atomic_json(state / name, {})
    atomic_json(state / "training_step.json", {"step": step})


@pytest.fixture
def policy(tmp_path, monkeypatch):
    policy = StoragePolicy.create(tmp_path / "storage")
    monkeypatch.setenv(POLICY_ENV, str(policy.path))
    return policy


def pretrain_path(policy, step):
    return policy.storage / f"outputs/pretrain/libero_pretrain_leak3/seed_0/checkpoints/{step:06d}"


def publish(path, step):
    with checkpoint_destination(path) as pending:
        checkpoint_files(pending, step)
    set_last(path)
    checkpoint_published(path)


def test_pretrain_retires_only_after_replacement_and_keeps_final_weights(policy):
    first, second, final = [pretrain_path(policy, step) for step in (5000, 10000, 30000)]
    publish(first, 5000)
    assert validate_checkpoint(first, resumable=True) == 5000
    with checkpoint_destination(second) as pending:
        checkpoint_files(pending, 10000)
        assert (first / "training_state/optimizer_state.safetensors").exists()
        assert (first.parent / "last").resolve() == first
    # Simulate an exit between committing the checkpoint and updating the last link.
    policy.maintain_pretrain(first.parent.parent)
    assert (first.parent / "last").resolve() == second
    assert not (first / "pretrained_model").exists()
    assert (first / "checkpoint_inventory.json").exists()
    assert validate_checkpoint(second, resumable=True) == 10000
    publish(final, 30000)
    policy.maintain_pretrain(final.parent.parent, finalize_at=30000)
    policy.maintain_pretrain(final.parent.parent, finalize_at=30000)
    assert validate_checkpoint(final) == 30000
    assert not (final / "training_state/optimizer_state.safetensors").exists()
    assert (final / "training_state/rng_state.safetensors").exists()


def test_failed_save_leaves_previous_resume_point_untouched(policy):
    first, second = pretrain_path(policy, 5000), pretrain_path(policy, 10000)
    publish(first, 5000)
    with pytest.raises(OSError, match="interrupted"):
        with checkpoint_destination(second) as pending:
            (pending / "partial").write_bytes(b"partial checkpoint")
            raise OSError("interrupted write")
    assert not second.exists()
    assert not second.with_name(".pending-010000").exists()
    assert (first.parent / "last").resolve() == first
    assert validate_checkpoint(first, resumable=True) == 5000


def test_recovers_partial_write_left_by_killed_worker(policy):
    path = pretrain_path(policy, 5000)
    pending = path.with_name(".pending-005000")
    atomic_json(pending / "transaction.json", {"target": str(path)})
    (pending / "partial").write_bytes(b"partial")
    publish(path, 5000)
    assert validate_checkpoint(path, resumable=True) == 5000


def test_budget_stops_before_write_and_excludes_shared_inputs(policy):
    shared = policy.storage / "datasets/large"
    shared.parent.mkdir()
    shared.write_bytes(b"shared" * 100)
    assert policy.usage() < shared.stat().st_size
    policy.data["budget_bytes"] = 4 * GB
    atomic_json(policy.path, policy.data)
    path = pretrain_path(policy, 5000)
    with pytest.raises(RuntimeError, match="storage guard"):
        with checkpoint_destination(path):
            pytest.fail("Write started without headroom")
    assert not path.exists()
    assert shared.exists()


def test_refuses_to_prune_external_and_symlinked_artifacts(policy, tmp_path):
    outside = tmp_path / "unrelated"
    checkpoint_files(outside)
    with pytest.raises(RuntimeError, match="outside"):
        policy.retire(outside, weights=True, reason="test")
    path = pretrain_path(policy, 5000)
    path.parent.mkdir(parents=True)
    path.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError):
        policy.retire(path, weights=True, reason="test")
    assert (outside / "pretrained_model/model.safetensors").exists()


def test_truncated_legacy_checkpoint_cannot_replace_last(policy):
    first, second = pretrain_path(policy, 5000), pretrain_path(policy, 10000)
    publish(first, 5000)
    checkpoint_files(second, 10000)
    tensor = second / "pretrained_model/model.safetensors"
    tensor.write_bytes(tensor.read_bytes()[:-1])
    set_last(second)  # upstream legacy behavior; corruption must not trigger pruning
    with pytest.raises(RuntimeError, match="Truncated"):
        policy.maintain_pretrain(first.parent.parent)
    assert validate_checkpoint(first, resumable=True) == 5000


def test_partial_legacy_save_is_removed_only_after_validating_resume_point(policy):
    first, second = pretrain_path(policy, 5000), pretrain_path(policy, 10000)
    # No transaction/inventory markers: these are old-launcher checkpoints.
    checkpoint_files(first, 5000)
    set_last(first)
    checkpoint_files(second, 10000)
    (second / "training_state/optimizer_state.safetensors").write_bytes(b"partial")
    assert policy.maintain_pretrain(first.parent.parent) == 5000
    assert not second.exists()
    assert validate_checkpoint(first, resumable=True) == 5000
    assert (first.parent / "discarded_010000.json").exists()


def test_interrupted_run_state_save_keeps_previous_committed_history(policy, monkeypatch):
    from iterative_fine_tuning.manifests import save_dataclass_json
    path = policy.storage / "outputs/reproduction/vfd/seed_1000/run_state.json"
    save_dataclass_json({"rounds": [0, 1]}, path)
    def fail(data, handle, **kwargs):
        handle.write('{"rounds":')
        raise OSError("interrupted manifest write")
    monkeypatch.setattr(json, "dump", fail)
    with pytest.raises(OSError, match="interrupted"):
        save_dataclass_json({"rounds": [0, 1, 2]}, path)
    assert read_json(path) == {"rounds": [0, 1]}


def evaluation_data(index):
    return dict(round_index=index, member_index=0, policy_type="smolvla", env_type="libero",
                model_path="synthetic-test-model", device="cuda", macro_avg_success_rate=0.,
                macro_avg_reward=0., macro_avg_steps=520., total_successes=0,
                pooled_success_rate=0., pooled_success_rate_ci_lower=0., pooled_success_rate_ci_upper=0.,
                eval_batch_size=15, n_rollouts_per_task=30, seed_start=100, seed_step=1,
                max_steps=520, use_async_envs=False, num_tasks=10, total_rollouts=300,
                per_task=[dict(task_group="libero_10", task_id=i, n_rollouts=30, task_instruction="test",
                               seed_start=100, seed_step=1, successes=0, success_rate=0.,
                               success_rate_ci_lower=0., success_rate_ci_upper=0., avg_steps=520., avg_reward=0.,
                               per_rollout_results=[{} for _ in range(30)]) for i in range(10)])


def make_round(manager, index, *, evaluated=True):
    directory = manager.round_dir(index)
    models = []
    for member in (0, 1):
        path = directory / f"training/member_{member:02d}/checkpoints/004000"
        publish(path, 4000)
        models.append(str(path.parent / "last/pretrained_model"))
    for name in ("selection_manifest.json", "candidate_scores.json", "training_manifest.json"):
        atomic_json(directory / name, {"round_index": index})
    atomic_json(directory / "training_manifest.json", dict(round_index=index, final_model_paths=models,
                                                           start_model_paths=models))
    if evaluated:
        atomic_json(directory / "training/member_00/member_evaluation.json", evaluation_data(index))
    record = dict(round_index=index, final_model_paths=models, completed_stages=["select", "train"],
                  selection_manifest_path=str(directory / "selection_manifest.json"),
                  training_manifest_path=str(directory / "training_manifest.json"))
    state_path = manager.root / "run_state.json"
    state = read_json(state_path) if state_path.exists() else {"rounds": []}
    state["rounds"].append(record)
    atomic_json(state_path, state)
    return record


@pytest.mark.parametrize("method,keep", [("random", {4, 9, 14}), ("vfd", {14})])
def test_all_fifteen_rounds_keep_milestones_and_resume_after_retirement(policy, method, keep):
    manager = RoundRetention(policy, policy.storage / f"outputs/reproduction/{method}/seed_1000", method)
    for index in range(15):
        record = make_round(manager, index)
        manager.commit_and_prune()
        manager.validate_resume()
        # The latest pair remains usable until its successor has been committed.
        for path in manager.models(record):
            validate_checkpoint(path)
    for record in read_json(manager.root / "run_state.json")["rounds"]:
        for path in manager.models(record):
            assert (path / "pretrained_model").exists() == (record["round_index"] in keep)
            assert not (path / "training_state/optimizer_state.safetensors").exists()
            assert (path / "checkpoint_inventory.json").exists()
        assert Path(record["selection_manifest_path"]).exists()
        assert manager.verify_commit(manager.round_dir(record["round_index"]))
    # A repeat of the completed workflow is idempotent.
    manager.commit_and_prune()
    manager.validate_resume()


def test_evaluation_failure_and_corrupted_evidence_prevent_retirement(policy):
    manager = RoundRetention(policy, policy.storage / "outputs/reproduction/vfd/seed_1000", "vfd")
    first = make_round(manager, 0)
    manager.commit_and_prune()
    second = make_round(manager, 1, evaluated=False)
    with pytest.raises(FileNotFoundError):
        manager.commit_and_prune()
    assert all((p / "pretrained_model").exists() for p in manager.models(first))
    atomic_json(manager.round_dir(1) / "training/member_00/member_evaluation.json", evaluation_data(1))
    manager.commit_and_prune()
    Path(first["selection_manifest_path"]).write_text("{}")
    with pytest.raises(RuntimeError, match="evidence missing or changed"):
        manager.validate_resume()
    assert all((p / "pretrained_model").exists() for p in manager.models(second))


def test_changed_evaluation_recipe_cannot_be_used_for_pruning(policy):
    manager = RoundRetention(policy, policy.storage / "outputs/reproduction/vfd/seed_1000", "vfd")
    record = make_round(manager, 0)
    path = manager.round_dir(0) / "training/member_00/member_evaluation.json"
    data = read_json(path)
    data["n_rollouts_per_task"] = 1
    atomic_json(path, data)
    with pytest.raises(RuntimeError, match="reference recipe"):
        manager.commit_and_prune()
    assert all((p / "training_state/optimizer_state.safetensors").exists() for p in manager.models(record))


def test_retry_uncommitted_round_keeps_previous_models(policy):
    manager = RoundRetention(policy, policy.storage / "outputs/reproduction/vfd/seed_1000", "vfd")
    previous = make_round(manager, 0)
    manager.commit_and_prune()
    failed = manager.round_dir(1) / "training/member_00/checkpoints/004000"
    publish(failed, 4000)
    record = SimpleNamespace(completed_stages=["select"], start_model_paths=previous["final_model_paths"])
    manager.before_round(None, SimpleNamespace(get_round=lambda _: record), 1)
    assert not failed.exists()
    assert all((p / "pretrained_model").exists() for p in manager.models(previous))
    assert (manager.round_dir(1) / "training/member_00/discarded_checkpoint_004000.json").exists()


def test_real_optimizer_resume_through_patched_upstream_save(policy):
    import torch
    from safetensors.torch import save_file
    from lerobot.utils.train_utils import save_checkpoint, update_last_checkpoint, load_training_state

    class TinyPolicy(torch.nn.Linear):
        def save_pretrained(self, path):
            path.mkdir(parents=True)
            save_file(self.state_dict(), str(path / "model.safetensors"))
            for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
                atomic_json(path / name, {"steps": []})

    model = TinyPolicy(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
    def step():
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
    step()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = SimpleNamespace(save_pretrained=lambda path: atomic_json(path / "train_config.json", {}))
    checkpoint = pretrain_path(policy, 5000)
    save_checkpoint(checkpoint, 5000, cfg, model, optimizer, scheduler)
    update_last_checkpoint(checkpoint)
    step()
    continued = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(expected)
    loaded_step, optimizer, scheduler = load_training_state(checkpoint, optimizer, scheduler)
    step()
    assert loaded_step == 5000
    for key in continued:
        torch.testing.assert_close(model.state_dict()[key], continued[key], rtol=0, atol=0)


def test_upstream_round_callbacks_preserve_selections_and_do_not_retrain_retired_rounds(policy, monkeypatch):
    from iterative_fine_tuning import main as upstream
    from iterative_fine_tuning.manifests import (
        EpisodeSelection, SelectionManifest, TrainingManifest, save_dataclass_json,
    )
    initial = pretrain_path(policy, 30000)
    publish(initial, 30000)
    manager = RoundRetention(policy, policy.storage / "outputs/reproduction/vfd/seed_1000", "vfd")
    cfg = SimpleNamespace(
        run_dir=lambda: manager.root, paths=SimpleNamespace(run_name="seed_1000"),
        execution=SimpleNamespace(resume=True, overwrite_existing_round=False, stages=["select", "train"]),
        iteration=SimpleNamespace(total_rounds=3, warm_start_from_previous_round=True),
        bootstrap_selection_paths=lambda: [],
        resolved_ensemble_model_paths=lambda: [str(initial / "pretrained_model")] * 2,
    )
    monkeypatch.setattr(upstream, "_save_config_snapshot", lambda cfg, path: atomic_json(path, {}))
    calls = []
    def select(cfg, *, round_index, round_layout, **kwargs):
        calls.append(("select", round_index))
        manifest = SelectionManifest(round_index=round_index,
                                     selected_episodes=[EpisodeSelection(episode_id=round_index)])
        save_dataclass_json(manifest, round_layout.selection_manifest_path)
        atomic_json(round_layout.candidate_scores_path, {"test": True})
        return manifest
    def train(cfg, *, round_index, model_references, selected_episode_ids, round_layout):
        calls.append(("train", round_index))
        assert selected_episode_ids == list(range(round_index + 1))
        models = []
        for member in (0, 1):
            path = round_layout.training_dir / f"member_{member:02d}/checkpoints/004000"
            publish(path, 4000)
            models.append(str(path.parent / "last/pretrained_model"))
        manifest = TrainingManifest(round_index=round_index, start_model_paths=model_references,
                                    final_model_paths=models)
        save_dataclass_json(manifest, round_layout.training_manifest_path)
        return manifest
    def after(cfg, state, index):
        path = manager.round_dir(index) / "training/member_00/member_evaluation.json"
        if not path.exists():
            atomic_json(path, evaluation_data(index))
        manager.commit_and_prune()
    monkeypatch.setattr(upstream, "run_selection_round", select)
    monkeypatch.setattr(upstream, "run_training_round", train)
    upstream._run_single(cfg, before_round=manager.before_round, after_round=after)
    expected = [(stage, index) for index in range(3) for stage in ("select", "train")]
    assert calls == expected
    manager.validate_resume()
    upstream._run_single(cfg, before_round=manager.before_round, after_round=after)
    assert calls == expected


def test_upstream_evaluator_rebuilds_complete_curve_after_weights_are_retired(policy, monkeypatch):
    from iterative_fine_tuning import evaluate_run
    from iterative_fine_tuning.manifests import TaskSpec
    from reproduction_rounds import evaluation_command
    manager = RoundRetention(policy, policy.storage / "outputs/reproduction/vfd/seed_1000", "vfd")
    initial = pretrain_path(policy, 30000)
    publish(initial, 30000)
    atomic_json(manager.root / "iterative_fine_tuning_config.json", {
        "dataset": {"libero_tasks": {"libero_10": list(range(10))}}, "policy": {"type": "smolvla"},
        "selection": {"ensemble_model_paths": [str(initial / "pretrained_model")]},
    })
    atomic_json(manager.root / "initial_evaluation/member_00/member_evaluation.json", evaluation_data(-1))
    records = [make_round(manager, index) for index in (0, 1)]
    manager.commit_and_prune()
    assert not (manager.models(records[0])[0] / "pretrained_model").exists()
    monkeypatch.setattr(evaluate_run, "_task_specs_from_config", lambda *a, **kw: [
        TaskSpec(task_group="libero_10", task_id=i, task_instruction="test") for i in range(10)])
    monkeypatch.setattr(evaluate_run, "_launch_member_evaluation",
                        lambda *a, **kw: pytest.fail("Attempted to load retired weights instead of cached outcomes"))
    monkeypatch.setattr(sys, "argv", ["evaluate_run"] + evaluation_command(manager.root)[3:])
    evaluate_run.main()
    assert [r["round_index"] for r in read_json(manager.root / "run_evaluation.json")["rounds"]] == [-1, 0, 1]
    manager.commit_and_prune()
    manager.validate_resume()
