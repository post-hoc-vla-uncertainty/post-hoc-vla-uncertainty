from pathlib import Path
import sys

import pytest

from uq_extensions.common import atomic_json, file_hash, identity, model_identity, plain, read_json
from uq_extensions.config import ExperimentConfig
from uq_extensions.reevaluate_preview import execute, main, preflight


@pytest.fixture
def preview(tmp_path, monkeypatch):
    import uq_extensions.runner as runner
    from uq_extensions.matrix import reference_recipes
    monkeypatch.setattr(runner, 'source_identity', lambda: 'corrected-code')
    root = tmp_path / 'workspace'
    base, trained, teacher = [tmp_path / name for name in ('base', 'trained', 'teacher')]
    for path in (base, trained, teacher):
        path.mkdir()
        (path / 'model.safetensors').write_bytes(path.name.encode())
    data = tmp_path / 'data.json'
    atomic_json(data, dict(manifest_hash='m', evaluator_directory=str(tmp_path),
                          observations=[dict(id='d', task='a')]))
    atomic_json(tmp_path / 'trajectory_index.json', [dict(id='d', dataset_from_index=0, dataset_to_index=10)])
    cfg = ExperimentConfig(name='preview', phase='smoke', arm='kv', rounds=1, fit_steps=100,
                           output_root=str(root / 'runs'), base_models=[str(base)], data_manifest=str(data))
    run = cfg.run_dir()
    atomic_json(run / 'experiment.json', dict(identity='old-training-identity', config=cfg.scientific_dict(),
                base_models=[str(base)], base_hashes=[model_identity(base)], manifest_hash='m'))
    atomic_json(run / 'complete.json', dict(models=[str(trained)], rounds=1))
    atomic_json(run / 'round_001/policy_fit.json', dict(models=[str(trained)], model_hashes=[model_identity(trained)],
                source_manifest='m', acquired_ids=['d'], seconds=12, requested_steps_per_member=100))
    atomic_json(run / 'acquisition_ledger.json', dict(pretraining_ids=[], rounds=[dict(ids=['d'])]))
    atomic_json(root / 'supervisor.json', dict(model=str(teacher), model_hash=model_identity(teacher)))
    result = dict(run=str(run), supervisor=str(teacher), report='original.html', matrix_hash='old')
    atomic_json(root / '01_preview/result.json', result)
    atomic_json(root / 'decisions/01_outputs.json', dict(evidence_hash=identity(result), outputs_ready=True))
    atomic_json(root / '01_preview/scoring/declaration.json', dict(config=plain(cfg), run=str(run), rounds=[0, 1],
                supervisor=str(teacher), condition=None, split=cfg.evaluation_split, recipes=reference_recipes(preview=True)))
    atomic_json(run / 'summary.json', dict(original=True))
    for r in (0, 1):
        atomic_json(run / f'round_{r:03d}/evaluation/evaluation.json', dict(original=True))
    return root, run, trained


def snapshot(root):
    return {str(p): file_hash(p) for p in root.rglob('*') if p.is_file()}


def test_plan_checks_real_checkpoint_files_without_gpu_or_writes(preview, monkeypatch):
    import uq_extensions.cli as cli
    root, _, _ = preview
    before = snapshot(root.parent)
    monkeypatch.setattr(cli, 'device_setup', lambda *a: pytest.fail('GPU initialization in --plan'))
    monkeypatch.setattr(sys, 'argv', ['reevaluate_preview', '--root', str(root), '--plan'])
    main()
    assert snapshot(root.parent) == before


def test_missing_or_modified_policy_never_triggers_training(preview):
    root, _, trained = preview
    weights = trained / 'model.safetensors'
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError, match='learner_after checkpoint changed'):
        preflight(root)
    weights.unlink()
    with pytest.raises(FileNotFoundError, match='No policy weights'):
        preflight(root)


def test_recovery_uses_saved_settings_preserves_history_and_resets_gate(preview, monkeypatch):
    import uq_extensions.acquisition as acquisition
    import uq_extensions.training as training
    import uq_extensions.supervisor as supervisor
    import uq_extensions.runner as runner
    import uq_extensions.matrix as matrix
    import uq_extensions.reporting as reporting
    from uq_extensions.evaluation import OBSERVATION_PREPROCESSING
    root, run, _ = preview
    before = snapshot(run)
    def forbidden(*args, **kwargs):
        pytest.fail('Reevaluation invoked training/acquisition')
    for module, name in [(runner, 'run'), (training, 'fit_policies'), (supervisor, 'train_supervisor'), (acquisition, 'acquire')]:
        monkeypatch.setattr(module, name, forbidden)
    def score(cfg, source, rounds, recipes, out, **kwargs):
        assert Path(source) == run and rounds == (0, 1)
        assert cfg.fit_steps == 100 and len(recipes) == 4
        for r in rounds:
            atomic_json(out / f'round_{r:03d}/learner/evaluation.json', dict(
                spec={'observation_preprocessing': OBSERVATION_PREPROCESSING},
                episodes=[dict(task='a', success=bool(r), q_time=0., q_path=0., relative_time=None, relative_path=None)]))
        return [dict(round=r, recipe=recipe, status='complete', metrics={'referral_global': {'supervisor_mean': 1.}})
                for r in rounds for recipe in recipes]
    monkeypatch.setattr(matrix, 'run_matrix', score)
    def report(out, **kwargs):
        assert kwargs['run_summaries'][str(run)]['final_success'] == 1.
        return out / 'report.html'
    monkeypatch.setattr(reporting, 'experiment_report', report)
    plan = preflight(root)
    result = execute(plan, device='cpu')
    assert snapshot(run) == before
    assert result['policy_training_updates'] == 0
    assert read_json(plan['output'] / 'active_learning/summary.json')['curve'][0]['success'] == 0.
    assert read_json(plan['output'] / 'active_learning/summary.json')['curve'][1]['success'] == 1.
    assert read_json(plan['output'] / 'previous_decision.json')['outputs_ready'] is True
    decision = read_json(root / 'decisions/01_outputs.json')
    assert decision['outputs_ready'] is None and decision['evidence_hash'] == identity(result)
    assert not (root / '02_histories').exists()
    # Completed reruns remain resumable and do not erase a review of identical evidence.
    atomic_json(root / 'decisions/01_outputs.json', dict(decision, outputs_ready=True))
    execute(preflight(root), device='cpu')
    assert read_json(root / 'decisions/01_outputs.json')['outputs_ready'] is True
    assert snapshot(run) == before


def test_interruption_leaves_old_results_intact_and_decision_pending(preview, monkeypatch):
    import uq_extensions.matrix as matrix
    root, run, _ = preview
    before = snapshot(run)
    original_result = read_json(root / '01_preview/result.json')
    def fail(*a, **kw):
        raise RuntimeError('interrupted GPU work')
    monkeypatch.setattr(matrix, 'run_matrix', fail)
    plan = preflight(root)
    with pytest.raises(RuntimeError, match='interrupted GPU'):
        execute(plan, device='cpu')
    assert read_json(root / 'decisions/01_outputs.json')['outputs_ready'] is None
    assert read_json(root / '01_preview/result.json') == original_result
    assert snapshot(run) == before
    assert preflight(root)['request'] == plan['request']
