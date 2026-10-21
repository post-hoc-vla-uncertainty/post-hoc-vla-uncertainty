"""Recompute preview evaluations and scoring from existing, verified policies.

This entry point never calls the policy trainer or acquisition runner. Original
evaluation artifacts remain in place; corrected results have their own directory.
"""
import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path

from .common import atomic_json, file_hash, identity, model_identity, plain, read_json
from .config import ExperimentConfig, FeatureConfig, GPConfig
from .evaluation import OBSERVATION_PREPROCESSING, RENDERING_PROTOCOL


def preflight(root, output=None):
    """Read and verify the completed preview before any GPU work or mutations."""
    from .runner import source_identity
    root = Path(root).resolve()
    original = root / '01_preview'
    declaration_path = original / 'scoring/declaration.json'
    declaration = read_json(declaration_path)
    values = dict(declaration['config'])
    values['features'] = FeatureConfig(**values['features'])
    values['gp'] = GPConfig(**values['gp'])
    cfg = ExperimentConfig(**values)
    cfg.validate()
    run = Path(declaration['run']).resolve()
    result = read_json(original / 'result.json')
    experiment = read_json(run / 'experiment.json')
    complete = read_json(run / 'complete.json')
    fit = read_json(run / 'round_001/policy_fit.json')
    ledger = read_json(run / 'acquisition_ledger.json')
    teacher = read_json(root / 'supervisor.json')
    manifest = read_json(cfg.data_manifest)
    if (cfg.phase != 'smoke' or cfg.rounds != 1 or cfg.policy_count != 1
            or declaration['rounds'] != [0, 1] or complete['rounds'] != 1):
        raise ValueError('This recovery command requires the completed one-round preview')
    if declaration.get('condition') is not None or declaration['split'] != cfg.evaluation_split:
        raise ValueError('Unexpected preview condition or evaluation split')
    if experiment['config'] != cfg.scientific_dict():
        raise ValueError('Saved preview configuration differs from the trained run')
    if Path(result['run']).resolve() != run or cfg.run_dir().resolve() != run:
        raise ValueError('Preview run paths disagree')
    if complete['models'] != fit['models'] or len(fit['models']) != 1:
        raise ValueError('Completed checkpoint differs from recorded policy fit')
    if len(ledger['rounds']) != 1 or fit['acquired_ids'] != ledger['rounds'][0]['ids']:
        raise ValueError('Recorded acquisition history differs from policy training')
    if not (manifest['manifest_hash'] == experiment['manifest_hash'] == fit['source_manifest']):
        raise ValueError('Training and scoring data manifests differ')
    if not (Path(declaration['supervisor']).resolve() == Path(teacher['model']).resolve()
            == Path(result['supervisor']).resolve()):
        raise ValueError('The fixed supervisor changed')
    models = {
        'learner_before': (experiment['base_models'][0], experiment['base_hashes'][0]),
        'learner_after': (fit['models'][0], fit['model_hashes'][0]),
        'supervisor': (teacher['model'], teacher['model_hash']),
    }
    verified = {}
    for role, (path, expected) in models.items():
        actual = model_identity(path)
        if actual != expected:
            raise ValueError(f'{role} checkpoint changed; refusing to score different weights: {path}')
        verified[role] = dict(path=str(Path(path).resolve()), model_hash=actual)
    output = Path(output).resolve() if output else original / 'reevaluations/libero_context_v3'
    # Never overlay a training directory or the original scoring/evaluation tree.
    if output == original or original not in output.parents or not output.is_relative_to(original / 'reevaluations'):
        raise ValueError('Output must be a new directory under 01_preview/reevaluations/')
    if output == original / 'reevaluations':
        raise ValueError('Choose a named reevaluation directory')
    source_files = [declaration_path, run / 'experiment.json', run / 'acquisition_ledger.json',
                    run / 'round_001/policy_fit.json', run / 'complete.json', Path(cfg.data_manifest), root / 'supervisor.json']
    request = dict(observation_preprocessing=OBSERVATION_PREPROCESSING, rendering_protocol=RENDERING_PROTOCOL, models=verified,
                   sources={str(p.resolve()): file_hash(p) for p in source_files},
                   source_hash=source_identity(), run=str(run), config=plain(cfg), recipes=declaration['recipes'])
    if (output / 'request.json').exists() and read_json(output / 'request.json') != request:
        raise ValueError('Reevaluation settings/code changed; choose another --output under 01_preview/reevaluations/')
    if output.exists() and any(output.iterdir()) and not (output / 'request.json').exists():
        raise ValueError('Reevaluation output is nonempty without a matching request')
    return dict(root=root, output=output, run=run, config=cfg, request=request,
                teacher=teacher['model'], recipes=declaration['recipes'])


def execute(plan, *, device):
    from .experiments import finish
    from .matrix import run_matrix
    from .reporting import experiment_report, report_run
    root, output, run = plan['root'], plan['output'], plan['run']
    cfg = plan['config']
    atomic_json(output / 'request.json', plan['request'])
    current_result = root / '01_preview/result.json'
    current_decision = root / 'decisions/01_outputs.json'
    for path, name in ((current_result, 'previous_result.json'), (current_decision, 'previous_decision.json')):
        if path.exists() and not (output / name).exists():
            atomic_json(output / name, read_json(path))
    # An interrupted reevaluation must not leave an old approval usable downstream.
    old_decision = read_json(current_decision) if current_decision.exists() else {}
    if not (output / 'result.json').exists():
        atomic_json(current_decision, dict(old_decision, outputs_ready=None,
                    notes=f'Reevaluation pending. Resume this command, then review {output / "report.html"}.'))
    print('EVALUATION ONLY: reuse three verified policy checkpoints and the recorded five-demo acquisition.', flush=True)
    for role, record in plan['request']['models'].items():
        print(f'  {role}: {record["path"]}', flush=True)
    print(f'Corrected output: {output}\nPolicy-training updates requested: 0', flush=True)
    # The source run is used only for immutable policies/history. Matrix evaluation
    # also supplies the learning curve, avoiding a duplicate learner rollout pass.
    rows = run_matrix(cfg, run, (0, 1), plan['recipes'], output / 'scoring',
                      device=device, supervisor=plan['teacher'], allow_nonrandom=True)
    evaluation_paths = {r: output / 'scoring' / f'round_{r:03d}/learner/evaluation.json' for r in (0, 1)}
    summary = report_run(run, evaluation_paths=evaluation_paths, output_root=output / 'active_learning')
    report = experiment_report(output, matrix=rows, runs=(run,), run_summaries={str(run): summary})
    result = dict(run=str(run), report=str(report), matrix_hash=identity(rows), smoke=True,
                  supervisor=plan['teacher'], observation_preprocessing=OBSERVATION_PREPROCESSING,
                  rendering_protocol=RENDERING_PROTOCOL,
                  reevaluation_request_hash=identity(plan['request']), reevaluation=str(output),
                  learning_curve=str(output / 'active_learning/learning_curve.csv'), policy_training_updates=0)
    atomic_json(output / 'result.json', result)
    finish(root, '01_preview', result)
    choice = read_json(current_decision) if current_decision.exists() else {}
    if choice.get('evidence_hash') != identity(result):
        atomic_json(current_decision, dict(evidence_hash=identity(result), outputs_ready=None,
                    notes=f'Review corrected evaluations and reports at {report}; previous results are retained.'))
    for row in summary['curve']:
        print(f'Round {row["round"]}: task-macro success={row["success"]:.1%}; '
              f'per-task={row["per_task_success"]}', flush=True)
    if rows:
        metrics = rows[0].get('metrics') or {}
        teacher_mean = metrics.get('referral_global', {}).get('supervisor_mean')
        if teacher_mean is not None:
            print(f'Supervisor task-macro success={teacher_mean:.1%}', flush=True)
        if teacher_mean == 0 and all(row['success'] == 0 for row in summary['curve']):
            print('All corrected rollouts still failed; inspect policy/environment behavior before interpreting referral.', flush=True)
    print(f'DONE: {report}\nSTOP: review {current_decision}; no next stage was started.', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='storage/extensions/experiments/v1')
    parser.add_argument('--output', help='Optional named directory under ROOT/01_preview/reevaluations/')
    parser.add_argument('--gpus', default='0', help='One available GPU is sufficient')
    parser.add_argument('--plan', action='store_true', help='Verify saved policies/settings without GPU work or output changes')
    parser.add_argument('--keep-feature-cache', action='store_true')
    parser.add_argument('--progress-seconds', type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.progress_seconds < float('inf'):
        parser.error('Progress interval must be positive and finite')
    if args.plan:
        plan = preflight(args.root, args.output)
        print(f'Verified three saved checkpoints. Policy training: 0 updates. Acquisition: reuse recorded IDs.\n'
              f'Evaluate learner before/after and supervisor; regenerate 8 scoring/referral cells.\n'
              f'Output: {plan["output"]}\nNo GPU work or output changes.')
        return
    root = Path(args.root).resolve()
    # Same lock as the numbered workflow, plus the lower-level source trajectory.
    with ExitStack() as stack:
        lock = stack.enter_context((root / 'workflow.lock').open('a'))
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another stage is using this workspace')
        declaration = read_json(root / '01_preview/scoring/declaration.json')
        run_lock = stack.enter_context((Path(declaration['run']) / 'run.lock').open('a'))
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('The source preview trajectory is still running')
        plan = preflight(root, args.output)
        from .cli import device_setup
        _, device = device_setup(args.gpus)
        os.environ['UQ_EXT_FEATURE_CACHE'] = str(root / 'feature_cache')
        os.environ['UQ_EXT_RETIRE_FEATURE_CACHE'] = '0' if args.keep_feature_cache else '1'
        os.environ['VLA_PROGRESS_SECONDS'] = str(args.progress_seconds)
        os.environ['PYTHONUNBUFFERED'] = '1'
        execute(plan, device=device)


if __name__ == '__main__':
    main()
