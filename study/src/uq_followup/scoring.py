"""One frozen-policy head job; simulator outcomes are read-only inputs."""
from pathlib import Path
import time
import numpy as np
from uq_extensions.common import atomic_json, identity, read_json
from uq_extensions.config import FeatureConfig
from uq_extensions.data import candidate_view, load_manifest
from uq_extensions.protocol import Candidate, CandidateView
from uq_extensions.baselines import baseline_scores, METHODS
from uq_extensions.preprocessing import UninformativeBank
from .design import MAPPINGS
from .features import cache_bank
from .estimators import Estimator


def score(study, spec, *, device, feature_batch_size=4):
    study = Path(study); started = time.monotonic()
    row = spec['recipe']; out = Path(spec['directory']); out.mkdir(parents=True, exist_ok=True)
    policy_id = spec['policy_id']
    if (out/'scores.json').exists() and (out/'diagnostics.json').exists():
        # A plotting interruption must not refit or rescore a completed head.
        diagnostics = read_json(out/'diagnostics.json')
        status = 'mapping_infeasible' if diagnostics.get('mapping_check', {}).get('usable') is False else 'complete'
        from .reporting import head_report
        metrics = head_report(study, policy_id, out)
        atomic_json(out/'result.json', dict(status=status, recipe=row, policy_id=policy_id,
            metrics=metrics, seconds=time.monotonic()-started, resumed_saved_scores=True))
        return
    declaration = read_json(study/'declaration.json')
    policy = next(p for p in declaration['policies'] if p['id'] == policy_id)
    manifest = load_manifest(declaration['manifest'])
    model = read_json(study/'policies'/policy_id/'model.json')['model']
    support = candidate_view(manifest, policy['ids'])
    pool = candidate_view(manifest, manifest['target_ids'])
    query = CandidateView([Candidate(**r) for r in read_json(study/'evaluations'/policy_id/'initial_inputs.json')])
    diagnostics = {}; status = 'complete'
    if row['method'] in METHODS:
        companion = spec['companion_id']
        models = [model, read_json(study/'policies'/companion/'model.json')['model']]
        scores = baseline_scores(row['method'], models, query, out/'scores_cache', device=device, seed=42)
        pooled = baseline_scores(row['method'], models, pool, out/'pool_cache', device=device, seed=42)
        values = np.asarray([scores[i] for i in query.ids]); pool_values = np.asarray([pooled[i] for i in pool.ids])
    else:
        features = FeatureConfig(**row['features'])
        target_needed = row['method'] == 'kv_t' or row['group'] in MAPPINGS
        banks = (features.bank, features.target_bank) if target_needed else (features.bank,)
        cache = study/'features'
        training = cache_bank(cache, model, support, features, banks, role='support', device=device, batch_size=feature_batch_size)
        query_features = cache_bank(cache, model, query, features, (features.bank,), role='query', device=device, batch_size=feature_batch_size)
        pool_features = cache_bank(cache, model, pool, features, (features.bank,), role='pool', device=device, batch_size=feature_batch_size)
        directory = out/'estimator'
        try:
            if (directory/'estimator_fit.json').exists() and (directory/'estimator.npz').exists():
                estimator = Estimator.load(row, directory, device)
            else:
                estimator = Estimator(row, device).fit(training[features.bank], training.get(features.target_bank), support.tasks, directory)
            values = estimator.score(query_features[features.bank])
            pool_values = estimator.score(pool_features[features.bank])
            diagnostics = estimator.diagnostics
            if diagnostics.get('mapping_check', {}).get('usable') is False: status = 'mapping_infeasible'
        except UninformativeBank as error:
            atomic_json(out/'result.json', dict(status='uninformative', reason=str(error), recipe=row,
                policy_id=policy_id, seconds=time.monotonic()-started))
            return
    if not np.isfinite(values).all() or not np.isfinite(pool_values).all():
        raise RuntimeError('Nonfinite uncertainty scores')
    pool_means = {task: float(np.mean(pool_values[np.asarray(pool.tasks) == task])) for task in sorted(set(pool.tasks))}
    atomic_json(out/'scores.json', dict(ids=query.ids, scores=values.tolist(), tasks=query.tasks, pool_task_means=pool_means))
    atomic_json(out/'diagnostics.json', diagnostics)
    from .reporting import head_report
    metrics = head_report(study, policy_id, out)
    atomic_json(out/'result.json', dict(status=status, recipe=row, policy_id=policy_id,
        metrics=metrics, seconds=time.monotonic()-started))
    from run_progress import emit
    emit('head', f'DONE: {policy_id} / {row["id"]}; {status}; wall={(time.monotonic()-started)/60:.1f} min; '
          f'nAUC={metrics["global_nauc"]}; -Spearman={metrics["neg_spearman"]}; -Pearson={metrics["neg_pearson"]}; report={out/"report.html"}')
