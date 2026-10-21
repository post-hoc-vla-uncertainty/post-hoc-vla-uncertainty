"""Capture completed artifacts and independently derive comparison metrics."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

from uq_extensions.common import identity
from uq_extensions.referral import PAIR_KEYS, pair_outcomes

POSITION_NAMES = dict(joint_intermediate='intermediate context', joint_late='final context',
    visual_local='early vision', visual_scene='late vision', visual_patches='visual patches',
    language='initial language', contextual_language='contextual language', contextual_vision='contextual vision',
    state='raw robot state', contextual_state_intermediate='intermediate fused state', contextual_state_late='final fused state',
    expert_0='final decoder', expert_1='decoder before last block', expert_3='decoder three blocks earlier',
    action_context='context + noisy action', final_action='context → final action')

def correlation(a, b, kind='spearman'):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.ptp(a) or not np.ptp(b):
        return None
    return float((spearmanr if kind == 'spearman' else pearsonr)(a, b).statistic)


def curve(scores, learner, supervisor, tasks, ids, allocation='global'):
    """Same tie rule and task weighting as referral_curve, without its bootstrap."""
    scores, learner, supervisor = (np.asarray(v, float) for v in (scores, learner, supervisor))
    tasks, ids = np.asarray(tasks), np.asarray(ids)
    if len(set(ids)) != len(ids) or not (len(scores) == len(learner) == len(supervisor) == len(tasks) == len(ids)):
        raise ValueError('Referral inputs must have matching unique episode IDs')
    if not all(np.isfinite(v).all() for v in (scores, learner, supervisor)):
        raise ValueError('Nonfinite referral inputs')
    names, counts = np.unique(tasks, return_counts=True)
    weights = np.array([1 / (len(names) * counts[np.flatnonzero(names == t)[0]]) for t in tasks])
    tie = dict(zip(sorted(ids), np.random.default_rng(42).random(len(ids))))
    order = sorted(range(len(ids)), key=lambda i: (scores[i], tie[ids[i]], ids[i]))
    indices = {t: np.array([i for i in order if tasks[i] == t]) for t in names}
    grids = np.arange(len(ids) + 1) / len(ids) if allocation == 'global' else sorted({j / len(ix) for ix in indices.values() for j in range(len(ix) + 1)})
    values = []
    for fraction in grids:
        mask = np.zeros(len(ids), bool)
        if allocation == 'global':mask[order[:round(fraction * len(ids))]] = True
        elif allocation == 'within_task':
            for ix in indices.values():mask[ix[:int(np.floor(fraction * len(ix) + 1e-9))]] = True
        else:raise ValueError('Unknown referral allocation')
        automated = float(weights @ mask)
        random = float(automated * (weights @ learner) + (1 - automated) * (weights @ supervisor))
        if allocation == 'within_task':
            random = float(np.mean([mask[ix].mean() * learner[ix].mean() + (1 - mask[ix].mean()) * supervisor[ix].mean() for ix in indices.values()]))
        values.append([automated, float(weights @ np.where(mask, learner, supervisor)), random])
    return np.asarray(values)


def curve_statistics(actual, oracle, tolerance=.02):
    gain = float(np.trapz(actual[:, 1] - actual[:, 2], actual[:, 0]))
    gap = float(np.trapz(oracle[:, 1] - oracle[:, 2], oracle[:, 0]))
    return dict(nauc=gain / gap if gap > 1e-12 else None, raw_gain=gain,
                automation=float(actual[actual[:, 1] >= actual[0, 1] - tolerance - 1e-12, 0].max()))


class Inputs:
    def __init__(self):self.files = {}

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        previous = self.files.get(str(path))
        if previous and previous['sha256'] != digest:
            raise ValueError(f'Input changed while the snapshot was being built: {path}')
        self.files[str(path)] = dict(sha256=digest, bytes=len(raw))
        return json.loads(raw)


def load_cohort(root, round_, declaration, inputs, supervisor_cache):
    learner_path = root / f'round_{round_:03d}/learner/evaluation.json'
    learner = inputs.read(learner_path)
    target_hash = declaration['supervisor_hash']
    wanted = {k:v for k,v in learner['spec'].items() if k != 'checkpoint'}
    matches = []
    for path in sorted(supervisor_cache.glob('*/evaluation.json')):
        # Do not bind unrelated evaluation files to this report snapshot.
        value = json.loads(path.read_text())
        if value['spec'].get('checkpoint') == target_hash and {k:v for k,v in value['spec'].items() if k != 'checkpoint'} == wanted:
            matches.append(path)
    if len(matches) != 1:raise ValueError(f'Expected one matching supervisor evaluation; found {len(matches)}')
    teacher = inputs.read(matches[0])
    pair_outcomes(learner['episodes'], teacher['episodes'])
    a = {e['id']:e for e in learner['episodes']};b = {e['id']:e for e in teacher['episodes']}
    if set(a) != set(b):raise ValueError('Learner/supervisor IDs differ')
    if any(any(a[i].get(k) != b[i].get(k) for k in PAIR_KEYS) for i in a):
        raise ValueError('An episode ID refers to different learner/supervisor initial states')
    ids = sorted(a);tasks = [a[i]['task'] for i in ids]
    lv = np.array([a[i]['success'] for i in ids], float);tv = np.array([b[i]['success'] for i in ids], float)
    rates = {t:float(np.mean([a[i]['success'] for i in ids if a[i]['task'] == t])) for t in sorted(set(tasks))}
    qualities = {}
    for name in ('time', 'path'):
        left = [a[i].get('q_' + name) for i in ids];right = [b[i].get('q_' + name) for i in ids]
        if all(v is not None and np.isfinite(v) for v in left + right):
            qualities[name] = (np.asarray(left, float), np.asarray(right, float))
    oracles = {allocation:curve(tv - lv, lv, tv, tasks, ids, allocation) for allocation in ('global', 'within_task')}
    summary = dict(round=round_, episodes=len(ids), tasks=len(rates),
        learner_success=float(np.mean(list(rates.values()))),
        supervisor_success=float(np.mean([np.mean([b[i]['success'] for i in ids if a[i]['task'] == t]) for t in rates])),
        learner_checkpoint=learner['spec']['checkpoint'], supervisor_checkpoint=target_hash,
        evaluation_identity=learner.get('identity'), split=learner['spec'].get('split'),
        continuous_qualities=list(qualities), oracle_global_nauc_denominator=float(np.trapz(oracles['global'][:, 1] - oracles['global'][:, 2], oracles['global'][:, 0])))
    return dict(ids=ids, tasks=tasks, clusters=[a[i]['init_state_hash'] for i in ids], learner=lv, supervisor=tv, rates=rates,
                qualities=qualities, oracles=oracles, summary=summary)


def describe(recipe):
    method = recipe['method'];f = recipe.get('features', {});g = recipe.get('gp', {})
    if recipe.get('scope') == 'paper':return 'Paper ' + method.replace('vlm_', '').replace('_', ' ').upper()
    key = recipe.get('group', f.get('bank', 'unknown'))
    group = POSITION_NAMES.get(key, key.replace('_', ' '))
    projection = g.get('input_projection') if method == 'kv_t' else f.get('compression')
    dimension = g.get('input_components') if method == 'kv_t' else f.get('components')
    label = f'{"KV-T" if method == "kv_t" else "KV"} · {group} · {projection}{dimension}'
    if 'adaptive' in recipe['id']:label += ' adaptive'
    if f.get('seed') != 42:label += f' s{f.get("seed")}'
    if method == 'kv_t':label += f' → {g.get("output_compression")}{g.get("output_components")}'
    if f.get('bank') == 'visual_patches':label += ' · ' + f.get('patch_aggregation', '')
    if g.get('lengthscale_initialization', 'geometry') != 'geometry':label += ' · variance init'
    return label


def fit_diagnostics(result, recipe):
    metrics = result.get('metrics') or {};g = metrics.get('gp_diagnostics', {})
    gp = recipe.get('gp', {});mapping = metrics.get('mapping_check', g.get('mapping_check', {}))
    geometry = g.get('geometry', {});cameras = g.get('cameras', [])
    kernels = g.get('kernels', [])
    jitters = [k['jitter_used'] for k in [g, *kernels, *cameras] if k.get('jitter_used') is not None]
    escalated = bool(jitters and max(jitters) > gp.get('jitter', 1e-6) * 1.01)
    history = g.get('training_records', {})
    milestones = []
    for phase, batches in history.items():
        for batch in batches:
            for entry in batch.get('milestones', []):
                row = dict(phase=phase, step=entry['step'], seconds=entry.get('seconds'))
                for name in ('validation_r2', 'validation_nll', 'objective'):
                    values = entry.get(name, [])
                    if values:
                        row[name + '_median'] = float(np.median(values))
                        row[name + '_mean'] = float(np.mean(values))
                milestones.append(row)
    r2 = mapping.get('gp_r2', [])
    pls = geometry.get('pls_checks', [])
    dims = g.get('input_components', g.get('components'))
    if dims is None and cameras:dims = cameras[0].get('components')
    return dict(actual_input_dim=dims, actual_output_dim=g.get('output_count'),
        precision=gp.get('precision', 'unknown') if recipe['method'] in ('kv', 'kv_t') else 'n/a',
        posterior_precision=g.get('prediction_numerics', {}).get('posterior_precision'),
        inference=g.get('inference', cameras[0].get('inference') if cameras else gp.get('inference')),
        max_kernel_jitter=max(jitters) if jitters else None, jitter_escalated=escalated,
        mapping_r2_median=float(np.median(r2)) if len(r2) else None,
        pls_target_explained=pls[-1]['retained_target_variance_explained'] if pls else None,
        pls_criterion_met=geometry.get('pls_criterion_met'),
        output_pca_variance=geometry.get('output_pca_variance_retained'),
        optimizer_seconds=sum(b.get('seconds', 0) for batches in history.values() for b in batches) or None,
        fit_feature_seconds=g.get('total_fit_and_feature_seconds'),
        worker_seconds=result.get('seconds'),
        peak_gpu_gib=((result.get('gpu_memory', {}).get('peak_reserved_bytes') or g.get('peak_gpu_allocated_bytes')) / 1024**3)
            if (result.get('gpu_memory', {}).get('peak_reserved_bytes') or g.get('peak_gpu_allocated_bytes')) else None,
        support_episodes=g.get('support_episodes', cameras[0].get('support_episodes') if cameras else None),
        support_rows=g.get('support_rows', cameras[0].get('support_rows') if cameras else None),
        milestones=milestones)


def extract_scores(saved, cohort, metrics):
    ids = cohort['ids'];scores = saved['scores']
    if set(scores) != set(ids):raise ValueError('Uncertainty scores and evaluation IDs differ')
    values = np.array([scores[i] for i in ids], float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite uncertainty scores')
    means = saved['pool_task_means'];names = sorted(cohort['rates'])
    if set(means) != set(names):raise ValueError('Pool task means and evaluation tasks differ')
    x = [means[t] for t in names];y = [cohort['rates'][t] for t in names]
    rho, r = correlation(x, y), correlation(x, y, 'pearson')
    result = dict(neg_spearman=-rho if rho is not None else None, neg_pearson=-r if r is not None else None,
        benefit_spearman=correlation(values, cohort['supervisor'] - cohort['learner']),
        benefit_pearson=correlation(values, cohort['supervisor'] - cohort['learner'], 'pearson'),
        unique_scores=len(np.unique(values)), episodes=len(ids), tasks=len(names))
    curves = {}
    for allocation in ('global', 'within_task'):
        actual = curve(values, cohort['learner'], cohort['supervisor'], cohort['tasks'], ids, allocation)
        stat = curve_statistics(actual, cohort['oracles'][allocation])
        prefix = 'global' if allocation == 'global' else 'within'
        result[prefix + '_nauc'] = stat['nauc'];result[prefix + '_raw_gain'] = stat['raw_gain']
        if allocation == 'global':result['automation'] = stat['automation']
        stored = metrics.get('referral_' + allocation, {})
        if stored.get('raw_gain') is not None:
            np.testing.assert_allclose(stat['raw_gain'], stored['raw_gain'], atol=1e-10, rtol=1e-8)
        curves[allocation] = actual
    for key, actual in [('between_task_spearman', rho), ('between_task_pearson', r), ('signed_gain_spearman', result['benefit_spearman'])]:
        if metrics.get(key) is not None:
            if actual is None:raise ValueError(f'Recorded {key} is defined but reconstructed correlation is undefined')
            np.testing.assert_allclose(actual, metrics[key], atol=1e-10, rtol=1e-8)
    for name, (learner, teacher) in cohort['qualities'].items():
        for kind in ('spearman', 'pearson'):
            result[f'benefit_{name}_{kind}'] = correlation(values, teacher - learner, kind)
    result['failure_auroc'] = metrics.get('macro_auroc')
    return result, dict(scores=values, task_means=x, task_names=names, success_rates=y, curves=curves)


def capture(study, supervisor_cache):
    study = Path(study).resolve();inputs = Inputs()
    started = datetime.now(timezone.utc).isoformat()
    declaration = inputs.read(study / 'declaration.json')
    queue = inputs.read(study / 'queue.json') if (study / 'queue.json').exists() else {}
    running = {(r['round'], r['head']):r for r in queue.get('running', [])}
    plans = [(round_, recipe, study / f'round_{round_:03d}' / recipe['id']) for recipe in declaration['recipes'] for round_ in declaration['rounds']]
    # Freeze availability once: newly finished cells appear in the next report.
    available = {str(out): (out / 'result.json').exists() for _, _, out in plans}
    rows = [];payload = {};cohorts = {};cohort_errors = {};issues = [];historical = [];recipes = {}
    training = inputs.read(study / 'training_data.json') if (study / 'training_data.json').exists() else {}
    for index, (round_, planned, out) in enumerate(plans):
        if index % 50 == 0:print(f'[report] Reading cells {index}/{len(plans)}', flush=True)
        request = None;row = dict(head=planned['id'], round=round_, status='pending', has_result=False, metrics_valid=False)
        recipe = planned;result = {};key = (round_, planned['id'])
        try:
            if (out / 'request.json').exists():
                request = inputs.read(out / 'request.json')
                if identity(request['spec']) != request['identity']:raise ValueError('Request checksum differs')
                recipe = request['spec']['recipe']
                if recipe['id'] != planned['id'] or request['spec']['round'] != round_:raise ValueError('Request belongs to another cell')
            if available[str(out)]:
                result = inputs.read(out / 'result.json');row['has_result'] = True
                if request is None or result.get('request_identity') != request['identity']:raise ValueError('Result/request identity differs')
                if result['recipe'] != recipe or result['round'] != round_:raise ValueError('Result recipe differs from its request')
                row['status'] = result['status'];row['reason'] = result.get('reason')
                if result.get('metrics'):
                    if round_ not in cohorts and round_ not in cohort_errors:
                        try:cohorts[round_] = load_cohort(study, round_, declaration, inputs, Path(supervisor_cache))
                        except (OSError, ValueError, KeyError) as error:cohort_errors[round_] = str(error)
                    if round_ in cohort_errors:raise ValueError(cohort_errors[round_])
                    saved = inputs.read(out / recipe['method'] / 'scores.json')
                    derived, detail = extract_scores(saved, cohorts[round_], result['metrics'])
                    row.update(derived, metrics_valid=True);payload[key] = detail
            elif key in running:
                row.update(status='running', gpu=running[key]['gpu'])
            if (out / 'failure.json').exists():
                failure = inputs.read(out / 'failure.json')
                if not request or failure.get('request_identity') != request['identity']:
                    historical.append(dict(head=planned['id'], round=round_, reason=failure.get('reason'), status=failure.get('status')))
                elif not row['has_result'] and row['status'] != 'running':
                    row.update(status=failure['status'], reason=failure.get('reason'))
                else:issues.append(dict(head=planned['id'], round=round_, kind='previous_attempt', reason=failure.get('reason')))
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
            row.update(status='data_error', metrics_valid=False, reason=f'{type(error).__name__}: {error}')
            issues.append(dict(head=planned['id'], round=round_, kind='data_error', reason=row['reason']))
        row.update(method=recipe['method'], scope=recipe.get('scope'), group=recipe.get('group'), label=describe(recipe),
            input_projection=recipe['gp'].get('input_projection') if recipe['method'] == 'kv_t' else recipe['features'].get('compression'),
            requested_input_dim=recipe['gp'].get('input_components') if recipe['method'] == 'kv_t' else recipe['features'].get('components'),
            projection_seed=recipe['features'].get('seed'),
            output_projection=recipe['gp'].get('output_compression') if recipe['method'] == 'kv_t' else None,
            output_dependence=recipe['gp'].get('output_dependence'),
            source=str(out / 'result.json'), detail_report=str(out / 'report/report.html'),
            **fit_diagnostics(result, recipe))
        row['legacy_coregionalized'] = recipe['method'] == 'kv_t' and row['output_dependence'] == 'coregionalized'
        row['eligible'] = (row['status'] == 'complete' and row['metrics_valid']
                           and not row['legacy_coregionalized'])
        row['quality'] = 'historical coregionalized' if row['legacy_coregionalized'] else 'jitter increased' if row['jitter_escalated'] else 'no recorded escalation'
        if row['method'] not in ('kv', 'kv_t'):row['quality'] = 'paper baseline'
        rows.append(row);recipes[key] = recipe
    meta = dict(started_at=started, finished_reading_at=datetime.now(timezone.utc).isoformat(), study=str(study),
        total=len(rows), recorded=sum(r['has_result'] for r in rows), successful=sum(r['status'] == 'complete' for r in rows),
        statuses=dict(Counter(r['status'] for r in rows)), eligible=sum(r['eligible'] for r in rows),
        historical_failure_markers=len(historical), current_failures=sum(r['status'] in ('failed', 'blocked') for r in rows),
        rounds=declaration['rounds'], queue_updated_at=queue.get('updated_at'),
        cohorts=[c['summary'] for c in cohorts.values()], training=training,
        declaration_source=declaration.get('source_hash'), precision_update=declaration.get('precision_update'),
        scientific_artifacts_written=False, test_used=declaration.get('split') == 'final')
    return dict(meta=meta, rows=rows, recipes=recipes, payload=payload, cohorts=cohorts,
                issues=issues, historical_failures=historical, files=inputs.files, declaration=declaration)
