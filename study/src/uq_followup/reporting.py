"""Incremental plots plus policy-seed summaries and paired cluster bootstrap."""
from pathlib import Path
import html
import json
import os
import tempfile
import numpy as np

from uq_extensions.common import atomic_json, atomic_npz, read_json
from head_report.bootstrap import Plan, METRICS


def cohort_for(study, policy_id):
    study = Path(study); seed = int(policy_id.rsplit('seed', 1)[1])
    learner = read_json(study/'evaluations'/policy_id/'conditions.json')
    supervisor = {r['id']: r for r in read_json(study/'evaluations'/f'supervisor_seed{seed}'/'conditions.json')}
    if set(supervisor) != {r['id'] for r in learner}: raise ValueError('Supervisor initial conditions differ')
    teacher = [supervisor[r['id']] for r in learner]
    for a, b in zip(learner, teacher):
        for key in ('task', 'init_state_hash', 'reset_seed', 'initial_input'):
            if key == 'initial_input':
                if a[key]['input_hash'] != b[key]['input_hash']: raise ValueError('Learner/supervisor observations differ')
            elif a[key] != b[key]: raise ValueError('Learner/supervisor condition mismatch')
    return dict(ids=[r['id'] for r in learner], tasks=[r['task'] for r in learner],
        clusters=[r['init_state_hash'] for r in learner], learner=[r['success'] for r in learner],
        supervisor=[r['success'] for r in teacher],
        qualities={k: ([r['q_'+k] for r in learner], [r['q_'+k] for r in teacher]) for k in ('time', 'path')})


def aligned_scores(path, cohort):
    data = read_json(Path(path)/'scores.json')
    values = dict(zip(data['ids'], data['scores']))
    if set(values) != set(cohort['ids']): raise ValueError('Scores and evaluated conditions differ')
    return np.array([values[i] for i in cohort['ids']]), data['pool_task_means']


def clean(value):
    return float(value) if np.isfinite(value) else None


def head_report(study, policy_id, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata
    directory = Path(directory); cohort = cohort_for(study, policy_id)
    scores, means = aligned_scores(directory, cohort)
    point = Plan(cohort, indices=np.arange(len(scores))[None, :])
    values, curves = point.evaluate(scores, means)
    # AUROC against averaged fractional success is not a binary ROC statistic.
    metrics = {name: clean(values[0, i]) for i, name in enumerate(METRICS) if name != 'failure_auroc'}
    bootstrap = Plan(cohort, repetitions=2000)
    samples, _ = bootstrap.evaluate(scores, means)
    atomic_npz(directory/'bootstrap.npz', metrics=samples, names=np.array(METRICS))
    atomic_json(directory/'metrics.json', metrics)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for kind, title in [('global_', 'Global referral'), ('within', 'Within-task referral')]:
        x, y, random = curves[kind]; ox, oy, _ = point.oracle[kind]
        ax = axes[0 if kind == 'global_' else 1]
        ax.plot(1-x[0], y[0], label='Head'); ax.plot(1-x[0], random[0], '--', label='Random')
        ax.plot(1-ox[0], oy[0], ':', label='Hindsight oracle')
        ax.set(xlabel='Fraction referred', ylabel='Success probability', title=title)
        ax.grid(alpha=.2); ax.legend()
    x, y, _ = curves['global_']
    losses = np.linspace(0, .2, 81)
    automated = [float(np.max(x[0][y[0] >= point.hbar[0]-loss-1e-12])) for loss in losses]
    axes[2].plot(losses, automated)
    axes[2].set(xlabel='Allowed loss relative to supervisor', ylabel='Automated fraction', title='Automation')
    axes[2].grid(alpha=.2)
    fig.savefig(directory/'referral.png', dpi=140); fig.savefig(directory/'referral.pdf'); plt.close(fig)
    names = sorted(set(cohort['tasks'])); tasks = np.asarray(cohort['tasks'])
    rates = [np.mean(np.asarray(cohort['learner'])[tasks == t]) for t in names]
    uncertainty = [means[t] for t in names]
    gain = np.asarray(cohort['supervisor'])-np.asarray(cohort['learner'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    axes[0].scatter(rankdata(uncertainty), rankdata(rates))
    for i, name in enumerate(names): axes[0].annotate(name, (rankdata(uncertainty)[i], rankdata(rates)[i]), fontsize=7)
    axes[0].set(xlabel='Training-pool uncertainty rank', ylabel='Policy success rank', title='Task Spearman correlation')
    axes[1].scatter(rankdata(scores), rankdata(gain), alpha=.3)
    axes[1].set(xlabel='Uncertainty rank', ylabel='Signed supervisor benefit rank', title='Condition-level Spearman correlation')
    fig.savefig(directory/'correlations.png', dpi=140); fig.savefig(directory/'correlations.pdf'); plt.close(fig)
    diagnostic_plot = ''
    if (directory/'diagnostics.json').exists():
        diagnostic = read_json(directory/'diagnostics.json')
        if diagnostic.get('training_records'):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
            histories = [r['loss_history'] for r in diagnostic.get('kernels', []) if r.get('loss_history')]
            if histories:
                axes[0].plot(np.mean(histories, axis=0))
            for batch in diagnostic['training_records'].get('mapping_validation', []):
                points = [r for r in batch['milestones'] if 'validation_r2' in r]
                if points: axes[1].plot([r['step'] for r in points], [np.mean(r['validation_r2']) for r in points], marker='o')
            axes[0].set(xlabel='GP optimizer iteration', ylabel='Objective', title='Full-support fit')
            axes[1].set(xlabel='GP optimizer iteration', ylabel='Mean held-out R²', title='Prediction feasibility')
            fig.savefig(directory/'gp_training.png', dpi=140); plt.close(fig)
            diagnostic_plot = '<img width="80%" src="gp_training.png">'
    lines = ['<!doctype html><meta charset="utf-8"><title>Follow-up head</title>',
             '<h1>'+html.escape(policy_id+' / '+directory.name)+'</h1>',
             '<p>500 initial conditions; outcomes averaged over three executions. nAUC uses the condition-level hindsight oracle.</p>',
             '<pre>'+html.escape(json.dumps(metrics, indent=2))+'</pre>',
             '<img width="100%" src="referral.png"><img width="80%" src="correlations.png">',
             diagnostic_plot, '<p><a href="diagnostics.json">GP fit, prediction and compression diagnostics</a></p>']
    (directory/'report.html').write_text('\n'.join(lines))
    return metrics


def refresh_index(study):
    study = Path(study)
    records = []
    for p in sorted((study/'heads').glob('*/*/result.json')):
        r = read_json(p); records.append(r)
    atomic_json(study/'head_results.json', records)
    rows = []
    for r in records:
        path = 'heads/'+r['policy_id']+'/'+r['recipe']['id']+'/report.html'
        rows.append('<tr><td>'+html.escape(r['policy_id'])+'</td><td><a href="'+path+'">'+
                    html.escape(r['recipe']['id'])+'</a></td><td>'+html.escape(r['status'])+'</td><td>'+str((r.get('metrics') or {}).get('global_nauc'))+'</td></tr>')
    destination = study/'index.html'
    fd, name = tempfile.mkstemp(prefix='index.', suffix='.partial', dir=study)
    os.close(fd); temporary = Path(name)
    temporary.write_text('<!doctype html><meta charset="utf-8"><h1>Follow-up study progress</h1>'
        f'<p>{len(records)} / 798 head evaluations recorded. <a href="report/report.html">Seed-averaged report</a></p>'
        '<p><a href="report/report.html#model-map">Model architecture and head locations</a></p>'
        '<p><a href="policy_quality.md">Policy quality</a> · <a href="queue.json">Queue and failures</a></p>'
        '<table><tr><th>Policy</th><th>Head</th><th>Status</th><th>Global nAUC</th></tr>'+''.join(rows)+'</table>')
    temporary.replace(destination)


def interval(values):
    values = np.asarray(values, float)
    if np.isfinite(values).sum() < .9*len(values): return [None, None]
    return np.nanquantile(values, [.025, .975]).tolist()


def atomic_text(path, text):
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f: f.write(text)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def aggregate_report(study):
    from uq_extensions.cache_lock import cache_lock
    with cache_lock(Path(study)/'report/.lock', label='follow-up report'):
        return _aggregate_report(study)


def _aggregate_report(study):
    # Report code is deliberately outside the pinned scientific source identity.
    from head_report.followup import build_report
    return build_report(study)
