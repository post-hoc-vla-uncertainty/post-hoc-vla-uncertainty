"""On-demand scheduled-round tables, seed uncertainty and paired bootstrap."""
import base64
import csv
import html
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from uq_extensions.common import atomic_json, atomic_npz, identity, read_json
from head_report.bootstrap import Plan, METRICS, combine
from .design import ARMS, SCORES, evaluation_rounds, round_dir

REPORT_METRICS = [m for m in METRICS if m not in ('failure_auroc', 'global_raw_gain', 'within_raw_gain')]
VERSION = 'final-save-v1-expected-ties'

HEAD_LABELS = {
    'random': 'Random', 'vfd': 'VFD', 'gu': 'GU',
    'h1': 'H1: final-context KV + whitening',
    'h2': 'H2: pooled-decoder KV + whitening',
    'h3': 'H3: decoder KV + PLS512 + whitening',
    'h4': 'H4: decoder KV-T', 'h1_ivr': 'H1 with IVR',
}


def report_progress(root):
    """Read committed job progress without changing the active dispatcher."""
    result = dict(generated_at=datetime.now(timezone.utc).isoformat(), jobs={}, failed={}, blocked=[])
    graph = root/'job_graph.json'
    if not graph.exists():
        return result
    from .launcher import completed
    jobs = read_json(graph)
    done = {j['id'] for j in jobs if completed(j, root)}
    state = read_json(root/'queue.json') if (root/'queue.json').exists() else {}
    failed = {id_:v for id_,v in state.get('failed', {}).items() if id_ not in done}
    blocked = set(failed)
    while True:
        more = {j['id'] for j in jobs if j['id'] not in done and set(j['dependencies']) & blocked}
        if more <= blocked: break
        blocked |= more
    result.update(total=len(jobs), complete=len(done), failed=failed, blocked=sorted(blocked-set(failed)),
                  dispatcher_updated_at=state.get('updated_at'), allowed_gpus=state.get('allowed_gpus'))
    for kind in sorted({j['kind'] for j in jobs}):
        rows = [j for j in jobs if j['kind']==kind]
        result['jobs'][kind] = dict(complete=sum(j['id'] in done for j in rows), total=len(rows))
    return result


def expected_ties(scores, gains):
    """Uniform expectation within exact ties, vectorized across bootstrap draws."""
    order = np.argsort(scores, axis=1, kind='stable')
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    values = np.take_along_axis(gains, order, axis=1)
    rows, n = values.shape
    start = np.column_stack([np.ones(rows, bool), sorted_scores[:, 1:] != sorted_scores[:, :-1]])
    groups = np.cumsum(start, axis=1)-1 + np.arange(rows)[:, None]*n
    sums = np.bincount(groups.ravel(), weights=values.ravel(), minlength=rows*n)
    counts = np.bincount(groups.ravel(), minlength=rows*n)
    return sums[groups]/counts[groups]


class ReferralPlan(Plan):
    def curves(self, scores):
        if not np.allclose(self.weights, self.weights[0]):
            raise ValueError('Final study expects the same number of conditions per task')
        sampled = np.asarray(scores)[self.indices]
        gain = expected_ties(sampled, self.gain)
        x = np.broadcast_to(np.linspace(0, 1, self.size+1), (self.repetitions, self.size+1))
        y = self.hbar[:, None]+np.column_stack([np.zeros(self.repetitions), np.cumsum(gain/self.size, axis=1)])
        random = self.hbar[:, None]+x*(self.lbar-self.hbar)[:, None]
        n = len(self.blocks[0]); grid = np.linspace(0, 1, n+1)
        within = np.zeros((self.repetitions, n+1))
        for block in self.blocks:
            if len(block) != n: raise ValueError('Unequal task sizes')
            g = expected_ties(sampled[:, block], self.gain[:, block])
            within += (self.supervisor[:, block].mean(1)[:, None] +
                       np.column_stack([np.zeros(self.repetitions), np.cumsum(g/n, axis=1)]))/len(self.blocks)
        expected = self.hbar[:, None]+grid*(self.lbar-self.hbar)[:, None]
        return dict(global_=(x, y, random), within=(np.broadcast_to(grid, within.shape), within, expected))


def cohort(root, policy_id, seed):
    root = Path(root)
    learner = read_json(root/f'evaluations/{policy_id}/conditions.json')
    supervisor = {r['id']:r for r in read_json(root/f'evaluations/supervisor_{seed}/conditions.json')}
    if set(supervisor) != {r['id'] for r in learner}: raise ValueError('Condition IDs do not align')
    teacher = [supervisor[r['id']] for r in learner]
    for left, right in zip(learner, teacher):
        if any(left[k] != right[k] for k in ('task', 'init_state_hash', 'reset_seed')):
            raise ValueError('Mismatched learner/supervisor conditions')
        if left['initial_input']['input_hash'] != right['initial_input']['input_hash']:
            raise ValueError('Unpaired head inputs')
    return dict(ids=[r['id'] for r in learner], tasks=[r['task'] for r in learner],
        clusters=[r['init_state_hash'] for r in learner], learner=[r['success'] for r in learner],
        supervisor=[r['success'] for r in teacher],
        qualities={k:([r['q_'+k] for r in learner], [r['q_'+k] for r in teacher]) for k in ('time', 'path')})


def aligned(data, cohort_):
    values = dict(zip(data['ids'], data['scores']))
    if set(values) != set(cohort_['ids']): raise ValueError('Scoring condition IDs differ')
    return np.asarray([values[i] for i in cohort_['ids']])


def clean(value):
    return float(value) if np.isfinite(value) else None


def point_metrics(cohort_, data):
    values, _ = ReferralPlan(cohort_, indices=np.arange(len(cohort_['ids']))[None, :]).evaluate(aligned(data, cohort_), data['pool_task_means'])
    return {m:clean(values[0, METRICS.index(m)]) for m in REPORT_METRICS}


def interval(values):
    values = np.asarray(values)
    return [None, None] if np.isfinite(values).mean() < .9 else np.nanquantile(values, [.025, .975]).tolist()


def statistics(points, replicates, seeds):
    points = np.asarray(points, float)
    evaluation, total = combine(np.asarray(replicates), seeds)
    result = {}
    for i, metric in enumerate(METRICS):
        if metric not in REPORT_METRICS: continue
        result[metric] = dict(mean=clean(np.mean(points[:, i])),
            seed_std=clean(np.std(points[:, i], ddof=1)) if len(seeds)>1 else None,
            evaluation_ci95=interval(evaluation[:, i]), total_ci95=interval(total[:, i]) if total is not None else [None, None])
    return result


def generate(root):
    root = Path(root); declaration = read_json(root/'declaration.json')
    common_rounds = evaluation_rounds(declaration, 'common')
    reference_rounds = evaluation_rounds(declaration, 'acquisition')
    output = root/'report'; output.mkdir(parents=True, exist_ok=True)
    progress = report_progress(root)
    records = []; samples = {}; points = {}; cohorts = {}; plans = {}; detail = []; valid = set()
    for path in sorted((root/'scores').glob('*/*/result.json')):
        result = read_json(path); pid = result['policy_id']; seed = result['seed']; head = result['head']; r = result['round']
        if r not in common_rounds: continue
        ckey = (pid, seed)
        if ckey not in cohorts: cohorts[ckey] = cohort(root, pid, seed)
        c = cohorts[ckey]; data = read_json(path.with_name('scores.json')); scores = aligned(data, c)
        point = ReferralPlan(c, indices=np.arange(len(scores))[None, :])
        estimate, curves = point.evaluate(scores, data['pool_task_means'])
        key = (head, seed, r); points[key] = estimate[0]
        if result['status'] == 'complete': valid.add(key)
        cachekey = identity(dict(version=VERSION, cohort=c, scores=data, repetitions=2000))
        cache = output/'bootstrap'/f'{cachekey}.npz'
        if cache.exists():
            with np.load(cache) as saved: samples[key] = saved['metrics']
        else:
            # Reuse one policy's vectorized bootstrap plan; release at policy boundaries.
            if ckey not in plans:
                plans.clear(); plans[ckey] = ReferralPlan(c, repetitions=2000)
            samples[key], _ = plans[ckey].evaluate(scores, data['pool_task_means'])
            atomic_npz(cache, metrics=samples[key], names=np.array(METRICS))
        result['metrics'] = {m:clean(estimate[0, METRICS.index(m)]) for m in REPORT_METRICS}
        records.append(result)
        names = point.names
        detail.append(dict(head=head, seed=seed, round=r, status=result['status'],
            metrics=result['metrics'], global_curve=[a[0].tolist() for a in curves['global_']],
            within_curve=[a[0].tolist() for a in curves['within']],
            oracle_global=[a[0].tolist() for a in point.oracle['global_']],
            oracle_within=[a[0].tolist() for a in point.oracle['within']],
            task_names=names, task_success=point.task_rates[0].tolist(), task_uncertainty=[data['pool_task_means'][t] for t in names]))
        print(f'Report: {pid} / {head}; metrics and bootstrap cached.', flush=True)
    summary = []; seeds = [g['seed'] for g in declaration['groups']]
    for head in SCORES:
        for round_ in common_rounds:
            found = [s for s in seeds if (head,s,round_) in valid]
            if not found: continue
            summary.append(dict(head=head, round=round_, seeds=found,
                metrics=statistics([points[head,s,round_] for s in found], [samples[head,s,round_] for s in found], found)))
    # Do not average unequal round coverage or silently change the averaging cohort.
    averages = []
    for head in SCORES:
        found = [s for s in seeds if all((head,s,r) in valid for r in common_rounds)]
        if not found: continue
        averages.append(dict(head=head, rounds=list(common_rounds), seeds=found,
            metrics=statistics([np.mean([points[head,s,r] for r in common_rounds], axis=0) for s in found],
                               [np.mean([samples[head,s,r] for r in common_rounds], axis=0) for s in found], found)))
    # Paired differences against each requested paper acquisition/scoring baseline.
    contrasts = []
    for head in ('h1','h2','h3','h4'):
        for baseline in ('vfd','gu'):
            found = [s for s in seeds if all((h,s,r) in valid for h in (head, baseline) for r in common_rounds)]
            if found:
                contrasts.append(dict(head=head, baseline=baseline, seeds=found, rounds=list(common_rounds),
                    metrics=statistics([np.mean([points[head,s,r]-points[baseline,s,r] for r in common_rounds], axis=0) for s in found],
                        [np.mean([samples[head,s,r]-samples[baseline,s,r] for r in common_rounds], axis=0) for s in found], found)))
    acquisition = []
    for arm in ARMS:
        for group in declaration['groups']:
            seed = group['seed']
            for r in reference_rounds:
                path = round_dir(root, arm if r else 'random', seed, r)/'reference.json'
                if not path.exists(): continue
                data = read_json(read_json(path)['source'])
                success = np.mean([np.mean([float(v['success']) for v in task['per_rollout_results']]) for task in data['per_task']])
                acquisition.append(dict(arm=arm, seed=seed, round=r, query_slots=5*r, success=float(success), provenance=read_json(path)['provenance']))
    acquisition_summaries = []
    for arm in ARMS:
        complete = []
        for seed in seeds:
            rows = sorted((v for v in acquisition if v['arm']==arm and v['seed']==seed), key=lambda v:v['round'])
            if [v['round'] for v in rows] != list(reference_rounds): continue
            values = np.asarray([v['success'] for v in rows]); slots = np.asarray(reference_rounds)*5
            selected = [i for r in range(1,16) for i in read_json(round_dir(root,arm,seed,r)/'selection.json')['selected_ids']]
            manifest = read_json(declaration['manifest'])
            complete.append(dict(seed=seed, final_success=float(values[-1]), normalized_learning_auc=float(np.trapz(values, slots)/75),
                slots_to_50=int(slots[values>=.5][0]) if np.any(values>=.5) else None,
                slots_to_60=int(slots[values>=.6][0]) if np.any(values>=.6) else None,
                unique_new_demos=len(set(selected)-set(manifest['pretraining_ids'])), unique_selected_demos=len(set(selected))))
        if complete:
            aggregate = {}
            for metric in ('final_success','normalized_learning_auc','unique_new_demos'):
                values = np.array([v[metric] for v in complete])
                aggregate[metric] = dict(mean=float(values.mean()), seed_std=float(values.std(ddof=1)) if len(values)>1 else None)
            acquisition_summaries.append(dict(arm=arm, rounds=list(reference_rounds), seeds=[v['seed'] for v in complete],
                metrics=aggregate, individual_seeds=complete,
                evaluation_note='AUC linearly interpolates evaluated rounds; thresholds are the first evaluated budget reaching the target, not an exact crossing round.'))
    quality = {p.parent.name:read_json(p) for p in (root/'evaluations').glob('*/quality.json')}
    warnings = [f'{pid}: {w}' for pid, record in quality.items() for w in record.get('warnings', [])]
    for r in common_rounds:
        values = {f'random_{s}_r{r:02d}':quality[f'random_{s}_r{r:02d}'] for s in seeds if f'random_{s}_r{r:02d}' in quality}
        if values and max(q['success'] for q in values.values())-min(q['success'] for q in values.values()) > .25:
            warnings.append(f'Round {r}: primary success differs by more than 25 percentage points across seeds; inspect, do not auto-exclude.')
    acquisition_by_round = []
    for arm in ARMS:
        for r in reference_rounds:
            rows = [v for v in acquisition if v['arm']==arm and v['round']==r]
            values = [v['success'] for v in rows]
            acquisition_by_round.append(dict(arm=arm, round=r, query_slots=5*r,
                seeds=[v['seed'] for v in rows], expected_seeds=len(seeds),
                mean=clean(np.mean(values)) if values else None,
                seed_std=clean(np.std(values, ddof=1)) if len(values)>1 else None))
    atomic_json(output/'summary.json', dict(version=VERSION, progress=progress, per_round=summary, round_averaged=averages,
        common_evaluation_rounds=list(common_rounds), acquisition_evaluation_rounds=list(reference_rounds),
        paired_differences=contrasts, acquisition=acquisition, acquisition_by_round=acquisition_by_round,
        acquisition_summary=acquisition_summaries, policy_quality=quality, warnings=warnings,
        note='Point estimates and sample SD across policy/supervisor pairs; total CI resamples pairs and 50 condition clusters per task. Same draws across heads and rounds. Three executions stay together. No bootstrap claim about new tasks.'))
    atomic_json(output/'details.json', detail)
    fields = ['head','seed','round','status', *REPORT_METRICS]
    with (output/'all_rounds.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in records: writer.writerow({k:row[k] for k in fields[:4]} | row['metrics'])
    with (output/'save_acquisition.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['arm','seed','round','query_slots','success','provenance']); writer.writeheader(); writer.writerows(acquisition)
    with (output/'save_acquisition_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['arm','round','query_slots','seeds','expected_seeds','mean','seed_std'])
        writer.writeheader(); writer.writerows(acquisition_by_round)
    _html(output, declaration, summary, averages, acquisition, detail, warnings, acquisition_summaries,
          progress, acquisition_by_round, quality)
    return output/'report.html'


def _html(output, declaration, summary, averages, acquisition, details, warnings, acquisition_summaries,
          progress, acquisition_by_round, quality):
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    common_label = ', '.join(map(str, evaluation_rounds(declaration, 'common')))
    reference_label = ', '.join(map(str, evaluation_rounds(declaration, 'acquisition')))
    fig, ax = plt.subplots(figsize=(9,5), layout='constrained')
    for arm in ARMS:
        rows = [v for v in acquisition if v['arm']==arm]
        rounds = sorted({v['round'] for v in rows})
        means = [np.mean([v['success'] for v in rows if v['round']==r]) for r in rounds]
        stds = [np.std([v['success'] for v in rows if v['round']==r], ddof=1) if sum(v['round']==r for v in rows)>1 else 0 for r in rounds]
        if rounds:
            x = np.array(rounds)*5
            counts = np.array([sum(v['round']==r for v in rows) for r in rounds])
            full = counts == len(declaration['groups'])
            line, = ax.plot([], [], label=HEAD_LABELS.get(arm,arm)); color=line.get_color()
            for i in range(len(rounds)-1):
                ax.plot(x[i:i+2], means[i:i+2], color=color, linestyle='-' if all(full[i:i+2]) else '--')
            for xi,yi,complete in zip(x,means,full):
                ax.plot(xi,yi,'o',color=color,markerfacecolor=color if complete else 'white')
            ax.fill_between(x, np.array(means)-stds, np.array(means)+stds, where=full, color=color, alpha=.08)
    ax.set(xlabel='Demonstration query slots', ylabel='Mean task success',
           title='SAVE acquisition: mean ± policy-seed SD\nSolid/filled: all seeds; dashed/hollow: partial seed coverage')
    if acquisition: ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.2); fig.savefig(output/'save_acquisition.png', dpi=150); plt.close(fig)
    image = base64.b64encode((output/'save_acquisition.png').read_bytes()).decode()
    def table(rows):
        if not rows:
            return '<p><strong>Pending:</strong> no completed head comparisons are available for this table yet.</p>'
        headers = ['Head','Round','Seeds', *REPORT_METRICS]
        lines = ['<table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>']
        for row in rows:
            cells = [html.escape(row['head']), str(row.get('round', common_label)), str(len(row['seeds']))]
            for metric in REPORT_METRICS:
                v = row['metrics'][metric]; mean=v['mean']; sd=v['seed_std']; ci=v['total_ci95']
                label = 'undefined' if mean is None else f'{mean:.3f}'+(f' ± {sd:.3f}' if sd is not None else ' (one seed)')
                label += f'<br><small>95% CI {ci[0]:.3f}, {ci[1]:.3f}</small>' if ci[0] is not None else ''
                cells.append(label)
            lines.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
        return ''.join(lines)+'</tbody></table>'
    body = '<!doctype html><meta charset="utf-8"><title>Final SAVE study</title><style>body{font:15px system-ui;margin:2em}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}th{cursor:pointer;position:sticky;top:0;background:#eee}small{white-space:nowrap}canvas{border:1px solid #ddd;margin:8px}pre{white-space:pre-wrap}</style>'
    body += f'<h1>Final SAVE study</h1><p>Eight acquisition methods × three policy pairs. Every acquisition round uses 4,000 updates. Acquisition-performance evaluation: rounds {reference_label}. Common-policy head comparison: Random rounds {common_label}. No fitted head is chosen using these outcomes.</p>'
    stamp = datetime.fromisoformat(progress['generated_at']).astimezone(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d %H:%M %Z')
    body += '<h2>Snapshot availability</h2><p>Generated '+stamp+'. This is an interim snapshot until all scheduled comparisons are complete. Empty sections mean pending work, not zero performance. Seed counts are shown for every acquisition checkpoint.</p>'
    if progress['jobs']:
        body += '<table><thead><tr><th>Stage</th><th>Completed</th><th>Scheduled</th></tr></thead><tbody>'
        stage_labels = dict(train='Policy training', acquire='Demonstration acquisition', reference='Acquisition-performance evaluations',
                            evaluate='Paired robot-evaluation batches (including reusable batches)', score='Correlation/referral head scoring')
        for kind,label in stage_labels.items():
            counts=progress['jobs'].get(kind,dict(complete=0,total=0))
            body += f'<tr><td>{label}</td><td>{counts["complete"]}</td><td>{counts["total"]}</td></tr>'
        body += '</tbody></table>'
    if progress['failed']:
        body += f'<p><strong>Run warning:</strong> {len(progress["failed"])} failed job(s), with {len(progress["blocked"])} dependent jobs blocked. Other independent work continues. These are execution failures, not measured zero-success results.</p><ul>'
        for id_,failure in progress['failed'].items():
            reason = 'CUDA driver initialization failed before acquisition scoring.' if 'driver initialization failed' in failure.get('tail','') else 'Worker exited unsuccessfully; inspect its job log.'
            body += '<li>'+html.escape(id_)+': '+reason+'</li>'
        body += '</ul>'
    from head_report.model_map import model_svg
    svg = base64.b64encode(model_svg({'C32','D32','F'}).encode()).decode()
    body += '<details><summary>Model blocks and the four selected head locations</summary><p>H1 reads C32: camera/language means plus state. H2 reads D32, averaging its 50 action tokens to 480 values. H3/H4 flatten D32 and append current actions, time and C32; F supplies their velocity targets (PLS for H3, GP fitting for H4). D32 is reused at multiple flow times; H2 uses the ninth evaluation only. Grey locations are not separate heads in this study.</p>'
    body += f'<img style="max-width:100%;width:900px" alt="SmolVLA blocks and selected feature taps" src="data:image/svg+xml;base64,{svg}"></details>'
    body += '<h2>What referral measures</h2><p>The head decides which initial conditions to give to a stronger, fixed supervisor. Random referral ignores the score. The hindsight oracle sorts by the actual signed supervisor improvement, including harm. nAUC is the area gained over random, divided by the oracle’s area over random. Within-task referral makes this comparison separately within each instruction. Exact score ties use their expected performance, so a task-constant score has zero within-task gain.</p>'
    body += '<p>Cells show mean ± sample SD across policy/supervisor pairs and a hierarchical 95% CI (2,000 paired condition/seed resamples). With only three pairs, seed uncertainty remains imprecise. Correlations are negative Pearson/Spearman of task uncertainty versus task success (higher is better). Automation allows two percentage points below supervisor success. All rounds are available in CSV.</p>'
    body += '<h2>Policy quality</h2><pre>'+html.escape('\n'.join(warnings) or ('No quality warning recorded; inspect per-task success in summary.json.' if quality else 'Pending: paired-policy quality audits have not been generated yet.'))+'</pre>'
    body += f'<h2>SAVE acquisition</h2><img style="max-width:100%" src="data:image/png;base64,{image}">'
    body += '<p>The learning-curve area uses linear interpolation between evaluated checkpoints. Threshold budgets are the first evaluated budgets reaching the target; crossings between evaluations are unknown. This is not an exact reproduction of the paper’s every-round sample-efficiency calculation.</p>'
    body += '<h3>Success at each evaluated checkpoint</h3><p>Values are success percentages: mean ± sample SD across available policy seeds. The n value states coverage; one seed has no estimated seed SD. Compare methods at checkpoints with all three seeds available. Later points with fewer seeds can change as results arrive, and the plotted curve can therefore mix seed cohorts.</p>'
    rounds=evaluation_rounds(declaration,'acquisition')
    body += '<table><thead><tr><th>Method</th>'+''.join(f'<th>Round {r} / {5*r} query slots</th>' for r in rounds)+'</tr></thead><tbody>'
    for arm in ARMS:
        body += '<tr><td>'+html.escape(HEAD_LABELS.get(arm,arm))+'</td>'
        for r in rounds:
            row=next(v for v in acquisition_by_round if v['arm']==arm and v['round']==r)
            value='Pending' if row['mean'] is None else f'{100*row["mean"]:.1f}%'+(f' ± {100*row["seed_std"]:.1f} pp' if row['seed_std'] is not None else '')
            body += '<td>'+value+f'<br><small>n={len(row["seeds"])}/{row["expected_seeds"]}; seeds '+html.escape(', '.join(map(str,row['seeds'])) or 'none')+'</small></td>'
        body += '</tr>'
    body += '</tbody></table><h3>Complete acquisition trajectories</h3><p>Only seeds with all four scheduled evaluation checkpoints enter the final-success and learning-curve summaries below. Partial seed coverage is provisional.</p>'
    body += '<table><thead><tr><th>Method</th><th>Complete seeds</th><th>Final success</th><th>Interpolated learning-curve AUC / 75</th><th>Unique new demonstrations</th><th>First evaluated slots at 50% / 60%, by seed</th></tr></thead><tbody>'
    for row in acquisition_summaries:
        cells = [row['arm'],str(len(row['seeds']))]
        for metric in ('final_success','normalized_learning_auc','unique_new_demos'):
            v=row['metrics'][metric]; cells.append(f'{v["mean"]:.3f}'+(f' ± {v["seed_std"]:.3f}' if v['seed_std'] is not None else ' (one seed)'))
        cells.append('; '.join(f'{v["seed"]}: {v["slots_to_50"] if v["slots_to_50"] is not None else "not observed"} / {v["slots_to_60"] if v["slots_to_60"] is not None else "not observed"}' for v in row['individual_seeds']))
        body += '<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>'
    body += '</tbody></table>'
    body += f'<h2>Mean over evaluated checkpoints</h2><p>Only seeds with every scheduled correlation/referral checkpoint ({common_label}) enter this table. The mean gives those checkpoints equal weight; it does not estimate an average across unobserved rounds. Per-round values remain available during the run. This is our matched 4,000-update protocol, not the paper’s separate 2,000-update correlation trajectory.</p>'+table(averages)
    body += '<h2>Correlation and referral by evaluated round</h2>'+table(summary)
    body += '<p>Mappings that fail the GP prediction check remain in the raw results and inspector with their status; they are excluded from comparison means. Seed counts are shown explicitly.</p>'
    body += '<h2>Inspect a completed head</h2>'
    if not details:
        body += '<p>Pending: correlation/referral scoring has not produced a completed head result. The running queue prioritizes acquisition and policy training before these comparisons.</p>'
    body += '<select id="inspect"'+(' disabled hidden' if not details else '')+'></select><div id="plots"></div><pre id="metrics"></pre>'
    payload = json.dumps(details, allow_nan=False).replace('<','\\u003c')
    body += '<script>const details='+payload+';const sel=document.getElementById("inspect");details.forEach((v,i)=>{let o=document.createElement("option");o.value=i;o.textContent=`${v.head} / seed ${v.seed} / round ${v.round} / ${v.status}`;sel.append(o)});'
    body += '''function plot(title,lines,xlabel,ylabel){const c=document.createElement('canvas');c.width=480;c.height=330;document.getElementById('plots').append(c);const g=c.getContext('2d');g.fillText(title,45,18);g.fillText(xlabel,170,320);g.fillText(ylabel,5,35);let ys=lines.flatMap(l=>l[1]),xs=lines.flatMap(l=>l[0]);let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);if(xmax===xmin)xmax++;if(ymax===ymin)ymax++;const X=x=>45+405*(x-xmin)/(xmax-xmin),Y=y=>285-240*(y-ymin)/(ymax-ymin);g.strokeRect(45,45,405,240);lines.forEach(([x,y,label],i)=>{g.strokeStyle=['#1268c4','#777','#eb8700'][i];g.beginPath();x.forEach((v,k)=>k?g.lineTo(X(v),Y(y[k])):g.moveTo(X(v),Y(y[k])));g.stroke();g.fillStyle=g.strokeStyle;g.fillText(label,60+i*125,303)});g.fillStyle='#000';g.fillText(ymin.toFixed(2),4,285);g.fillText(ymax.toFixed(2),4,49)}
function show(){if(!details.length)return;const d=details[Number(sel.value)];document.getElementById('plots').innerHTML='';document.getElementById('metrics').textContent=JSON.stringify(d.metrics,null,2);for(const k of ['global','within']){let c=d[k+'_curve'],o=d['oracle_'+k],x=c[0].map(v=>1-v);plot(k+' referral',[[x,c[1],'Head'],[x,c[2],'Random'],[o[0].map(v=>1-v),o[1],'Oracle']],'Fraction referred','Success')}let c=d.global_curve,loss=Array.from({length:81},(_,i)=>i*.0025),h=c[1][0];plot('Automation',[[loss,loss.map(l=>Math.max(0,...c[0].filter((v,i)=>c[1][i]>=h-l-1e-12))),'Head']],'Allowed loss','Automated');let order=d.task_uncertainty.map((v,i)=>i).sort((a,b)=>d.task_uncertainty[a]-d.task_uncertainty[b]);plot('Task uncertainty versus success',[[order.map(i=>d.task_uncertainty[i]),order.map(i=>d.task_success[i]),'Tasks']],'Task uncertainty','Success')};sel.onchange=show;show();document.querySelectorAll('th').forEach(th=>th.onclick=()=>{const tb=th.closest('table').tBodies[0],i=th.cellIndex;th.asc=!th.asc;Array.from(tb.rows).sort((a,b)=>{let x=a.cells[i].innerText,y=b.cells[i].innerText;let nx=parseFloat(x),ny=parseFloat(y);return(th.asc?1:-1)*(Number.isFinite(nx)&&Number.isFinite(ny)?nx-ny:x.localeCompare(y))}).forEach(r=>tb.append(r))});</script>'''
    (output/'report.html').write_text(body)
    availability = f'Generated {stamp}. Interim snapshot. {len(details)} head/checkpoint results; '+str(len(acquisition_summaries))+' acquisition methods have at least one fully evaluated seed.\n\n'
    (output/'report.md').write_text('# Final SAVE study\n\n'+availability+'Open `report.html` for the portable, sortable report with embedded plots.\n\n'
        'Raw all-round head metrics: `all_rounds.csv`. SAVE curves: `save_acquisition.csv`. '
        'Per-checkpoint seed coverage, mean and sample SD: `save_acquisition_summary.csv`. '
        'Seed SD, evaluation-only and total 95% intervals, paired differences, and quality audits: `summary.json`.\n')
