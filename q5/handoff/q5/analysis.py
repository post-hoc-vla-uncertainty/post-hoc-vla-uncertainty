"""Derived statistics and LaTeX tables from the frozen study tables in data/final_save_v1.

Requires numpy, pandas and scipy. Recomputes the study's referral and correlation
metrics, verifies them against the exported report values, then writes data/derived/*.csv
and tables/generated/*.tex. Run from anywhere:

    python3 scripts/analysis.py
"""
from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/final_save_v1'
DERIVED = ROOT/'data/derived'
TABLES = ROOT/'tables/generated'
SEEDS = [1000, 1001, 1002]
ROUNDS = [0, 10]
TASKS = [f'libero_10/{t}' for t in range(10)]
BASELINES = ['vfd', 'gu', 'action_l2', 'ace', 'decu', 'vlm_token_entropy', 'vlm_perplexity']
OURS = ['h1', 'h2']
LABEL = {'vfd': 'VFD', 'gu': 'GU', 'action_l2': 'Action-L2', 'ace': 'ACE', 'decu': 'DECU',
         'vlm_token_entropy': 'VLM entropy', 'vlm_perplexity': 'VLM perplexity',
         'h1': r'\KVctx{}', 'h2': r'\KVdec{}', 'h3': 'KV-PLS (decoder)', 'h4': 'KV-T (decoder)',
         'random': 'Random', 'h1_ivr': r'\KVctx{}+IVR', 'count': 'Demonstration count'}
# Policies needed at scoring time, and the kind of quantity each score measures.
POLICIES = {'vfd': 2, 'gu': 2, 'action_l2': 2, 'ace': 1, 'decu': 2, 'vlm_token_entropy': 1,
            'vlm_perplexity': 1, 'h1': 1, 'h2': 1}
KIND = {'vfd': 'E', 'gu': 'E', 'action_l2': 'E', 'decu': 'E', 'ace': 'A', 'vlm_token_entropy': 'A',
        'vlm_perplexity': 'A', 'h1': 'K', 'h2': 'K'}
# Römer et al. (arXiv:2606.18043v1), Table 1: mean over 15 rounds of a separate 2,000-update run.
PAPER_TABLE1 = {'action_l2': (0.50, 0.48), 'ace': (0.31, 0.36), 'decu': (0.31, 0.23), 'gu': (0.62, 0.65),
                'vlm_token_entropy': (0.10, 0.23), 'vlm_perplexity': (-0.04, 0.02), 'vfd': (0.71, 0.71)}
BOOT = 2000
trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # NumPy < 2 compatibility


# ---------------------------------------------------------------- referral metrics
def expected_ties(scores, values):
    """Sort ascending by score; replace values inside exact-score ties by their mean."""
    order = np.argsort(scores, kind='stable')
    s, v = scores[order], values[order]
    out = np.empty_like(v)
    start = 0
    for end in range(1, len(s)+1):
        if end == len(s) or s[end] != s[start]:
            out[start:end] = v[start:end].mean(); start = end
    return out


def curves(scores, primary, supervisor, tasks):
    """Automation curves as in the study: the lowest-uncertainty conditions return to the primary."""
    n = len(scores); gain = primary-supervisor
    hbar, lbar = supervisor.mean(), primary.mean()
    x = np.linspace(0, 1, n+1)
    y = hbar+np.concatenate([[0], np.cumsum(expected_ties(scores, gain)/n)])
    blocks = [np.flatnonzero(tasks == t) for t in TASKS]
    m = len(blocks[0]); grid = np.linspace(0, 1, m+1); within = np.zeros(m+1)
    for b in blocks:
        within += (supervisor[b].mean()+np.concatenate([[0], np.cumsum(expected_ties(scores[b], gain[b])/m)]))/len(blocks)
    return dict(global_=(x, y, hbar+x*(lbar-hbar)), within=(grid, within, hbar+grid*(lbar-hbar)))


def area(curve):
    x, y, ref = curve
    return trapezoid(y-ref, x)


def referral(scores, primary, supervisor, tasks):
    c = curves(scores, primary, supervisor, tasks)
    o = curves(supervisor-primary, primary, supervisor, tasks)
    out = {}
    for key, name in (('global_', 'global_nauc'), ('within', 'within_nauc')):
        d = area(o[key]); out[name] = area(c[key])/d if d > 1e-12 else np.nan
    x, y, _ = c['global_']
    out['automation'] = float(np.clip(x[y >= supervisor.mean()-.02-1e-12].max(), 0, 1))
    return out, c, o


def within_failure_auroc(scores, primary, tasks):
    """Macro AUROC over tasks; failures are fractional (1 - success over three executions)."""
    values = []
    for t in TASKS:
        b = tasks == t; s = scores[b]; f = 1-primary[b]; n = b.sum(); pos = f.sum()
        if 0 < pos < n:
            values.append(((rankdata(s)*f).sum()-pos*(pos+1)/2)/(pos*(n-pos)))
    return np.mean(values) if values else np.nan


def within_spearman(scores, target, tasks):
    values = [spearmanr(scores[tasks == t], target[tasks == t])[0] for t in TASKS
              if np.ptp(target[tasks == t]) > 0 and np.ptp(scores[tasks == t]) > 0]
    return np.nanmean(values) if values else np.nan


def task_correlations(task_values, task_success):
    x, y = np.asarray(task_values, float), np.asarray(task_success, float)
    return -np.corrcoef(rankdata(x), rankdata(y))[0, 1], -np.corrcoef(x, y)[0, 1]


# ---------------------------------------------------------------- helpers
def cohort(conditions, seed, rnd):
    d = conditions[(conditions.seed == seed) & (conditions['round'] == rnd)].sort_values('condition')
    return d.reset_index(drop=True)


def boot_indices(tasks, rng):
    """Resample 50 condition clusters with replacement inside each task."""
    idx = []
    for t in TASKS:
        members = np.flatnonzero(tasks == t)
        idx.append(members[rng.integers(len(members), size=(BOOT, len(members)))])
    return np.concatenate(idx, axis=1)


def summarize(values_by_seed, draws_by_seed=None, rng=None):
    """Mean and sample SD across seeds; with draws, the evaluation-plus-seed 95% interval."""
    v = np.asarray(values_by_seed, float)
    out = dict(mean=np.nanmean(v), sd=np.nanstd(v, ddof=1))
    if draws_by_seed is not None:
        draws = np.asarray(draws_by_seed)            # seeds x BOOT
        pick = rng.integers(len(v), size=(len(v), BOOT))
        total = draws[pick, np.arange(BOOT)[None, :]].mean(0)
        out['lo'], out['hi'] = np.nanquantile(total, [.025, .975])
    return out


def nz(x, digits):
    return round(float(x), digits)+0.0  # avoid printing -0.00


def pm(mean, sd, digits=2):
    mean = nz(mean, digits)
    return f'${mean:.{digits}f}\\pm{sd:.{digits}f}$' if np.isfinite(sd) else f'${mean:.{digits}f}$'


def num(x, digits=2):
    return f'${nz(x, digits):.{digits}f}$'


def bold_best(cells, values, higher=True):
    finite = [v for v in values if np.isfinite(v)]
    best = max(finite) if higher else min(finite)
    return [f'{{\\boldmath{c}}}' if np.isfinite(v) and abs(v-best) < 1e-12 else c for c, v in zip(cells, values)]


# ---------------------------------------------------------------- analyses
def verify(conditions, pools, reported):
    """Recompute every reported head metric from the frozen tables."""
    worst = 0.0
    for seed, rnd in itertools.product(SEEDS, ROUNDS):
        d = cohort(conditions, seed, rnd); tasks = d.task.values
        p, s = d.primary.values, d.supervisor.values
        rates = d.groupby('task').primary.mean().reindex(TASKS).values
        for head in reported['head'].unique():
            r = reported[(reported['head'] == head) & (reported.seed == seed) & (reported['round'] == rnd)].iloc[0]
            m, _, _ = referral(d[head].values, p, s, tasks)
            pool = pools[(pools.seed == seed) & (pools['round'] == rnd) & (pools['head'] == head)].set_index('task').pool_mean.reindex(TASKS)
            m['neg_spearman'], m['neg_pearson'] = task_correlations(pool.values, rates)
            for k, v in m.items():
                worst = max(worst, abs(v-float(r[k])))
    if worst > 1e-9:
        raise AssertionError(f'Recomputed metrics differ from the report by {worst:.3g}')
    return worst


def demonstration_count(conditions, selections, pools_info):
    """Uncertainty = 1/(1+unique demonstrations per task in the scoring support)."""
    prep = pools_info.set_index('task').preparatory_demos
    rows = []
    for seed, rnd in itertools.product(SEEDS, ROUNDS):
        sel = selections[(selections.arm == 'random') & (selections.seed == seed) & (selections['round'] <= rnd)]
        new = sel[sel.preparatory == 0].drop_duplicates('demo_id').groupby('task').size()
        counts = np.array([prep[t]+new.get(t, 0) for t in range(10)])
        d = cohort(conditions, seed, rnd)
        rates = d.groupby('task').primary.mean().reindex(TASKS).values
        rho, r = task_correlations(1/(1+counts), rates)
        rows.append(dict(seed=seed, round=rnd, neg_spearman=rho, neg_pearson=r, counts=' '.join(map(str, counts))))
    return pd.DataFrame(rows)


def variance_decomposition(conditions):
    """Success variance across executions: between tasks, between conditions, within conditions."""
    rows = []
    policies = [(s, r, 'primary') for s in SEEDS for r in ROUNDS]+[(s, 0, 'supervisor') for s in SEEDS]
    for seed, rnd, role in policies:
        d = cohort(conditions, seed, rnd)
        y = d[[f'{role}_rep{i}' for i in range(3)]].values      # conditions x 3
        cond = y.mean(1); task = d.assign(c=cond).groupby('task').c.transform('mean').values
        total = y.var()
        between_task = np.mean((task-y.mean())**2)
        between_cond = np.mean((cond-task)**2)
        within = np.mean((y-cond[:, None])**2)
        reproducible = max(between_cond-within/2, 0.0)        # E[(3-run mean - p)^2] = within/2
        rows.append(dict(seed=seed, round=rnd if role == 'primary' else 'all', role=role, total=total,
                         between_task=between_task, between_condition=between_cond,
                         between_condition_reproducible=reproducible, within_condition=within,
                         reliability=reproducible/(reproducible+within/2) if reproducible+within > 0 else np.nan))
    return pd.DataFrame(rows)


def within_task(conditions, heads):
    """Within-task failure ranking and referral, with a cross-replicate reference for referral."""
    rng = np.random.default_rng(20261011)
    rows, draws = [], {}
    for seed, rnd in itertools.product(SEEDS, ROUNDS):
        d = cohort(conditions, seed, rnd); tasks = d.task.values
        p, s = d.primary.values, d.supervisor.values
        idx = boot_indices(tasks, rng)
        for head in heads:
            x = d[head].values
            m, _, _ = referral(x, p, s, tasks)
            rows.append(dict(head=head, seed=seed, round=rnd, within_nauc=m['within_nauc'],
                             within_failure_auroc=within_failure_auroc(x, p, tasks),
                             within_spearman_failure=within_spearman(x, 1-p, tasks),
                             within_spearman_gain=within_spearman(x, s-p, tasks)))
            draws[(head, seed, rnd)] = np.array([within_failure_auroc(x[i], p[i], tasks[i]) for i in idx])
        # Rank by the signed gain of one held-out execution, evaluate on the other two.
        P = d[[f'primary_rep{i}' for i in range(3)]].values; S = d[[f'supervisor_rep{i}' for i in range(3)]].values
        values = []
        for hold in range(3):
            keep = [i for i in range(3) if i != hold]
            jitter = 1e-9*np.random.default_rng(hold).random(len(d))
            m, _, _ = referral(S[:, hold]-P[:, hold]+jitter, P[:, keep].mean(1), S[:, keep].mean(1), tasks)
            values.append(m['within_nauc'])
        rows.append(dict(head='cross_replicate', seed=seed, round=rnd, within_nauc=np.mean(values)))
    return pd.DataFrame(rows), draws


def within_task_by_coverage(conditions, heads, rnd=10):
    """Within-task Spearman of score with primary failure, preparatory tasks 0-2 versus acquired tasks 3-9."""
    rows = []
    for seed in SEEDS:
        d = cohort(conditions, seed, rnd)
        for head in heads:
            for name, tasks in (('preparatory', TASKS[:3]), ('acquired', TASKS[3:])):
                values = [spearmanr(g[head], 1-g.primary)[0] for t, g in d.groupby('task')
                          if t in tasks and g.primary.nunique() > 1]
                rows.append(dict(head=head, seed=seed, round=rnd, tasks=name, spearman_failure=np.nanmean(values)))
    return pd.DataFrame(rows)


def coverage_ambiguity(conditions):
    """Does single-policy action entropy (ACE) follow coverage? Spearman across task means and within tasks."""
    rows = []
    for seed, rnd in itertools.product(SEEDS, ROUNDS):
        d = cohort(conditions, seed, rnd)
        for other in ('h1', 'vfd'):
            means = d.groupby('task')[['ace', other]].mean()
            rows.append(dict(seed=seed, round=rnd, pair=f'ace~{other}', task_level=spearmanr(means.ace, means[other])[0],
                             within_task=within_spearman(d.ace.values, d[other].values, d.task.values)))
    return pd.DataFrame(rows)


def rank_agreement(conditions, heads):
    rows = []
    for seed, rnd in itertools.product(SEEDS, ROUNDS):
        d = cohort(conditions, seed, rnd); tasks = d.task.values
        for a, b in itertools.combinations(heads, 2):
            rows.append(dict(seed=seed, round=rnd, a=a, b=b, spearman_all=spearmanr(d[a], d[b])[0],
                             spearman_within=within_spearman(d[a].values, d[b].values, tasks)))
    return pd.DataFrame(rows)


def acquisition(success, selections):
    rows = []
    for arm, g in success.groupby('arm'):
        for seed, h in g.groupby('seed'):
            h = h.sort_values('round')
            auc = trapezoid(h.success.values, h.query_slots.values)/75
            sel = selections[(selections.arm == arm) & (selections.seed == seed)]
            new = sel[sel.preparatory == 0].demo_id.nunique()
            row = dict(arm=arm, seed=seed, auc=auc, new_demos=new,
                       repeated_slots=int((sel.previously_acquired == 1).sum()+(sel.preparatory == 1).sum()))
            for _, r in h.iterrows(): row[f'success_r{int(r["round"])}'] = r.success
            rows.append(row)
    return pd.DataFrame(rows)


def costs(jobs):
    """Median wall-clock minutes of completed single-GPU jobs."""
    m = lambda q: jobs.query(q).seconds.median()/60 if len(jobs.query(q)) else np.nan
    rows = []
    for arm in ['random', 'vfd', 'gu', 'h1', 'h2', 'h3', 'h4', 'h1_ivr']:
        rows.append(dict(arm=arm,
                         acquisition_min=m(f"kind=='acquire' and arm=='{arm}' and round>0"),
                         primary_training_min=m(f"kind=='train' and arm=='{arm}' and head=='0'"),
                         companion_training_min=m(f"kind=='train' and arm=='{arm}' and head=='1'"),
                         scoring_500_min=m(f"kind=='score' and head=='{arm}'"),
                         devices=';'.join(sorted(jobs.query(f"kind=='acquire' and arm=='{arm}'").device.dropna().unique())),
                         peak_gib=jobs.query(f"kind=='acquire' and arm=='{arm}'").peak_reserved_gib.median()))
    for head in BASELINES[2:]:
        rows.append(dict(arm=head, scoring_500_min=m(f"kind=='score' and head=='{head}'")))
    return pd.DataFrame(rows)


def acquisition_diagnostics(diag, selections):
    """Internal: why the decoder-target heads failed as SAVE acquisition scores."""
    rows = []
    for arm, g in diag.groupby('arm'):
        if arm == 'gu': continue  # GU's raw scores are log-variances; SAVE shifts them before weighting.
        means = g[[f'task_mean_{t}' for t in range(10)]].values
        weights = np.where(means > 0, means, np.nan)**2.5
        weights = weights/np.nansum(weights, 1, keepdims=True)
        entropy = -np.nansum(weights*np.log(weights), 1)/np.log(10)
        ratio = (g[[f'selected_next_score_{r}' for r in range(5)]].values/g[[f'selected_score_{r}' for r in range(5)]].values).ravel()
        sel = selections[selections.arm == arm]
        rows.append(dict(arm=arm, task_weight_entropy_r1=np.nanmean(entropy[g['round'].values == 1]),
                         task_weight_entropy_r15=np.nanmean(entropy[g['round'].values == 15]),
                         task_mean_ratio_max_min=np.nanmedian(np.nanmax(means, 1)/np.nanmin(np.where(means > 0, means, np.nan), 1)),
                         score_after_acquisition_ratio=np.nanmedian(ratio),
                         repeated_or_preparatory_slot_fraction=((sel.previously_acquired == 1) | (sel.preparatory == 1)).mean(),
                         max_times_one_demo=sel.groupby(['seed', 'demo_id']).size().max()))
    return pd.DataFrame(rows)


def ivr_coverage(diag, pools_info):
    """Candidate-pool variance in the seven initially uncovered tasks under each arm's own head."""
    sizes = pools_info.set_index('task').candidate_pool
    rows = []
    for arm in ['h1', 'h1_ivr']:
        for (seed, rnd), g in diag[diag.arm == arm].groupby(['seed', 'round']):
            r = g.iloc[0]
            mean = sum(r[f'task_mean_{t}']*sizes[t] for t in range(3, 10))/sum(sizes[t] for t in range(3, 10))
            rows.append(dict(arm=arm, seed=seed, round=rnd, pool_mean_variance=mean,
                             pool_max_variance=max(r[f'task_max_{t}'] for t in range(3, 10))))
    return pd.DataFrame(rows)


def condition_agreement(conditions):
    """Do independently trained policies find the same conditions hard? Mean within-task Spearman."""
    rows = []
    for rnd in ROUNDS:
        d = conditions[conditions['round'] == rnd]
        primary = d.pivot_table(index='condition', columns='seed', values='primary')
        supervisor = d.pivot_table(index='condition', columns='seed', values='supervisor')
        tasks = np.array([c.rsplit('/', 1)[0] for c in primary.index])
        for a, b in itertools.combinations(SEEDS, 2):
            rows.append(dict(round=rnd, pair=f'{a}-{b}', kind='primary-primary',
                             spearman=within_spearman(primary[a].values, primary[b].values, tasks)))
            if rnd == 10:
                rows.append(dict(round='all', pair=f'{a}-{b}', kind='supervisor-supervisor',
                                 spearman=within_spearman(supervisor[a].values, supervisor[b].values, tasks)))
        for seed in SEEDS:
            rows.append(dict(round=rnd, pair=str(seed), kind='primary-supervisor',
                             spearman=within_spearman(primary[seed].values, supervisor[seed].values, tasks)))
    return pd.DataFrame(rows)


def exploratory_summary(exploratory):
    """Within-task referral across all configurations of the earlier head studies."""
    rows = []
    for (study, condition), g in exploratory.groupby(['study', 'condition']):
        for family, h in [('KV', g[g.method == 'kv']), ('KV-T', g[g.method == 'kv_t']),
                          ('paper scores', g[~g.method.isin(['kv', 'kv_t'])])]:
            w = h.within_nauc.dropna()
            if not len(w): continue
            rows.append(dict(study=study, condition=condition, family=family, configurations=len(w),
                             mean=w.mean(), median=w.median(), maximum=w.max(),
                             lower_bound_above_zero=int((h.within_nauc_total_low > 0).sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tables

# Complete tabular environments; \input inside a tabular would break \bottomrule.
TABULAR = {
    'calibration': ('@{}lcccccc@{}', r"""& & & \multicolumn{2}{c}{This study} & \multicolumn{2}{c}{Reported}\\
\cmidrule(lr){4-5}\cmidrule(l){6-7}
Score & Policies & Type & $-\rho_{\rm S}$ & $-r_{\rm P}$ & $-\rho_{\rm S}$ & $-r_{\rm P}$\\"""),
    'referral': ('@{}lcccc@{}', r'''& Global & Within-task & Automated & Within-task\\
Score & nAUC & nAUC (r10) & fraction & failure AUROC (r10)\\'''),
    'acquisition': ('@{}lccccccc@{}', r"""& & \multicolumn{4}{c}{Success (\%)} & & Overhead\\
\cmidrule(lr){3-6}
Acquisition & Policies & 5 & 10 & 15 & Mean & New & (min/round)\\"""),
    'per_round': ('@{}lcccccccc@{}', r"""& \multicolumn{4}{c}{Round 0} & \multicolumn{4}{c}{Round 10}\\
\cmidrule(lr){2-5}\cmidrule(l){6-9}
Score & $-\rho_{\rm S}$ & $-r_{\rm P}$ & Global & Within & $-\rho_{\rm S}$ & $-r_{\rm P}$ & Global & Within\\"""),
    'paired': ('@{}lccccc@{}', r"Difference & $-\rho_{\rm S}$ & $-r_{\rm P}$ & Global nAUC & Within nAUC & Automated fraction\\"),
    'exploratory': ('@{}llcccc@{}', r"Study and condition & Family & Configurations & Mean & Maximum & Interval $>0$\\"),
    'decoder': ('@{}lcccccccc@{}', r"Head & $-\rho_{\rm S}$ & $-r_{\rm P}$ & Global nAUC & Success (\%) & New & Ratio & Max. & Minutes\\"),
}


def write_tabular(name, lines):
    spec, header = TABULAR[name]
    body = [f'\\begin{{tabular}}{{{spec}}}', r'\toprule', header, r'\midrule', *lines, r'\bottomrule', r'\end{tabular}']
    (TABLES/f'{name}.tex').write_text('\n'.join(body)+'\n')

def table_calibration(summary, count):
    get = lambda h, m: summary[(summary['head'] == h) & (summary['round'] == 'mean_0_10') & (summary.metric == m)].iloc[0]
    heads = BASELINES+OURS
    rho = [get(h, 'neg_spearman') for h in heads]; r = [get(h, 'neg_pearson') for h in heads]
    c_rho = pm(count.neg_spearman.mean(), count.groupby('seed').neg_spearman.mean().std(ddof=1))
    c_r = pm(count.neg_pearson.mean(), count.groupby('seed').neg_pearson.mean().std(ddof=1))
    rho_cells = bold_best([pm(x['mean'], x.seed_sd) for x in rho], [x['mean'] for x in rho])
    r_cells = bold_best([pm(x['mean'], x.seed_sd) for x in r], [x['mean'] for x in r])
    lines = []
    for i, h in enumerate(heads):
        if h == OURS[0]: lines.append(r'\midrule')
        paper = PAPER_TABLE1.get(h)
        lines.append(f"{LABEL[h]} & {POLICIES[h]} & {KIND[h]} & {rho_cells[i]} & {r_cells[i]} & "
                     + (f"{num(paper[0])} & {num(paper[1])}" if paper else '-- & --') + r'\\')
    lines += [r'\midrule', f"{LABEL['count']}$^\\dagger$ & -- & -- & {c_rho} & {c_r} & -- & --\\\\"]
    write_tabular('calibration', lines)


def table_referral(summary, within):
    get = lambda h, rnd, m: summary[(summary['head'] == h) & (summary['round'] == rnd) & (summary.metric == m)].iloc[0]
    heads = BASELINES+OURS
    cols = [('mean_0_10', 'global_nauc'), ('10', 'within_nauc'), ('mean_0_10', 'automation')]
    cells, values = [], []
    for rnd, m in cols:
        xs = [get(h, rnd, m) for h in heads]
        values.append([x['mean'] for x in xs])
        cells.append(bold_best([pm(x['mean'], x.seed_sd) for x in xs], values[-1]))
    w = within[(within['round'] == 10)]
    au = [w[w['head'] == h].within_failure_auroc for h in heads]
    cells.append(bold_best([pm(a.mean(), a.std(ddof=1)) for a in au], [a.mean() for a in au]))
    lines = []
    for i, h in enumerate(heads):
        if h == OURS[0]: lines.append(r'\midrule')
        lines.append(f"{LABEL[h]} & " + ' & '.join(c[i] for c in cells)+r'\\')
    x = within[(within['head'] == 'cross_replicate') & (within['round'] == 10)].within_nauc
    lines += [r'\midrule', f"Cross-replicate reference$^\\ddagger$ & -- & {pm(x.mean(), x.std(ddof=1))} & -- & --\\\\"]
    write_tabular('referral', lines)


def table_acquisition(acq, cost):
    arms = ['random', 'vfd', 'gu', 'h1', 'h2']
    policies = {'random': 1, 'vfd': 2, 'gu': 2, 'h1': 1, 'h2': 1, 'h1_ivr': 1}
    cols = ['success_r5', 'success_r10', 'success_r15', 'auc']
    stats = {c: [(acq[acq.arm == a][c].mean(), acq[acq.arm == a][c].std(ddof=1)) for a in arms] for c in cols}
    cells = {c: bold_best([pm(100*m, 100*s, 1) for m, s in stats[c]],
                          [m for m, _ in stats[c]]) for c in cols}
    cost = cost.set_index('arm')
    lines = []
    for i, a in enumerate(arms):
        if a == 'h1': lines.append(r'\midrule')
        new = acq[acq.arm == a].new_demos
        overhead = cost.acquisition_min[a]+(cost.companion_training_min[a] if policies[a] == 2 else 0) if a != 'random' else 0.0
        lines.append(f"{LABEL[a]} & {policies[a]} & " + ' & '.join(cells[c][i] for c in cols)
                     + f" & {pm(new.mean(), new.std(ddof=1), 1)} & {('--' if a == 'random' else num(overhead, 1))}\\\\")
    write_tabular('acquisition', lines)


def table_per_round(summary):
    heads = BASELINES+OURS
    get = lambda h, rnd, m: summary[(summary['head'] == h) & (summary['round'] == rnd) & (summary.metric == m)].iloc[0]
    lines = []
    for h in heads:
        if h == OURS[0]: lines.append(r'\midrule')
        cells = []
        for rnd in ('0', '10'):
            for m in ('neg_spearman', 'neg_pearson', 'global_nauc', 'within_nauc'):
                x = get(h, rnd, m); cells.append(pm(x['mean'], x.seed_sd))
        lines.append(f"{LABEL[h]} & "+' & '.join(cells)+r'\\')
    write_tabular('per_round', lines)


def table_paired(paired):
    lines = []
    for head in OURS:
        for base in ('vfd', 'gu'):
            cells = []
            for m in ('neg_spearman', 'neg_pearson', 'global_nauc', 'within_nauc', 'automation'):
                x = paired[(paired['head'] == head) & (paired.baseline == base) & (paired.metric == m)].iloc[0]
                cells.append(f"${nz(x['mean'], 3):+.3f}$ $[{nz(x.total_lo, 3):+.3f}, {nz(x.total_hi, 3):+.3f}]$")
            lines.append(f"{LABEL[head]} $-$ {LABEL[base]} & "+' & '.join(cells)+r'\\')
    write_tabular('paired', lines)


def table_exploratory(exploratory):
    names = {('original_study', 'round_0'): 'Head study, round 0', ('original_study', 'round_10'): 'Head study, round 10',
             ('followup_study', 'original'): 'Follow-up, original data', ('followup_study', 'plus35'): 'Follow-up, +35 demos'}
    lines = []
    for key, label in names.items():
        for family in ('KV', 'KV-T', 'paper scores'):
            r = exploratory[(exploratory.study == key[0]) & (exploratory.condition == key[1]) & (exploratory.family == family)]
            if not len(r): continue
            r = r.iloc[0]
            lines.append(f"{label} & {family} & {int(r.configurations)} & {num(r['mean'])} & {num(r.maximum)} & {int(r.lower_bound_above_zero)}\\\\")
        if key != list(names)[-1]: lines.append(r'\addlinespace')
    write_tabular('exploratory', lines)


def table_decoder(summary, acq, cost, diagnostics):
    get = lambda h, rnd, m: summary[(summary['head'] == h) & (summary['round'] == rnd) & (summary.metric == m)].iloc[0]
    d = diagnostics.set_index('arm'); c = cost.set_index('arm')
    lines = []
    for h in ['h1', 'h3', 'h4']:
        cells = [pm(get(h, 'mean_0_10', m)['mean'], get(h, 'mean_0_10', m).seed_sd) for m in ('neg_spearman', 'neg_pearson', 'global_nauc')]
        a = acq[acq.arm == h]
        cells += [pm(100*a.success_r15.mean(), 100*a.success_r15.std(ddof=1), 1), pm(a.new_demos.mean(), a.new_demos.std(ddof=1), 1),
                  num(d.score_after_acquisition_ratio[h], 3), str(int(d.max_times_one_demo[h])), num(c.acquisition_min[h], 1)]
        lines.append(f"{LABEL[h]} & "+' & '.join(cells)+r'\\')
    write_tabular('decoder', lines)


def main():
    DERIVED.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    conditions = pd.read_csv(DATA/'conditions.csv')
    pools = pd.read_csv(DATA/'pool_task_means.csv')
    reported = pd.read_csv(DATA/'head_metrics_by_seed.csv')
    summary = pd.read_csv(DATA/'head_metrics_summary.csv')
    paired = pd.read_csv(DATA/'paired_differences.csv')
    success = pd.read_csv(DATA/'acquisition_success.csv')
    selections = pd.read_csv(DATA/'acquisition_selections.csv')
    diag = pd.read_csv(DATA/'acquisition_diagnostics.csv')
    jobs = pd.read_csv(DATA/'job_costs.csv', dtype={'head': str})
    pools_info = pd.read_csv(DATA/'libero10_pools.csv')

    worst = verify(conditions, pools, reported)
    count = demonstration_count(conditions, selections, pools_info)
    variance = variance_decomposition(conditions)
    heads = BASELINES+OURS+['h3', 'h4']
    within, draws = within_task(conditions, heads)
    agreement = rank_agreement(conditions, ['h1', 'h2', 'vfd', 'gu'])
    acq = acquisition(success, selections)
    cost = costs(jobs)
    diagnostics = acquisition_diagnostics(diag, selections)
    coverage = ivr_coverage(diag, pools_info)
    agreement_conditions = condition_agreement(conditions)
    by_coverage = within_task_by_coverage(conditions, ['h1', 'h2', 'vfd', 'gu'])
    ambiguity = coverage_ambiguity(conditions)
    exploratory = exploratory_summary(pd.read_csv(DATA/'exploratory_head_studies.csv'))

    rng = np.random.default_rng(7)
    within_summary = []
    for head in heads:
        for rnd in ROUNDS:
            w = within[(within['head'] == head) & (within['round'] == rnd)].sort_values('seed')
            auc = summarize(w.within_failure_auroc.values, [draws[(head, s, rnd)] for s in SEEDS], rng)
            within_summary.append(dict(head=head, round=rnd, within_failure_auroc=auc['mean'], sd=auc['sd'], lo=auc['lo'], hi=auc['hi'],
                                       within_spearman_failure=w.within_spearman_failure.mean(),
                                       within_spearman_gain=w.within_spearman_gain.mean(), within_nauc=w.within_nauc.mean()))
    for name, frame in [('demonstration_count', count), ('variance_decomposition', variance), ('within_task', within),
                        ('within_task_summary', pd.DataFrame(within_summary)), ('rank_agreement', agreement),
                        ('acquisition_by_seed', acq), ('costs', cost), ('acquisition_diagnostics', diagnostics),
                        ('ivr_coverage', coverage), ('condition_agreement', agreement_conditions),
                        ('within_task_by_coverage', by_coverage), ('coverage_ambiguity', ambiguity),
                        ('exploratory_within_task', exploratory)]:
        frame.to_csv(DERIVED/f'{name}.csv', index=False, float_format='%.6g')

    table_calibration(summary, count)
    table_referral(summary, within)
    table_acquisition(acq, cost)
    table_per_round(summary)
    table_paired(paired)
    table_exploratory(exploratory)
    table_decoder(summary, acq, cost, diagnostics)

    # Console summary of the numbers quoted in the prose.
    pd.set_option('display.width', 200)
    print(f'Recomputed report metrics: max abs difference {worst:.2e}')
    print('Demonstration count:', count[['neg_spearman', 'neg_pearson']].mean().round(3).to_dict(),
          'per round', count.groupby('round')[['neg_spearman', 'neg_pearson']].mean().round(3).to_dict())
    v = variance.assign(f_task=variance.between_task/variance.total, f_repro=variance.between_condition_reproducible/variance.total,
                        f_exec=variance.within_condition/variance.total)
    print(v.groupby(['role', 'round'])[['f_task', 'f_repro', 'f_exec', 'reliability']].agg(['min', 'max']).round(3))
    print(pd.DataFrame(within_summary).round(3).to_string())
    print(within[within['head'] == 'cross_replicate'].round(3).to_string())
    print(agreement.groupby(['round', 'a', 'b'])[['spearman_all', 'spearman_within']].mean().round(3))
    print(acq.groupby('arm').agg(['mean', 'std']).round(3).to_string())
    print(cost.round(2).to_string())
    print(diagnostics.round(3).to_string())
    print(agreement_conditions.groupby(['round', 'kind']).spearman.agg(['min', 'max', 'mean']).round(3))
    print(exploratory.round(3).to_string())
    print(by_coverage.groupby(['head', 'tasks']).spearman_failure.mean().unstack().round(3))
    print(ambiguity.groupby(['pair', 'round'])[['task_level', 'within_task']].mean().round(2))
    print(coverage.groupby(['arm', 'round'])[['pool_mean_variance', 'pool_max_variance']].mean().unstack(0).round(3).to_string())


if __name__ == '__main__':
    main()
