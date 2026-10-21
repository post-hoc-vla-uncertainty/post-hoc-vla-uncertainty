"""Q5 metrics (E1, E2, E3) recomputed from the per-observation export in data/q5, and checked against it.

Requires numpy and pandas (and scripts/analysis.py for the referral metrics). Reads only data/q5:
  e1_scores.csv.gz, e1_scores_dense.csv.gz   raw head scores of every E1 observation (support: initial / dense)
  e3_traces.csv.gz, e3_episodes.csv          raw head scores every 5 steps of every E3 run, and the run outcomes
  e2_conditions.csv                          primary and supervisor outcomes of every E2 condition
and recomputes the summaries that the research repository exported (e1_summary.json, e2_summary.json,
e3_summary.json): within-task AUROC (E1), window AUROC, percentile rise, detection and false alarms (E3), referral
nAUC, automated fraction and failure AUROC (E2), each with the hierarchical bootstrap (seeds, then initial states
or pairs within tasks; 2,000 draws in the research code's order). It aborts if a recomputed value differs from the
export by more than the tolerance.

It then recomputes everything with near-identical scores treated as ties. Kernel variances are computed in float32 as
1 - |v|^2, so scores at the variance floor (e.g. the raw instruction head, about 0.000999 for every training
instruction) differ only in their last bits (by up to 8e-7); ordering them by these differences gives spurious
AUROCs. Before any ranking, the scores of each kernel head are grouped greedily from the smallest, each group spanning
at most TIE_TOL (1e-6, about seventeen float32 steps below 1), and replaced by the group's smallest value. VFD is not
at a floor and is left unchanged. The tie-corrected summaries, used by the paper, are written
to data/derived/q5_*_summary.json, together with paired differences of the paper heads against VFD
(data/derived/q5_paired.json, same bootstrap draws for both scores).

    python3 scripts/q5_analysis.py
"""
from pathlib import Path
from collections import defaultdict
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as paper  # noqa: E402  (referral and within_failure_auroc of the main study)

ROOT = Path(__file__).resolve().parents[1]
DATA, DERIVED = ROOT/'data/q5', ROOT/'data/derived'
SEEDS, TASKS, BOOT = (0, 1, 2), tuple(range(10)), 2000
CONTEXT = ('pre_ws', 'pre_wrist', 'pre_lang', 'pre_state', 'ctx_ws', 'ctx_wrist', 'ctx_lang', 'ctx_state')
INPUTS = ('kv_ctx', 'kv_dec', *CONTEXT)
HEADS = (*INPUTS, 'comb', 'vfd')
LIBRARY = json.loads((ROOT/'handoff/q5/perturbations.json').read_text())


TIE_TOL = 1e-6


def snap(values, tol=TIE_TOL):
    """Group the sorted scores greedily from the smallest, each group spanning at most tol, and replace every score by
    the smallest value of its group. Unlike single linkage, this never chains: scores more than tol apart stay apart,
    so densely spaced but genuinely different scores (e.g. the raw state head around 1e-5) keep their order."""
    v = np.asarray(values, float)
    if not len(v):
        return v
    sorted_v = np.sort(v)
    anchors, i = [], 0
    while i < len(sorted_v):
        anchors.append(sorted_v[i])
        i = np.searchsorted(sorted_v, sorted_v[i]+tol, 'right')
    anchors = np.array(anchors)
    return anchors[np.searchsorted(anchors, v, 'right')-1]


def snap_frame(d, columns, ties):
    """Numeric copies of the kernel-head columns, snapped per head when ties is set."""
    d = d.copy()
    for h in columns:
        values = pd.to_numeric(d[h], errors='coerce').astype(float)
        if ties:
            ok = values.notna()
            values.loc[ok] = snap(values[ok].values)
        d[h] = values
    return d


def midrank(ref, x):
    """Percentile among the reference values, ties counted half (a constant head sits at 50)."""
    return 100*(np.searchsorted(ref, x, 'left')+np.searchsorted(ref, x, 'right'))/(2*len(ref))


def auroc(pos, neg):
    if not len(pos) or not len(neg):
        return np.nan
    d = pos[:, None]-neg[None, :]
    return float((d > 0).mean()+.5*(d == 0).mean())


# ---------------------------------------------------------------- E1
E1_TEST, E1_CALIB, E1_VFD = range(0, 25), range(25, 50), range(0, 10)
# Variant order of the research code (q5_e1.FAMILIES): clean, then each family's levels.
E1_LEVELS = {'I1': ['P2', 'P3', 'P4', 'P5'], 'I2': ['one', 'all'], 'I3': ['please', 'could_you', 'typo'], 'I4': ['pronouns'],
             'I5': ['negated', 'shuffled', 'german', 'empty'],
             'V1': ['x0.8', 'x0.6', 'x0.4', 'x1.4', 'x2.0', 'warm', 'cool', 'dir45'],
             'V2': ['sigma0.01', 'sigma0.03', 'sigma0.08', 'blur3', 'blur7', 'blur11', 'jpeg60', 'jpeg25', 'jpeg10'],
             'V3': ['1cm1deg', '2cm3deg', '4cm6deg'], 'V4': ['25', '50', '75'], 'V5': ['table', 'tint30', 'tint90'],
             'V6': ['blackout', 'blur15'], 'V7': ['noise0.35', 'replaced', 'grey', 'swapped', 'mirrored'],
             'S1': ['3cm', '6cm', '10cm'], 'S2': ['far', 'near'], 'S3': ['30deg', '90deg', '180deg'],
             'R1': ['0.05rad', '0.1rad', '0.2rad'], 'R2': ['1cm', '3cm', '6cm'], 'M1': ['same_scene', 'other_scene'],
             'M2': ['same_task'], 'M3': ['same_task', 'other_task']}
E1_VARIANTS = [f'{f}:{lv}' for f, levels in E1_LEVELS.items() for lv in levels]
PAPER_HEADS = ('kv_ctx', 'kv_dec')
PAIRED_E1 = tuple(f'{h}@vfd' for h in PAPER_HEADS)  # paper heads on the same states as VFD (0-9), for paired differences


def e1_rows(path, seed, ties=False):
    d = pd.read_csv(path, keep_default_na=False)
    d = snap_frame(d[d.seed == seed], INPUTS, ties)
    rows = []
    for r in d.itertuples(index=False):
        row = {'task': int(r.task.split('/')[1]), 'state': int(r.init_state_id), 'variant': r.variant}
        for h in INPUTS:
            row[h] = float(getattr(r, h))
        row['vfd'] = float(r.vfd) if r.vfd != '' else np.nan
        rows.append(row)
    for task in TASKS:  # percentiles among the task's clean calibration observations; COMB from the context heads
        calib = [r for r in rows if r['task'] == task and r['state'] in E1_CALIB and r['variant'] == 'clean']
        ref = {h: np.sort([r[h] for r in calib]) for h in (*INPUTS, 'vfd')}
        task_rows = [r for r in rows if r['task'] == task]
        for r in task_rows:
            for h, values in ref.items():
                r[f'pct_{h}'] = float(midrank(values, r[h]))
            r['comb'] = max(r[f'pct_{h}'] for h in CONTEXT)
        comb = np.sort([r['comb'] for r in calib])
        for r in task_rows:
            r['pct_comb'] = float(midrank(comb, r['comb']))
    return rows


def e1_seed_stats(rows, counts):
    index = {(r['task'], r['state'], r['variant']): r for r in rows}
    states = {t: [s for s in E1_TEST if (t, s, 'clean') in index] for t in TASKS}
    point, draws, pct, ntasks = {}, {}, {}, {}
    for v in E1_VARIANTS:
        for h in (*HEADS, *PAIRED_E1):
            base = h.split('@')[0]
            p_task, b_task, pc = [], [], []
            for t, st in states.items():
                keep = [i for i, s in enumerate(st) if (h != 'vfd' and '@' not in h) or s in E1_VFD]
                pert = [i for i in keep if (t, st[i], v) in index]
                if not pert:
                    continue
                p = np.array([index[t, st[i], v][base] for i in pert])
                c = np.array([index[t, st[i], 'clean'][base] for i in keep])
                if np.isnan(p).all() or np.isnan(c).all():
                    continue
                m = (p[:, None] > c[None, :])+.5*(p[:, None] == c[None, :])
                p_task.append(m.mean())
                wp, wc = counts[t][:, pert], counts[t][:, keep]
                with np.errstate(invalid='ignore', divide='ignore'):
                    b_task.append(np.einsum('bi,ij,bj->b', wp, m, wc)/(wp.sum(1)*wc.sum(1)))
                pc.append(np.nanmean([index[t, st[i], v][f'pct_{base}'] for i in pert]))
            if p_task:
                point[v, h] = float(np.mean(p_task))
                draws[v, h] = np.nanmean(np.stack(b_task), axis=0)
                pct[v, h] = float(np.mean(pc))
                ntasks[v, h] = len(p_task)
    return point, draws, pct, ntasks


def e1(ties=False):
    rng = np.random.default_rng(0)
    seed_draws = rng.integers(0, len(SEEDS), size=(BOOT, len(SEEDS)))
    out, paired = {}, {}
    for support, name in (('initial', 'e1_scores.csv.gz'), ('dense', 'e1_scores_dense.csv.gz')):
        stats = {}
        for seed in SEEDS:
            rows = e1_rows(DATA/name, seed, ties)
            counts = {t: np.stack([np.bincount(rng.integers(0, 25, 25), minlength=25) for _ in range(BOOT)]).astype(float)
                      for t in TASKS}
            stats[seed] = e1_seed_stats(rows, counts)
        summary = {}
        for v in E1_VARIANTS:
            summary[v] = {}
            for h in HEADS:
                vals = [stats[s][0].get((v, h)) for s in SEEDS]
                if any(x is None for x in vals):
                    summary[v][h] = {'auroc': None, 'pct': None, 'ci': None, 'sd': None, 'tasks': 0}
                    continue
                d = np.stack([stats[s][1][v, h] for s in SEEDS])
                hier = d[seed_draws, np.arange(BOOT)[:, None]].mean(axis=1)
                hier = hier[~np.isnan(hier)]
                summary[v][h] = {'auroc': float(np.mean(vals)), 'sd': float(np.std(vals, ddof=1)),
                                 'per_seed': [float(x) for x in vals],
                                 'pct': float(np.mean([stats[s][2][v, h] for s in SEEDS])),
                                 'ci': [float(np.quantile(hier, .025)), float(np.quantile(hier, .975))],
                                 'tasks': int(stats[SEEDS[0]][3][v, h])}
        out[support] = summary
        paired[support] = {}
        for v in E1_VARIANTS:
            paired[support][v] = {}
            for h in PAPER_HEADS:
                if any((v, f'{h}@vfd') not in stats[s][0] or (v, 'vfd') not in stats[s][0] for s in SEEDS):
                    continue
                diffs = [stats[s][0][v, f'{h}@vfd']-stats[s][0][v, 'vfd'] for s in SEEDS]
                d = np.stack([stats[s][1][v, f'{h}@vfd']-stats[s][1][v, 'vfd'] for s in SEEDS])
                hier = d[seed_draws, np.arange(BOOT)[:, None]].mean(axis=1)
                hier = hier[~np.isnan(hier)]
                paired[support][v][f'{h}-vfd'] = {'mean': float(np.mean(diffs)), 'sd': float(np.std(diffs, ddof=1)),
                                                  'ci': [float(np.quantile(hier, .025)), float(np.quantile(hier, .975))]}
    return {'variants': out}, paired


# ---------------------------------------------------------------- E3
E3_STATES, E3_CALIB, ONSET, WINDOW = (0, 1), (2, 3, 4), 100, 50
E3_EVENTS = [('I1', False), ('M1', False), ('I5', False), ('V1', False), ('V3', False), ('V4', True), ('V6', True),
             ('V7', False), ('S1', False), ('S2', False), ('S4', False), ('R3', False), ('D1', False)]


def e3_runs(traces, episodes, seed, tasks):
    t = traces[(traces.seed == seed) & (traces['head'].isin([*INPUTS, 'vfd']))]  # already snapped in e3() if ties
    runs = {}
    for (task, state, event), g in t.groupby(['task', 'init_state_id', 'event']):
        task = int(task.split('/')[1])
        if task not in tasks:
            continue
        wide = g.pivot_table(index='env_step', columns='head', values='score', aggfunc='first').sort_index()
        r = {'steps': wide.index.values, **{h: wide[h].values.astype(float) for h in (*INPUTS, 'vfd')}}
        ep = episodes[(episodes.seed == seed) & (episodes.task == f'libero_10/{task}') & (episodes.init_state_id == state)
                      & (episodes.event == event)].iloc[0]
        r['success'] = str(ep.success) == 'True'
        runs[task, int(state), event] = r
    for task in tasks:
        clean = [runs[task, s, 'clean'] for s in E3_CALIB if (task, s, 'clean') in runs]
        ref = {h: np.sort(np.concatenate([r[h] for r in clean])) for h in (*INPUTS, 'vfd')}
        for key, r in runs.items():
            if key[0] != task:
                continue
            for h in ref:
                r[f'pct_{h}'] = midrank(ref[h], r[h])
            r['comb'] = np.max([r[f'pct_{h}'] for h in CONTEXT], axis=0)
        comb_ref = np.sort(np.concatenate([r['comb'] for r in clean]))
        for key, r in runs.items():
            if key[0] == task:
                r['pct_comb'] = midrank(comb_ref, r['comb'])
                r['thr'] = {h: np.quantile(np.concatenate([c[h] for c in clean]), .99) for h in HEADS}
    return runs


def e3_pair(e, c, head, transient):
    se, sc = e['steps'], c['steps']
    common = np.intersect1d(se[(se >= ONSET) & (se < ONSET+WINDOW)], sc)
    if not len(common):
        return {}
    ie, ic = np.searchsorted(se, common), np.searchsorted(sc, common)
    pe, pc = e[f'pct_{head}'][ie], c[f'pct_{head}'][ic]
    window = se[(se >= ONSET) & (se < ONSET+WINDOW)]
    above = window[e[head][np.searchsorted(se, window)] > e['thr'][head]]
    out = {'auroc': auroc(e[head][ie], c[head][ic]), 'rise': float(pe.mean()-pc.mean()), 'pct_event': float(pe.mean()),
           'pct_clean': float(pc.mean()), 'detect': float(len(above) > 0),
           'delay': float(above[0]-ONSET) if len(above) else np.nan}
    if transient:
        after = np.intersect1d(se[(se >= ONSET+60) & (se < ONSET+110)], sc)
        if len(after):
            out['recovery'] = float(e[f'pct_{head}'][np.searchsorted(se, after)].mean()
                                    - c[f'pct_{head}'][np.searchsorted(sc, after)].mean())
    return out


def e3(tasks=TASKS, ties=False):
    traces = pd.read_csv(DATA/'e3_traces.csv.gz')
    if ties:
        for h in INPUTS:
            rows = traces['head'] == h
            traces.loc[rows, 'score'] = snap(traces.loc[rows, 'score'].astype(float).values)
    episodes = pd.read_csv(DATA/'e3_episodes.csv')
    rng = np.random.default_rng(0)
    per_seed, false_alarm = {}, {}
    for seed in SEEDS:
        runs = e3_runs(traces, episodes, seed, tasks)
        false_alarm[seed] = {h: float(np.mean([(runs[t, s, 'clean'][h] > runs[t, s, 'clean']['thr'][h]).any()
                                               for t in tasks for s in E3_STATES if (t, s, 'clean') in runs])) for h in HEADS}
        per_seed[seed] = {}
        for event, transient in E3_EVENTS:
            pairs = [(t, s) for t in tasks for s in E3_STATES if (t, s, event) in runs and (t, s, 'clean') in runs]
            per_seed[seed][event] = {
                'pairs': pairs,
                'success_event': float(np.mean([runs[p+(event,)]['success'] for p in pairs])),
                'success_clean': float(np.mean([runs[p+('clean',)]['success'] for p in pairs])),
                'values': {h: [e3_pair(runs[p+(event,)], runs[p+('clean',)], h, transient) for p in pairs] for h in HEADS}}
    seed_draws = rng.integers(0, len(SEEDS), size=(BOOT, len(SEEDS)))
    summary = {}
    for event, transient in E3_EVENTS:
        summary[event] = {'success_clean': float(np.mean([per_seed[s][event]['success_clean'] for s in SEEDS])),
                          'success_event': float(np.mean([per_seed[s][event]['success_event'] for s in SEEDS])),
                          'pairs_per_seed': [len(per_seed[s][event]['pairs']) for s in SEEDS], 'heads': {}}
        pair_draws = {s: rng.integers(0, len(per_seed[s][event]['pairs']), size=(BOOT, len(per_seed[s][event]['pairs'])))
                      for s in SEEDS}
        for h in HEADS:
            o = {}
            for metric in ('auroc', 'rise', 'pct_event', 'pct_clean', 'detect', 'delay', 'recovery'):
                arrays = {s: np.array([v.get(metric, np.nan) for v in per_seed[s][event]['values'][h]], float) for s in SEEDS}
                if all(np.isnan(a).all() for a in arrays.values()):
                    continue
                with np.errstate(invalid='ignore'), np.testing.suppress_warnings() as sup:
                    sup.filter(RuntimeWarning)
                    means = [float(np.nanmedian(a)) if metric == 'delay' else float(np.nanmean(a)) for a in arrays.values()]
                    o[metric] = float(np.nanmean(means))
                    o[f'{metric}_sd'] = float(np.nanstd(means, ddof=1))
                    if metric in ('auroc', 'rise', 'detect'):
                        boot = np.stack([np.nanmean(arrays[s][pair_draws[s]], axis=1) for s in SEEDS])
                        hier = boot[seed_draws, np.arange(BOOT)[:, None]].mean(axis=1)
                        o[f'{metric}_ci'] = [float(np.nanquantile(hier, .025)), float(np.nanquantile(hier, .975))]
            o['false_alarm'] = float(np.mean([false_alarm[s][h] for s in SEEDS]))
            summary[event]['heads'][h] = o
        summary[event]['paired'] = {}
        for h in PAPER_HEADS:
            arrays = {s: np.array([a.get('auroc', np.nan)-b.get('auroc', np.nan) for a, b in
                                   zip(per_seed[s][event]['values'][h], per_seed[s][event]['values']['vfd'])], float) for s in SEEDS}
            means = [float(np.nanmean(a)) for a in arrays.values()]
            boot = np.stack([np.nanmean(arrays[s][pair_draws[s]], axis=1) for s in SEEDS])
            hier = boot[seed_draws, np.arange(BOOT)[:, None]].mean(axis=1)
            summary[event]['paired'][f'{h}-vfd'] = {'mean': float(np.mean(means)), 'sd': float(np.std(means, ddof=1)),
                                                    'ci': [float(np.nanquantile(hier, .025)), float(np.nanquantile(hier, .975))]}
    return {'events': summary}


# ---------------------------------------------------------------- E2
E2_STATES = tuple(range(10))
E2_SCORES = (*INPUTS[:2], *CONTEXT, 'vfd', 'comb', 'comb_mean', 'cross_replicate')


def e2_tables(conditions, scores_path, ties=False):
    s = snap_frame(pd.read_csv(scores_path, keep_default_na=False), INPUTS, ties)
    calib = s[(s.split == 'calibration') & (s.variant == 'clean')]
    ref = {(int(seed), int(task.split('/')[1])): {h: np.sort(g[h].astype(float).values) for h in CONTEXT}
           for (seed, task), g in calib.groupby(['seed', 'task'])}
    test = s[(s.split == 'test') & (s.init_state_id < 10)]
    scores = {(int(r.seed), int(r.task.split('/')[1]), int(r.init_state_id), r.variant):
              {h: (float(getattr(r, h)) if getattr(r, h) != '' else np.nan) for h in ('vfd',)} |
              {h: float(getattr(r, h)) for h in INPUTS}
              for r in test.itertuples(index=False)}
    tables = {}
    for seed in SEEDS:
        c = conditions[conditions.seed == seed]
        rollouts = {(int(r.task.split('/')[1]), int(r.init_state_id), r.variant):
                    {'primary': [r.primary_rep0, r.primary_rep1, r.primary_rep2],
                     'supervisor': [r.supervisor_rep0, r.supervisor_rep1, r.supervisor_rep2]} for r in c.itertuples(index=False)}
        keys = sorted(k for k in rollouts if (seed, *k) in scores)
        t = {'keys': keys, 'tasks': np.array([f'libero_10/{k[0]}' for k in keys]), 'state': np.array([k[1] for k in keys]),
             'primary': np.array([np.mean(rollouts[k]['primary']) for k in keys]),
             'supervisor': np.array([np.mean(rollouts[k]['supervisor']) for k in keys])}
        for h in (*INPUTS, 'vfd'):
            t[h] = np.array([scores[(seed, *k)][h] for k in keys])
        pct = np.array([[midrank(ref[seed, k[0]][h], scores[(seed, *k)][h]) for h in CONTEXT] for k in keys])
        t['comb'], t['comb_mean'] = pct.max(axis=1), pct.mean(axis=1)
        t['cross_replicate'] = np.array([rollouts[k]['supervisor'][0]-rollouts[k]['primary'][0] for k in keys])
        t['primary_eval'] = np.array([np.mean(rollouts[k]['primary'][1:]) for k in keys])
        t['supervisor_eval'] = np.array([np.mean(rollouts[k]['supervisor'][1:]) for k in keys])
        tables[seed] = t
    return tables


def e2_metrics(t, idx=None):
    sel = slice(None) if idx is None else idx
    out = {}
    for name in E2_SCORES:
        p, s = (t['primary_eval'], t['supervisor_eval']) if name == 'cross_replicate' else (t['primary'], t['supervisor'])
        score = t[name][sel]
        if np.isnan(score).any():
            continue
        r, _, _ = paper.referral(score, p[sel], s[sel], t['tasks'][sel])
        out[name] = {'within_nauc': r['within_nauc'], 'global_nauc': r['global_nauc'], 'automation': r['automation'],
                     'failure_auroc': paper.within_failure_auroc(score, p[sel], t['tasks'][sel])}
    return out


def e2_boot_indices(t, rng):
    idx = []
    for task in paper.TASKS:
        members = {st: np.flatnonzero((t['tasks'] == task) & (t['state'] == st)) for st in E2_STATES}
        draws = rng.integers(len(E2_STATES), size=(BOOT, len(E2_STATES)))
        idx.append(np.stack([np.concatenate([members[E2_STATES[i]] for i in row]) for row in draws]))
    return np.concatenate(idx, axis=1)


def e2(ties=False):
    conditions = pd.read_csv(DATA/'e2_conditions.csv')
    rng = np.random.default_rng(0)
    seed_pick = rng.integers(len(SEEDS), size=(len(SEEDS), BOOT))
    out, paired = {}, {}
    for support, name in (('dense', 'e1_scores_dense.csv.gz'), ('initial', 'e1_scores.csv.gz')):
        tables = e2_tables(conditions, DATA/name, ties)
        point = {s: e2_metrics(tables[s]) for s in SEEDS}
        draws = {}
        for s in SEEDS:
            idx = e2_boot_indices(tables[s], rng)
            draws[s] = [e2_metrics(tables[s], idx[b]) for b in range(BOOT)]
        summary = {}
        for name in E2_SCORES:
            summary[name] = {}
            for m in ('within_nauc', 'global_nauc', 'automation', 'failure_auroc'):
                vals = [point[s][name][m] for s in SEEDS]
                d = np.array([[draws[s][b][name][m] for b in range(BOOT)] for s in SEEDS])
                total = d[seed_pick, np.arange(BOOT)[None, :]].mean(0)
                summary[name][m] = {'mean': float(np.nanmean(vals)), 'sd': float(np.nanstd(vals, ddof=1)),
                                    'per_seed': [float(v) for v in vals],
                                    'ci': [float(np.nanquantile(total, .025)), float(np.nanquantile(total, .975))]}
        out[support] = summary
        paired[support] = {}
        for h in PAPER_HEADS:
            paired[support][f'{h}-vfd'] = {}
            for m in ('within_nauc', 'global_nauc', 'automation', 'failure_auroc'):
                vals = [point[s][h][m]-point[s]['vfd'][m] for s in SEEDS]
                d = np.array([[draws[s][b][h][m]-draws[s][b]['vfd'][m] for b in range(BOOT)] for s in SEEDS])
                total = d[seed_pick, np.arange(BOOT)[None, :]].mean(0)
                paired[support][f'{h}-vfd'][m] = {'mean': float(np.nanmean(vals)), 'sd': float(np.nanstd(vals, ddof=1)),
                                                  'ci': [float(np.nanquantile(total, .025)), float(np.nanquantile(total, .975))]}
    return {'supports': out}, paired


# ---------------------------------------------------------------- verification
def numbers(obj, prefix=''):
    """Flatten nested dicts/lists to {path: number}."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from numbers(v, f'{prefix}/{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from numbers(v, f'{prefix}[{i}]')
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and obj is not None:
        yield prefix, float(obj)


# E1 and E3 recompute from scores exported with six significant digits, which can merge or split a few ties:
# AUROC-scale values may move by up to about 1e-3, percentile-scale values (0-100) and delays (steps) by up to
# about 0.1. E2 reads the same exported scores as the research code and must agree exactly.
POINT_SCALE = ('pct', 'rise', 'pct_event', 'pct_clean', 'recovery', 'delay')


def tolerance_of(path, exact):
    if exact:
        return 1e-9
    leaf = path.rsplit('/', 1)[1].split('[')[0]
    return .1 if any(leaf == k or leaf.startswith(k + '_') for k in POINT_SCALE) else 2e-3


def verify(name, recomputed, exported, keys, exact):
    ref = dict(numbers({k: exported[k] for k in keys}))
    new = dict(numbers(recomputed))
    common = [k for k in ref if k in new and not (np.isnan(ref[k]) and np.isnan(new[k]))]
    diffs = np.array([abs(ref[k]-new[k]) for k in common])
    over = [k for k, d in zip(common, diffs) if d > tolerance_of(k, exact)]
    worst = common[int(np.nanargmax(diffs))] if len(diffs) else None
    print(f'{name}: {len(common)} values compared, max |difference| {np.nanmax(diffs):.2e}'
          + (f' at {worst}' if worst else '') + f', {len(over)} beyond tolerance')
    if over:
        raise SystemExit(f'{name}: recomputed values differ from data/q5 beyond tolerance, e.g. {over[:3]}')


def strip_paired(result):
    """Remove the paired entries that the export does not contain, for verification."""
    if 'events' in result:
        return {'events': {e: {k: v for k, v in x.items() if k != 'paired'} for e, x in result['events'].items()}}
    return result


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    skip = '--skip-verify' in sys.argv
    jobs = [('e1', e1, ['variants'], False), ('e3', e3, ['events'], False), ('e2', e2, ['supports'], True)]
    paired = {}
    for name, fn, keys, exact in jobs:
        export = json.loads((DATA/f'{name}_summary.json').read_text())
        if not skip:  # 1. the pipeline reproduces the research code's export
            result = fn()
            result = result[0] if isinstance(result, tuple) else result
            verify(name.upper(), strip_paired(result), export, keys, exact)
        result = fn(ties=True)  # 2. near-identical kernel scores treated as ties (used in the paper)
        if isinstance(result, tuple):
            result, paired[name] = result
        else:
            paired[name] = {e: x.pop('paired') for e, x in result['events'].items()}
        (DERIVED/f'q5_{name}_summary.json').write_text(json.dumps(result, indent=1)+'\n')
    (DERIVED/'q5_paired.json').write_text(json.dumps(paired, indent=1)+'\n')
    print('wrote tie-corrected data/derived/q5_{e1,e2,e3}_summary.json and data/derived/q5_paired.json')


if __name__ == '__main__':
    main()
