"""Configuration-level summaries; a feature seed is a replicate, never a leader."""
from copy import deepcopy
import re
import time

import numpy as np
import pandas as pd

from uq_extensions.common import identity
from . import analysis
from .bootstrap import METRICS, Plan, cached_metrics, combine, interval, interpolate_rows
from .data import describe


DIAGNOSTICS = ['mapping_r2_median', 'pls_target_explained', 'output_pca_variance',
               'optimizer_seconds', 'fit_feature_seconds', 'worker_seconds', 'peak_gpu_gib']


def configuration(recipe, row, *, numerics=True):
    canonical = re.sub(r'_s\d+(?=_|$)', '', recipe['id'])
    f, g = deepcopy(recipe['features']), deepcopy(recipe['gp'])
    f.pop('seed', None)
    if not numerics:g.pop('precision', None)
    spec = dict(id=canonical, method=recipe['method'], scope=recipe.get('scope'), group=recipe.get('group'), features=f, gp=g)
    if numerics:
        # Numerical changes do not masquerade as feature-seed variation.
        spec['numerics'] = dict(precision=row['precision'], inference=row['inference'],
            posterior_precision=row.get('posterior_precision') or row['precision'],
            escalated_jitter=row.get('max_kernel_jitter') if row['jitter_escalated'] else None)
    return canonical+'__'+identity(spec)[:10]


def summarize_metrics(row, observed, evaluation, total, prefix=''):
    """Require every observed seed to define a point metric; never silently drop one."""
    observed = np.asarray(observed, float)
    for i, metric in enumerate(METRICS):
        key = prefix+metric;values = observed[:, i];defined = np.isfinite(values)
        row[key+'_n'] = int(defined.sum())
        row[key] = float(values.mean()) if defined.all() else np.nan
        row[key+'_sd'] = float(values.std(ddof=1)) if defined.all() and len(values)>1 else np.nan
        for name, samples in [('eval', evaluation), ('total', total)]:
            lo, hi, count = interval(samples[:, i]) if samples is not None and defined.all() else (np.nan, np.nan, 0)
            row[f'{key}_{name}_low'] = lo;row[f'{key}_{name}_high'] = hi;row[f'{key}_{name}_valid'] = count


def prepare(snapshot, raw, cache, *, repetitions=2000, seed=20261001):
    started = time.monotonic()
    plans = {r:Plan(c, repetitions, seed) for r,c in snapshot['cohorts'].items()}
    by_key = {(r['round'],r['head']):r for r in snapshot['rows']}
    families = {};buckets = {};mapping = {};group_recipes = {}
    for key, row in by_key.items():
        recipe = snapshot['recipes'][key]
        family = configuration(recipe, row, numerics=False);group = configuration(recipe, row)
        families.setdefault((key[0],family), []).append(row)
        buckets.setdefault((key[0],group), []).append(row);mapping[key] = group
        group_recipes[(key[0],group)] = recipe
    bootstrap = {};cache_hits = 0;completed = raw[raw.eligible]
    print(f'[report] Bootstrapping {len(completed)} runs × {repetitions} resamples (CPU, vectorized)', flush=True)
    for index, cell in enumerate(completed.to_dict('records')):
        key = (cell['round'],cell['head'])
        values, hit = cached_metrics(plans[key[0]], snapshot['payload'][key], cache)
        bootstrap[key] = values;cache_hits += hit
        if index%25 == 0 or index+1 == len(completed):
            print(f'[report] Bootstrap {index+1}/{len(completed)}; {cache_hits} cached; {time.monotonic()-started:.1f}s', flush=True)
    rows = [];bundles = {};view_payload = {};view_recipes = {}
    for (r, group), candidates in buckets.items():
        representative = candidates[0];recipe = deepcopy(group_recipes[(r,group)])
        family = configuration(recipe, representative, numerics=False)
        available = sorted((row for row in candidates if row['eligible']), key=lambda row:row['projection_seed'])
        seeds = [row['projection_seed'] for row in available]
        if len(seeds) != len(set(seeds)):raise ValueError(f'Duplicate feature seed in configuration {group}')
        recipe['features']['seed'] = 42  # Display/config matching only: actual seeds remain in members.
        row = {k:v for k,v in representative.items() if k!='milestones'}
        expected = sorted({row['projection_seed'] for row in families[(r,family)]})
        row.update(head=group, label=describe(recipe), family_id=family, status='complete' if available else representative['status'],
            eligible=bool(available), seed_count=len(seeds), planned_seeds=len(expected), seed_ids=','.join(map(str,seeds)),
            seed_coverage=f'{len(seeds)}/{len(expected)}', member_heads=[v['head'] for v in available],
            seed_note='Two seeds: provisional total CI' if len(seeds)==2 else 'Seed variation unestimated' if len(seeds)<2 else '',
            total_ci_available=len(seeds)>=2)
        if len(seeds)<len(expected):row['seed_note'] += '; incomplete or numerically separated seed coverage'
        row['quality'] = representative['quality']
        view_recipes[(r,group)] = recipe
        for name in DIAGNOSTICS:
            vals = np.asarray([v.get(name) if v.get(name) is not None else np.nan for v in available], float)
            row[name] = float(vals.mean()) if len(vals) and np.isfinite(vals).all() else np.nan
            row[name+'_sd'] = float(vals.std(ddof=1)) if len(vals)>1 and np.isfinite(vals).all() else np.nan
        if available:
            observed = np.array([[v.get(m) if v.get(m) is not None else np.nan for m in METRICS] for v in available])
            samples = np.stack([bootstrap[(r,v['head'])] for v in available])
            evaluation, total = combine(samples, seeds, seed)
            summarize_metrics(row, observed, evaluation, total)
            bundles[(r,group)] = dict(seeds=seeds, observed=observed, samples=samples, evaluation=evaluation, total=total, members=available)
            details = [snapshot['payload'][(r,v['head'])] for v in available]
            curves = {a:np.mean([d['curves'][a] for d in details],axis=0) for a in ('global','within_task')}
            view_payload[(r,group)] = dict(curves=curves, members=[dict(seed=s, detail=d, row=v) for s,d,v in zip(seeds,details,available)])
        else:
            for name in METRICS:row[name] = row[name+'_sd'] = np.nan
        rows.append(row)
    data = pd.DataFrame(rows)
    # Retain only common seed IDs when averaging checkpoints, and average rounds
    # within a seed before calculating across-seed variation.
    balanced_rows = [];balanced_bundles = {};rounds = snapshot['meta']['rounds']
    for group in data[data.eligible]['head'].unique():
        if not all((r,group) in bundles for r in rounds):continue
        values = [bundles[(r,group)] for r in rounds]
        common = sorted(set.intersection(*(set(v['seeds']) for v in values)))
        if not common:continue
        matched_states = len({identity(dict(ids=snapshot['cohorts'][r]['ids'], tasks=snapshot['cohorts'][r]['tasks'],
            clusters=snapshot['cohorts'][r].get('clusters'))) for r in rounds}) == 1
        obs = np.mean([v['observed'][[v['seeds'].index(s) for s in common]] for v in values],axis=0)
        sampled = np.mean([v['samples'][[v['seeds'].index(s) for s in common]] for v in values],axis=0)
        evaluation, total = combine(sampled, common, seed)
        if not matched_states:evaluation = np.full_like(evaluation, np.nan);total = None
        source = data[(data['head']==group)&(data['round']==rounds[0])].iloc[0].to_dict()
        record = dict(head=group,label=source['label'],method=source['method'],group=source['group'],scope=source['scope'],
            policies=len(rounds),precisions=source['precision'],jitter_escalated=source['jitter_escalated'],
            seed_count=len(common),seed_ids=','.join(map(str,common)),seed_coverage=f'{len(common)}/{source["planned_seeds"]}',
            seed_note=source['seed_note'],quality=source['quality'])
        summarize_metrics(record,obs,evaluation,total,prefix='mean_')
        balanced_rows.append(record)
        balanced_bundles[group] = dict(seeds=common, observed=obs, evaluation=evaluation,total=total)
    balanced = pd.DataFrame(balanced_rows)
    if len(balanced):balanced = balanced.sort_values('mean_global_nauc',ascending=False)
    view = dict(snapshot, rows=rows, recipes=view_recipes, payload=view_payload)
    pairs = analysis.matched_pairs(view,data)
    if len(pairs):
        lookup=data.set_index(['round','head'])
        for index,pair in pairs.iterrows():
            a=lookup.loc[(pair['round'],pair.control)];b=lookup.loc[(pair['round'],pair.treatment)]
            if a.seed_ids != b.seed_ids:
                pairs.loc[index,'comparable_numerics']=False
                pairs.loc[index,'caveats']=(pair.caveats+'; different available seed IDs').strip('; ')
    meta=dict(repetitions=repetitions,seed=seed,cache_hits=cache_hits,runs=len(completed),seconds=time.monotonic()-started,
        resampling='Initial-state clusters within each fixed task; same episodes across heads/seeds/checkpoints; independent feature-seed resampling',
        interval='2.5/97.5 percentiles; nominal approximate 95%; at least 90% defined replicates required',
        scope='Feature-selection seeds and evaluation initial states; conditional on policy, GP support and these tasks',
        numerical_grouping='Separate requested precision, recorded posterior precision and escalated-jitter versions',
        single_seed='Total CI unavailable when seed variability cannot be estimated',
        cohort_plans={r:p.fingerprint for r,p in plans.items()})
    return dict(data=data, balanced=balanced, pairs=pairs, ablations=analysis.ablation_summary(pairs),
        bundles=bundles,balanced_bundles=balanced_bundles,view=view,plans=plans,mapping=mapping,meta=meta,seed=seed)


def select(review, top=3):
    data=review['data'];current=data[data.eligible];balanced=review['balanced'];chosen={}
    def add(head,reason):chosen.setdefault(head,[]).append(reason)
    for head in current[current.scope=='paper']['head'].unique():add(head,'Paper baseline')
    if len(balanced):
        for method in ('kv','kv_t'):
            for head in balanced[balanced.method==method].head(top)['head']:add(head,'Top seed-averaged referral across checkpoints')
    for r,panel in current[current.scope!='paper'].groupby('round'):
        for metric in analysis.METRICS:
            order=panel.dropna(subset=[metric]).sort_values([metric,'head'],ascending=[False,True])
            if len(order):add(order.iloc[0]['head'],f'Best observed seed-mean {metric}, round {r}')
        for (scope,method),group in panel.groupby(['scope','method']):
            add(group.sort_values('global_nauc',ascending=False).iloc[0]['head'],f'Best seed-mean {scope} {method}, round {r}')
        new=panel[(panel.method=='kv_t')&(panel.precision=='float64')].sort_values('global_nauc',ascending=False)
        if len(new):add(new.iloc[0]['head'],f'Best float64 KV-T seed mean, round {r}')
    curves=[]
    for method in ('vfd','gu'):
        candidates=current[current.method==method]
        if len(candidates):curves.append(candidates.iloc[0]['head'])
    if len(balanced):
        for method in ('kv','kv_t'):curves.extend(balanced[balanced.method==method].head(2)['head'])
    latest=current[(current['round']==current['round'].max())&(current.method=='kv_t')&(current.precision=='float64')]
    if len(latest):curves.append(latest.sort_values('global_nauc',ascending=False).iloc[0]['head'])
    return chosen,list(dict.fromkeys(curves))[:7]


def curve_intervals(review, snapshot, heads):
    for (r,head),bundle in review['bundles'].items():
        if head not in heads:continue
        detail=review['view']['payload'][(r,head)];plan=review['plans'][r]
        samples={'global':[],'within_task':[]}
        for member in bundle['members']:
            source=snapshot['payload'][(r,member['head'])];curves=plan.curves(source['scores'])
            for name,key in [('global','global_'),('within_task','within')]:
                x,y,_=curves[key];grid=detail['curves'][name][:,0]
                samples[name].append(interpolate_rows(x,y,grid))
        detail['bands']={}
        for name,values in samples.items():
            evaluation,total=combine(np.stack(values),bundle['seeds'],review['seed'])
            use=total if total is not None else evaluation
            detail['bands'][name]=dict(low=np.quantile(use,.025,axis=0),high=np.quantile(use,.975,axis=0),
                kind='total' if total is not None else 'evaluation only')


def comparisons(review, selected):
    """Paired episode differences: matched ablations and selected heads vs VFD/GU."""
    data=review['data'];lookup=data.set_index(['round','head']);requests={}
    for pair in review['pairs'].to_dict('records'):
        key=(pair['round'],pair['control'],pair['treatment'])
        requests[key]=dict(family=pair['family'],caveats=pair['caveats'])
    for r,panel in data[data.eligible].groupby('round'):
        for _,control in panel[panel.method.isin(['vfd','gu'])].iterrows():
            for _,treatment in panel[(panel['head'].isin(selected))&(panel.scope!='paper')].iterrows():
                requests.setdefault((r,control['head'],treatment['head']),dict(family='Versus paper baseline',caveats=''))
    rows=[]
    for (r,a,b),info in requests.items():
        left,right=review['bundles'][(r,a)],review['bundles'][(r,b)]
        evaluation=right['evaluation']-left['evaluation']
        total=right['total']-left['total'] if left['total'] is not None and right['total'] is not None else None
        for i,metric in enumerate(METRICS):
            delta=lookup.loc[(r,b),metric]-lookup.loc[(r,a),metric]
            row=dict(round=r,control=a,treatment=b,control_label=lookup.loc[(r,a),'label'],treatment_label=lookup.loc[(r,b),'label'],
                metric=metric,mean_difference=delta,control_seeds=len(left['seeds']),treatment_seeds=len(right['seeds']),**info)
            for name,values in [('eval',evaluation),('total',total)]:
                lo,hi,n=interval(values[:,i]) if values is not None and np.isfinite(delta) else (np.nan,np.nan,0)
                row[name+'_low']=lo;row[name+'_high']=hi;row[name+'_valid']=n
                valid=values[:,i][np.isfinite(values[:,i])] if values is not None else []
                row[name+'_fraction_positive']=float(np.mean(np.asarray(valid)>0)) if len(valid) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)
