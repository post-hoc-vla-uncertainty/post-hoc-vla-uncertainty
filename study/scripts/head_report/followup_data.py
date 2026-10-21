"""Read-only follow-up adapters and paired policy-seed statistics for the shared review."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import io

import numpy as np
import pandas as pd

from .analysis import ablation_summary
from .bootstrap import METRICS, Plan
from .data import Inputs, POSITION_NAMES, fit_diagnostics
from .html import clean

METRIC_NAMES = tuple(m for m in METRICS if m != 'failure_auroc')
DATASETS = {'original': 'Original training data', 'plus35': 'Original +35 demonstrations'}
POSITIONS = dict(POSITION_NAMES, early_to_decoder='early fused context → final decoder',
    middle_to_decoder='intermediate fused context → final decoder',
    middle_to_final_context='intermediate → final observation context',
    decoder_to_velocity='final decoder → velocity')


def family(recipe):
    return recipe['id'].replace('_s42', '').replace('_s43', '')


def method(recipe):
    return recipe.get('method', 'kv_t' if '__kv_t_' in recipe['id'] else 'kv')


def label(recipe):
    if recipe.get('scope') == 'paper':return 'Paper ' + method(recipe).replace('vlm_', '').replace('_', ' ').upper()
    name = recipe.get('group', recipe['id'].split('__')[0])
    variant = recipe.get('variant', recipe.get('features', {}).get('compression', ''))
    text = f'{method(recipe).upper().replace("_", "-")} · {POSITIONS.get(name, name)} · {variant}'
    if recipe.get('whiten'):text += ' · whitened'
    if method(recipe) == 'kv_t':
        gp = recipe.get('gp', {})
        text += f' → {gp.get("output_compression", "pca")}{gp.get("output_components", 16)}'
    return text


def interval(values):
    values = np.asarray(values, float)
    if np.isfinite(values).sum() < .9*len(values):return [None, None]
    return clean(np.nanquantile(values, [.025, .975]).tolist())


def combine(samples):
    """Keep the existing follow-up seed draws exactly, shared by paired groups."""
    samples = np.asarray(samples)
    evaluation = samples.mean(0)
    if len(samples) < 2:return evaluation, None
    draws = np.random.default_rng(20261001).integers(len(samples), size=(samples.shape[1], len(samples)))
    return evaluation, samples[draws, np.arange(samples.shape[1])[:, None]].mean(1)


def summarize(observed, samples, *, prefix=''):
    observed = np.asarray(observed); evaluation, total = combine(samples)
    average = observed.mean(0)
    sd = observed.std(0, ddof=1) if len(observed)>1 else np.full(len(METRIC_NAMES), np.nan)
    row = {}
    for i, name in enumerate(METRIC_NAMES):
        key = prefix+name
        row.update({key:clean(average[i]), key+'_sd':clean(sd[i])})
        for kind, values in [('eval', evaluation), ('total', total)]:
            lo, hi = interval(values[:, i]) if values is not None else [None, None]
            row.update({key+'_'+kind+'_low':lo, key+'_'+kind+'_high':hi})
    return row


def capture(study):
    from uq_followup.reporting import cohort_for
    study = Path(study); inputs = Inputs()
    declaration = inputs.read(study/'declaration.json')
    queue = inputs.read(study/'queue.json') if (study/'queue.json').exists() else {}
    paths = sorted((study/'heads').glob('*/*/result.json'))
    records = [(p, inputs.read(p)) for p in paths]
    policies = [p for p in declaration.get('policies', []) if p['condition'] in DATASETS]
    if not policies:
        policies = [dict(id=pid, condition=pid.rsplit('_seed', 1)[0], seed=int(pid.rsplit('_seed', 1)[1]))
                    for pid in sorted({r['policy_id'] for _, r in records})]
    recipes = {r['id']:r for r in declaration['recipes']}
    groups = defaultdict(lambda:defaultdict(list)); cohorts = {}; points = {}; raw = []; details = {}; artifacts = {}
    for path, record in records:
        recipe = record['recipe']; pid = record['policy_id']; name = recipe['id']; directory = path.parent
        condition, seed = pid.rsplit('_seed', 1); seed = int(seed)
        row = dict(policy_id=pid, round=pid, condition=condition, dataset=DATASETS.get(condition, condition),
            seed=seed, head=name, family=family(recipe), label=label(recipe), method=method(recipe),
            scope=recipe.get('scope', 'observation'), group=recipe.get('group', name.split('__')[0]),
            status=record['status'], reason=record.get('reason'), seconds=record.get('seconds'),
            checkpoint_label=f'{DATASETS.get(condition, condition)} · policy seed {seed}', **record.get('metrics', {}))
        request = study/'jobs'/(pid+'__'+name)/'request.json'
        if request.exists():inputs.read(request)
        diagnostic = inputs.read(directory/'diagnostics.json') if (directory/'diagnostics.json').exists() else {}
        diagnostic_recipe = dict(recipe, method=method(recipe))
        row.update(fit_diagnostics(dict(metrics=dict(gp_diagnostics=diagnostic,
            mapping_check=diagnostic.get('mapping_check', {})), seconds=record.get('seconds')), diagnostic_recipe))
        row['quality'] = 'Recorded jitter increase' if row['jitter_escalated'] else 'No recorded jitter increase'
        if method(recipe) not in ('kv','kv_t'):row['quality'] = 'Paper baseline'
        raw.append(row); artifacts[(pid,name)] = dict(recipe=recipe, diagnostic=diagnostic)
        if record['status'] != 'complete':continue
        if pid not in cohorts:
            for policy in (pid, f'supervisor_seed{seed}'):
                inputs.read(study/'evaluations'/policy/'conditions.json')
            cohort = cohort_for(study, pid)
            # The same initial-state IDs and ordering make saved bootstrap draws paired.
            if cohorts:
                previous = next(iter(cohorts.values()))
                for key in ('ids', 'tasks', 'clusters'):
                    if cohort[key] != previous[key]:raise ValueError(f'Unpaired evaluation {key} across policies')
            point = Plan(cohort, indices=np.arange(len(cohort['ids']))[None, :]); points[pid] = point
            cohort['oracles'] = {name:np.column_stack([v[0] for v in point.oracle[key]])
                                 for name,key in [('global','global_'),('within_task','within')]}
            cohorts[pid] = cohort
        cohort = cohorts[pid]; point = points[pid]
        saved = inputs.read(directory/'scores.json')
        scores = dict(zip(saved['ids'], saved['scores']))
        if set(scores) != set(cohort['ids']) or len(scores) != len(saved['ids']):raise ValueError('Unpaired or duplicate score IDs')
        values = np.array([scores[i] for i in cohort['ids']]); means = saved['pool_task_means']
        measured, curves = point.evaluate(values, means)
        for key in METRIC_NAMES:
            actual = measured[0, METRICS.index(key)]; stored = record.get('metrics', {}).get(key)
            if stored is not None:np.testing.assert_allclose(actual, stored, rtol=1e-7, atol=1e-9, err_msg=pid+'/'+name+'/'+key)
        task_names = sorted(set(cohort['tasks'])); tasks = np.array(cohort['tasks'])
        detail = dict(scores=values, task_names=task_names, task_means=[means[t] for t in task_names],
            success_rates=[np.mean(np.asarray(cohort['learner'])[tasks==t]) for t in task_names],
            curves={name:np.column_stack([v[0] for v in curves[key]])
                    for name,key in [('global','global_'),('within_task','within')]})
        details[(pid,name)] = detail
        bootpath = directory/'bootstrap.npz'; blob = bootpath.read_bytes()
        inputs.files[str(bootpath.resolve())] = dict(sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            names = data['names'].tolist(); samples = data['metrics'][:, [names.index(m) for m in METRIC_NAMES]]
        if samples.shape != (2000,len(METRIC_NAMES)):raise ValueError('Unexpected bootstrap shape')
        observed = np.array([record.get('metrics', {}).get(m) if record.get('metrics', {}).get(m) is not None else np.nan for m in METRIC_NAMES])
        groups[(condition,family(recipe))][seed].append(dict(row=row, recipe=recipe,
            observed=observed, samples=samples, detail=detail, diagnostic=diagnostic))
    # Also expose uncomputed and currently running evaluations, as in the original report.
    present = {(r['policy_id'],r['head']) for r in raw}
    running = {r['job'] for r in queue.get('running', [])}
    failed = set(queue.get('failed', [])); deferred = set(queue.get('deferred', []))
    for policy in policies:
        for name, recipe in recipes.items():
            if (policy['id'],name) in present:continue
            job = policy['id']+'__'+name
            status = 'running' if job in running else 'failed' if job in failed else 'deferred' if job in deferred else 'pending'
            raw.append(dict(policy_id=policy['id'], round=policy['id'], condition=policy['condition'],
                dataset=DATASETS.get(policy['condition'],policy['condition']), seed=policy['seed'], head=name,
                family=family(recipe), label=label(recipe), method=method(recipe), scope=recipe.get('scope','observation'),
                group=recipe.get('group',name.split('__')[0]), status=status, milestones=[]))
    return dict(study=study, declaration=declaration, queue=queue, policies=policies, recipes=recipes,
        groups=groups, cohorts=cohorts, raw=raw, details=details, artifacts=artifacts, files=inputs.files,
        captured_at=datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S %Z'))


def aggregate(snapshot):
    declaration = snapshot['declaration']; planned = defaultdict(set); representatives = {}
    for recipe in declaration['recipes']:
        planned[family(recipe)].add(recipe['id']); representatives.setdefault(family(recipe),recipe)
    expected_seeds = set(declaration.get('seeds',[0,1,2])); summaries = []; rows = []; bundles = {}
    for (condition,head), seeds in sorted(snapshot['groups'].items()):
        ids = sorted(seeds)
        point = np.array([np.mean([r['observed'] for r in seeds[s]],axis=0) for s in ids])
        samples = np.array([np.mean([r['samples'] for r in seeds[s]],axis=0) for s in ids])
        complete = sum({r['recipe']['id'] for r in seeds[s]}==planned[head] for s in ids)
        coverage = 'complete' if set(ids)==expected_seeds and complete==len(expected_seeds) else 'partial'
        stats = summarize(point,samples); recipe = representatives[head]
        legacy = dict(condition=condition, head=head, policy_seeds=len(ids), planned_policy_seeds=len(expected_seeds),
            policy_seed_ids=','.join(map(str,ids)), complete_policy_seeds=complete,
            coordinate_repetitions=','.join(str(len(seeds[s])) for s in ids), planned_coordinate_repetitions=len(planned[head]),coverage=coverage)
        for metric in METRIC_NAMES:
            legacy.update({metric+'_mean':stats[metric], metric+'_std':stats[metric+'_sd'],
                metric+'_eval95':[stats[metric+'_eval_low'],stats[metric+'_eval_high']],
                metric+'_total95':[stats[metric+'_total_low'],stats[metric+'_total_high']]})
        summaries.append(legacy)
        runs=[r for s in ids for r in seeds[s]]
        row=dict(condition=condition, round=condition, dataset=DATASETS.get(condition,condition), head=head,
            label=label(recipe), method=method(recipe), group=recipe.get('group',head.split('__')[0]),scope=recipe.get('scope','observation'),
            seed_coverage=f'{len(ids)}/{len(expected_seeds)}', seed_ids=legacy['policy_seed_ids'], coverage=coverage,
            coverage_note=f'{coverage.title()}: {complete}/{len(expected_seeds)} fully scored policies',
            seed_note='Feature subsets per policy: '+legacy['coordinate_repetitions'], **stats)
        for key in ('precision','inference','actual_input_dim','actual_output_dim','quality'):
            values=sorted({str(r['row'].get(key)) for r in runs if r['row'].get(key) is not None})
            row[key]=' / '.join(values) if values else None
        for key in ('mapping_r2_median','pls_target_explained','output_pca_variance','max_kernel_jitter','optimizer_seconds','worker_seconds','peak_gpu_gib'):
            values=[r['row'][key] for r in runs if r['row'].get(key) is not None]
            row[key]=float(np.mean(values)) if values else None
        row['pls_criterion_met']=' / '.join(sorted({str(r['row']['pls_criterion_met']) for r in runs if r['row'].get('pls_criterion_met') is not None}))
        rows.append(row);bundles[(condition,head)]=dict(seeds=ids, observed=point, samples=samples, runs=seeds, recipe=recipe)
    existing={(r['condition'],r['head']) for r in rows}
    for condition in sorted({p['condition'] for p in snapshot['policies']}):
        for head, recipe in representatives.items():
            if (condition,head) in existing:continue
            related=[r for r in snapshot['raw'] if r['condition']==condition and r['family']==head]
            status=', '.join(sorted({r['status'] for r in related}))
            rows.append(dict(condition=condition,round=condition,dataset=DATASETS.get(condition,condition),head=head,label=label(recipe),
                method=method(recipe),scope=recipe.get('scope','observation'),group=recipe.get('group',head.split('__')[0]),
                coverage='pending',coverage_note=status,seed_coverage=f'0/{len(expected_seeds)}',seed_ids='',
                **{m:None for m in METRIC_NAMES}))
    rows.sort(key=lambda r:(r['coverage']!='complete',r['condition'],r['head']))
    summaries.sort(key=lambda r:(r['coverage']!='complete',r['condition'],r['head']))
    data=pd.DataFrame(rows); balanced=[]
    conditions=sorted({p['condition'] for p in snapshot['policies']})
    if len(conditions)>1:
        for head in representatives:
            cells=[next((r for r in rows if r['condition']==c and r['head']==head),None) for c in conditions]
            if not all(r and r['coverage']=='complete' for r in cells):continue
            values=[bundles[(c,head)] for c in conditions]
            if len({tuple(v['seeds']) for v in values})!=1:continue
            row={k:cells[0].get(k) for k in ('head','label','method','scope','group','precision','seed_coverage','seed_ids')}
            row.update(summarize(np.mean([v['observed'] for v in values],axis=0),np.mean([v['samples'] for v in values],axis=0),prefix='mean_'))
            balanced.append(row)
    balanced=pd.DataFrame(balanced)
    if len(balanced):balanced=balanced.sort_values('mean_global_nauc',ascending=False)
    return dict(data=data, summaries=summaries, bundles=bundles, balanced=balanced)


def select(review):
    data=review['data'];complete=data[data.coverage=='complete'];selected=set(complete[complete.scope=='paper']['head'])
    balanced=review['balanced']
    if len(balanced):
        for m in ('kv','kv_t'):selected.update(balanced[balanced.method==m].head(3)['head'])
    for _, panel in complete.groupby('condition'):
        for metric in ('global_nauc','within_nauc','neg_spearman','neg_pearson','benefit_spearman','automation'):
            values=panel[panel.scope!='paper'].dropna(subset=[metric]).sort_values(metric,ascending=False)
            if len(values):selected.add(values.iloc[0]['head'])
        for _, subset in panel[panel.scope!='paper'].groupby(['scope','method']):
            selected.add(subset.sort_values('global_nauc',ascending=False).iloc[0]['head'])
    curves=[h for h in ('paper__vfd','paper__gu') if h in selected]
    if len(balanced):
        for m in ('kv','kv_t'):curves.extend(balanced[balanced.method==m].head(2)['head'])
    if len(complete):
        last=sorted(complete.condition.unique())[-1]
        latest=complete[(complete.condition==last)&(complete.method=='kv_t')].sort_values('global_nauc',ascending=False)
        if len(latest):curves.append(latest.iloc[0]['head'])
    if not curves:
        curves=list(complete.sort_values('global_nauc',ascending=False)['head'].unique())[:6]
    return sorted(selected),list(dict.fromkeys(curves))[:7]


def pair_requests(review):
    """The declared one-factor ablations, with policy seeds paired throughout."""
    requests={};bundles=review['bundles'];data=review['data'];complete=data[data.coverage=='complete']
    for condition,panel in complete.groupby('condition'):
        records=panel[panel.method.isin(['kv','kv_t'])].to_dict('records')
        def descriptor(row):
            r=bundles[(condition,row['head'])]['recipe'];gp=r.get('gp',{})
            return (row['group'], row['method'], r.get('variant'), bool(r.get('whiten')),
                    gp.get('output_compression','pca'),gp.get('output_components',16),r.get('conditioning_columns',0))
        lookup={descriptor(r):r for r in records}
        for control in records:
            key=descriptor(control);group,m,variant,white,out,width,conditioning=key
            variants=[]
            if m=='kv':variants.append(('KV → KV-T (matched inputs)',(group,'kv_t',variant,white,out,width,conditioning)))
            if not white:variants.append(('Covariance whitening',(group,m,variant,True,out,width,conditioning)))
            for a,b,name in [('random512','pca512','Random512 → PCA512'),('pca512','pls512','PCA512 → PLS512'),
                ('pca_adaptive','pls_adaptive','Adaptive PCA → adaptive PLS'),('pca512','pca_adaptive','PCA512 → adaptive PCA'),
                ('pls512','pls_adaptive','PLS512 → adaptive PLS')]:
                if variant==a:variants.append((name,(group,m,b,white,out,width,conditioning)))
            if m=='kv_t' and out=='pca':variants.append(('Output PCA16 → random16',(group,m,variant,white,'random',width,conditioning)))
            for title,target in variants:
                treatment=lookup.get(target)
                if treatment is None:continue
                requests[(condition,control['head'],treatment['head'])]=title
    return requests


def paired_comparisons(review, selected):
    requests=pair_requests(review);ablation_keys=set(requests);data=review['data'];complete=data[data.coverage=='complete']
    for condition,panel in complete.groupby('condition'):
        for baseline in ('paper__vfd','paper__gu'):
            if baseline not in set(panel['head']):continue
            for head in selected:
                if head!=baseline and head in set(panel['head']):requests.setdefault((condition,baseline,head),'Versus paper baseline')
    pairs=[];comparisons=[];lookup=data.set_index(['condition','head'])
    for (condition,control,treatment),title in requests.items():
        a=review['bundles'][(condition,control)];b=review['bundles'][(condition,treatment)]
        if a['seeds']!=b['seeds']:continue
        point=b['observed']-a['observed'];samples=b['samples']-a['samples'];ev,total=combine(samples)
        ar=lookup.loc[(condition,control)];br=lookup.loc[(condition,treatment)]
        numerics=ar.precision==br.precision and ar.inference==br.inference
        caveats=[]
        if ar.precision!=br.precision:caveats.append(f'{ar.precision} → {br.precision}')
        if ar.inference!=br.inference:caveats.append('Different inference settings')
        if ar.seed_note!=br.seed_note:caveats.append('Different numbers of feature subsets, averaged within each policy')
        if any('increase' in str(x) and not str(x).startswith('No') for x in (ar.quality,br.quality)):caveats.append('Inspect recorded jitter')
        row=dict(round=condition,condition=condition,dataset=DATASETS.get(condition,condition),family=title,contrast=title,
            control=control,treatment=treatment,control_label=ar.label,treatment_label=br.label,
            comparable_numerics=numerics,caveats='; '.join(caveats),policy_seeds=len(a['seeds']))
        for i,metric in enumerate(METRIC_NAMES):
            delta=clean(point[:,i].mean());row['delta_'+metric]=delta
            lo,hi=interval(ev[:,i]);tl,th=interval(total[:,i]) if total is not None else [None,None]
            comparisons.append(dict(row,metric=metric,mean_difference=delta,eval_low=lo,eval_high=hi,
                total_low=tl,total_high=th,bootstrap_fraction_positive=clean(np.mean(total[:,i]>0)) if total is not None and np.isfinite(total[:,i]).all() else None))
        if (condition,control,treatment) in ablation_keys:pairs.append(row)
    pairs=pd.DataFrame(pairs);summaries=ablation_summary(pairs)
    if len(summaries):summaries['interpretation']=summaries.interpretation.str.replace('between checkpoints','between training datasets',regex=False)
    return pairs,summaries,pd.DataFrame(comparisons)
