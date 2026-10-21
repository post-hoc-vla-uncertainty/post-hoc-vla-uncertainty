"""Publish the follow-up using the original study's standalone review structure."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import base64
import html
import json
import os
import shutil
import tempfile
import time
import uuid

import numpy as np
import pandas as pd

from uq_extensions.common import file_hash
from . import followup_data as stats
from . import followup_plots
from .analysis import ALL_PAIRS, MATCHED_PAIRS
from .html import clean, document, inspector_markup, number, table
from .model_map import model_map_html

NAV = [('overview','Overview'),('model-map','Model & head locations'),('data','Data'),
       ('metrics','Metrics & intervals'),('leaders','Baselines & leaders'),('balanced','Both datasets'),
       ('referral','Referral'),('correlations','Correlation'),('ablations','Ablations'),
       ('differences','Paired differences'),('inspector','Inspect head'),('all','All results'),
       ('diagnostics','Diagnostics'),('exports','Exports')]
METRIC_COLUMNS = [('label','Configuration'),('seed_coverage','Policy seeds available/planned'),
    ('neg_spearman','−Spearman (task)'),('neg_pearson','−Pearson (task)'),
    ('benefit_spearman','Spearman (signed benefit)'),('global_nauc','Global referral nAUC'),
    ('within_nauc','Within-task nAUC'),('automation','Automation ≤2 pp'),('precision','Fit precision'),
    ('coverage_note','Coverage'),('seed_note','Feature subsets')]


def render(snapshot, review, figures, artifact_dir, artifact_url):
    esc=html.escape;parts=[];data=review['data'];complete=data[data.coverage=='complete'];selected=review['selected']
    balanced=review['balanced'];pairs=review['pairs'];ablations=review['ablations'];comparisons=review['comparisons']
    conditions=sorted({p['condition'] for p in snapshot['policies']});counts=Counter(r['status'] for r in snapshot['raw'])
    def section(anchor,title):parts.append(f'<section id="{anchor}"><h2>{esc(title)}</h2>')
    def end():parts.append('</section>')
    def paragraph(text):parts.append('<p>'+text+'</p>')
    def figure(name):
        record=next((f for f in figures if f['name']==name),None)
        if record is None:return
        raw=(Path(artifact_dir)/'figures'/(name+'.png')).read_bytes()
        parts.append('<figure><h3>'+esc(record['title'])+'</h3><img loading="lazy" alt="'+esc(record['title'])+'" src="data:image/png;base64,'+base64.b64encode(raw).decode()+'"><figcaption>'+esc(record['caption'])+
            f' <a href="{artifact_url}figures/{name}.png">PNG</a> · <a href="{artifact_url}figures/{name}.pdf">PDF</a></figcaption></figure>')
    section('overview','Current evidence')
    cards=[('Successful evaluations',counts.get('complete',0)),('Declared evaluations',len(snapshot['raw'])),
        ('Complete head/dataset groups',len(complete)),('Running in queue snapshot',counts.get('running',0)),
        ('Pending',counts.get('pending',0)),('Failures / infeasible',sum(v for k,v in counts.items() if k not in ('complete','running','pending')))]
    parts.append('<div class="cards">'+''.join(f'<div><strong>{v}</strong><span>{esc(k)}</span></div>' for k,v in cards)+'</div>')
    paragraph('Snapshot captured <b>'+esc(snapshot['captured_at'])+'</b>. One evaluation is one head configuration on one trained policy. '
        'This report reads saved results; it neither runs models nor changes the worker queue. Newly completed evaluations appear on the next report regeneration.')
    paragraph('<b>Main comparisons average policy-training seeds.</b> First average random-feature subset results within each policy, then average the three independently trained policies. '
        'Each metric shows mean ± sample policy-seed SD, evaluation-only 95% CI, and total 95% CI. The original and +35 training datasets are reported separately.')
    paragraph('Rankings, selected plots, and matched ablations use groups with every planned policy seed and feature subset completed. '
        'Partial groups remain visible in the full table, with provisional estimates and explicit coverage. A single policy has no estimable policy-seed SD or total CI.')
    if len(balanced):
        winner=balanced.iloc[0]
        paragraph('<b>Highest observed mean global referral nAUC across both datasets:</b> '+esc(winner.label)+
            f' ({number(winner.mean_global_nauc)} ± {number(winner.mean_global_nauc_sd)} policy-seed SD). '
            'This is an exploratory ranking among many configurations, not a confirmed winner.')
    if len(pairs):
        matched=pairs[pairs.family=='KV → KV-T (matched inputs)'];deltas=matched.delta_global_nauc.dropna()
        if len(deltas):paragraph(f'<b>KV versus KV-T with matched inputs:</b> median KV-T minus KV nAUC {number(deltas.median())}; '
            f'KV-T is higher in {int((deltas>0).sum())}/{len(deltas)} available mapping/dataset comparisons. '
            'These comparisons share policies and evaluation samples; their count is not an independent sample size.')
    paragraph('Paper baselines are evaluated on this follow-up protocol, not copied from the published paper. '
        'All 50 initial conditions per task are used for evaluation; this study makes no held-out test claim. Comparisons are exploratory and are not adjusted for searching many heads.')
    end();parts.append(model_map_html(snapshot['declaration'].get('recipes',[]),study='followup'))

    section('data','Training and evaluation data')
    policy_rows=[]
    for p in snapshot['declaration'].get('policies',snapshot['policies']):
        cohort=snapshot['cohorts'].get(p['id']);success=np.mean(cohort['learner']) if cohort else None
        nconditions=len(cohort['ids']) if cohort else None
        if p['condition']=='supervisor':
            c=snapshot['cohorts'].get(f'original_seed{p["seed"]}');success=np.mean(c['supervisor']) if c else None
            nconditions=len(c['ids']) if c else None
        policy_rows.append(dict(policy=p['id'],demonstrations=len(p['ids']) if 'ids' in p else None,frames=p.get('frames'),
            epochs=p.get('effective_epochs'),success=success,conditions=nconditions))
    parts.append(table(policy_rows,[('policy','Policy'),('demonstrations','Unique training demonstrations'),('frames','Training frames'),('epochs','Effective epochs'),('success','Mean evaluation success'),('conditions','Evaluated initial conditions')]))
    paragraph('Original policies use 1,437 demonstrations. Each +35 policy starts from the same initialization procedure and trains once on the original data plus 35 independently sampled additions: '
        'five demonstrations from each of the seven previously missing LIBERO-10 tasks. It uses uniform sampling and the same number of effective data passes, with no SAVE acquisition rounds. '
        'Each policy seed has a corresponding full-data supervisor trained on 1,693 demonstrations.')
    paragraph('Evaluation uses 50 initial conditions per task × 10 tasks × three action-noise executions per policy. The three outcomes are averaged within each initial condition before referral and correlation calculations. '
        'Robot outcomes are evaluated once and reused for every head. Each head scores the initial observation, optionally probing internal action-generation trajectories; these probes are not extra robot rollouts.')
    paragraph('Policy rollout performance was checked before head evaluation. All heads for a policy use the same training support and saved initial observations. '
        'Learner and supervisor inputs are paired by simulator state, instruction, and robot state; bounded RGB rendering differences were audited. '
        'A supervisor trained on more data is not guaranteed to improve every episode, so signed negative referral benefits are retained.')
    end()

    section('metrics','How to read the metrics')
    parts.append('<dl><dt>−Spearman / −Pearson (task)</dt><dd>Negative correlation between each task’s average uncertainty over its demonstration pool and that policy’s evaluation success rate. Higher is better. There are ten task points per policy.</dd>'
        '<dt>Spearman / Pearson (signed benefit)</dt><dd>Correlation between initial-observation uncertainty and supervisor mean success minus learner mean success. Each success rate averages three executions, so benefits can take fractional values from −1 to +1. Negative improvement is retained.</dd>'
        '<dt>Oracle-normalized referral nAUC</dt><dd>(actual referral area − expected random-referral area) / (hindsight-oracle area − random-referral area). Zero is random referral; one is the oracle. Values can be negative. Undefined oracle headroom remains unavailable.</dd>'
        '<dt>Global versus within-task referral</dt><dd>Global referral ranks all initial conditions together. Within-task referral uses the same automated fraction within each task. Tasks receive equal weight. Original-data within-task results are diagnostic rather than designated main-paper results.</dd>'
        '<dt>Automation ≤2 pp</dt><dd>Largest measured fraction handled by the learner while combined success remains within two percentage points of supervisor-only success. This is an evaluation-set operating point.</dd>'
        '<dt>Continuous time/path benefit</dt><dd>Quality is zero on failure; on success it is the clipped ratio of reference-demonstration duration/path to achieved duration/path. Correlate uncertainty with supervisor quality minus learner quality. These measure success-aware efficiency, not raw speed.</dd></dl>')
    paragraph('<b>Policy seeds and feature subsets:</b> metrics are calculated separately for each head evaluation. Seeds 42/43 select random embedding coordinates; their metrics are averaged within a policy. '
        'Means and sample SD (n−1 denominator) then use policy seeds 0/1/2. Feature subsets are not extra trained policies, and raw uncertainty scores are not averaged into an ensemble.')
    paragraph('<b>Bootstrap:</b> 2,000 paired resamples. Each draw samples 50 initial conditions with replacement within each of the ten tasks; all three executions remain together in their condition mean. '
        'Heads, learner/supervisor policies, and training datasets use the same condition draws. Evaluation-only intervals hold the observed policy seeds fixed. '
        'Total intervals additionally resample the policy/supervisor seed pairs, retaining their sampled +35 training data. Oracle normalization is recalculated for each evaluation resample.')
    paragraph('Intervals are 2.5th/97.5th percentiles and require at least 90% defined replicates. Three policy seeds still give a limited estimate of training variability. '
        'Intervals are conditional on the observed tasks and protocol and do not cover selection bias or new task families. Undefined correlations remain unavailable. '
        'The deterministic randomized tie order is fixed across bootstrap draws.')
    if snapshot['declaration'].get('companion_protocol')=='study_seed':paragraph('<b>Ensemble-baseline limitation:</b> ensemble baselines reuse other study-seed policies. Pairs share members, and +35 companions have different added demonstrations. Their policy-seed bootstrap is a descriptive approximation for dependent ensemble pairs.')
    end()

    section('leaders','Selected heads versus paper baselines')
    paragraph('The display rule includes all completed paper baselines, the top three KV and KV-T heads by mean global nAUC across both datasets, '
        'each dataset’s leaders on the main metrics, and its best observation/action head per estimator. It is reapplied on every report update; every configuration remains in the full table.')
    for condition in conditions:
        panel=complete[(complete.condition==condition)&complete['head'].isin(selected)].sort_values('global_nauc',ascending=False)
        parts.append('<h3>'+esc(stats.DATASETS.get(condition,condition))+'</h3>'+table(panel,METRIC_COLUMNS));figure('leaders_'+condition)
    end()
    section('balanced','Comparisons across both training datasets')
    paragraph('Only heads complete for both datasets enter this comparison. Average original/+35 metrics within each policy seed, then calculate the mean, SD, and paired intervals across the three seeds. '
        'Each dataset has equal weight; the six policies are not treated as six independent seeds. Separate dataset results remain the primary interpretation.')
    if len(balanced):parts.append(table(balanced[balanced['head'].isin(selected)], [('label','Configuration'),('seed_coverage','Common policy seeds'),('mean_global_nauc','Mean global nAUC'),('mean_within_nauc','Mean within-task nAUC'),('mean_neg_spearman','Mean −Spearman'),('mean_neg_pearson','Mean −Pearson'),('mean_benefit_spearman','Mean signed-benefit Spearman'),('mean_automation','Mean automation ≤2 pp')]))
    figure('balanced');figure('positions');end()
    section('referral','Referral and automation')
    paragraph('These plots show VFD/GU, up to two KV and two KV-T leaders complete on both datasets, and the +35 dataset’s leading KV-T head. '
        'Referral shading gives pointwise total 95% bootstrap intervals; automation shading gives policy-seed SD. Both average feature subsets within policies first.')
    figure('referral');figure('automation');end()
    section('correlations','Correlations and signed supervisor improvement')
    for condition in conditions:figure('correlations_'+condition)
    parts.append(table(complete[complete['head'].isin(selected)],[('dataset','Training dataset'),('label','Head'),('benefit_spearman','Success benefit Spearman'),('benefit_pearson','Success benefit Pearson'),('benefit_time_spearman','Time-quality Spearman'),('benefit_time_pearson','Time-quality Pearson'),('benefit_path_spearman','Path-quality Spearman'),('benefit_path_pearson','Path-quality Pearson')]))
    end()
    section('ablations','Matched ablations')
    paragraph('Each comparison holds the policy seeds and evaluation observations fixed while changing the named estimator or preprocessing choice at the same mapping. '
        'KV versus KV-T includes float32 KV and float64 KV-T as implemented; KV does not need double precision to enter the comparison. '
        'PLS-KV uses output-informed preprocessing, so Random/PCA are the strictly input-only controls. All feature subsets are averaged within policies before comparison.')
    columns=[('contrast','Treatment'),('pairs','Pairs'),('median_delta_global_nauc','Median Δ global nAUC'),('median_delta_within_nauc','Median Δ within-task nAUC'),
        ('median_delta_neg_spearman','Median Δ −Spearman'),('median_delta_neg_pearson','Median Δ −Pearson'),('median_delta_benefit_spearman','Median Δ benefit Spearman'),('median_delta_benefit_pearson','Median Δ benefit Pearson'),('fraction_positive_global_nauc','Fraction global improved'),('interpretation','Short interpretation')]
    if len(ablations):
        parts.append(table(ablations[ablations.subset==ALL_PAIRS],columns))
        parts.append('<details><summary>Sensitivity check: matching numerical settings</summary>'+table(ablations[ablations.subset==MATCHED_PAIRS],columns)+'</details>')
        parts.append('<details><summary>Every matched mapping/dataset comparison</summary>'+table(pairs,[('dataset','Dataset'),('contrast','Treatment'),('control_label','Control'),('treatment_label','Treatment'),('delta_global_nauc','Δ global nAUC'),('delta_within_nauc','Δ within-task nAUC'),('delta_neg_spearman','Δ −Spearman'),('delta_neg_pearson','Δ −Pearson'),('caveats','Numerical / repetition differences')])+'</details>')
    else:paragraph('No complete matched comparisons are available yet.')
    paragraph('Interpretations describe medians and directional consistency, not significance or equivalence. Mapping-specific changes are shown in the expanded table and CSV so averages do not hide improvements in particular locations. '
        'A sign reversal between training datasets is called out. Comparisons share models and data, so their count is not an independent sample size.')
    for name in ('ablations', 'ablations_correlations', 'ablations_continuous'):figure(name)
    end()
    section('differences','Paired differences and ranking uncertainty')
    paragraph('Each difference uses identical initial-condition and policy-seed resamples on both sides. An interval spanning zero means the comparison remains unresolved at that interval level. '
        'Bootstrap fractions above zero are stability diagnostics, not posterior probabilities. No multiple-comparison correction is applied.')
    if len(comparisons):
        shown=comparisons[(comparisons.metric=='global_nauc')&comparisons.treatment.isin(selected)&comparisons.family.isin(['Versus paper baseline','KV → KV-T (matched inputs)'])]
        parts.append(table(shown,[('dataset','Dataset'),('control_label','Control'),('treatment_label','Treatment'),('mean_difference','Δ mean nAUC'),('eval_low','Eval CI lower'),('eval_high','Eval CI upper'),('total_low','Total CI lower'),('total_high','Total CI upper'),('caveats','Caveats')]))
    paragraph(f'<a href="{artifact_url}paired_comparisons.csv">All paired comparisons, including every ablation and continuous-benefit metric (CSV)</a>.');end()
    section('inspector','Inspect any completed head')
    paragraph('Choose a head and policy to inspect the individual feature-subset result, referral curves, task correlation, signed benefit, and GP optimizer milestones. '
        'The main comparisons above average policy seeds; this inspector preserves each individual run. Its charts and data are embedded and work offline.')
    parts.append(inspector_markup(fractional=True));end()
    section('all','Every configuration, averaged over available policy seeds')
    parts.append('<label for="search">Filter all columns </label><input id="search" placeholder="e.g. decoder, plus35, partial, whitened"><span id="visible-count"></span>')
    paragraph('Click a column heading to sort. Complete means all planned feature subsets and policy seeds are present. Partial results are provisional and excluded from the main rankings. '
        'Pending or failed configurations remain visible. Mapping infeasibility is a prediction-quality result, not a numerical factorization error.')
    parts.append(table(data,[('dataset','Dataset'),('label','Configuration'),('seed_coverage','Policy seeds'),('seed_ids','Policy seed IDs'),('coverage_note','Coverage / status'),
        ('global_nauc','Global nAUC'),('within_nauc','Within-task nAUC'),('neg_spearman','−Spearman'),('neg_pearson','−Pearson'),('benefit_spearman','Benefit Spearman'),('automation','Automation ≤2 pp'),('actual_input_dim','Input dimensions'),('actual_output_dim','Output dimensions'),('precision','Precision'),('quality','Numerical note'),('seed_note','Feature subsets')],identifier='all-heads'))
    parts.append('<details><summary>Every individual evaluation, including unfinished runs</summary>'+table(snapshot['raw'],[('policy_id','Policy'),('head','Run ID'),('status','Status'),('global_nauc','Global nAUC'),('neg_spearman','−Spearman'),('neg_pearson','−Pearson'),('seconds','Recorded seconds'),('reason','Reason')])+'</details>');end()
    section('diagnostics','Feasibility, numerical issues, and cost')
    diagnostic=data[(data.method=='kv_t')&(data.coverage!='pending')]
    parts.append(table(diagnostic,[('dataset','Dataset'),('label','Head'),('actual_input_dim','Input dimensions'),('actual_output_dim','Output dimensions'),('mapping_r2_median','Mean policy-level median mapping R²'),('pls_target_explained','Mean PLS target variance explained'),('pls_criterion_met','PLS ≥99% met'),('output_pca_variance','Mean output PCA variance retained'),('max_kernel_jitter','Mean policy-level max jitter'),('optimizer_seconds','Mean optimizer seconds'),('worker_seconds','Mean worker seconds'),('peak_gpu_gib','Mean recorded peak GPU GiB')]))
    paragraph('Prediction R² checks mapping feasibility, not whether uncertainty predicts referral benefit. Adaptive PLS is capped at 512 dimensions and reports the achieved criterion. '
        'The inspector retains 200/1,000/3,000/10,000-step diagnostics. Worker time can include feature extraction, waiting for shared caches, and resumed work; it is not a uniform fresh-run benchmark. '
        'Missing memory or numerical diagnostics remain unavailable. Individual values are exported in all_heads.csv.')
    problem=[r for r in snapshot['raw'] if r['status'] not in ('complete','pending','running')]
    if problem:parts.append(table(problem,[('policy_id','Policy'),('head','Head'),('status','Status'),('reason','Reason')]))
    else:paragraph('No failed or infeasible results were recorded in this snapshot.')
    end()
    section('exports','Exports and provenance')
    names=['summary.json','summary.csv','all_heads.csv','configuration_summary.csv','balanced_heads.csv','selected_heads.csv','matched_pairs.csv','ablation_summary.csv','paired_comparisons.csv','fit_milestones.csv','episode_scores.csv','task_scores.csv','referral_curves.csv','snapshot.json','input_hashes.json']
    parts.append('<p>'+' · '.join(f'<a href="{artifact_url}{name}">{name}</a>' for name in names)+'</p>')
    paragraph('Every report publishes an immutable snapshot. Plots and inspector data are embedded in the HTML; opening or copying the HTML alone retains them. PNG/PDF and CSV downloads are stored alongside its snapshot. '
        'Hashes identify the completed results, scores, bootstrap arrays, paired evaluations, requests, and report source used. GPU jobs and their scientific identities are unchanged.')
    parts.append('<pre>bash scripts/experiments/10_followup_study.sh --report-only</pre>')
    parts.append('<details><summary>Snapshot metadata and display selections</summary><pre>'+esc(json.dumps(clean(snapshot['meta']),indent=2))+'</pre></details>');end()
    inspector=[]
    for row in snapshot['raw']:
        key=(row['policy_id'],row['head'])
        if key not in snapshot['details']:continue
        inspector.append(dict(key=':'.join(key),row=row,recipe=snapshot['artifacts'][key]['recipe'],detail=snapshot['details'][key],report=None))
    inspector.sort(key=lambda h:(h['row']['policy_id'],h['row']['head']))
    cohorts={pid:{k:c[k] for k in ('ids','tasks','learner','supervisor','oracles','qualities')} for pid,c in snapshot['cohorts'].items()}
    embedded=json.dumps(clean(dict(heads=inspector,cohorts=cohorts)),separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    return document(''.join(parts),embedded,NAV,title='VLA uncertainty · three-seed follow-up review',
        eyebrow='VLA UNCERTAINTY · THREE-SEED FOLLOW-UP',heading='Uncertainty head comparison',
        subtitle='Original and +35 training datasets · matched policy/supervisor seeds · saved evaluation outcomes')


def write_json(path,value):
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def exports(snapshot,review,destination):
    data=review['data'];tables=dict(summary=pd.DataFrame(review['summaries']),all_heads=pd.DataFrame(snapshot['raw']),
        configuration_summary=data,balanced_heads=review['balanced'],selected_heads=data[data['head'].isin(review['selected'])],
        matched_pairs=review['pairs'],ablation_summary=review['ablations'],paired_comparisons=review['comparisons'])
    milestones=[];episodes=[];tasks=[];curves=[]
    for row in snapshot['raw']:
        pid=row['policy_id'];head=row['head'];key=(pid,head)
        for m in row.get('milestones',[]):milestones.append(dict(policy_id=pid,head=head,**m))
        if key not in snapshot['details']:continue
        detail=snapshot['details'][key];cohort=snapshot['cohorts'][pid]
        for i,id_ in enumerate(cohort['ids']):
            record=dict(policy_id=pid,head=head,initial_condition=id_,task=cohort['tasks'][i],uncertainty=detail['scores'][i],learner_success=cohort['learner'][i],supervisor_success=cohort['supervisor'][i],signed_success_benefit=cohort['supervisor'][i]-cohort['learner'][i])
            for quality,(l,h) in cohort['qualities'].items():record['signed_'+quality+'_benefit']=h[i]-l[i]
            episodes.append(record)
        tasks.extend(dict(policy_id=pid,head=head,task=t,pool_mean_uncertainty=u,learner_success_rate=s) for t,u,s in zip(detail['task_names'],detail['task_means'],detail['success_rates']))
        for name,c in detail['curves'].items():
            curves.extend(dict(policy_id=pid,head=head,allocation=name,automated=x,combined_success=y,random_success=z,oracle_success=cohort['oracles'][name][i,1]) for i,(x,y,z) in enumerate(c))
    tables.update(fit_milestones=pd.DataFrame(milestones),episode_scores=pd.DataFrame(episodes),task_scores=pd.DataFrame(tasks),referral_curves=pd.DataFrame(curves))
    for name,df in tables.items():df.to_csv(destination/(name+'.csv'),index=False)
    write_json(destination/'summary.json',review['summaries']);write_json(destination/'input_hashes.json',snapshot['files'])


def build_report(study, *, make_plots=True):
    from uq_followup.reporting import atomic_text
    started=time.monotonic();study=Path(study);output=study/'report';output.mkdir(parents=True,exist_ok=True)
    print('[report] Reading completed follow-up results and validating paired metrics',flush=True)
    snapshot=stats.capture(study);review=stats.aggregate(snapshot)
    review['selected'],curve_heads=stats.select(review)
    review['pairs'],review['ablations'],review['comparisons']=stats.paired_comparisons(review,review['selected'])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
    destination=output/'snapshots'/stamp;destination.parent.mkdir(exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.building-',dir=output))
    try:
        print('[report] Writing summaries, matched comparisons, and per-policy exports',flush=True)
        exports(snapshot,review,temporary)
        figures=followup_plots.generate(snapshot,review,review['selected'],curve_heads,temporary/'figures',output/'.curve_cache') if make_plots else []
        snapshot['meta']=dict(snapshot_id=stamp,captured_at=snapshot['captured_at'],successful=sum(r['status']=='complete' for r in snapshot['raw']),
            total=len(snapshot['raw']),statuses=dict(Counter(r['status'] for r in snapshot['raw'])),complete_groups=int((review['data'].coverage=='complete').sum()),
            partial_groups=int((review['data'].coverage=='partial').sum()),policy_seeds=snapshot['declaration'].get('seeds',[0,1,2]),
            bootstrap_repetitions=2000,bootstrap_seed=20261001,selection=review['selected'],curve_heads=curve_heads,
            generation_seconds=time.monotonic()-started,figures=figures,
            report_code={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            scientific_source_hash=snapshot['declaration'].get('source_hash'))
        write_json(temporary/'snapshot.json',dict(meta=snapshot['meta'],configurations=review['data'].to_dict('records'),
            raw_rows=snapshot['raw'],balanced=review['balanced'].to_dict('records')))
        print('[report] Building self-contained HTML using the original report layout',flush=True)
        page=render(snapshot,review,figures,temporary,'')
        (temporary/'report.html').write_text(page)
        latest=render(snapshot,review,figures,temporary,f'snapshots/{stamp}/')
        os.replace(temporary,destination)
        # Publish compatibility summaries only after the complete report is ready.
        for name in ('summary.json','summary.csv'):atomic_text(output/name,(destination/name).read_text())
        atomic_text(output/'report.html',latest)
        atomic_text(output/'latest.json',json.dumps(clean(dict(snapshot=stamp,meta=snapshot['meta'])),indent=2)+'\n')
        print(f'[report] Published {output/"report.html"}: {snapshot["meta"]["successful"]}/{snapshot["meta"]["total"]} evaluations, {len(figures)} figures, {len(review["pairs"])} matched pairs',flush=True)
        return output/'report.html'
    finally:
        if temporary.exists():shutil.rmtree(temporary)
