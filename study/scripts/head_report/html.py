"""Self-contained HTML review: no server, third-party assets, or GPU required."""
import base64
import html
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import ALL_PAIRS, MATCHED_PAIRS
from .model_map import model_map_html


def clean(value):
    if isinstance(value, dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):return int(value)
    if isinstance(value, (np.bool_,)):return bool(value)
    if isinstance(value, (float, np.floating)):return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):return str(value)
    return value


def number(value, percent=False):
    if value is None or pd.isna(value):return '—'
    if not percent and 0 < abs(value) < .001:return f'{value:.2e}'
    return f'{value:.1%}' if percent else f'{value:.3f}'


def table(data, columns, *, searchable=False, identifier=''):
    """Columns are (field, visible heading); every artifact string is escaped."""
    def cell(row, field):
        value = row.get(field)
        if value is None or (isinstance(value, float) and not math.isfinite(value)):return '—'
        if field+'_sd' in row:
            percent=field in ('automation','mean_automation')
            result=f'<span class="estimate">{number(value,percent)} ± {number(row.get(field+"_sd"),percent)}</span>'
            for kind,label in [('eval','Eval 95%'),('total','Total 95%')]:
                if field+'_'+kind+'_low' not in row:continue
                lo,hi=row.get(field+'_'+kind+'_low'),row.get(field+'_'+kind+'_high')
                interval=f'[{number(lo,percent)}, {number(hi,percent)}]' if lo is not None and pd.notna(lo) else 'unavailable'
                result+=f'<small class="ci">{label}: {interval}</small>'
            return result
        if field in ('automation', 'mean_automation'):return number(value, True)
        if isinstance(value, (float, np.floating)):return number(value)
        if field == 'head':return f'<code>{html.escape(str(value))}</code>'
        return html.escape(str(value))
    rows = data.to_dict('records') if isinstance(data, pd.DataFrame) else data
    headings = ''.join(
        f'<th scope="col" data-column="{i}" aria-sort="none"><button type="button" class="column-sort" '
        f'title="Sort by {html.escape(label, quote=True)}"><span>{html.escape(label)}</span>'
        '<span class="sort-indicator" aria-hidden="true">↕</span></button></th>'
        for i,(_,label) in enumerate(columns))
    body = []
    for row in rows:
        cells = []
        for key,_ in columns:
            value = clean(row.get(key))
            sort = '' if value is None else str(value)
            style = ' class="interpretation"' if key == 'interpretation' else ''
            cells.append(f'<td{style} data-sort="{html.escape(sort, quote=True)}">{cell(row,key)}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    attrs = f' id="{identifier}"' if identifier else ''
    return f'<div class="table-scroll"><table{attrs} class="sortable"><thead><tr>{headings}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


METRIC_COLUMNS = [('label','Configuration'), ('seed_coverage','Seeds available/planned'), ('neg_spearman','−Spearman (task)'), ('neg_pearson','−Pearson (task)'),
                  ('benefit_spearman','Spearman (signed benefit)'), ('global_nauc','Global referral nAUC'),
                  ('within_nauc','Within-task nAUC'), ('automation','Automation ≤2 pp'),
                  ('precision','Fit precision'), ('quality','Numerical note'), ('seed_note','Seed coverage note')]


def render(snapshot, data, balanced, pairs, ablations, selected, figures, *, artifact_dir, document_dir, artifact_url, review=None):
    raw_data=data
    if review is not None:
        data=review['data'];balanced=review['balanced'];pairs=review['pairs'];ablations=review['ablations']
    meta = snapshot['meta'];valid = data[data.eligible];rounds = meta['rounds']
    esc = html.escape;parts = []
    def section(anchor, title):parts.append(f'<section id="{anchor}"><h2>{esc(title)}</h2>')
    def end():parts.append('</section>')
    def paragraph(text):parts.append('<p>'+text+'</p>')
    def figure(name):
        record = next((f for f in figures if f['name']==name), None)
        if record is None:return
        raw = (artifact_dir/'figures'/f'{name}.png').read_bytes()
        parts.append(f'<figure><h3>{esc(record["title"])}</h3><img loading="lazy" alt="{esc(record["title"])}" src="data:image/png;base64,{base64.b64encode(raw).decode()}"><figcaption>{esc(record["caption"])} '
                     f'<a href="{artifact_url}figures/{name}.png">PNG</a> · <a href="{artifact_url}figures/{name}.pdf">PDF</a></figcaption></figure>')

    section('overview','Current evidence')
    counts = meta['statuses']
    cards = [('Successful evaluations',meta['successful']), ('Eligible for comparisons',meta['eligible']),
             ('Declared evaluations',meta['total']), ('Running in queue snapshot',counts.get('running',0)),
             ('Pending',counts.get('pending',0)), ('Current failures / blocked',meta['current_failures'])]
    parts.append('<div class="cards">'+''.join(f'<div><strong>{v}</strong><span>{esc(k)}</span></div>' for k,v in cards)+'</div>')
    paragraph(f'This report captured available results starting <b>{esc(meta["started_at"])}</b>. '
              'One evaluation means one head configuration at one policy checkpoint. Running status comes from the saved queue; this report does not poll or control processes. '
              f'{meta["recorded"]} cells have recorded outcomes, including skips and infeasible mappings. '
              f'{len(valid)} configuration/checkpoint groups contain eligible results; {len(balanced)} configurations have common eligible seeds at all requested checkpoints.')
    paragraph('<b>Main comparisons now average feature-selection seeds.</b> Each metric shows mean ± sample seed SD, an evaluation-only 95% CI, '
              'and an approximate total 95% CI combining seed and evaluation resampling. The seed count is always shown. '
              'A single seed has no estimable seed SD or total CI; two-seed total intervals are provisional. Individual runs remain in the inspector.')
    if counts.get('data_error',0):paragraph(f'<b class="warning">{counts["data_error"]} artifact validation errors were excluded. See diagnostics before interpreting coverage.</b>')
    if len(balanced):
        winner = balanced.iloc[0]
        paragraph(f'<b>Highest observed mean referral nAUC:</b> {esc(winner.label)} '
                  f'({number(winner.mean_global_nauc)} ± {number(winner.get("mean_global_nauc_sd"))} seed SD across checkpoints, n={winner.get("seed_count",1)}). '
                  'This is a validation-selected leader, not a confirmed winner; a single-seed lead has unmeasured seed variation.')
    method_pairs = pairs[pairs.family=='KV → KV-T (PCA16 targets)'] if len(pairs) else pairs
    if len(method_pairs):
        deltas = method_pairs.delta_global_nauc.dropna()
        if len(deltas):
            paragraph(f'<b>KV versus KV-T, holding the input representation fixed:</b> all {len(deltas)} available pairs have '
                      f'median KV-T minus KV referral nAUC {number(deltas.median())}; KV-T is higher in {int((deltas>0).sum())}/{len(deltas)} pairs. '
                      'This includes float32 KV versus float64 KV-T. The ablation section separately checks matching numerical settings and flags differences in inference, jitter, or seed coverage. '
                      'Pairs share data and policies, so their count is not a sample size for significance.')
    paragraph('All displayed results use the same saved validation outcomes within each checkpoint. '
              'They compare implementations of the paper baselines on this study’s policies and data; they are not the paper’s published aggregate results. '
              'Selection among many heads is exploratory. Test data and independent training seeds are needed for a final verdict.')
    end()

    parts.append(model_map_html(snapshot['recipes'].values(), study='old'))

    section('data','Training, validation, and test data')
    training = meta['training'];train_rows=[]
    for key,label in [(f'round_{r}',f'Learner, round {r}') for r in rounds]+[('supervisor','Supervisor')]:
        value=training.get(key,{})
        train_rows.append(dict(model=label, **{k:value.get(k) for k in ('unique_demonstrations','available_frames','policy_updates','tasks')}))
    parts.append(table(train_rows,[('model','Policy'),('unique_demonstrations','Unique training demonstrations'),('available_frames','Available training frames'),('policy_updates','Optimizer updates'),('tasks','Training tasks')]))
    parts.append(table(meta['cohorts'],[('round','Learner round'),('split','Saved split name'),('episodes','Validation episodes'),('tasks','Validation tasks'),('learner_success','Learner success'),('supervisor_success','Supervisor success')]))
    paragraph('Round 0 is the learner before active learning; round 10 is the saved learner after ten Random-acquisition rounds. '
              'The heads are fitted to these frozen policies: this reporting pipeline performs no policy training, GP fitting, or simulation. '
              'The same initial simulator states and paired learner/supervisor outcomes are reused across heads. '
              'A validation episode is one complete attempt at a task from one initial state, rather than one training demonstration.')
    if training:
        paragraph(f'GP support: {esc(training.get("gp_training_unit", "see saved configuration"))}. '
                  f'The training recipe is {esc(training.get("additional_training_recipe","see saved configuration"))}; batch size {training.get("batch_size","unknown")}. '
                  f'The separate test set contains {training.get("test_episodes","unknown")} episodes. '
                  + ('<b>Warning: this study declares the final split.</b>' if meta['test_used'] else 'This report reads validation results only; the test set has not been used by this pipeline.'))
    paragraph('A supervisor trained on more data is not automatically less epistemically uncertain. Here its empirical success advantage supplies referral headroom; '
              'episode-level gains can still be negative when the learner succeeds and the supervisor fails. A useful uncertainty ranking should identify where referral helps.')
    end()

    section('metrics','How to read the metrics')
    parts.append('<dl><dt>−Spearman / −Pearson (task)</dt><dd>Negative correlation between each task’s average uncertainty over the initial demonstration pool and the learner’s validation success rate on that task. Higher is better. There are only ten task points. Raw correlations with success have the opposite sign.</dd>'
        '<dt>Spearman / Pearson (signed benefit)</dt><dd>Correlation between episode uncertainty and supervisor success minus learner success, taking values −1, 0, +1. Negative gains are retained; higher correlation means better alignment with referral benefit.</dd>'
        '<dt>Oracle-normalized referral nAUC</dt><dd>(area under the actual success curve − area under expected random referral) / (area under hindsight-oracle referral − area under random referral). Zero is random referral, one is the oracle; values can be negative. Undefined oracle headroom is shown as —. The oracle ranks signed benefit on these same paired episodes.</dd>'
        '<dt>Global versus within-task referral</dt><dd>Global referral ranks all episodes together; within-task referral applies the same automation fraction separately to each task. Both metrics weight tasks equally. Within-task performance helps distinguish task recognition from useful ranking within a task.</dd>'
        '<dt>Automation ≤2 pp</dt><dd>Largest evaluated fraction handled by the learner while combined success stays within two percentage points of supervisor-only success. No interpolation; a validation operating point rather than a guarantee.</dd>'
        '<dt>Continuous improvement</dt><dd>For time or path length, q = 0 on failure; on success q = min(1, reference demonstration mean / achieved duration or path length). Correlate uncertainty with q(supervisor) − q(learner), retaining negative improvement. This measures clipped, success-aware efficiency benefit, not raw execution speed.</dd></dl>')
    paragraph('Tied uncertainty scores use the existing deterministic, seed-42 randomized tie order. Metric definitions and ties match the per-head reports. '
              'Missing, constant, or invalid correlations are not replaced with zero. Correlation does not establish probability calibration.')
    if review is not None:
        boot=review['meta']
        paragraph(f'<b>Seed aggregation and bootstrap:</b> {boot["repetitions"]:,} paired resamples, reporting RNG seed {boot["seed"]}. '
            'Compute every metric separately for each feature seed, then average metrics; raw uncertainty scores are never averaged into an ensemble. '
            'Mean ± SD uses sample SD (n−1 denominator), not standard error. Same-configuration results with different precision or recorded escalated jitter are kept in separate groups.')
        paragraph('Each repetition draws the observed number of initial states with replacement within every task: here 20 per task, 200 episodes total. '
            'All outcomes, scores, and seeds use the same episode draws. Repeated rollouts of a state remain together. '
            'The evaluation CI holds observed seeds fixed; the total CI also draws S seed IDs with replacement from the S observed seeds and averages their resampled metrics. '
            'Oracle normalization, supervisor success, and task success rates are recalculated for every resample. '
            'The original seeded tie order is fixed across repetitions; bootstrap copies of an episode retain that order.')
        paragraph('Intervals are the 2.5th and 97.5th percentiles, conditional on this policy, its fitted heads/support data, and these ten tasks. '
            '“Total” covers the two measured sources only; it does not include policy-training variation, new task families, or selection bias. '
            'Two seeds cannot support a well-calibrated estimate of seed variability, regardless of the number of resamples. '
            'Intervals require at least 90% defined replicates; otherwise they are unavailable. Per-metric valid-replicate counts and defined-seed counts are exported. '
            'Undefined oracle headroom or constant correlations remain undefined. All intervals and ranking comparisons are exploratory, without multiple-comparison correction.')
    end()

    section('leaders','Selected heads versus paper baselines')
    paragraph(f'The selection rule includes every paper baseline; the top {meta.get("display_top_per_method",3)} KV and KV-T configurations by seed-mean global nAUC across all checkpoints; '
              'each checkpoint’s leader on each principal metric; the best observation and action heads for each method; and the best completed float64 KV-T head. '
              'The rule is reapplied on every update. All other heads remain available in the full table and inspector.')
    for r in rounds:
        subset=valid[(valid['round']==r)&valid['head'].isin(selected)].sort_values('global_nauc',ascending=False)
        parts.append(f'<h3>Round {r}</h3>'+table(subset,METRIC_COLUMNS))
    figure('leaders');end()

    section('balanced','Comparisons across both checkpoints')
    paragraph('Only configurations complete and eligible at every requested checkpoint enter this table. Each checkpoint has equal weight. '
              'Only common seed IDs and matching numerical versions are used. First average checkpoints within each seed, then calculate mean ± SD across those seeds. '
              'The same bootstrap episodes and seed draws remain paired across checkpoints. Historical completed fits with increased jitter are retained in separate flagged groups.')
    if len(balanced):
        subset=balanced[balanced['head'].isin(selected)]
        parts.append(table(subset,[('label','Configuration'),('seed_coverage','Common seeds/planned'),('mean_neg_spearman','Mean −Spearman'),('mean_neg_pearson','Mean −Pearson'),('mean_benefit_spearman','Mean benefit Spearman'),('mean_global_nauc','Mean global nAUC'),('mean_within_nauc','Mean within-task nAUC'),('mean_automation','Mean automation ≤2 pp'),('precisions','Fit precision'),('jitter_escalated','Recorded jitter increase')]))
    figure('balanced');figure('positions');end()

    section('referral','Referral and automation')
    paragraph('To keep these plots readable, they show VFD and GU, two KV and two KV-T leaders complete at both checkpoints, '
              'and the highest seed-mean referral float64 KV-T configuration at the latest checkpoint. Referral lines average the curves across seeds; '
              'shading gives pointwise total 95% bootstrap intervals where multiple seeds exist, or evaluation-only intervals for one seed. '
              'These are pointwise intervals, not a simultaneous confidence band for the whole curve. Automation plots show mean ± seed SD.')
    figure('referral');figure('automation');end()

    section('correlations','Correlations and signed supervisor improvement')
    for r in rounds:figure(f'correlations_r{r:03d}')
    columns=[('round','Round'),('label','Head'),('benefit_spearman','Success benefit Spearman'),('benefit_pearson','Success benefit Pearson'),('benefit_time_spearman','Time-quality benefit Spearman'),('benefit_time_pearson','Time-quality benefit Pearson'),('benefit_path_spearman','Path-quality benefit Spearman'),('benefit_path_pearson','Path-quality benefit Pearson')]
    parts.append(table(valid[valid['head'].isin(selected)].sort_values(['round','global_nauc'],ascending=[True,False]),columns))
    end()

    section('ablations','Matched ablations')
    paragraph('Each pair uses the same policy checkpoint and feature-extraction settings except the named treatment. '
              'The main table includes all available pairs, including float32 KV versus float64 KV-T. '
              'Every change is treatment minus control; positive values favor the treatment, including the sign-reversed task correlations. '
              'Pairs share policies and evaluation data, so their count is not a sample size for significance. '
              'Output-compression comparisons with PLS also refit PLS to the changed targets.')
    paragraph('<b>Earlier action-decoder embedding:</b> position 0 is the embedding immediately before the final linear projection into action velocities. '
              'Position 1 is one transformer block earlier; position 3 is three blocks earlier. '
              'For KV-T the target remains the same velocity output, so earlier inputs leave more transformer computation between input and target. '
              'For KV only the location of the input embedding changes. This is network depth, not an earlier robot state or an earlier flow time. '
              'The table separates 0→1, 0→3, and 1→3 comparisons, and holds the other extraction settings fixed.')
    paragraph('<b>Precision is a numerical setting, not an extra KV requirement.</b> KV keeps its existing float32 configuration; '
              'the unfinished/problematic KV-T fits moved to float64 for stable GP calculations. KV does not need to be retrained in float64 to enter the main comparison. '
              'The optional sensitivity table requires the same recorded fit and posterior precision, inference type, available seed IDs, '
              'no recorded jitter escalation, and no unintended difference in effective input width. It is a stricter descriptive subset, not a test that mixed-precision comparisons are invalid. '
              'Matching float32 settings also does not certify numerical accuracy.')
    paragraph('Interpretations describe observed median changes and the fraction of pairs that improved; “usually” means at least 75% in the stated direction. '
              'A change below 0.001 or a median rounded to zero is a description of scale, not evidence of equivalence. '
              'They are not significance tests; use the paired bootstrap intervals below for individual comparisons. '
              'Checkpoint reversals are called out rather than hidden in an overall median. Undefined correlations are omitted per metric, with defined-pair counts in the CSV.')
    columns=[('contrast','Treatment'),('pairs','Pairs'),('median_delta_global_nauc','Median Δ global nAUC'),
             ('median_delta_within_nauc','Median Δ within-task nAUC'),('median_delta_neg_spearman','Median Δ −Spearman (task)'),
             ('median_delta_neg_pearson','Median Δ −Pearson (task)'),('median_delta_benefit_spearman','Median Δ signed-benefit Spearman'),
             ('median_delta_benefit_pearson','Median Δ signed-benefit Pearson'),('fraction_positive_global_nauc','Fraction global improved'),
             ('interpretation','Short interpretation')]
    if len(ablations):
        parts.append(table(ablations[ablations.subset==ALL_PAIRS],columns))
        parts.append('<details><summary>Sensitivity check: matching numerical settings and seed IDs</summary>')
        parts.append(table(ablations[ablations.subset==MATCHED_PAIRS],columns));parts.append('</details>')
        parts.append('<details><summary>Matched changes in continuous time/path benefit correlations</summary>')
        parts.append(table(ablations[ablations.subset==ALL_PAIRS],[('contrast','Treatment'),('pairs','Pairs'),
            ('median_delta_benefit_time_spearman','Median Δ time-benefit Spearman'),('median_delta_benefit_time_pearson','Median Δ time-benefit Pearson'),
            ('median_delta_benefit_path_spearman','Median Δ path-benefit Spearman'),('median_delta_benefit_path_pearson','Median Δ path-benefit Pearson')]))
        parts.append('</details>')
    else:paragraph('No fully matched pairs are available yet.')
    for name in ('ablations', 'ablations_correlations', 'ablations_continuous'):figure(name)
    paragraph(f'<a href="{artifact_url}matched_pairs.csv">Every matched pair, its metric changes, and numerical caveats (CSV)</a>.');end()

    if review is not None:
        section('differences','Paired differences and ranking uncertainty')
        paragraph('These comparisons subtract configuration means using identical bootstrap episode draws. Matching seed sets also share seed draws. '
            'A total interval requires at least two observed seeds on both sides; otherwise only the evaluation interval is estimated. '
            'An interval spanning zero means the comparison is unresolved at that interval level. It is not evidence of equivalence. '
            'The CSV includes all matched ablations and selected configurations versus VFD/GU, all performance metrics, and bootstrap fractions above zero; those fractions are stability diagnostics, not posterior probabilities.')
        contrast=review['comparisons']
        if len(contrast):
            shown=contrast[(contrast.metric=='global_nauc') & (contrast.treatment.isin(selected)) &
                           contrast.family.isin(['KV → KV-T (PCA16 targets)','Versus paper baseline'])].copy()
            parts.append(table(shown,[('round','Round'),('control_label','Control'),('treatment_label','Treatment'),('mean_difference','Δ mean nAUC'),
                ('eval_low','Eval CI lower'),('eval_high','Eval CI upper'),('total_low','Total CI lower'),('total_high','Total CI upper'),('caveats','Caveats')]))
        paragraph(f'<a href="{artifact_url}paired_comparisons.csv">All paired bootstrap comparisons (CSV)</a>.')
        end()

    section('inspector','Inspect any completed head')
    paragraph('This inspector shows individual seed runs for diagnosis; the rankings above use configuration means. Choose a head/checkpoint to inspect its exact configuration, referral curves, task correlation, signed improvement, and GP fitting milestones. '
              'Charts are interactive and work offline. Scatter-plot points show their values on hover. The benefit selector includes continuous time and path quality where available.')
    parts.append(inspector_markup())
    end()

    section('all','Every configuration, averaged over available seeds')
    parts.append('<label for="search">Filter all columns </label><input id="search" placeholder="e.g. joint_late, float64, pending, jitter increased"><span id="visible-count"></span>')
    paragraph('Click a column heading to sort. A status of “not_needed” is a satisfied conditional PLS rule, not a failure. '
              '“Mapping infeasible” is a GP prediction feasibility result, not a Cholesky failure. Historical coregionalized results are listed but excluded from comparisons at your request.')
    parts.append(table(data.sort_values(['round','head']),[('round','Round'),('label','Configuration'),('seed_coverage','Seeds available/planned'),('seed_ids','Seed IDs'),('method','Method'),('scope','Input type'),('status','Status'),('global_nauc','Global nAUC'),('within_nauc','Within-task nAUC'),('neg_spearman','−Spearman'),('neg_pearson','−Pearson'),('benefit_spearman','Benefit Spearman'),('automation','Automation ≤2 pp'),('actual_input_dim','Actual input dim'),('actual_output_dim','Output dim'),('precision','Precision'),('quality','Note'),('seed_note','Seed note')],identifier='all-heads'))
    parts.append('<details><summary>Every individual run, including failures and pending seeds</summary>')
    parts.append(table(raw_data.sort_values(['round','head']),[('round','Round'),('head','Run ID'),('status','Status'),('global_nauc','Global nAUC'),('neg_spearman','−Spearman'),('neg_pearson','−Pearson'),('precision','Precision'),('quality','Note')]))
    parts.append('</details>')
    end()

    section('diagnostics','Feasibility, numerical issues, and cost')
    diagnostic=valid[(valid.method=='kv_t')&valid['head'].isin(selected)]
    parts.append(table(diagnostic,[('round','Round'),('label','Head'),('actual_input_dim','Actual input dim'),('actual_output_dim','Output dim'),('mapping_r2_median','Median mapping R²'),('pls_target_explained','PLS target variance explained'),('pls_criterion_met','PLS ≥99% met'),('output_pca_variance','Output PCA variance retained'),('max_kernel_jitter','Max recorded jitter'),('optimizer_seconds','Optimizer seconds'),('worker_seconds','Worker seconds'),('peak_gpu_gib','Peak recorded GPU GiB')]))
    paragraph('GP predictive R² checks whether the chosen mapping can be learned; it does not determine whether epistemic uncertainty is useful. '
              'The inspector exposes the recorded 200 / 1,000 / 3,000 / 10,000-step milestones. Adaptive PLS is capped at 512 dimensions, and achieved target variance is reported even when 99% is missed. '
              'Completed legacy float32 runs were preserved. “No recorded escalation” does not certify numerical accuracy; some early runs have limited diagnostics. '
              'Float64 results and runs with increased jitter should be distinguished in comparisons. Recorded timing may include cached work or retries, and hardware differs; it is not a fresh-run estimate. GPU memory is the recorded per-worker peak, not total device usage.')
    problematic=raw_data[~raw_data.status.isin(['complete','pending','running'])]
    if len(problematic):parts.append(table(problematic,[('round','Round'),('head','Configuration'),('status','Outcome'),('reason','Recorded reason')]))
    paragraph(f'{meta["historical_failure_markers"]} old failure markers belong to superseded requests. They are retained in provenance and are not counted as current failures.')
    if snapshot['issues']:parts.append('<details><summary>Artifact validation / previous attempt notes</summary><pre>'+esc(json.dumps(clean(snapshot['issues']),indent=2))+'</pre></details>')
    end()

    section('exports','Exports and provenance')
    names=['all_heads.csv','balanced_heads.csv','selected_heads.csv','matched_pairs.csv','ablation_summary.csv','fit_milestones.csv','episode_scores.csv','task_scores.csv','referral_curves.csv','snapshot.json','input_hashes.json']
    if review is not None:names += ['configuration_summary.csv','paired_comparisons.csv','seed_mean_referral_curves.csv','seed_sensitivity.csv','individual_run_pairs.csv']
    parts.append('<p>'+' · '.join(f'<a href="{artifact_url}{name}">{name}</a>' for name in names)+'</p>')
    paragraph('The report stores hashes of every result, request, score file, and paired evaluation it actually read. '
              'Each successful rerun creates an immutable report snapshot and replaces only the latest report link/document. '
              'The study’s results, caches, request identities, and running queue are never written by this pipeline. Newly completed cells appear on the next report run.')
    parts.append('<pre>'+esc('bash scripts/experiments/09_head_report.sh')+'</pre>')
    parts.append('<details><summary>Snapshot metadata and display selection rules</summary><pre>'+esc(json.dumps(clean(dict(meta=meta,selected=selected)),indent=2))+'</pre></details>')
    end()

    inspector=[]
    lookup={(r['round'],r['head']):r for r in snapshot['rows']}
    for (r,head),detail in snapshot['payload'].items():
        row=lookup[(r,head)]
        if not row['eligible']:continue
        inspector.append(dict(key=f'{r}:{head}',row=row,recipe=snapshot['recipes'][(r,head)],detail=detail,
            report=os.path.relpath(row['detail_report'],document_dir) if Path(row['detail_report']).exists() else None))
    inspector.sort(key=lambda x:(x['row']['round'],x['row']['head']))
    cohorts={r:{k:c[k] for k in ('ids','tasks','learner','supervisor','oracles','qualities')} for r,c in snapshot['cohorts'].items()}
    embedded=json.dumps(clean(dict(heads=inspector,cohorts=cohorts)),separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    nav=[('overview','Overview'),('model-map','Model & head locations'),('data','Data'),('metrics','Metrics & intervals'),('leaders','Baselines & leaders'),('balanced','Both checkpoints'),('referral','Referral'),('correlations','Correlation'),('ablations','Ablations'),('differences','Paired differences'),('inspector','Inspect head'),('all','All results'),('diagnostics','Diagnostics'),('exports','Exports')]
    return document(''.join(parts), embedded, nav)


def document(body, embedded, nav, *, title='VLA uncertainty · head study review',
             eyebrow='VLA UNCERTAINTY · VALIDATION STUDY', heading='Uncertainty head comparison',
             subtitle='A reproducible review of the currently available evidence.'):
    """Shared standalone presentation for the original and follow-up studies."""
    esc = html.escape
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+CSS+'</style></head><body><header><p class="eyebrow">'+esc(eyebrow)+'</p><h1>'+esc(heading)+'</h1><p>'+esc(subtitle)+'</p></header><nav>'+''.join(f'<a href="#{esc(a)}">{esc(b)}</a>' for a,b in nav)+'</nav><main><p id="table-interactivity-note">Column sorting requires JavaScript. If the buttons do not respond in a preview, open this HTML file in a web browser.</p>'+body+'</main><script>'+TABLE_JS+'</script><script id="report-data" type="application/json">'+embedded+'</script><script>'+JS+'</script></body></html>'


def inspector_markup(*, fractional=False):
    benefit = 'Mean success improvement (−1 to +1)' if fractional else 'Success (−1, 0, +1)'
    return '<p id="inspector-status" role="status">Loading embedded head data. If this message remains, download the complete HTML and open it directly in a browser with JavaScript enabled.</p>' + \
        '<noscript><p class="warning">The interactive head inspector requires JavaScript. All static plots and tables remain available.</p></noscript>' + \
        '<label for="head-choice">Head / policy </label><select id="head-choice" disabled><option>Loading embedded head data…</option></select><div id="head-meta"></div>' + \
        '<div class="charts">' + ''.join(
            '<div><h3>'+heading+'</h3>'+extra+'<svg id="'+id_+'" viewBox="0 0 540 330" role="img" aria-label="'+heading+'"></svg></div>'
            for id_, heading, extra in [
                ('inspect-global', 'Global referral', ''), ('inspect-within', 'Within-task referral', ''),
                ('inspect-task', 'Task-level rank correlation', ''),
                ('inspect-benefit', 'Signed referral benefit', '<label for="quality-choice">Benefit </label><select id="quality-choice"><option value="success">'+benefit+'</option><option value="time">Continuous time quality</option><option value="path">Continuous path quality</option></select>'),
                ('inspect-fit', 'GP mapping validation R²', '')]) + \
        '</div><details><summary>Recorded configuration and fit diagnostics</summary><pre id="head-config"></pre></details>'


CSS = '''
:root{--ink:#182b3b;--muted:#506575;--line:#dbe4eb;--blue:#185eb9}*{box-sizing:border-box}body{margin:0;background:#f1f5f8;color:var(--ink);font:16px/1.6 system-ui,sans-serif}header{padding:40px max(4vw,24px) 28px;background:#16324c;color:white}header h1{font-size:38px;margin:3px 0}.eyebrow{font-size:12px;letter-spacing:2px}header p{margin:5px 0}nav{display:flex;gap:20px;flex-wrap:wrap;padding:12px 4vw;background:white;border-bottom:1px solid var(--line)}nav a{font-size:14px}main{max-width:1680px;margin:28px auto;padding:0 24px}section{background:white;padding:28px;margin:0 0 24px;border:1px solid var(--line);border-radius:10px;scroll-margin-top:10px}h2{font-size:25px;margin:0 0 16px}h3{font-size:19px;margin:22px 0 12px}p{max-width:1180px;margin:12px 0}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px}.cards div{padding:16px;background:#f2f7fc;border:1px solid #dce7f1;border-radius:6px}.cards strong{display:block;font-size:29px;line-height:1.2}.cards span{display:block;font-size:13px;color:var(--muted);margin-top:8px}.table-scroll{max-width:100%;overflow:auto;margin:16px 0}table{border-collapse:collapse;min-width:100%;font-size:12px;line-height:1.45}td,th{text-align:left;padding:10px 11px;border-bottom:1px solid var(--line);vertical-align:top}th{background:#edf3f8;color:#294559;cursor:pointer;white-space:normal}td:not(:first-child){font-variant-numeric:tabular-nums}tr:hover td{background:#f1f8fc}td code{font-size:11px;overflow-wrap:anywhere;display:block;min-width:210px}dt{font-weight:650;margin-top:14px}dd{margin:3px 0 0;color:var(--muted);max-width:1150px}figure{margin:28px 0;padding-top:8px;border-top:1px solid var(--line)}figure img{display:block;max-width:100%;max-height:1350px;height:auto;margin:0 auto}figcaption{font-size:13px;color:var(--muted);margin-top:14px;max-width:1150px}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,470px),1fr));gap:20px}.charts>div{background:#f8fafc;padding:10px;border:1px solid var(--line);border-radius:6px}.charts svg{width:100%;font:12px system-ui,sans-serif}select,input{font:inherit;padding:8px;border:1px solid #acbdca;border-radius:5px;background:white;max-width:100%}#head-choice{width:100%;font-size:14px}#search{min-width:50%}#visible-count{padding:10px;font-size:13px}pre{font-size:12px;line-height:1.5;overflow:auto;max-height:550px;padding:16px;background:#f2f5f8;white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;color:var(--blue);margin:14px 0}.warning{color:#a33620}.legend{font-size:13px}.muted{color:var(--muted)}@media(max-width:650px){section{padding:16px}main{padding:0 10px}header h1{font-size:29px}}@media print{nav,input,select{display:none}section{break-inside:avoid;border:0}.table-scroll{overflow:visible}body{background:white}}
'''

CSS += '\n.estimate{display:block;white-space:nowrap;font-weight:600}.ci{display:block;white-space:nowrap;font-size:10px;color:#4a6073;line-height:1.6}.ci:last-child{color:#185eb9}td:has(.estimate){min-width:150px}\n'
CSS += '\ntd.interpretation{min-width:340px;max-width:520px}\n'
CSS += '\n.sortable th{padding:0}.column-sort{appearance:none;border:0;border-radius:0;background:transparent;color:inherit;font:inherit;font-weight:650;display:flex;align-items:center;justify-content:space-between;gap:8px;width:100%;padding:10px 11px;text-align:left;cursor:pointer}.column-sort:hover{background:#dfeaf4}.column-sort:focus-visible{outline:2px solid var(--blue);outline-offset:-2px}.sort-indicator{color:#506575;white-space:nowrap}th[aria-sort="ascending"] .sort-indicator,th[aria-sort="descending"] .sort-indicator{color:var(--blue)}#table-interactivity-note[hidden]{display:none}\n'

TABLE_JS = r'''
(() => {
  // Tables work independently of the much larger interactive-inspector payload.
  document.querySelectorAll('table.sortable').forEach(table => {
    const headings = [...table.querySelectorAll('th[data-column]')];
    headings.forEach(th => {
      const button = th.querySelector('button.column-sort');
      if (!button) return;
      button.addEventListener('click', () => {
        const column = Number(th.dataset.column), body = table.tBodies[0];
        const ascending = th.getAttribute('aria-sort') !== 'ascending';
        const direction = ascending ? 1 : -1;
        [...body.rows].sort((a, b) => {
          const x = a.cells[column].dataset.sort || '', y = b.cells[column].dataset.sort || '';
          if (x === '') return y === '' ? 0 : 1;
          if (y === '') return -1;
          return direction * ((Number.isFinite(Number(x)) && Number.isFinite(Number(y)))
            ? Number(x) - Number(y) : x.localeCompare(y));
        }).forEach(row => body.appendChild(row));
        headings.forEach(other => {
          const active = other === th;
          other.setAttribute('aria-sort', active ? (ascending ? 'ascending' : 'descending') : 'none');
          other.querySelector('.sort-indicator').textContent = active ? (ascending ? '↑' : '↓') : '↕';
        });
      });
    });
  });
  const search = document.getElementById('search');
  if (search) {
    function filter() {
      let count = 0;
      document.querySelectorAll('#all-heads tbody tr').forEach(row => {
        row.hidden = !row.textContent.toLowerCase().includes(search.value.toLowerCase());
        if (!row.hidden) count++;
      });
      const output = document.getElementById('visible-count');
      if (output) output.textContent = `${count} configuration groups shown`;
    }
    search.addEventListener('input', filter); filter();
  }
  const note = document.getElementById('table-interactivity-note');
  if (note) note.hidden = true;
})();
'''

JS = r'''
(() => {
const status=document.getElementById('inspector-status');
function showError(error){
  if(status){status.className='warning';status.textContent='Could not load this head: '+error.message+'. Re-download the complete HTML and open it directly in a browser; no companion files are needed for this inspector.';}
  console.error('Head inspector:',error);
}
try {
const DATA=JSON.parse(document.getElementById('report-data').textContent);
if(!Array.isArray(DATA.heads)||!DATA.cohorts)throw Error('The embedded report data are incomplete');
const select=document.getElementById('head-choice');select.replaceChildren();for(const h of DATA.heads){const o=document.createElement('option');o.value=h.key;o.textContent=`${h.row.checkpoint_label||('Round '+h.row.round)} · ${h.row.label} · ${h.row.precision}`;select.appendChild(o);}
function fmt(x){return x==null?'—':Number(x).toFixed(3);}
function ranks(a){const sorted=a.map((v,i)=>[v,i]).sort((a,b)=>a[0]-b[0]),out=[];let i=0;
  while(i<sorted.length){let j=i+1;while(j<sorted.length&&sorted[j][0]===sorted[i][0])j++;for(let k=i;k<j;k++)out[sorted[k][1]]=(i+j+1)/2;i=j;}return out;}
function node(svg,name,attrs,text){const el=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));if(text!==undefined)el.textContent=text;svg.appendChild(el);return el;}
function chart(id,series,points,xlabel,ylabel,domain){const svg=document.getElementById(id);svg.replaceChildren();
  const all=[...series.flatMap(s=>s.points),...points.map(p=>[p[0],p[1]])].filter(p=>p.every(Number.isFinite));
  if(!all.length){node(svg,'text',{x:60,y:100,fill:'#506575'},'No recorded data for this chart.');return;}
  let [xmin,xmax,ymin,ymax]=domain||[Math.min(...all.map(p=>p[0])),Math.max(...all.map(p=>p[0])),Math.min(...all.map(p=>p[1])),Math.max(...all.map(p=>p[1]))];
  if(xmax===xmin){xmin-=.5;xmax+=.5;}if(ymax===ymin){ymin-=.5;ymax+=.5;}
  const X=x=>65+(x-xmin)/(xmax-xmin)*450,Y=y=>275-(y-ymin)/(ymax-ymin)*240;
  for(let i=0;i<6;i++){const x=xmin+(xmax-xmin)*i/5,y=ymin+(ymax-ymin)*i/5;
    node(svg,'line',{x1:65,y1:Y(y),x2:515,y2:Y(y),stroke:'#dfe6eb'});node(svg,'text',{x:55,y:Y(y)+4,'text-anchor':'end',fill:'#506575'},y.toFixed(2));
    node(svg,'text',{x:X(x),y:295,'text-anchor':'middle',fill:'#506575'},x.toFixed(2));}
  node(svg,'text',{x:290,y:325,'text-anchor':'middle'},xlabel);node(svg,'text',{x:65,y:16},ylabel);
  for(const s of series)node(svg,'polyline',{points:s.points.map(p=>`${X(p[0])},${Y(p[1])}`).join(' '),fill:'none',stroke:s.color,'stroke-width':2,'stroke-dasharray':s.dash||''});
  points.forEach(p=>{const c=node(svg,'circle',{cx:X(p[0]),cy:Y(p[1]),r:3,fill:'#087f8c',opacity:.55});node(c,'title',{},p[2]||`${p[0]}, ${p[1]}`);});
  series.forEach((s,i)=>node(svg,'text',{x:75+i*145,y:34,fill:s.color,'font-size':10},s.name));
}
function inspect(){const h=DATA.heads.find(h=>h.key===select.value);if(!h)return;
  const c=DATA.cohorts[h.row.round],d=h.detail,r=h.row,meta=document.getElementById('head-meta');meta.replaceChildren();
  const text=document.createElement('p');text.textContent=`${r.head} | global nAUC ${fmt(r.global_nauc)} | within-task nAUC ${fmt(r.within_nauc)} | −Spearman ${fmt(r.neg_spearman)} | −Pearson ${fmt(r.neg_pearson)} | success-benefit Spearman ${fmt(r.benefit_spearman)} | ${r.quality}`;meta.appendChild(text);
  for(const [allocation,id] of [['global','inspect-global'],['within_task','inspect-within']]){const v=d.curves[allocation],o=c.oracles[allocation];
    chart(id,[{name:'Head',points:v.map(p=>[p[0],p[1]]),color:'#185eb9'},{name:'Random referral',points:v.map(p=>[p[0],p[2]]),color:'#667788',dash:'5 4'},{name:'Hindsight oracle',points:o.map(p=>[p[0],p[1]]),color:'#c66a28',dash:'3 3'}],[],'Fraction handled by learner','Combined success',[0,1,0,1]);}
  const tx=ranks(d.task_means),ty=ranks(d.success_rates.map(x=>1-x));
  chart('inspect-task',[],tx.map((x,i)=>[x,ty[i],`${d.task_names[i]} | uncertainty ${d.task_means[i]} | success ${d.success_rates[i]}`]),'Rank of task-average uncertainty','Rank of task failure rate');
  const quality=document.getElementById('quality-choice').value;
  const benefitKey=quality==='success'?'benefit':`benefit_${quality}`;
  const qualityStats=document.createElement('p');qualityStats.textContent=`Selected benefit (${quality}): Spearman ${fmt(r[benefitKey+'_spearman'])} · Pearson ${fmt(r[benefitKey+'_pearson'])}`;meta.appendChild(qualityStats);
  const outcomes=quality==='success'?[c.learner,c.supervisor]:c.qualities[quality];
  const scoreRanks=ranks(d.scores);const points=outcomes?scoreRanks.map((x,i)=>[x/d.scores.length,outcomes[1][i]-outcomes[0][i],`${c.ids[i]} | uncertainty ${d.scores[i]} | signed gain ${outcomes[1][i]-outcomes[0][i]}`]):[];
  chart('inspect-benefit',[],points,'Uncertainty percentile','Supervisor − learner benefit',[0,1,-1.1,1.1]);
  const milestones=(r.milestones||[]).filter(m=>m.validation_r2_median!=null).sort((a,b)=>a.step-b.step);
  chart('inspect-fit',milestones.length?[{name:'Median output R²',points:milestones.map(m=>[m.step,m.validation_r2_median]),color:'#185eb9'}]:[],[],'Optimizer steps','Median validation R²');
  document.getElementById('head-config').textContent=JSON.stringify({recipe:h.recipe,diagnostics:r},null,2);
}
function renderSelected(){
  try {
    inspect();
    if(status){status.className='muted';status.textContent=`${DATA.heads.length} completed head evaluations available. Charts and data are stored in this HTML; no companion files are needed.`;}
  }catch(error){showError(error);}
}
if(DATA.heads.length){
  select.disabled=false;
  select.addEventListener('change',renderSelected);
  document.getElementById('quality-choice').addEventListener('change',renderSelected);
  renderSelected();
}else if(status){status.textContent='No completed head evaluations are available in this snapshot.';}
}catch(error){showError(error);}
})();
'''
