#!/usr/bin/env python3
"""Build a reviewable, repeatable report without changing study artifacts."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

# The report is intentionally outside uq_extensions' scientific source hash.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLBACKEND', 'Agg')

import pandas as pd

from head_report import analysis, plots, seed_summary
from head_report.data import capture
from head_report.html import clean, render


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def atomic_text(path, text):
    descriptor, temporary = tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()


def exports(snapshot, data, balanced, pairs, ablations, selected, destination, review=None):
    tables = dict(all_heads=data, balanced_heads=balanced, matched_pairs=pairs, ablation_summary=ablations,
                  selected_heads=data[data['head'].isin(selected)])
    tables['fit_milestones'] = pd.DataFrame([dict(head=r['head'], round=r['round'], **milestone)
                                          for r in snapshot['rows'] for milestone in r['milestones']])
    episodes=[];tasks=[];curves=[]
    for (round_,head),detail in snapshot['payload'].items():
        cohort=snapshot['cohorts'][round_]
        for index,episode in enumerate(cohort['ids']):
            record=dict(round=round_,head=head,episode=episode,task=cohort['tasks'][index],uncertainty=detail['scores'][index],
                        learner_success=cohort['learner'][index],supervisor_success=cohort['supervisor'][index],
                        signed_success_benefit=cohort['supervisor'][index]-cohort['learner'][index])
            for quality,(learner,teacher) in cohort['qualities'].items():record['signed_'+quality+'_benefit']=teacher[index]-learner[index]
            episodes.append(record)
        tasks.extend(dict(round=round_,head=head,task=task,pool_mean_uncertainty=mean,learner_success_rate=rate)
                     for task,mean,rate in zip(detail['task_names'],detail['task_means'],detail['success_rates']))
        for allocation,curve in detail['curves'].items():
            oracle=cohort['oracles'][allocation]
            curves.extend(dict(round=round_,head=head,allocation=allocation,automated=x,combined_success=y,
                               random_success=z,oracle_success=oracle[i,1]) for i,(x,y,z) in enumerate(curve))
    tables.update(episode_scores=pd.DataFrame(episodes),task_scores=pd.DataFrame(tasks),referral_curves=pd.DataFrame(curves))
    if review is not None:
        tables.update(configuration_summary=review['data'],balanced_heads=review['balanced'],
            selected_heads=review['data'][review['data']['head'].isin(review['selected'])],
            matched_pairs=review['pairs'],ablation_summary=review['ablations'],
            paired_comparisons=review['comparisons'],individual_run_pairs=pairs,
            seed_sensitivity=pairs[pairs.family=='Random coordinate seed 42 → 43'] if len(pairs) else pairs)
        mean_curves=[]
        for (r,head),detail in review['view']['payload'].items():
            for allocation,curve in detail['curves'].items():
                bands=detail.get('bands',{}).get(allocation)
                for i,(x,y,z) in enumerate(curve):
                    mean_curves.append(dict(round=r,configuration=head,allocation=allocation,automated=x,
                        seed_mean_success=y,random_success=z,ci_low=bands['low'][i] if bands else None,
                        ci_high=bands['high'][i] if bands else None,ci_kind=bands['kind'] if bands else None))
        tables['seed_mean_referral_curves']=pd.DataFrame(mean_curves)
    for name,table in tables.items():table.to_csv(destination/(name+'.csv'),index=False)


def build_report(study, output, supervisor_cache, *, top=3, make_plots=True, bootstrap_repetitions=2000, bootstrap_seed=20261001):
    started=time.monotonic();study=Path(study).resolve();output=Path(output).resolve();cache=Path(supervisor_cache).resolve()
    for source in (study,cache):
        if output==source or output in source.parents or source in output.parents:
            raise ValueError('Report output must be separate from study and supervisor artifacts')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.report.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another report generator is writing to this output directory') from None
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6]
        destination=output/'snapshots'/stamp
        destination.parent.mkdir(exist_ok=True)
        temporary=Path(tempfile.mkdtemp(prefix='.building-',dir=output))
        try:
            snapshot=capture(study,cache)
            print('[report] Computing balanced comparisons and matched ablations',flush=True)
            data=analysis.frame(snapshot);balanced=analysis.balanced_summary(data,snapshot['meta']['rounds'])
            pairs=analysis.matched_pairs(snapshot,data);ablations=analysis.ablation_summary(pairs)
            review=seed_summary.prepare(snapshot,data,output/'.bootstrap_cache',repetitions=bootstrap_repetitions,seed=bootstrap_seed)
            selected,curve_heads=seed_summary.select(review,top=top);review['selected']=selected
            review['comparisons']=seed_summary.comparisons(review,selected)
            seed_summary.curve_intervals(review,snapshot,curve_heads)
            code=[Path(__file__),*(Path(__file__).parent/'head_report').glob('*.py')]
            snapshot['meta'].update(snapshot_id=stamp, display_top_per_method=top,
                report_code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code)},
                curve_heads=curve_heads,comparison_rule='Mean ± sample SD across feature seeds; checkpoints averaged within common seeds',
                bootstrap=review['meta'],configuration_groups=len(review['data']),eligible_configuration_groups=int(review['data'].eligible.sum()))
            print('[report] Writing metric, curve, episode, and fit-diagnostic exports',flush=True)
            exports(snapshot,data,balanced,pairs,ablations,selected,temporary,review)
            figures=[]
            if make_plots:
                print('[report] Rendering PNG/PDF comparison figures',flush=True)
                figures=plots.generate(review['view'],review['data'],review['balanced'],review['pairs'],selected,curve_heads,temporary/'figures')
            snapshot['meta']['generation_seconds']=time.monotonic()-started
            write_json(temporary/'input_hashes.json',snapshot['files'])
            write_json(temporary/'snapshot.json',dict(meta=snapshot['meta'],rows=snapshot['rows'],selection=selected,
                issues=snapshot['issues'],historical_failures=snapshot['historical_failures'],figures=figures,
                configurations=review['data'].to_dict('records'),balanced_configurations=review['balanced'].to_dict('records')))
            print('[report] Building standalone HTML with an interactive head inspector',flush=True)
            arguments=(snapshot,data,balanced,pairs,ablations,selected,figures)
            # Resolve links against the eventual location, not the staging folder.
            page=render(*arguments,artifact_dir=temporary,document_dir=destination,artifact_url='',review=review)
            (temporary/'report.html').write_text(page)
            latest_page=render(*arguments,artifact_dir=temporary,document_dir=output,artifact_url=f'snapshots/{stamp}/',review=review)
            os.replace(temporary,destination)
            # Publish only after all assets and the immutable snapshot exist.
            atomic_text(output/'report.html',latest_page)
            atomic_text(output/'latest.json',json.dumps(dict(snapshot=stamp,report=str(destination/'report.html'),meta=snapshot['meta']),indent=2)+'\n')
            print(f'[report] Published {output / "report.html"}',flush=True)
            print(f'[report] {snapshot["meta"]["successful"]}/{snapshot["meta"]["total"]} successful; '
                  f'{snapshot["meta"]["eligible"]} eligible; statuses {snapshot["meta"]["statuses"]}',flush=True)
            print(f'[report] {len(figures)} figures; {len(pairs)} matched pairs; {time.monotonic()-started:.1f}s total',flush=True)
            return dict(path=output/'report.html',snapshot=destination,meta=snapshot['meta'])
        finally:
            if temporary.exists():shutil.rmtree(temporary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'storage/extensions/head_ablation_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/head_study')
    parser.add_argument('--supervisor-cache',type=Path,default=ROOT/'storage/extensions/supervisor/evaluation')
    parser.add_argument('--top',type=int,default=3,help='Balanced leaders per method to include in selected comparisons')
    parser.add_argument('--no-plots',action='store_true',help='Skip static figures (interactive inspector and CSVs are still generated)')
    parser.add_argument('--bootstrap-resamples',type=int,default=2000,help='Paired evaluation/seed bootstrap repetitions (default: 2000)')
    parser.add_argument('--bootstrap-seed',type=int,default=20261001,help='Reproducible reporting RNG seed; independent of feature seeds')
    args=parser.parse_args()
    if args.top<1:parser.error('--top must be positive')
    if args.bootstrap_resamples<100:parser.error('--bootstrap-resamples must be at least 100')
    build_report(args.study,args.output,args.supervisor_cache,top=args.top,make_plots=not args.no_plots,
        bootstrap_repetitions=args.bootstrap_resamples,bootstrap_seed=args.bootstrap_seed)


if __name__=='__main__':main()
