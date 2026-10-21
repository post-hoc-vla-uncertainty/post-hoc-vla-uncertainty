"""Resumable uncertainty-head study on existing round-0/10 SAVE policies.

No policy training or acquisition is called by this entry point. A fresh child
process owns each head's GPU allocations; raw features and fitted estimators
are shared through content-addressed disk caches.
"""
import argparse
from dataclasses import replace
import fcntl
import html
import os
from pathlib import Path
import signal
import time

from .common import atomic_json, identity, model_identity, plain, read_json
from .config import ExperimentConfig, FeatureConfig, GPConfig
from .head_study_design import study_recipes
from .head_queue import parse_gpus,resolve_slots,start_child,run_queue,execution_phase
from .cache_lock import cache_lock


def training_data_summary(manifest, source, supervisor):
    rows=read_json(Path(manifest['evaluator_directory'])/'trajectory_index.json')
    lengths={r['id']:r['dataset_to_index']-r['dataset_from_index'] for r in rows}
    mapping={v:k for k,v in manifest['episode_map'].items()}
    acquired=[]
    for i in range(10):
        p=Path(source)/f'round_{i:03d}/selection_manifest.json'
        acquired.extend(mapping[r['episode_id']] for r in read_json(p)['selected_episodes'])
    pre=set(manifest['pretraining_ids']); teacher=pre|set(manifest['target_ids'])
    def describe(ids):
        return dict(unique_demonstrations=len(ids), available_frames=sum(lengths[i] for i in ids))
    return dict(round_0=dict(**describe(pre),policy_updates=30000,tasks=33),
        round_10=dict(**describe(pre|set(acquired)),policy_updates=70000,query_slots=len(acquired),
                      unique_selected_demonstrations=len(set(acquired)),new_demonstrations=len(set(acquired)-pre)),
        supervisor=dict(**describe(teacher),policy_updates=30000,tasks=40,model=str(supervisor)),
        batch_size=32,additional_training_recipe='10 x 4000 updates, 50:50 preparatory/cumulative-selected replay',
        gp_training_unit='Initial observation of each unique available demonstration; action heads add generated probes',
        validation_episodes=len(manifest['evaluation_splits']['development']),
        test_episodes=len(manifest['evaluation_splits']['final']))


def configure_caches(root):
    root=Path(root).resolve()
    for env,folder in [('UQ_EXT_FEATURE_CACHE','feature_cache'),('UQ_EXT_PROJECTION_CACHE','projection_cache'),
                       ('UQ_EXT_ESTIMATOR_CACHE','estimator_cache')]:
        os.environ[env]=str(root/folder)
    os.environ['UQ_EXT_BUNDLE_FEATURES']='1'
    os.environ['UQ_EXT_RETIRE_FEATURE_CACHE']='0'
    os.environ['UQ_EXT_STUDY_MEMORY_GUARD']='1'
    os.environ['PYTHONUNBUFFERED']='1'


def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(text);os.replace(tmp,path)


def completed_records(root):
    records=[]
    for path in sorted(Path(root).glob('round_*/*/result.json')):
        try:
            record=read_json(path)
            for field in ('round','recipe','status','seconds','directory','request_identity'):record[field]
            record['recipe']['id']
            request=path.with_name('request.json')
            if request.exists() and record['request_identity']!=read_json(request)['identity']:continue
        except (OSError,ValueError,KeyError,TypeError):
            # The worker validates this artifact and records its failure. One
            # broken result must not also break the cumulative progress report.
            continue
        records.append(record)
    return records


def failure_records(root):
    return [read_json(p) for p in sorted(Path(root).glob('round_*/*/failure.json'))]


def record_failure(request,event):
    """Coordinator-owned failure metadata, separate from completed scientific scores."""
    from .runner import source_identity
    spec=request['spec'];out=Path(spec['directory'])
    record=dict(event,round=spec['round'],head=spec['recipe']['id'],directory=str(out),
        request_identity=request['identity'],updated_at=time.time(),execution_source_hash=source_identity())
    history_path=out/'failure_history.json'
    history=read_json(history_path) if history_path.exists() else []
    history.append(record)
    atomic_json(history_path,history);atomic_json(out/'failure.json',record)


def head_report(record):
    from .reporting import experiment_report
    directory=Path(record['directory'])/'report'
    path=experiment_report(directory,matrix=[record])
    diagnostics=(record.get('metrics') or {}).get('gp_diagnostics',{})
    extra='<h2>GPU memory and GP fitting</h2><pre>'+html.escape(str(record.get('gpu_memory',{})))+'</pre>'
    if diagnostics.get('prediction_numerics'):
        extra+='<h3>Posterior numerical precision</h3><pre>'+html.escape(str(diagnostics['prediction_numerics']))+'</pre>'
    geometry=diagnostics.get('geometry',{})
    if geometry:extra+='<h3>Projection diagnostics</h3><pre>'+html.escape(str(geometry))+'</pre>'
    whitening=diagnostics.get('whitening') or [c['whitening'] for c in diagnostics.get('cameras',[]) if 'whitening' in c]
    if whitening:extra+='<h3>Training-only covariance whitening</h3><pre>'+html.escape(str(whitening))+'</pre>'
    training=diagnostics.get('training_records',{})
    if training:
        initializations={phase:[b['initialization'] for b in batches if b.get('initialization')]
                         for phase,batches in training.items()}
        initializations={phase:rows for phase,rows in initializations.items() if rows}
        if initializations:
            extra+='<h3>Variance-informed length-scale initialization</h3><pre>'+html.escape(str(initializations))+'</pre>'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import base64
        import numpy as np
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        histories=[k['loss_history'] for k in diagnostics.get('kernels',[]) if k.get('loss_history')]
        if not histories and diagnostics.get('loss_history'):
            histories=[diagnostics['loss_history']]
        if histories:
            values=np.asarray(histories);steps=np.arange(1,values.shape[1]+1)
            axes[0].plot(steps,values.mean(0));axes[0].fill_between(steps,values.min(0),values.max(0),alpha=.2)
        for batch in training.get('mapping_validation',[]):
            for metric,label in [('validation_r2','Held-out GP R²')]:
                rows=[r for r in batch['milestones'] if metric in r]
                if rows:axes[1].plot([r['step'] for r in rows],[np.mean(r[metric]) for r in rows],marker='o',label=label)
        axes[0].set(title='Final fit: objective across outputs',xlabel='Optimizer updates',ylabel='Negative log likelihood / bound')
        axes[1].set(title='GP regression validation',xlabel='Optimizer updates',ylabel='Mean R²')
        for ax in axes:ax.grid(alpha=.25)
        fig.savefig(directory/'gp_training.png',dpi=140);fig.savefig(directory/'gp_training.pdf');plt.close(fig)
        extra+='<img class="plot" src="data:image/png;base64,'+base64.b64encode((directory/'gp_training.png').read_bytes()).decode()+'">'
        atomic_json(directory/'gp_training.json',training)
    atomic_text(path,path.read_text().replace('</html>',extra+'</html>'))
    atomic_json(directory/'complete.json',dict(recipe=record['recipe']['id'],round=record['round']))
    return path


def refresh_index(root, *, active=None):
    """Cheap cumulative report; detailed plots live in each completed head report."""
    import pandas as pd
    from .matrix import write_table
    root=Path(root);records=completed_records(root);failures=failure_records(root)
    atomic_json(root/'matrix.json',records)
    if records:write_table(records,root)
    rows=[]
    for record in records:
        metrics=record.get('metrics') or {}
        report=Path(record['directory'])/'report'
        stats={}
        p=report/'referral_diagnostics/summary.json'
        if p.exists():
            try:stats=next((r for r in read_json(p)['records'] if r['allocation']=='global'),{})
            except (OSError,ValueError,KeyError,TypeError):pass
        row=dict(head=record['recipe']['id'],round=record['round'],status=record['status'],
            gp_precision=record['recipe'].get('gp',{}).get('precision'),
            referral_nAUC=stats.get('oracle_normalized_gain'),referral_gain=metrics.get('referral_global',{}).get('raw_gain'),
            automated_fraction=stats.get('automation_within_two_points'),task_spearman=metrics.get('between_task_spearman'),
            within_task_auroc=metrics.get('macro_auroc'),seconds=round(record['seconds'],1),
            peak_GPU_GiB=record.get('gpu_memory',{}).get('peak_reserved_bytes',0)/1024**3,
            plots=str((report/'report.html').relative_to(root)))
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(root/'summary.csv',index=False)
    failure_frame=pd.DataFrame(failures,columns=['round','head','status','reason','exit_code','gpu','log','updated_at'])
    atomic_json(root/'failures.json',failures)
    atomic_text(root/'failures.csv',failure_frame.to_csv(index=False))
    display=frame.copy()
    if len(display):
        display['plots']=display['plots'].map(lambda p:f'<a href="{html.escape(p)}">Metrics and plots</a>')
    declaration=read_json(root/'declaration.json') if (root/'declaration.json').exists() else {}
    total=len(declaration.get('recipes',[]))*len(declaration.get('rounds',[]))
    text='''<!doctype html><html><meta charset="utf-8"><title>Uncertainty head validation study</title>
<style>body{font:15px system-ui;margin:2rem}table{border-collapse:collapse;font-size:12px}th,td{border:1px solid #ccc;padding:6px}pre{white-space:pre-wrap}</style>
<h1>Uncertainty head validation study</h1><p>Frozen round-0 and round-10 robot policies. 200 validation attempts per model; no policy training. Test data remain unused.</p>'''
    text+=f'<p>{len(records)} / {total} configurations have saved scores or an explicit skip. '
    text+=f'{len(failures)} unresolved failures/blocked heads (including any incomplete reports). Reload to see progress.</p>'
    if active:text+='<p><b>Current work:</b> '+html.escape(active)+'</p>'
    text+='<p>Each plot link includes normalized referral AUC, referral curves, Spearman plots, automation curves and applicable GP training diagnostics. '
    text+='Negative task Spearman is desirable; positive referral gain is desirable. Infeasible mappings and skipped duplicates are retained in the table.</p>'
    if len(display):text+=display.to_html(index=False,escape=False,float_format=lambda x:f'{x:.4g}',na_rep='—')
    if failures:
        text+='<h2>Failures and blocked heads</h2><p>These are retried on restart. Saved scientific scores are retained when only reporting failed.</p>'
        text+=failure_frame.to_html(index=False,escape=True)
    if (root/'training_data.json').exists():
        text+='<h2>Training data</h2><pre>'+html.escape(str(read_json(root/'training_data.json')))+'</pre>'
    atomic_text(root/'report.html',text+'</html>')
    return records


def worker(request_path, gpu):
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    out=Path(read_json(request_path)['spec']['directory'])
    with cache_lock(out/'.worker.lock',label='head worker'):
        if (out/'result.json').exists():
            return finish_job(read_json(request_path))
        return _worker(request_path,gpu)


def _worker(request_path, gpu):
    request=read_json(request_path)
    if request['identity']!=identity(request['spec']):raise RuntimeError('Head request checksum changed')
    spec=request['spec'];root=Path(spec['root']);out=Path(spec['directory']);row=spec['recipe']
    from .study_precision import validate_worker_request,assert_worker_hardware
    validate_worker_request(root,request)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    configure_caches(root)
    from .cli import device_setup
    from .gpu_memory import configure_memory,memory_record
    _,device=device_setup(gpu);budget=configure_memory(device)
    assert_worker_hardware(request,device)
    cfg=ExperimentConfig.load(spec['config_path'])
    cfg=replace(cfg,features=FeatureConfig(**row['features']),gp=GPConfig(**row['gp']))
    cfg.validate()
    started=time.perf_counter();status='complete';metrics=None;reason=None
    from .panels import score_panel
    from .preprocessing import UninformativeBank
    try:
        metrics=score_panel(cfg,spec['run_dir'],spec['round'],out,device=device,methods=(row['method'],),
            supervisor_model=spec['supervisor'],evaluation_directory=root/f'round_{spec["round"]:03d}/learner')[row['method']]
        mapping=metrics.get('mapping_check',{})
        if mapping.get('performed') and not mapping.get('usable'):status='mapping_infeasible'
    except UninformativeBank as error:
        status='uninformative';reason=str(error)
    except ValueError as error:
        if 'Repeated inputs' not in str(error):raise
        status='missing_conditioning';reason=str(error)
    from .runner import source_identity
    record=dict(round=spec['round'],recipe=row,status=status,reason=reason,metrics=metrics,
        seconds=time.perf_counter()-started,directory=str(out),request_identity=request['identity'],
        execution_source_hash=source_identity(),
        gpu_memory=memory_record(device,budget))
    atomic_json(out/'result.json',record)
    head_report(record)
    print(f'HEAD DONE: {row["id"]}; round={spec["round"]}; status={status}; wall={record["seconds"]:.1f}s; report={out/"report/report.html"}',flush=True)


def queued_jobs(root,recipes,rounds,run_dir,teacher,source_hash,only=None):
    jobs=[]
    for row in recipes:
        if only and only not in row['id']:continue
        for round_ in rounds:
            out=root/f'round_{round_:03d}'/row['id']
            spec=dict(root=str(root),directory=str(out),config_path=str(root/'config.json'),run_dir=str(run_dir),
                supervisor=teacher,round=round_,recipe=row,source_hash=source_hash)
            request=dict(identity=identity(spec),spec=spec)
            from .study_precision import precision_request,archive_previous_request
            request=precision_request(root,request)
            if (out/'result.json').exists():
                try:finish_job(request)
                except Exception as error:
                    # Repair/validate this one result in its worker; do not stop
                    # the queue for an individual report or artifact error.
                    print(f'Requeue saved head for validation/report repair: {row["id"]}: {error}',flush=True)
                else:
                    print(f'Already complete: round {round_}, {row["id"]}',flush=True)
                    continue
            archive_previous_request(root,request)
            atomic_json(out/'request.json',request)
            jobs.append(request)
    return sorted(jobs,key=execution_phase)


def finish_job(request):
    out=Path(request['spec']['directory'])
    record=read_json(out/'result.json')
    if record['request_identity']!=request['identity']:raise RuntimeError('Completed head identity changed')
    if not (out/'report/complete.json').exists():head_report(record)
    (out/'failure.json').unlink(missing_ok=True)


def resume_declaration(root, stored, requested):
    """Accept only recorded exact transitions for the repair or appended matrix.

    Existing declarations/results retain their original provenance. New result
    files additionally record execution_source_hash. Any further source or
    experimental-setting change still requires a new study. The append manifest
    additionally pins the full old/new matrices and an unchanged original prefix.
    """
    if stored == requested:
        return stored
    from .study_precision import resume_precision_declaration
    upgraded=resume_precision_declaration(root,stored,requested)
    if upgraded is not None:
        print('Verified precision migration: unfinished KV-T fits restart in float64; completed results retained.',flush=True)
        return upgraded
    from .study_compatibility import append_declaration,scheduler_declaration
    scheduled=scheduler_declaration(stored,requested)
    if scheduled is not None:
        print('Verified scheduler update: action heads run last; individual failures are recorded and retried on restart.',flush=True)
        return scheduled
    appended=append_declaration(stored,requested)
    if appended is not None:
        print('Verified append-only whitening extension: original heads/results and optimizer checkpoints retained.',flush=True)
        return appended
    repair_path=Path(root)/'numerical_repair.json'
    if repair_path.exists():
        repair=read_json(repair_path)
        if (repair.get('kind')=='kvt_float64_posterior_retry_v1'
                and repair.get('source_before')==stored.get('source_hash')
                and repair.get('source_after')==requested.get('source_hash')
                and dict(requested,source_hash=stored.get('source_hash'))==stored):
            print(f'Resuming with verified numerical repair: {repair_path}; completed results retained.',flush=True)
            return stored
    raise RuntimeError('Study settings/source changed. Use a new --root; existing results are preserved.')


def job_ready(request):
    spec=request['spec'];out=Path(spec['directory']);row=spec['recipe']
    if not row.get('conditional_on'):return 'ready'
    predecessor=out.parent/row['conditional_on']/'result.json'
    if not predecessor.exists():return 'wait'
    base=read_json(predecessor)
    previous_request=predecessor.with_name('request.json')
    if previous_request.exists() and base.get('request_identity')!=read_json(previous_request)['identity']:
        raise RuntimeError('Prerequisite result identity changed')
    geometry=(base.get('metrics') or {}).get('gp_diagnostics',{}).get('geometry',{})
    if geometry.get('pls_criterion_met') or base['status'] in ('uninformative','missing_conditioning'):
        record=dict(round=spec['round'],recipe=row,status='not_needed',
            reason='PLS-32 already reaches 99%, or the input mapping is undefined',metrics=None,
            seconds=0,directory=str(out),request_identity=request['identity'])
        atomic_json(out/'result.json',record);head_report(record)
        (out/'failure.json').unlink(missing_ok=True)
        print(f'SKIP round {spec["round"]}: {row["id"]}; {record["reason"]}',flush=True)
        return 'skip'
    return 'ready'


def queue_report(root,running,pending):
    active=[dict(gpu=w.slot['label'],device=w.slot['device'],pid=w.process.pid,
        round=w.job['spec']['round'],head=w.job['spec']['recipe']['id'],
        seconds=time.monotonic()-w.started,log=str(w.log_path)) for w in running]
    failures=failure_records(root)
    atomic_json(root/'queue.json',dict(running=active,pending=len(pending),
        completed=len(completed_records(root)),failed=sum(r['status']=='failed' for r in failures),
        blocked=sum(r['status']=='blocked' for r in failures),updated_at=time.time()))
    message='; '.join(f'GPU {r["gpu"]}: round {r["round"]}, {r["head"]}' for r in active)
    refresh_index(root,active=message or (f'{len(pending)} heads queued.' if pending else
        f'Queue drained; {len(failures)} unresolved failures/blocked heads.' if failures else 'All selected work completed.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu','--gpus',dest='gpu',default='6')
    parser.add_argument('--root',default='storage/extensions/head_ablation_v1')
    parser.add_argument('--scope',choices=('all','observation'),default='all')
    parser.add_argument('--rounds',default='0,10')
    parser.add_argument('--gp-steps',type=int,default=10000)
    parser.add_argument('--output-batch-size',type=int,default=16)
    parser.add_argument('--flow-samples',type=int,default=50)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--only',help='Execute matching recipe IDs; keep the complete declaration')
    parser.add_argument('--gpu-routing',choices=('auto','model'),default='auto',
        help='Model routing: A100 for KV-T, RTX 3090 for other heads. Mandatory for the precision migration.')
    parser.add_argument('--worker-request',help=argparse.SUPPRESS)
    args=parser.parse_args()
    try:gpus=parse_gpus(args.gpu)
    except ValueError as error:parser.error(str(error))
    if min(args.gp_steps,args.output_batch_size,args.flow_samples)<1:parser.error('Budgets must be positive')
    if args.worker_request:
        if len(gpus)!=1:parser.error('A worker must receive exactly one GPU')
        return worker(args.worker_request,gpus[0])
    root=Path(args.root).resolve()
    rounds=tuple(map(int,args.rounds.split(',')))
    if not rounds or any(r not in (0,5,10,15) for r in rounds) or len(set(rounds))!=len(rounds):
        parser.error('Choose distinct retained rounds from 0,5,10,15')
    recipes=study_recipes(scope=args.scope,fit_steps=args.gp_steps,output_batch_size=args.output_batch_size,flow_samples=args.flow_samples)
    if args.plan:
        from collections import Counter
        print(f'Frozen-policy head study: {len(recipes)} declared recipes x {len(rounds)} models = {len(recipes)*len(rounds)} cells, including conditional PLS checks.')
        print(dict(Counter((r['scope'],r['method']) for r in recipes)))
        added=[r for r in recipes if r.get('extension')=='whitening_v1']
        print(f'Original recipe definitions retained: {len(recipes)-len(added)}; appended: {len(added)} '
              f'({sum(r["method"]=="kv" for r in added)} KV, {sum(r["method"]=="kv_t" for r in added)} KV-T per policy).')
        phases=Counter(execution_phase(dict(spec=dict(recipe=r))) for r in recipes)
        print(f'Execution order with stage barriers: {phases[0]*len(rounds)} observation/reference, '
              f'{phases[1]*len(rounds)} action KV, {phases[2]*len(rounds)} action KV-T. Individual failures do not stop the queue.')
        print(f'Validation: 200 attempts/model. GP cap: {args.gp_steps}. Outputs trained together: {args.output_batch_size}. No policy training.')
        print(f'GPUs: {", ".join(gpus)}; concurrent heads: {len(gpus)} (one per GPU); output: {root}; report: {root/"report.html"}')
        return
    root.mkdir(parents=True,exist_ok=True)
    with (root/'study.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This head study already has a running process')
        if args.report_only:
            for record in completed_records(root):head_report(record)
            return refresh_index(root)
        from .data import load_manifest
        from .reference import import_reproduction
        from .runner import source_identity
        manifest=load_manifest('storage/extensions/data/data_manifest.json')
        teacher_record=read_json('storage/extensions/experiments/v1/supervisor.json')
        teacher=teacher_record['model']
        if model_identity(teacher)!=teacher_record['model_hash']:raise RuntimeError('Fixed supervisor weights changed')
        cfg=ExperimentConfig(name='head_study_common',arm='random',companion=True,
            base_models=[str(Path('storage/outputs/pretrain/libero_pretrain_leak3')/f'seed_{i}/checkpoints/last/pretrained_model') for i in (0,1)],
            output_root=str(root/'imported'),feature_batch_size=4,evaluation_batch_size=15)
        source=Path('storage/outputs/reproduction/random/seed_1000')
        run_dir=import_reproduction(cfg,source,manifest,through_round=max(rounds))
        data=training_data_summary(manifest,source,teacher)
        declaration=dict(config=plain(cfg),recipes=recipes,rounds=list(rounds),supervisor=teacher,
            supervisor_hash=model_identity(teacher),split='development',split_label='validation',condition=None,
            manifest=manifest['manifest_hash'],source_hash=source_identity(),run=str(run_dir),memory_limit_gib=24)
        path=root/'declaration.json'
        if path.exists():
            stored=read_json(path)
            declaration=resume_declaration(root,stored,plain(declaration))
            if declaration.get('append_only_extension') and not stored.get('append_only_extension'):
                atomic_json(root/'declaration_before_whitening.json',stored)
            if declaration.get('scheduler_update') and not stored.get('scheduler_update'):
                atomic_json(root/'declaration_before_scheduler.json',stored)
        atomic_json(path,declaration);atomic_json(root/'training_data.json',data)
        atomic_json(root/'config.json',cfg)
        print(f'Saved models reused. Training data: {data}',flush=True)
        print(f'{len(recipes)} recipes x {len(rounds)} policies. Live report: {root/"report.html"}',flush=True)
        configure_caches(root)
        signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        jobs=queued_jobs(root,recipes,rounds,run_dir,teacher,declaration['source_hash'],args.only)
        if jobs:(root/'complete.json').unlink(missing_ok=True)
        slots=resolve_slots(gpus) if jobs else []
        from .study_precision import migration_record
        if slots and (args.gpu_routing=='model' or migration_record(root)):
            from .head_queue import assign_model_roles
            slots=assign_model_roles(slots)
        from .head_queue import slot_accepts,validate_slots
        validate_slots(jobs,slots)
        eligible_slots=[s for s in slots if any(slot_accepts(j,s) for j in jobs)]
        atomic_json(root/'execution.json',dict(gpus=gpus,slots=slots,concurrent_heads=len(eligible_slots),started_at=time.time(),
            stage_order=['observation_reference','action_kv','action_kv_t'],failure_policy='continue_and_retry_on_restart'))
        print(f'Dispatching {len(jobs)} remaining heads across {len(eligible_slots)} eligible GPUs '
              f'({len(slots)} selected); one head per GPU.',flush=True)
        counts=[sum(execution_phase(j)==phase for j in jobs) for phase in range(3)]
        print(f'Stage order: observation/reference ({counts[0]} remaining), action KV ({counts[1]}), '
              f'action KV-T ({counts[2]}). Each stage drains before the next; individual errors are recorded and skipped.',flush=True)
        queue_report(root,[],jobs)
        try:
            run_queue(jobs,slots,ready=job_ready,on_complete=finish_job,
                on_update=lambda running,pending:queue_report(root,running,pending),on_failure=record_failure,start=start_child)
        except (KeyboardInterrupt,Exception):
            # run_queue checkpoints/stops all active children before this lock is released.
            atomic_json(root/'queue.json',dict(running=[],status='stopped',updated_at=time.time()))
            refresh_index(root,active='Stopped. Rerun the same command to resume; GPU count may be changed.')
            raise
        records=refresh_index(root)
        failures=failure_records(root)
        if failures:
            (root/'complete.json').unlink(missing_ok=True)
            queue=read_json(root/'queue.json');queue['status']='finished_with_failures';atomic_json(root/'queue.json',queue)
            print(f'All runnable selected heads attempted; {len(failures)} failures/blocked heads remain. '
                  f'Details: {root/"failures.csv"}. Rerun after fixes; completed heads will be skipped.',flush=True)
            raise SystemExit(1)
        if len(records)==len(recipes)*len(rounds):atomic_json(root/'complete.json',dict(cells=len(records)))
        print(f'Head study finished. Report: {root/"report.html"}',flush=True)


if __name__=='__main__':main()
