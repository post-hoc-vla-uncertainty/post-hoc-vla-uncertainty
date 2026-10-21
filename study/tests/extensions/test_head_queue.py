"""Real subprocess scheduling, cancellation, and shared-cache contention."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from uq_extensions.common import atomic_json,identity,read_json
from uq_extensions.head_queue import parse_gpus,run_queue


WORKER='''
import json,os,signal,sys,time
from pathlib import Path
job=json.loads(Path(sys.argv[1]).read_text());spec=job['spec'];out=Path(spec['directory'])
def stop(*_):
    (out/'terminated').write_text('checkpoint opportunity')
    sys.exit(143)
signal.signal(signal.SIGTERM,stop)
started=time.monotonic();(out/'started').write_text(str(started))
time.sleep(spec['duration'])
if spec.get('fail'):sys.exit(7)
record=dict(start=started,end=time.monotonic(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],
    egl=os.environ.get('MUJOCO_EGL_DEVICE_ID'),request_identity=job['identity'])
(out/'result.json').write_text(json.dumps(record))
'''


def queue_fixture(tmp_path,monkeypatch,durations):
    import uq_extensions.head_queue as queue
    helper=tmp_path/'worker.py';helper.write_text(WORKER)
    jobs=[]
    for name,duration in durations.items():
        spec=dict(directory=str(tmp_path/name),recipe=dict(id=name),round=0,duration=duration)
        job=dict(spec=spec,identity=identity(spec));atomic_json(Path(spec['directory'])/'request.json',job)
        jobs.append(job)
    processes=[];actual_popen=subprocess.Popen
    def popen(command,**kwargs):
        assert command[command.index('--gpu')+1]==kwargs['env']['CUDA_VISIBLE_DEVICES']
        assert ',' not in kwargs['env']['CUDA_VISIBLE_DEVICES']
        assert kwargs['start_new_session'] is True
        request=command[command.index('--worker-request')+1]
        process=actual_popen([sys.executable,str(helper),request],**kwargs);processes.append(process)
        return process
    monkeypatch.setattr(queue.subprocess,'Popen',popen)
    slots=[dict(label=str(i),device=f'GPU-slot-{i}',environment=dict(CUDA_VISIBLE_DEVICES=f'GPU-slot-{i}',
        MUJOCO_EGL_DEVICE_ID=str(i+4))) for i in range(2)]
    return jobs,slots,processes


def test_queue_overlaps_heads_and_immediately_refills_free_gpu(tmp_path,monkeypatch):
    jobs,slots,processes=queue_fixture(tmp_path,monkeypatch,{'slow':1.,'fast':.12,'next':.12})
    completed=[];snapshots=[]
    run_queue(jobs,slots,ready=lambda _: 'ready',on_complete=lambda j:completed.append(j['spec']['recipe']['id']),
        on_update=lambda active,pending:snapshots.append([w.slot['label'] for w in active]),poll_seconds=.01)
    results={n:read_json(tmp_path/n/'result.json') for n in ('slow','fast','next')}
    assert completed==['fast','next','slow']
    assert results['fast']['start']<results['slow']['end']
    assert results['next']['end']<results['slow']['end']
    assert results['next']['gpu']==results['fast']['gpu']!=results['slow']['gpu']
    assert results['next']['egl']=='5'
    assert all(len(active)==len(set(active))<=2 for active in snapshots)
    assert all(p.poll()==0 for p in processes)


def test_model_routing_overlaps_methods_without_using_wrong_gpu(tmp_path,monkeypatch):
    jobs,slots,_=queue_fixture(tmp_path,monkeypatch,{'target':.3,'geometry':.05,'target2':.05})
    slots[0]['role']='kv_t';slots[1]['role']='other'
    for job,method in zip(jobs,('kv_t','kv','kv_t')):
        job['spec']['recipe']['method']=method
        atomic_json(Path(job['spec']['directory'])/'request.json',job)
    run_queue(jobs,slots,ready=lambda _:'ready',on_complete=lambda _:None,on_update=lambda *_:None,poll_seconds=.01)
    first=read_json(tmp_path/'target/result.json');kv=read_json(tmp_path/'geometry/result.json')
    second=read_json(tmp_path/'target2/result.json')
    assert first['gpu']==second['gpu']=='GPU-slot-0'
    assert kv['gpu']=='GPU-slot-1' and kv['start']<first['end']
    assert second['start']>=first['end']


def test_conditional_jobs_wait_for_predecessor_while_other_heads_run(tmp_path,monkeypatch):
    jobs,slots,_=queue_fixture(tmp_path,monkeypatch,{'adaptive':.05,'base':.25,'independent':.05})
    def ready(job):
        if job['spec']['recipe']['id']=='adaptive' and not (tmp_path/'base/result.json').exists():return 'wait'
        return 'ready'
    run_queue(jobs,slots,ready=ready,on_complete=lambda _:None,on_update=lambda *_:None,poll_seconds=.01)
    assert read_json(tmp_path/'adaptive/result.json')['start']>=read_json(tmp_path/'base/result.json')['end']
    assert read_json(tmp_path/'independent/result.json')['end']<read_json(tmp_path/'base/result.json')['end']


def test_failed_worker_is_recorded_while_other_workers_and_pending_heads_continue(tmp_path,monkeypatch):
    jobs,slots,processes=queue_fixture(tmp_path,monkeypatch,{'long':.8,'failure':.15,'pending':.05})
    jobs[1]['spec']['fail']=True
    atomic_json(tmp_path/'failure/request.json',jobs[1])
    recorded=[];completed=[]
    failures=run_queue(jobs,slots,ready=lambda _:'ready',on_complete=lambda j:completed.append(j['spec']['recipe']['id']),
        on_failure=lambda j,e:recorded.append(e),on_update=lambda *_:None,poll_seconds=.01)
    assert len(failures)==len(recorded)==1 and recorded[0]['exit_code']==7
    assert set(completed)=={'long','pending'}
    assert len(processes)==3 and all(p.poll() is not None for p in processes)
    assert not (tmp_path/'long/terminated').exists()
    assert read_json(tmp_path/'pending/result.json')['end']<read_json(tmp_path/'long/result.json')['end']


def test_interrupt_stops_all_active_process_groups(tmp_path,monkeypatch):
    jobs,slots,processes=queue_fixture(tmp_path,monkeypatch,{'a':10.,'b':10.})
    def interrupt(active,pending):
        deadline=time.monotonic()+3
        while not all((tmp_path/n/'started').exists() for n in ('a','b')):
            if time.monotonic()>deadline:raise RuntimeError('worker startup timeout')
            time.sleep(.01)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run_queue(jobs,slots,ready=lambda _:'ready',on_complete=lambda _:None,on_update=interrupt,poll_seconds=.01)
    assert all(p.poll() is not None for p in processes)
    assert all((tmp_path/n/'terminated').exists() for n in ('a','b'))


def test_no_runnable_dependency_is_reported(tmp_path):
    failures=run_queue([{'spec':{'round':0,'directory':str(tmp_path),'recipe':{'id':'missing_base'}}}],
        [dict(label='0')],ready=lambda _:'wait',on_complete=lambda _:None,on_update=lambda *_:None)
    assert len(failures)==1 and failures[0]['status']=='blocked'
    assert 'predecessor' in failures[0]['reason']


def test_action_stages_wait_for_all_preceding_stage_workers(tmp_path,monkeypatch):
    jobs,slots,_=queue_fixture(tmp_path,monkeypatch,{'action_t':.03,'obs_slow':.35,'action_kv':.06,'obs_fast':.03})
    for job in jobs:
        row=job['spec']['recipe']
        if row['id'].startswith('action'):
            row.update(scope='action',method='kv_t' if row['id']=='action_t' else 'kv')
    failures=run_queue(jobs,slots,ready=lambda _:'ready',on_complete=lambda _:None,on_update=lambda *_:None,poll_seconds=.01)
    result={name:read_json(tmp_path/name/'result.json') for name in ('action_t','obs_slow','action_kv','obs_fast')}
    assert not failures
    assert result['action_kv']['start']>=max(result[n]['end'] for n in ('obs_slow','obs_fast'))
    assert result['action_t']['start']>=result['action_kv']['end']


def test_failed_dependency_is_blocked_without_blocking_next_phase_and_retried_on_restart(tmp_path,monkeypatch):
    jobs,slots,_=queue_fixture(tmp_path,monkeypatch,{'adaptive':.02,'base':.08,'action':.02})
    jobs[0]['spec']['recipe']['conditional_on']='base'
    jobs[1]['spec']['fail']=True
    jobs[2]['spec']['recipe'].update(scope='action',method='kv_t')
    atomic_json(tmp_path/'base/request.json',jobs[1])
    def ready(job):
        name=job['spec']['recipe']['id']
        if (tmp_path/name/'result.json').exists():return 'skip'
        if name=='adaptive' and not (tmp_path/'base/result.json').exists():return 'wait'
        return 'ready'
    failures=run_queue(jobs,slots,ready=ready,on_complete=lambda _:None,on_update=lambda *_:None,poll_seconds=.01)
    assert [(f['job']['spec']['recipe']['id'],f['status']) for f in failures]==[('base','failed'),('adaptive','blocked')]
    assert not (tmp_path/'adaptive/started').exists() and (tmp_path/'action/result.json').exists()
    previous_action=(tmp_path/'action/result.json').read_bytes()
    jobs[1]['spec']['fail']=False
    atomic_json(tmp_path/'base/request.json',jobs[1])
    assert not run_queue(jobs,slots,ready=ready,on_complete=lambda _:None,on_update=lambda *_:None,poll_seconds=.01)
    assert read_json(tmp_path/'adaptive/result.json')['start']>=read_json(tmp_path/'base/result.json')['end']
    assert (tmp_path/'action/result.json').read_bytes()==previous_action


@pytest.mark.parametrize('failure_stage',['ready','start','report'])
def test_individual_callback_errors_do_not_stop_other_heads(tmp_path,monkeypatch,failure_stage):
    import uq_extensions.head_queue as queue
    jobs,slots,_=queue_fixture(tmp_path,monkeypatch,{'bad':.02,'good':.02})
    def fault(job):
        if job['spec']['recipe']['id']=='bad':raise RuntimeError('deliberate failure')
    def ready(job):
        if failure_stage=='ready':fault(job)
        return 'ready'
    def start(job,slot):
        if failure_stage=='start':fault(job)
        return queue.start_child(job,slot)
    def complete(job):
        if failure_stage=='report':fault(job)
    failures=run_queue(jobs,slots,ready=ready,start=start,on_complete=complete,on_update=lambda *_:None,poll_seconds=.01)
    assert len(failures)==1 and 'deliberate failure' in failures[0]['reason']
    assert (tmp_path/'good/result.json').exists()


def test_adaptive_pls_is_skipped_only_after_its_own_base_meets_criterion(tmp_path,monkeypatch):
    import uq_extensions.head_study as study
    reports=[];monkeypatch.setattr(study,'head_report',lambda record:reports.append(record))
    spec=dict(round=10,directory=str(tmp_path/'round_010/adaptive'),
        recipe=dict(id='adaptive',conditional_on='base'))
    request=dict(spec=spec,identity=identity(spec))
    assert study.job_ready(request)=='wait'
    base=dict(status='complete',metrics={'gp_diagnostics':{'geometry':{'pls_criterion_met':True}}})
    atomic_json(tmp_path/'round_000/base/result.json',base)
    assert study.job_ready(request)=='wait'
    atomic_json(tmp_path/'round_010/base/result.json',base)
    atomic_json(tmp_path/'round_010/adaptive/failure.json',{'status':'blocked'})
    assert study.job_ready(request)=='skip'
    assert not (tmp_path/'round_010/adaptive/failure.json').exists()
    assert read_json(tmp_path/'round_010/adaptive/result.json')['status']=='not_needed'
    assert len(reports)==1
    base['metrics']['gp_diagnostics']['geometry']['pls_criterion_met']=False
    atomic_json(tmp_path/'round_010/base/result.json',base)
    assert study.job_ready(request)=='ready'


@pytest.mark.parametrize('bad',['','1,,2','1,1','01,1','cpu,1','cuda:0'])
def test_invalid_gpu_lists_are_rejected(bad):
    with pytest.raises(ValueError):parse_gpus(bad)


def test_parse_gpu_list_keeps_physical_selections():
    assert parse_gpus(' 0, 6,7 ')==['0','6','7']


def test_uuid_aliases_cannot_schedule_two_workers_on_same_physical_gpu(monkeypatch):
    import lerobot.utils.gpu_utils as gpu
    from uq_extensions.head_queue import resolve_slots
    uuid='gpu-12345678-1234-1234-1234-123456789abc'
    monkeypatch.setattr(gpu,'nvidia_gpu_uuids',lambda:{'6':uuid})
    with pytest.raises(RuntimeError,match='duplicate'):
        resolve_slots(['6',uuid])


CACHE_WORKER='''
import os,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from uq_extensions.common import atomic_json,read_json
from uq_extensions.config import ExperimentConfig,FeatureConfig
from uq_extensions.cache_lock import lock_directory
import uq_extensions.acquisition as acquisition
import uq_extensions.features as features
import uq_extensions.preprocessing as preprocessing
from uq_extensions.kernel_variance import KernelVariance
root=Path(sys.argv[1]);number=sys.argv[2]
os.environ['UQ_EXT_FEATURE_CACHE']=str(root/'features')
os.environ['UQ_EXT_ESTIMATOR_CACHE']=str(root/'estimators')
os.environ['UQ_EXT_PROJECTION_CACHE']=str(root/'projections')
os.environ.pop('UQ_EXT_BUNDLE_FEATURES',None)
def count(name):
    with (root/(name+'.calls')).open('a') as f:f.write(number+'\\n')
    time.sleep(.15)
original_fit=KernelVariance.fit
def fit(self,*a,**kw):count('fit');return original_fit(self,*a,**kw)
KernelVariance.fit=fit
original_projection=preprocessing._fit_projection
def project(*a,**kw):count('projection');return original_projection(*a,**kw)
preprocessing._fit_projection=project
features.model_identity=acquisition.model_identity=lambda _: 'fixed_weights'
features.load_policy=lambda *a:(None,None,None)
features.preprocess_inputs=lambda observations,_:observations
class Extractor:
    spec={'fake':True}
    def __init__(self,*args):pass
    def extract(self,observations,banks,**kwargs):
        count('extract')
        return {bank:np.asarray(observations,dtype=np.float32) for bank in banks},{}
features.FeatureExtractor=Extractor
values=np.random.default_rng(7).normal(size=(6,5))
support=SimpleNamespace(ids=[str(i) for i in range(6)],tasks=['a']*6,
    candidates=[SimpleNamespace(input_hash=str(i)) for i in range(6)],observation=lambda i:values[int(i)])
cfg=ExperimentConfig(arm='kv',feature_batch_size=6,features=FeatureConfig(components=3))
model=acquisition.fit_estimator(cfg,'checkpoint',support,root/number,'cpu')
@lock_directory('directory','.evaluation.lock',label='test evaluation')
def evaluate(directory):
    result=directory/'evaluation.json'
    if result.exists():return read_json(result)
    count('evaluation');atomic_json(result,{'success':True});return {'success':True}
assert evaluate(root/'shared_evaluation')['success']
atomic_json(root/(number+'.json'),{'scores':model.score(values).tolist()})
'''


def test_parallel_workers_fit_shared_features_projections_estimators_and_evaluations_once(tmp_path):
    helper=tmp_path/'cache_worker.py';helper.write_text(CACHE_WORKER)
    processes=[]
    try:
        for i in range(3):
            with (tmp_path/f'{i}.log').open('w') as log:
                processes.append(subprocess.Popen([sys.executable,str(helper),str(tmp_path),str(i)],
                    stdout=log,stderr=subprocess.STDOUT))
        for i,process in enumerate(processes):
            assert process.wait(timeout=45)==0,(tmp_path/f'{i}.log').read_text()
    finally:
        for process in processes:
            if process.poll() is None:process.kill();process.wait()
    for operation in ('extract','projection','fit','evaluation'):
        assert len((tmp_path/f'{operation}.calls').read_text().splitlines())==1
    assert read_json(tmp_path/'0.json')==read_json(tmp_path/'1.json')==read_json(tmp_path/'2.json')


def test_process_death_releases_cache_lock(tmp_path):
    from uq_extensions.cache_lock import cache_lock
    helper=tmp_path/'lock_worker.py'
    helper.write_text('from pathlib import Path\nimport sys,time\nfrom uq_extensions.cache_lock import cache_lock\n'
        'p=Path(sys.argv[1])\nwith cache_lock(p/"test.lock"):\n (p/"ready").touch()\n time.sleep(30)\n')
    process=subprocess.Popen([sys.executable,str(helper),str(tmp_path)])
    try:
        deadline=time.monotonic()+10
        while not (tmp_path/'ready').exists():
            if time.monotonic()>deadline:raise RuntimeError('lock worker startup timeout')
            time.sleep(.01)
        process.kill();process.wait(timeout=5)
        with cache_lock(tmp_path/'test.lock'):
            assert (tmp_path/'test.lock').exists()
    finally:
        if process.poll() is None:process.kill();process.wait()
