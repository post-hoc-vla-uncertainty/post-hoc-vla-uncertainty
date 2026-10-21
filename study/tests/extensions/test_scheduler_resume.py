"""Operational source migration and durable failure/retry reporting."""
from pathlib import Path

import pytest

from uq_extensions.common import atomic_json,identity,read_json


def test_scheduler_upgrade_preserves_both_legacy_and_appended_request_identities(tmp_path,monkeypatch):
    from uq_extensions import study_compatibility as compat
    from uq_extensions.head_study import resume_declaration,queued_jobs
    before=[dict(id='old',method='kv',scope='action')]
    after=before+[dict(id='new',method='kv',scope='observation')]
    whitening=dict(kind='head_study_whitening_append_v1',source_before=['original'],source_after='whitening',
        recipes_before_hash=identity(before),recipes_after_hash=identity(after),original_recipe_count=1)
    scheduler=dict(kind='head_study_scheduler_v1',base_source_hash='whitening',source_after='scheduler')
    for attr,name,data in [('COMPATIBILITY_PATH','whitening.json',whitening),
                           ('SCHEDULER_COMPATIBILITY_PATH','scheduler.json',scheduler)]:
        path=tmp_path/name;atomic_json(path,data);monkeypatch.setattr(compat,attr,path)
    original=dict(source_hash='original',recipes=before,rounds=[0,10],supervisor='fixed')
    expanded=dict(original,source_hash='whitening',recipes=after)
    appended=compat.append_declaration(original,expanded)
    requested=dict(expanded,source_hash='scheduler')
    accepted=resume_declaration(tmp_path,original,requested)
    assert resume_declaration(tmp_path,appended,requested)==accepted
    assert resume_declaration(tmp_path,accepted,requested)==accepted
    assert accepted['source_hash']=='original'
    assert accepted['append_only_extension']==appended['append_only_extension']
    fresh_upgrade=resume_declaration(tmp_path,expanded,requested)
    assert fresh_upgrade['source_hash']=='whitening'
    assert resume_declaration(tmp_path,fresh_upgrade,requested)==fresh_upgrade
    for change in (dict(rounds=[0]),dict(source_hash='unknown'),dict(recipes=after[::-1]),dict(supervisor='different')):
        with pytest.raises(RuntimeError,match='settings/source changed'):
            resume_declaration(tmp_path,original,dict(requested,**change))
    previous=queued_jobs(tmp_path,before,[0],tmp_path/'run','fixed','original')[0]
    current=queued_jobs(tmp_path,accepted['recipes'],[0],tmp_path/'run','fixed',accepted['source_hash'])
    assert [j['spec']['recipe']['id'] for j in current]==['new','old']
    assert current[1]==previous
    assert compat.compatible_execution_source('scheduler','whitening')
    assert not compat.compatible_execution_source('unknown','whitening')


def test_failures_are_visible_and_successful_report_retry_preserves_scores(tmp_path,monkeypatch):
    import uq_extensions.head_study as study
    import uq_extensions.runner as runner
    monkeypatch.setattr(runner,'source_identity',lambda:'current')
    recipe=dict(id='head',method='kv',features={},gp={},question='test')
    atomic_json(tmp_path/'declaration.json',dict(recipes=[recipe],rounds=[0]))
    job=study.queued_jobs(tmp_path,[recipe],[0],tmp_path/'run','fixed','original')[0]
    out=Path(job['spec']['directory'])
    record=dict(round=0,recipe=recipe,status='complete',metrics=None,reason=None,seconds=1.,
                directory=str(out),request_identity=job['identity'])
    atomic_json(out/'result.json',record)
    original=(out/'result.json').read_bytes()
    def broken(_):raise RuntimeError('report failed')
    monkeypatch.setattr(study,'head_report',broken)
    assert study.queued_jobs(tmp_path,[recipe],[0],tmp_path/'run','fixed','original')==[job]
    study.record_failure(job,dict(status='failed',reason='report failed',gpu='6',exit_code=1,log=str(out/'run.log')))
    study.refresh_index(tmp_path)
    assert len(read_json(tmp_path/'failures.json'))==1
    assert 'report failed' in (tmp_path/'report.html').read_text()
    assert 'report failed' in (tmp_path/'failures.csv').read_text()
    monkeypatch.setattr(study,'head_report',lambda _:atomic_json(out/'report/complete.json',{}))
    # worker exits through report repair before reaching policy scoring.
    monkeypatch.setattr(study,'_worker',lambda *a:pytest.fail('Finished scoring must not repeat'))
    study.worker(out/'request.json','cpu')
    assert not (out/'failure.json').exists()
    assert len(read_json(out/'failure_history.json'))==1
    assert (out/'result.json').read_bytes()==original
    assert study.queued_jobs(tmp_path,[recipe],[0],tmp_path/'run','fixed','original')==[]
    study.refresh_index(tmp_path)
    assert read_json(tmp_path/'failures.json')==[]


def test_broken_individual_result_does_not_break_progress_report(tmp_path):
    import uq_extensions.head_study as study
    broken=tmp_path/'round_000/broken/result.json';broken.parent.mkdir(parents=True)
    broken.write_text('{incomplete')
    atomic_json(tmp_path/'round_000/wrong/result.json',{'status':'complete'})
    assert study.refresh_index(tmp_path)==[]
    assert (tmp_path/'report.html').exists()


def test_launcher_drains_queue_before_nonzero_exit_and_retries_only_failed_head(tmp_path,monkeypatch):
    import os
    import sys
    import time
    from types import SimpleNamespace
    import uq_extensions.head_study as study
    import uq_extensions.data as data
    import uq_extensions.reference as reference
    import uq_extensions.runner as runner
    from uq_extensions.head_queue import RunningHead
    from uq_extensions.head_study_design import study_recipes
    root=tmp_path/'study';recipes=study_recipes()[:2]
    for env in ('UQ_EXT_FEATURE_CACHE','UQ_EXT_PROJECTION_CACHE','UQ_EXT_ESTIMATOR_CACHE',
                'UQ_EXT_BUNDLE_FEATURES','UQ_EXT_RETIRE_FEATURE_CACHE','UQ_EXT_STUDY_MEMORY_GUARD','PYTHONUNBUFFERED'):
        monkeypatch.setenv(env,os.environ.get(env,''))
    monkeypatch.setattr(study,'study_recipes',lambda **_:recipes)
    monkeypatch.setattr(study,'training_data_summary',lambda *_:{})
    monkeypatch.setattr(data,'load_manifest',lambda _:dict(manifest_hash='manifest'))
    monkeypatch.setattr(reference,'import_reproduction',lambda *a,**k:tmp_path/'imported')
    monkeypatch.setattr(runner,'source_identity',lambda:'source')
    monkeypatch.setattr(study,'model_identity',lambda _:'teacher')
    original_read=study.read_json
    monkeypatch.setattr(study,'read_json',lambda p:dict(model='teacher',model_hash='teacher')
        if str(p)=='storage/extensions/experiments/v1/supervisor.json' else original_read(p))
    monkeypatch.setattr(study,'resolve_slots',lambda ids:[dict(label=i,device='cpu',environment={}) for i in ids])
    monkeypatch.setattr(study,'head_report',lambda record:atomic_json(Path(record['directory'])/'report/complete.json',{}))
    launched=[];fail=True
    def child(job,slot):
        row=job['spec']['recipe'];out=Path(job['spec']['directory']);launched.append(row['id'])
        status=7 if fail and row['id']==recipes[0]['id'] else 0
        if not status:
            atomic_json(out/'result.json',dict(round=0,recipe=row,status='complete',reason=None,
                metrics=None,seconds=1.,directory=str(out),request_identity=job['identity']))
        process=SimpleNamespace(pid=os.getpid(),poll=lambda:status,wait=lambda:status)
        return RunningHead(process,None,job,slot,out/'run.log',time.monotonic())
    monkeypatch.setattr(study,'start_child',child)
    monkeypatch.setattr(sys,'argv',['head_study','--root',str(root),'--rounds','0','--gpus','1,2'])
    with pytest.raises(SystemExit) as stopped:study.main()
    assert stopped.value.code==1
    assert launched==[r['id'] for r in recipes]
    assert not (root/'complete.json').exists()
    assert read_json(root/'queue.json')['status']=='finished_with_failures'
    assert len(read_json(root/'failures.json'))==1
    good=root/'round_000'/recipes[1]['id']/'result.json';unchanged=good.read_bytes()
    fail=False;launched.clear()
    study.main()
    assert launched==[recipes[0]['id']]
    assert good.read_bytes()==unchanged
    assert read_json(root/'complete.json')['cells']==2
    assert read_json(root/'failures.json')==[]
