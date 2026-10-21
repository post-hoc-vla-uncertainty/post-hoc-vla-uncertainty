"""Numerical equivalence, optimizer continuation and frozen-study contracts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uq_extensions.batched_gp import fit_batched
from uq_extensions.config import FeatureConfig, GPConfig
from uq_extensions.target_gp import ScalarGP, TargetGP
from uq_extensions.common import atomic_json, read_json


def test_study_resume_keeps_completed_scores_and_repairs_reports(tmp_path,monkeypatch):
    """Exercise the launcher/worker/report boundary without running a robot policy."""
    import os
    import sys
    import time
    import uq_extensions.head_study as study
    import uq_extensions.data as data
    import uq_extensions.reference as reference
    import uq_extensions.runner as runner
    import uq_extensions.cli as cli
    import uq_extensions.panels as panels
    import uq_extensions.reporting as reporting
    import uq_extensions.common as common
    from uq_extensions.head_study_design import study_recipes
    from uq_extensions.head_queue import RunningHead

    root=tmp_path/'study';teacher_cache=tmp_path/'teacher_cache'
    for env in ('UQ_EXT_FEATURE_CACHE','UQ_EXT_PROJECTION_CACHE','UQ_EXT_ESTIMATOR_CACHE',
                'UQ_EXT_BUNDLE_FEATURES','UQ_EXT_RETIRE_FEATURE_CACHE','UQ_EXT_STUDY_MEMORY_GUARD','PYTHONUNBUFFERED'):
        monkeypatch.setenv(env,os.environ.get(env,''))
    recipes=study_recipes()[:2]
    monkeypatch.setattr(study,'study_recipes',lambda **_:recipes)
    monkeypatch.setattr(study,'training_data_summary',lambda *_:dict(round_0={'unique_demonstrations':8}))
    monkeypatch.setattr(data,'load_manifest',lambda _:dict(manifest_hash='manifest'))
    monkeypatch.setattr(reference,'import_reproduction',lambda *a,**k:tmp_path/'imported')
    monkeypatch.setattr(runner,'source_identity',lambda:'source')
    monkeypatch.setattr(study,'model_identity',lambda _:'teacher')
    monkeypatch.setattr(common,'model_identity',lambda _:'teacher')
    original_read=study.read_json
    monkeypatch.setattr(study,'read_json',lambda path:dict(model=str(tmp_path/'teacher'),model_hash='teacher')
        if str(path)=='storage/extensions/experiments/v1/supervisor.json' else original_read(path))
    monkeypatch.setattr(cli,'device_setup',lambda _:('cpu','cpu'))
    monkeypatch.setattr(study,'resolve_slots',lambda ids:[dict(label=i,device='cpu',environment={}) for i in ids])
    original_report=reporting.referral_diagnostic_report
    monkeypatch.setattr(reporting,'referral_diagnostic_report',
        lambda *a,**kw:original_report(*a,**dict(kw,supervisor_cache=teacher_cache)))

    scoring_calls=[]
    def score_panel(cfg,run,round_,out,*,methods,evaluation_directory,**kwargs):
        scoring_calls.append(Path(out).name)
        ids=list('abcdefgh');tasks=['a']*4+['b']*4
        def episodes(successes):
            return [dict(id=i,task=t,success=s,init_state_hash=i,reset_seed=0,repetition=0,
                condition='original',split='development',rendering_protocol='correct')
                for i,t,s in zip(ids,tasks,successes)]
        learner=episodes([1,0,1,0,0,0,1,0]);teacher=episodes([1]*8)
        spec=dict(checkpoint='learner',starts=[dict(id=i) for i in ids],rendering_protocol='correct',
            observation_preprocessing='rotate180',split='development')
        atomic_json(Path(evaluation_directory)/'evaluation.json',dict(spec=spec,episodes=learner))
        atomic_json(teacher_cache/'paired/evaluation.json',dict(spec=dict(spec,checkpoint='teacher'),episodes=teacher))
        scores={r['id']:1-r['success'] for r in learner}
        metrics=panels.assess_scores(scores,[(i,t,scores[i]) for i,t in zip(ids,tasks)],
            SimpleNamespace(ids=ids,tasks=tasks),learner,teacher,Path(out)/methods[0])
        return {methods[0]:metrics}
    monkeypatch.setattr(panels,'score_panel',score_panel)

    calls=[]
    def completed_child(job,slot):
        request=Path(job['spec']['directory'])/'request.json'
        study.worker(request,'cpu')
        process=SimpleNamespace(pid=os.getpid(),poll=lambda:0,wait=lambda:0)
        return RunningHead(process,None,job,slot,request.parent/'run.log',time.monotonic())
    def interrupted_child(job,slot):
        calls.append(job)
        if len(calls)==2:raise KeyboardInterrupt()
        return completed_child(job,slot)
    monkeypatch.setattr(study,'start_child',interrupted_child)
    monkeypatch.setattr(sys,'argv',['head_study','--root',str(root),'--rounds','0'])
    with pytest.raises(KeyboardInterrupt):study.main()
    assert scoring_calls==[recipes[0]['id']]
    first=root/'round_000'/recipes[0]['id']
    assert (first/'report/complete.json').exists()
    assert 'Stopped' in (root/'report.html').read_text()
    # A report can be interrupted independently of expensive scoring.
    (first/'report/complete.json').unlink()
    monkeypatch.setattr(study,'start_child',completed_child)
    # GPU assignments are operational settings: changing their count must preserve results.
    monkeypatch.setattr(sys,'argv',['head_study','--root',str(root),'--rounds','0','--gpus','1,2'])
    study.main()
    assert scoring_calls==[r['id'] for r in recipes]
    assert read_json(root/'complete.json')['cells']==2
    assert len(read_json(root/'matrix.json'))==2
    assert (first/'report/complete.json').exists()
    for row in recipes:
        report=root/'round_000'/row['id']/'report'
        assert (report/'report.html').exists()
        stats=read_json(report/'referral_diagnostics/summary.json')
        assert not stats['unavailable']
        assert all(r['oracle_normalized_gain']==pytest.approx(1) for r in stats['records'])


@pytest.mark.parametrize('inference',['exact','sparse'])
def test_batched_training_matches_independent_scalar_optimization(inference):
    rng=np.random.default_rng(9);x=rng.normal(size=(25,5));y=rng.normal(size=(25,3))
    config=GPConfig(precision='float64',fit_steps=5,inference=inference,inducing_points=9,
                    sparse_justification='numerical equivalence' if inference=='sparse' else None)
    batched,_=fit_batched(x,y,config,seed=41)
    for j,model in enumerate(batched):
        reference=ScalarGP(config).fit(x,y[:,j],seed=41+j)
        np.testing.assert_allclose(model.history,reference.history,rtol=1e-10,atol=1e-9)
        for actual,expected in zip(model.predict(x[:4]),reference.predict(x[:4])):
            torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-9)


def test_interrupted_batched_fit_resumes_identical_adam_trajectory(tmp_path):
    rng=np.random.default_rng(2);x=rng.normal(size=(20,4));y=rng.normal(size=(20,3))
    config=GPConfig(precision='float64',fit_steps=7,checkpoint_every=2,diagnostic_steps=(2,5,7))
    def stop(step):
        if step==3:raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):fit_batched(x,y,config,directory=tmp_path,step_callback=stop)
    assert torch.load(tmp_path/'outputs_00_03.pt',weights_only=True)['step']==3
    resumed,records=fit_batched(x,y,config,directory=tmp_path)
    uninterrupted,_=fit_batched(x,y,config)
    assert records[0]['resumed_from_step']==3
    assert [m['step'] for m in records[0]['milestones']]==[2,5,7]
    for left,right in zip(resumed,uninterrupted):
        torch.testing.assert_close(left.length,right.length,rtol=0,atol=0)
        assert left.history==right.history
    with pytest.raises(RuntimeError,match='identity changed'):
        fit_batched(x+.1,y,config,directory=tmp_path)


def test_interrupted_coregionalized_fit_resumes_optimizer_state(tmp_path,monkeypatch):
    from uq_extensions.coregionalized import CoregionalizedGP
    rng=np.random.default_rng(14);x=rng.normal(size=(20,4));y=rng.normal(size=(20,3))
    config=GPConfig(precision='float64',fit_steps=5,checkpoint_every=1,diagnostic_steps=(2,5),
        inference='sparse',sparse_justification='resume test',output_dependence='coregionalized',inducing_points=8)
    model=CoregionalizedGP(FeatureConfig(),config);model.training_directory=tmp_path
    original_step=torch.optim.Adam.step;steps=[]
    def interrupt(optimizer,*args,**kwargs):
        steps.append(1)
        if len(steps)==3:raise KeyboardInterrupt()
        return original_step(optimizer,*args,**kwargs)
    monkeypatch.setattr(torch.optim.Adam,'step',interrupt)
    with pytest.raises(KeyboardInterrupt):model._train_joint(x,y)
    monkeypatch.setattr(torch.optim.Adam,'step',original_step)
    resumed,likelihood,history=model._train_joint(x,y)
    reference,reference_likelihood,reference_history=CoregionalizedGP(FeatureConfig(),config)._train_joint(x,y)
    assert model.training_records['full'][0]['resumed_from_step']==2
    np.testing.assert_allclose(history,reference_history,rtol=1e-10)
    for name,value in resumed.state_dict().items():
        torch.testing.assert_close(value,reference.state_dict()[name],atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(likelihood.task_noises,reference_likelihood.task_noises,atol=1e-10,rtol=1e-10)


def test_input_selection_seeds_do_not_change_targets_or_validation_split():
    rng=np.random.default_rng(3);x=rng.normal(size=(35,10));y=rng.normal(size=(35,5));tasks=['a']*35
    g=GPConfig(input_projection='random',input_components=4,output_components=3,output_compression='random',
        output_seed=142,split_seed=242,training_seed=342,fit_steps=2)
    a=TargetGP(FeatureConfig(seed=42),g).fit(x,y,tasks=tasks)
    b=TargetGP(FeatureConfig(seed=43),g).fit(x,y,tasks=tasks)
    assert a.diagnostics['mapping_check']['test_indices']==b.diagnostics['mapping_check']['test_indices']
    np.testing.assert_array_equal(a.output_projection.weights,b.output_projection.weights)
    assert not np.array_equal(a.projection.weights,b.projection.weights)


def test_adaptive_pls_reports_unmet_criterion_and_respects_available_dimensions():
    rng=np.random.default_rng(5);x=rng.normal(size=(80,5));y=x*x+rng.normal(size=(80,5))
    g=GPConfig(input_components=2,output_components=3,pls_variance_target=.99,pls_max_components=512)
    model=TargetGP(FeatureConfig(),g)
    projection,_,_=model._geometry(x,y)
    assert projection.weights.shape[1]<=5
    assert model.geometry_diagnostics['pls_criterion_met'] is False
    assert model.geometry_diagnostics['pls_checks'][-1]['retained_target_variance_explained']<.99
    assert len(model.geometry_diagnostics['pls_checks'])==2 # 2 then rank-limited larger projection; no duplicate 512


def test_full_matrix_includes_declared_positions_and_conditional_variants():
    from uq_extensions.head_study_design import study_recipes,HEADS
    rows=study_recipes()
    assert {h['id'] for h in HEADS}<={r['group'] for r in rows}
    assert {r['features']['expert_distance'] for r in rows if r['features']['bank']=='action_expert'}=={0,1,3}
    ids={r['id'] for r in rows}
    for row in rows:
        if row.get('conditional_on'):assert row['conditional_on'] in ids
        if row['method']=='kv_t':
            assert row['gp']['output_components']==16 and row['gp']['output_batch_size']==16
            assert row['gp']['output_seed']==142 and row['gp']['pls_max_components']==512
        if row['features']['bank']=='state':
            assert row['features']['components']==8 and row['features']['seed']==42
    assert all(r['scope']!='action' for r in study_recipes(scope='observation'))


def test_feature_bundles_reuse_all_layers_and_resume_at_last_completed_batch(tmp_path,monkeypatch):
    import uq_extensions.features as feature
    monkeypatch.setenv('UQ_EXT_FEATURE_CACHE',str(tmp_path/'shared'))
    monkeypatch.setenv('UQ_EXT_BUNDLE_FEATURES','1')
    monkeypatch.setattr(feature,'model_identity',lambda p:'model')
    monkeypatch.setattr(feature,'load_policy',lambda *a:(object(),object(),object()))
    monkeypatch.setattr(feature,'preprocess_inputs',lambda obs,pre:obs)
    calls=[];fail=[True]
    class Extractor:
        def __init__(self,*args):self.spec={'dimensions':{}}
        def extract(self,batch,banks,**kwargs):
            calls.append(list(batch))
            if batch==[2,3] and fail[0]:raise KeyboardInterrupt()
            return {k:np.asarray(batch,dtype='float32')[:,None]+j for j,k in enumerate(banks)},{}
    monkeypatch.setattr(feature,'FeatureExtractor',Extractor)
    view=SimpleNamespace(ids=list(range(5)),candidates=[SimpleNamespace(input_hash=str(i)) for i in range(5)],observation=lambda i:i)
    f=FeatureConfig(bank='joint_late')
    with pytest.raises(KeyboardInterrupt):
        feature.cache_features('model',view,f,tmp_path/'a',batch_size=2,banks=('joint_late',))
    fail[0]=False
    a=feature.cache_features('model',view,f,tmp_path/'a',batch_size=2,banks=('joint_late',))
    assert calls==[[0,1],[2,3],[2,3],[4]]
    b=feature.cache_features('model',view,replace(f,bank='contextual_state_late',seed=43),tmp_path/'b',
                             batch_size=2,banks=('contextual_state_late',))
    assert len(calls)==4 and len(a['joint_late'])==len(b['contextual_state_late'])==5


def test_partial_reference_import_does_not_wait_for_or_depend_on_future_rounds(tmp_path,monkeypatch):
    from test_reference_reuse import fixture
    from uq_extensions.reference import import_reproduction,available_panel_rounds
    cfg,source,manifest=fixture(tmp_path,monkeypatch)
    state=read_json(source/'run_state.json')
    atomic_json(source/'run_state.json',dict(rounds=state['rounds'][:11]))
    out=import_reproduction(cfg,source,manifest,through_round=10)
    before=read_json(out/'reuse.json')
    assert available_panel_rounds(out,[0,10,15])==[0,10]
    atomic_json(source/'run_state.json',state)
    assert import_reproduction(cfg,source,manifest,through_round=10)==out
    assert read_json(out/'reuse.json')==before
