import json
from pathlib import Path
import numpy as np
import pytest
import torch
from uq_extensions.common import array_hash,atomic_npz,identity,atomic_json,isolated_rng
from uq_extensions.config import ExperimentConfig
from uq_extensions.protocol import Candidate,CandidateView,AcquisitionLedger,select_save,task_weights
from uq_extensions.training import starting_models,make_training_config
from uq_extensions.conditions import apply_condition,transform_training_batch,validate_condition


def view(tmp_path,n=9):
    rows=[]
    for i in range(n):
        obs=dict(image=np.zeros((4,4,3),dtype=np.uint8),image2=np.zeros((4,4,3),dtype=np.uint8),state=np.ones(8)*i,instruction=np.asarray('move'))
        path=tmp_path/f'{i}.npz';atomic_npz(path,**obs)
        rows.append(Candidate(str(i),i,str(i%3),str(path),identity({k:array_hash(v) for k,v in obs.items()})))
    return CandidateView(rows)


def test_candidate_rejects_actions_and_changed_inputs(tmp_path):
    v=view(tmp_path)
    o=v.observation('0');o['action']=np.zeros(7);atomic_npz(v.candidates[0].observation_file,**o)
    with pytest.raises(ValueError,match='undeclared'):v.observation('0')
    del o['action'];o['state']+=1;atomic_npz(v.candidates[0].observation_file,**o)
    with pytest.raises(RuntimeError,match='changed'):v.observation('0')


def test_ledger_disjoint_resume_and_commit_before_reveal(tmp_path):
    v=view(tmp_path);args=dict(manifest_hash='x',protocol='unique_demo',pretraining_ids=['0','1'],candidates=v.ids)
    ledger=AcquisitionLedger(tmp_path/'ledger.json',**args)
    assert set(ledger.eligible())==set(v.ids)-{'0','1'}
    with pytest.raises(RuntimeError):ledger.reveal_indices(0,{})
    with pytest.raises(ValueError):ledger.commit(0,['0','2'],audit={},expected_count=2)
    ledger.commit(0,['2','3'],audit={},expected_count=2)
    ledger=AcquisitionLedger(tmp_path/'ledger.json',**args)
    assert ledger.support==['0','1','2','3']
    assert ledger.reveal_indices(0,{id:int(id) for id in v.ids})==[2,3]
    with pytest.raises(RuntimeError):ledger.commit(0,['4','5'],audit={},expected_count=2)
    with pytest.raises(ValueError):ledger.commit(1,['3','5'],audit={},expected_count=2)


def test_save_stable_ties_exhaustion_and_uniform_tasks(tmp_path):
    v=view(tmp_path);scores=dict.fromkeys(v.ids,1.)
    a,_=select_save(v,scores,count=9,temperature=2.5,seed=11)
    b,_=select_save(CandidateView(v.candidates[::-1]),scores,count=9,temperature=2.5,seed=11)
    assert a==b and len(set(a))==9
    assert sum(task_weights(['a','a','a','b'])[:3])==pytest.approx(.5)
    with pytest.raises(ValueError):select_save(v,scores,count=10,temperature=0,seed=1)


def test_modes_and_native_policy_counts():
    for arm,count in [('random',1),('vfd',2),('gu',2),('kv',1),('kv_t',1),('kv_ivr',1)]:
        assert ExperimentConfig(arm=arm).policy_count==count
    cfg=ExperimentConfig(training_mode='pristine_refit')
    assert starting_models(cfg,['base'],['different'],7)==['base']
    cfg.training_mode='save_warm_50_50'
    assert starting_models(cfg,['base'],['previous'],7)==['previous']
    assert starting_models(cfg,['base'],[],0)==['base']


def test_rng_scoring_does_not_change_training_stream():
    torch.manual_seed(3);np.random.seed(4)
    state=torch.get_rng_state().clone();numpy_state=np.random.get_state()
    with isolated_rng(93):torch.rand(7);np.random.rand(4)
    assert torch.equal(state,torch.get_rng_state())
    assert np.array_equal(numpy_state[1],np.random.get_state()[1])


def test_condition_consistent_training_and_evaluation():
    c=dict(name='test',family='undertrained_appearance',tasks=['a'],instructions={'a':'new task'},
           recolor=dict(lower_rgb=[0,0,0],upper_rgb=[40,40,40],replacement_rgb=[90,120,150],camera='both'))
    validate_condition(c)
    obs=dict(image=np.zeros((3,3,3),np.uint8),image2=np.zeros((3,3,3),np.uint8),state=np.zeros(8),instruction=np.asarray('old'))
    transformed=apply_condition(obs,c,'a')
    batch={'episode_index':torch.tensor([9,10]),'task':['old','old']}
    for k in ('image','image2'):batch['observation.images.'+k]=torch.zeros(2,3,3,3)
    changed=transform_training_batch(batch,{'9':dict(task='a',condition=c)})
    np.testing.assert_allclose(changed['observation.images.image'][0].permute(1,2,0).numpy()*255,transformed['image'],atol=1e-5)
    assert changed['task']==['new task','old'] and changed['observation.images.image'][1].sum()==0
    with pytest.raises(ValueError,match='verification'):validate_condition(c,final=True)


def test_final_requires_freeze_and_smoke_cannot_masquerade():
    with pytest.raises(ValueError):ExperimentConfig(phase='final').validate()
    with pytest.raises(ValueError):ExperimentConfig(allow_smoke_base=True).validate()
    with pytest.raises(ValueError):ExperimentConfig(fit_steps=2).validate()
