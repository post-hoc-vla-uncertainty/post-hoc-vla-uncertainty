"""Precision migration isolates old fits and enforces heterogeneous GPU routing."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from uq_extensions.common import atomic_json,identity,read_json
from uq_extensions.config import GPConfig


def migration_fixture(tmp_path,monkeypatch):
    import uq_extensions.study_precision as precision
    import uq_extensions.study_compatibility as compat
    from uq_extensions.head_study import queued_jobs
    recipes=[dict(id='old',method='kv_t',scope='action',gp=dict(precision='float32',output_dependence='independent')),
             dict(id='pending',method='kv_t',scope='action',gp=dict(precision='float32',output_dependence='independent')),
             dict(id='kv',method='kv',scope='observation',gp=dict(precision='float32'))]
    declaration=dict(source_hash='before',recipes=recipes,rounds=[0],supervisor='fixed')
    jobs=queued_jobs(tmp_path,recipes,[0],tmp_path/'run','fixed','before')
    old=next(j for j in jobs if j['spec']['recipe']['id']=='old')
    out=Path(old['spec']['directory'])
    atomic_json(out/'result.json',dict(request_identity=old['identity'],status='complete'))
    atomic_json(out/'report/complete.json',{})
    pending=next(j for j in jobs if j['spec']['recipe']['id']=='pending')
    config=tmp_path/'compatibility.json'
    record=dict(kind=precision.KIND,source_before='before',source_after='after',
                estimator_code_before='old-code',estimator_code_after='new-code')
    atomic_json(config,record);monkeypatch.setattr(precision,'COMPATIBILITY_PATH',config)
    atomic_json(tmp_path/'precision_migration.json',dict(kind=precision.KIND,compatibility_hash=identity(record),
        declaration_before_hash=identity(declaration),request_source_before='before',
        cells={'round_000/pending':dict(request_before_identity=pending['identity'])}))
    return declaration,recipes,pending,old


def test_migration_preserves_finished_results_restarts_only_declared_cells_and_is_idempotent(tmp_path,monkeypatch):
    from uq_extensions import head_study as study,study_precision as precision,runner
    declaration,recipes,pending,old=migration_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'source_identity',lambda:'after')
    requested=dict(declaration,source_hash='after')
    updated=study.resume_declaration(tmp_path,declaration,requested)
    assert updated['recipes']==recipes and updated['source_hash']=='before'
    assert study.resume_declaration(tmp_path,updated,requested)==updated
    old_result=Path(old['spec']['directory'])/'result.json';before=old_result.read_bytes()
    jobs=study.queued_jobs(tmp_path,recipes,[0],tmp_path/'run','fixed',updated['source_hash'])
    new=next(j for j in jobs if j['spec']['recipe']['id']=='pending')
    assert new['spec']['recipe']['gp']['precision']=='float64'
    assert new['identity']!=pending['identity']
    assert next(j for j in jobs if j['spec']['recipe']['id']=='kv')['spec']['recipe']['gp']['precision']=='float32'
    assert old_result.read_bytes()==before
    assert read_json(tmp_path/'precision_archive/round_000/pending/request.json')==pending
    assert study.queued_jobs(tmp_path,recipes,[0],tmp_path/'run','fixed','before')==jobs
    precision.validate_worker_request(tmp_path,new)
    with pytest.raises(RuntimeError,match='float64 request'):precision.validate_worker_request(tmp_path,pending)
    for change in (dict(rounds=[10]),dict(source_hash='unknown'),dict(supervisor='different')):
        with pytest.raises(RuntimeError,match='settings/source changed'):
            study.resume_declaration(tmp_path,updated,dict(requested,**change))
    corrupt=deepcopy(new);corrupt['spec']['recipe']['gp']['noise']=1.
    with pytest.raises(RuntimeError):precision.validate_worker_request(tmp_path,corrupt)


def test_new_estimator_code_reuses_only_unchanged_kv_caches(tmp_path,monkeypatch):
    from uq_extensions import study_compatibility as compat,runner
    migration_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'source_identity',lambda:'after')
    assert compat.legacy_estimator_codes(dict(method='kv',code='new-code'))==['old-code']
    assert compat.legacy_estimator_codes(dict(method='kv_t',code='new-code'))==[]
    monkeypatch.setattr(runner,'source_identity',lambda:'unknown')
    assert compat.legacy_estimator_codes(dict(method='kv',code='new-code'))==[]


def test_gpu_roles_are_by_uuid_not_index_and_missing_models_fail_before_launch(monkeypatch):
    import uq_extensions.head_queue as queue
    monkeypatch.setattr(queue.subprocess,'check_output',lambda *a,**k:
        'GPU-a, NVIDIA A100-PCIE-40GB\nGPU-b, NVIDIA GeForce RTX 3090\nGPU-unselected, NVIDIA A100-PCIE-40GB\n')
    slots=queue.assign_model_roles([dict(label='4',device='GPU-a'),dict(label='6',device='GPU-b')])
    assert [s['role'] for s in slots]==['kv_t','other']
    kvt=dict(spec=dict(recipe=dict(method='kv_t')));kv=dict(spec=dict(recipe=dict(method='kv')))
    assert queue.slot_accepts(kvt,slots[0]) and not queue.slot_accepts(kvt,slots[1])
    assert queue.slot_accepts(kv,slots[1]) and not queue.slot_accepts(kv,slots[0])
    with pytest.raises(ValueError,match='A100'):queue.validate_slots([kvt],slots[1:])
    with pytest.raises(ValueError,match='RTX 3090'):queue.validate_slots([kv],slots[:1])
    queue.validate_slots([kvt,kv],slots)


def test_worker_refuses_3090_for_migrated_kvt(monkeypatch):
    from uq_extensions.study_precision import assert_worker_hardware,KIND
    request=dict(spec=dict(precision_migration=KIND))
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'NVIDIA GeForce RTX 3090')
    with pytest.raises(RuntimeError,match='A100'):assert_worker_hardware(request,'cuda')
    with pytest.raises(RuntimeError,match='A100'):assert_worker_hardware(request,'cpu')
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'NVIDIA A100-PCIE-40GB')
    assert_worker_hardware(request,'cuda')


def test_stable_sparse_objective_and_gradient_match_dense_variational_bound():
    from uq_extensions.batched_gp import terms,kernel
    torch.manual_seed(12);torch.set_num_threads(2)
    x=torch.randn(45,4,dtype=torch.float64);y=torch.randn(2,45,dtype=torch.float64)
    z=torch.stack([x[:10],x[10:20]])
    logs=torch.zeros((2,6),dtype=torch.float64,requires_grad=True)
    config=GPConfig(precision='float64',inference='sparse',jitter=0.)
    actual,_=terms(x,y,z,logs,config)
    length,amplitude,noise=logs[:,:-2].exp(),logs[:,-2].exp(),logs[:,-1].exp()
    kuu=kernel(z,z,length,amplitude);kuf=kernel(z,x,length,amplitude)
    q=kuf.transpose(-1,-2)@torch.linalg.solve(kuu,kuf)
    covariance=q+noise[:,None,None]*torch.eye(len(x),dtype=x.dtype)
    trace=len(x)*amplitude-q.diagonal(dim1=-2,dim2=-1).sum(-1)
    reference=.5*((y*torch.linalg.solve(covariance,y.unsqueeze(-1)).squeeze(-1)).sum(-1)
                   +torch.linalg.slogdet(covariance).logabsdet+trace/noise)
    torch.testing.assert_close(actual,reference,rtol=1e-11,atol=1e-11)
    ga=torch.autograd.grad(actual.sum(),logs,retain_graph=True)[0]
    gr=torch.autograd.grad(reference.sum(),logs)[0]
    torch.testing.assert_close(ga,gr,rtol=1e-9,atol=1e-10)


def test_double_kernel_construction_handles_near_duplicates_without_jitter_escalation():
    from uq_extensions.batched_gp import terms
    torch.manual_seed(5);torch.set_num_threads(2)
    x=torch.randn(600,8);x[1:64]=x[0]+torch.randn(63,8)*1e-5
    y=torch.stack([x[:,0],x[:,1]]);z=torch.stack([x[:64],x[:64]])
    logs=torch.zeros((2,10),requires_grad=True)
    with torch.no_grad():logs[:,:8]=3.;logs[:,-2]=4.;logs[:,-1]=-13.
    loss,posterior=terms(x,y,z,logs,GPConfig(precision='float64',inference='sparse'))
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(logs.grad).all()
    assert posterior[0].dtype==torch.float64 and posterior[-1]==1e-6


def test_float32_optimizer_cannot_resume_as_float64(tmp_path):
    from dataclasses import replace
    from uq_extensions.batched_gp import fit_batched
    rng=np.random.default_rng(3);x=rng.normal(size=(12,3));y=rng.normal(size=(12,2))
    config=GPConfig(fit_steps=1)
    fit_batched(x,y,config,directory=tmp_path)
    with pytest.raises(RuntimeError,match='identity changed'):
        fit_batched(x,y,replace(config,precision='float64'),directory=tmp_path)
