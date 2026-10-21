"""Numerical contracts for grouped flow, patch conditioning and multi-output IVR."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from uq_extensions.config import ExperimentConfig,FeatureConfig,GPConfig
from uq_extensions.estimators import EpisodeEstimator,PatchEstimator,probe_weights
from uq_extensions.ivr import select_ivr
from uq_extensions.target_gp import split_episodes
from uq_extensions.protocol import task_weights


def data():
    rng=np.random.default_rng(41);x=rng.normal(size=(24,2,5));y=x@rng.normal(size=(5,4))
    return x,y,[str(i%3) for i in range(24)]


def test_feasibility_split_keeps_all_probes_of_an_episode_together():
    _,_,tasks=data();groups=np.repeat(np.arange(len(tasks)),6)
    fit,test=split_episodes(np.repeat(tasks,6),groups=groups)
    assert not set(groups[fit])&set(groups[test])
    assert len(fit)+len(test)==len(groups)


@pytest.mark.parametrize('dependence',['independent','coregionalized'])
def test_multigroup_ivr_equals_direct_joint_trace_reduction(tmp_path,dependence):
    x,y,tasks=data();f=FeatureConfig(components=3)
    g=GPConfig(input_components=3,output_components=2,fit_steps=3,precision='float64',inference='sparse',
        inducing_points=8,sparse_justification='numerical test',output_dependence=dependence,output_compression='random')
    model=EpisodeEstimator(f,g,target=True).fit(x[:18],y[:18],tasks=tasks[:18])
    c=x[18:21];r=x[21:];ids=['a','b','c'];rt=tasks[21:]
    before=model.score(r).copy();chosen,audit=select_ivr(model,r,rt,c,ids,count=2,seed=3)
    if dependence=='coregionalized':
        raw=r.reshape(-1,5)
        diagonal=model.model.joint_covariance(raw,raw).diagonal().numpy().reshape(3,2,2).mean((1,2))
        np.testing.assert_allclose(before,diagonal,atol=1e-8)
    reordered,_=select_ivr(model,r,rt,c[::-1],ids[::-1],count=2,seed=3)
    assert chosen==reordered
    selected=c[[ids.index(id) for id in chosen]].reshape(-1,5);reference=r.reshape(-1,5)
    total=0.
    for covariance,noise,scale in model.covariance_components():
        rc=covariance(reference,selected).numpy();cc=covariance(selected,selected).numpy()
        reduction=(rc*np.linalg.solve(cc+np.diag(np.tile(noise,len(selected))),rc.T).T).sum(1)
        weights=np.repeat(np.repeat(task_weights(rt)/2,2),len(noise))
        total+=float(weights@reduction)*scale
    np.testing.assert_allclose(audit['initial_integrated_variance']-audit['final_integrated_variance'],total,atol=1e-7)
    np.testing.assert_array_equal(before,model.score(r))
    assert audit['probes_per_acquired_demo']==2
    model.save(tmp_path);loaded=EpisodeEstimator.load(tmp_path,f,g,target=True)
    np.testing.assert_allclose(before,loaded.score(r),atol=1e-7)


def test_patch_support_uses_more_than_geometry_sample_and_roundtrips(tmp_path):
    rng=np.random.default_rng(2);x=rng.normal(size=(5,2,9,4))
    f=FeatureConfig(bank='visual_patches',components=3,geometry_samples=12,patch_aggregation='top_mean',patch_top_fraction=.2)
    model=PatchEstimator(f,GPConfig(precision='float64')).fit(x)
    assert all(m.diagnostics['unique_rows']==45 for m in model.models)
    assert all(m.diagnostics['geometry_fit_rows']==12 for m in model.models)
    scores=model.patch_scores(x);expected=np.sort(scores,axis=-1)[...,-2:].mean(-1).max(-1)
    np.testing.assert_allclose(model.score(x),expected,atol=1e-7)
    model.save(tmp_path);loaded=PatchEstimator.load(tmp_path,f,GPConfig(precision='float64'))
    np.testing.assert_allclose(model.score(x),loaded.score(x),atol=1e-7)


def test_time_weights_do_not_change_total_episode_mass():
    f=FeatureConfig(flow_indices=(0,5,9),flow_weights=(1,2,1))
    w=probe_weights(12,f);assert w.sum()==1
    np.testing.assert_allclose(w.reshape(4,3).sum(0),[.25,.5,.25])


def test_new_arms_and_patch_restrictions_are_validated():
    ExperimentConfig(arm='kv_t_ivr').validate()
    with pytest.raises(ValueError,match='Patch heads'):
        ExperimentConfig(arm='kv_ivr',features=FeatureConfig(bank='visual_patches')).validate()
    with pytest.raises(ValueError,match='Coregionalized'):
        ExperimentConfig(gp=GPConfig(output_dependence='coregionalized')).validate()
