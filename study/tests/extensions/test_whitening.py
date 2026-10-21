"""Geometry, target isolation, batched controls and append-only study contracts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uq_extensions.common import atomic_json, identity, plain, read_json
from uq_extensions.config import ExperimentConfig, FeatureConfig, GPConfig
from uq_extensions.preprocessing import fit_projection, shrunk_covariance
from uq_extensions.kernel_variance import KernelVariance, initial_log_parameters, geometry_lengthscale, rbf
from uq_extensions.target_gp import TargetGP


def sample():
    rng=np.random.default_rng(71)
    x=(rng.normal(size=(48,9))*np.geomspace(.03,3,9))@rng.normal(size=(9,12))
    y=np.sin(x[:,:4])+x[:,4:8]
    return x,y


def test_whitened_random_distances_equal_regularized_mahalanobis(tmp_path,monkeypatch):
    x,_=sample()
    monkeypatch.setenv('UQ_EXT_PROJECTION_CACHE',str(tmp_path))
    base=fit_projection(x,method='random',components=8)
    projection=fit_projection(x,method='random_whiten',components=8)
    z=((x-base.mean)/base.scale-base.center)@base.weights
    covariance,details=shrunk_covariance(z)
    delta=z[0]-z[1]
    expected=delta@np.linalg.solve(covariance,delta)
    actual=np.square(projection.transform(x)[0]-projection.transform(x)[1]).sum()
    assert actual==pytest.approx(expected,rel=2e-6)
    assert projection.diagnostics['whitening']['shrinkage']==details['shrinkage']
    # Restored cached transforms remain fixed when the query distribution changes.
    restored=fit_projection(x,method='random_whiten',components=8)
    np.testing.assert_array_equal(restored.transform(x+20),projection.transform(x+20))
    assert restored.diagnostics==projection.diagnostics
    assert len(list(tmp_path.glob('*.npz')))==2 # base projection is shared


@pytest.mark.parametrize('method',['pca','pls'])
def test_component_whitening_only_rescales_and_survives_rank_deficiency(method):
    x,y=sample();x=np.column_stack([x,x[:,0],np.ones(len(x))])
    base=fit_projection(x,method=method,components=4,targets=y)
    white=fit_projection(x,method=method+'_whiten',components=4,targets=y)
    scales=white.diagnostics['whitening']['scales']
    np.testing.assert_allclose(white.weights,base.weights/np.asarray(scales),atol=1e-12)
    assert np.isfinite(white.transform(x+1)).all()
    assert white.diagnostics['whitening']['basis']=='existing_component_axes'


def test_pls_kv_reuses_target_gp_projection_and_needs_no_optimizer(tmp_path):
    x,y=sample()
    gp=GPConfig(input_projection='pls',input_components=4,output_components=3,output_seed=142,precision='float64')
    features=FeatureConfig(compression='pls',components=4)
    reference=TargetGP(features,gp)._geometry(x,y)[0]
    model=KernelVariance(features,gp).fit(x,y)
    np.testing.assert_array_equal(model.projection.weights,reference.weights)
    scores=model.score(x[:5]+.2)
    model.save(tmp_path)
    loaded=KernelVariance.load(tmp_path,features,gp)
    np.testing.assert_array_equal(loaded.score(x[:5]+.2),scores)
    assert loaded.diagnostics['target_informed_projection']
    with pytest.raises(ValueError,match='aligned'):KernelVariance(features,gp).fit(x)


def test_variance_initialization_matches_whitened_isotropic_kernel():
    x,_=sample();z=fit_projection(x,method='pls',components=4,targets=x[:,:3]).transform(x)
    tx=torch.as_tensor(z,dtype=torch.float64)
    gp=GPConfig(lengthscale_initialization='variance',precision='float64')
    logs,diagnostics=initial_log_parameters(tx,gp)
    covariance,_=shrunk_covariance(z)
    scales=np.sqrt(np.maximum(covariance.diagonal(),diagnostics['variance_floor']))
    normalized=tx/torch.tensor(scales)
    length,_=geometry_lengthscale(normalized)
    torch.testing.assert_close(rbf(tx,tx,torch.tensor(np.exp(logs[:-2]))),rbf(normalized,normalized,length),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('projection,initialization',[('random_whiten','geometry'),('pls','variance')])
def test_target_gp_new_methods_fit_save_restore_and_report(tmp_path,projection,initialization):
    x,y=sample()
    gp=GPConfig(fit_steps=2,precision='float64',input_projection=projection,input_components=4,
                output_components=3,lengthscale_initialization=initialization)
    model=TargetGP(FeatureConfig(),gp).fit(x,y,tasks=['a']*len(x))
    model.save(tmp_path)
    loaded=TargetGP.load(tmp_path,FeatureConfig(),gp)
    np.testing.assert_allclose(loaded.score(x[:5]+.3),model.score(x[:5]+.3),rtol=1e-8,atol=1e-9)
    if projection.endswith('_whiten'):
        assert model.diagnostics['mapping_check']['geometry']['whitening']['training_rows']==38
        assert model.diagnostics['geometry']['whitening']['training_rows']==48
    else:
        assert model.diagnostics['training_records']['full'][0]['initialization']['kind']=='variance'


def test_sparse_whitening_keeps_original_inducing_examples():
    x,_=sample()
    f=FeatureConfig(compression='random',components=6,geometry_seed=2026)
    gp=GPConfig(inference='sparse',inducing_points=12,sparse_justification='test',precision='float64')
    base=KernelVariance(f,gp).fit(x)
    white=KernelVariance(replace(f,compression='random_whiten'),gp).fit(x)
    transform=np.linalg.lstsq(base.projection.weights,white.projection.weights,rcond=None)[0]
    np.testing.assert_allclose(base.inducing.numpy()@transform,white.inducing.numpy(),atol=3e-7)


def test_patch_aggregation_reuses_saved_variances(tmp_path,monkeypatch):
    from uq_extensions.estimators import PatchEstimator
    rng=np.random.default_rng(2);x=rng.normal(size=(8,2,4,6))
    features=FeatureConfig(bank='visual_patches',compression='random_whiten',components=4,geometry_seed=2026)
    gp=GPConfig(precision='float64')
    model=PatchEstimator(features,gp).fit(x)
    model.save(tmp_path)
    expected=model.patch_scores(x+.1)
    restored=PatchEstimator.load(tmp_path,replace(features,patch_aggregation='mean'),gp)
    def forbidden(*_):raise AssertionError('Patch posterior was recomputed')
    monkeypatch.setattr(restored,'_patch_scores',forbidden)
    np.testing.assert_array_equal(restored.patch_scores(x+.1),expected)
    np.testing.assert_allclose(restored.score(x+.1),expected.mean(-1).max(-1))
    assert all('whitening' in camera for camera in restored.diagnostics['cameras'])


def test_appended_matrix_preserves_original_prefix_and_limits_expensive_fits():
    from uq_extensions.head_study_design import study_recipes,original_study_recipes
    before=original_study_recipes();after=study_recipes()
    assert len(before)==191 and len(after)==262
    assert after[:191]==before
    extras=after[191:]
    assert sum(r['method']=='kv' for r in extras)==68
    assert sum(r['method']=='kv_t' for r in extras)==3
    for row in after:
        assert 'lengthscale_initialization' not in row['gp'] or row['gp']['lengthscale_initialization']=='variance'
        ExperimentConfig(features=FeatureConfig(**row['features']),gp=GPConfig(**row['gp'])).validate()
    assert len(study_recipes(scope='observation'))==170


def test_append_resume_preserves_request_identities_and_rejects_other_changes(tmp_path,monkeypatch):
    from uq_extensions import study_compatibility as compat
    from uq_extensions.head_study import resume_declaration,queued_jobs
    before=[dict(id='old',method='kv')];after=before+[dict(id='new',method='kv')]
    registry=dict(kind='head_study_whitening_append_v1',source_before=['original'],source_after='new-source',
        recipes_before_hash=identity(before),recipes_after_hash=identity(after),original_recipe_count=1)
    path=tmp_path/'compat.json';atomic_json(path,registry);monkeypatch.setattr(compat,'COMPATIBILITY_PATH',path)
    old=dict(source_hash='original',recipes=before,rounds=[0,10],supervisor='fixed')
    requested=dict(old,source_hash='new-source',recipes=after)
    accepted=resume_declaration(tmp_path,old,requested)
    assert accepted['source_hash']=='original'
    assert resume_declaration(tmp_path,accepted,requested)==accepted
    for bad in (dict(requested,rounds=[0]),dict(requested,source_hash='unknown'),
                dict(requested,recipes=after[::-1]),dict(requested,supervisor='changed')):
        with pytest.raises(RuntimeError,match='settings/source changed'):resume_declaration(tmp_path,old,bad)
    jobs=queued_jobs(tmp_path,before,[0],tmp_path/'run','fixed','original')
    out=Path(jobs[0]['spec']['directory'])
    atomic_json(out/'result.json',dict(request_identity=jobs[0]['identity']))
    atomic_json(out/'report/complete.json',{})
    remaining=queued_jobs(tmp_path,accepted['recipes'],[0],tmp_path/'run','fixed',accepted['source_hash'])
    assert [r['spec']['recipe']['id'] for r in remaining]==['new']


@pytest.mark.parametrize('partial',[False,True])
def test_original_estimator_or_optimizer_is_reused_after_append(tmp_path,monkeypatch,partial):
    import uq_extensions.acquisition as acquisition
    import uq_extensions.runner as runner
    from uq_extensions import study_compatibility as compat
    from uq_extensions import study_precision as precision
    # This test represents the earlier append-only source transition, before
    # the later migration deliberately invalidates unfinished KV-T optimizers.
    monkeypatch.setattr(precision,'COMPATIBILITY_PATH',tmp_path/'no_precision_migration.json')
    from run_progress import strip_reporting_blocks
    cfg=ExperimentConfig(arm='kv_t')
    support=SimpleNamespace(ids=['one','two'],tasks=['a','a'],candidates=[SimpleNamespace(input_hash='a'),SimpleNamespace(input_hash='b')])
    modules=('features','estimators','kernel_variance','target_gp','batched_gp','preprocessing','coregionalized')
    code=identity({n:strip_reporting_blocks(Path(acquisition.__file__).with_name(n+'.py').read_text()) for n in modules})
    old=dict(method='kv_t',checkpoint='policy',support=support.ids,inputs=['a','b'],features=cfg.features,gp=cfg.gp,code='old-code')
    path=tmp_path/'estimator_cache'/identity(old)
    atomic_json(path/('optimizer/marker.json' if partial else 'identity.json'),dict(identity=identity(old),**old))
    registry=tmp_path/'compat.json'
    atomic_json(registry,dict(kind='head_study_whitening_append_v1',source_after='source',estimator_code_after=code,
        estimator_code_before=['old-code'],legacy_estimator_config_hashes=[identity({k:old[k] for k in ('method','features','gp')})]))
    monkeypatch.setattr(compat,'COMPATIBILITY_PATH',registry)
    monkeypatch.setattr(runner,'source_identity',lambda:'source')
    monkeypatch.setenv('UQ_EXT_ESTIMATOR_CACHE',str(tmp_path/'estimator_cache'))
    monkeypatch.setattr(acquisition,'model_identity',lambda _:'policy')
    calls=[]
    class Fake:
        def __init__(self):self.model=SimpleNamespace(training_directory=None);self.diagnostics={}
        def fit(self,*a,**kw):
            assert self.model.training_directory==path/'optimizer';calls.append('resume')
        def save(self,directory):assert directory==path
    def make(*a,**kw):
        if 'path' in kw:assert kw['path']==path;calls.append('load')
        return Fake()
    monkeypatch.setattr(acquisition,'make_estimator',make)
    monkeypatch.setattr(acquisition,'cache_features',lambda *a,**kw:{b:np.zeros((2,4)) for b in kw['banks']})
    current=dict(old,code=code)
    (tmp_path/'estimator_cache'/identity(current)/'support_features').mkdir(parents=True)
    acquisition.fit_estimator(cfg,'policy',support,tmp_path/'head','cpu')
    assert calls==['resume' if partial else 'load']
    assert compat.legacy_estimator_codes(dict(current,gp=replace(cfg.gp,fit_steps=1)))==[]
    monkeypatch.setattr(runner,'source_identity',lambda:'unapproved')
    assert compat.legacy_estimator_codes(current)==[]


def test_pls_kv_requests_targets_only_for_fitting(tmp_path,monkeypatch):
    from uq_extensions import acquisition
    x,y=sample();cfg=ExperimentConfig(arm='kv',features=FeatureConfig(compression='pls',components=4),
                                    gp=GPConfig(output_components=3,precision='float64'))
    monkeypatch.delenv('UQ_EXT_ESTIMATOR_CACHE',raising=False)
    monkeypatch.setattr(acquisition,'model_identity',lambda _:'policy')
    support=SimpleNamespace(ids=list(map(str,range(len(x)))),tasks=['a']*len(x),
                            candidates=[SimpleNamespace(input_hash=str(i)) for i in range(len(x))])
    requested=[]
    def cached(*a,**kwargs):
        requested.append(kwargs['banks'])
        return {cfg.features.bank:x,cfg.features.target_bank:y}
    monkeypatch.setattr(acquisition,'cache_features',cached)
    model=acquisition.fit_estimator(cfg,'policy',support,tmp_path,'cpu')
    assert requested==[(cfg.features.bank,cfg.features.target_bank)]
    assert np.isfinite(model.score(x[:3])).all()
