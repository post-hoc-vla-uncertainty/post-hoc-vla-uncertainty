"""Posterior recovery must preserve the fit and resume only an audited change."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uq_extensions.common import atomic_json, identity
from uq_extensions.config import ExperimentConfig, FeatureConfig, GPConfig
from uq_extensions.target_gp import ScalarGP, TargetGP


@pytest.mark.parametrize('inference', ['exact', 'sparse'])
def test_precision_retry_rebuilds_from_parameters_and_preserves_jitter(inference):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(24, 5)).astype('float32')
    y = rng.normal(size=(24, 2)).astype('float32')
    config = GPConfig(fit_steps=2, input_components=4, output_components=2,
                      inference=inference, inducing_points=8,
                      sparse_justification='recovery test' if inference == 'sparse' else None)
    model = TargetGP(FeatureConfig(), config).fit(x, y, tasks=['a'] * len(x), mapping_check=False)
    healthy = model.score(x)
    assert model.prediction_numerics['float64_retries'] == 0
    originals = model.models
    expected = []
    for original in originals:
        # Retain an already escalated jitter, rather than changing the GP on retry.
        original.rebuild(kernel_jitter=1e-3)
        reference = ScalarGP(replace(config, precision='float64', jitter=1e-3))
        for name in ('x', 'y', 'inducing', 'length', 'amplitude', 'noise'):
            setattr(reference, name, getattr(original, name).double())
        reference.rebuild()
        expected.append(reference.predict(model.projection.transform(x))[1].diag().numpy())
    reference_score = np.mean(expected, axis=0)

    # Simulate loss of accuracy in an intermediate float32 factor, not the fit.
    # Casting this factor to double would still give the wrong answer.
    bad_factor = originals[0].terms[0] * .5
    originals[0].terms = (bad_factor, *originals[0].terms[1:])
    actual = model.score(x, batch_size=5)
    np.testing.assert_allclose(actual, reference_score, rtol=1e-7, atol=1e-8)
    assert np.isfinite(actual).all() and (actual >= 0).all()
    assert model.prediction_numerics['float64_retries'] == 1
    assert model.prediction_numerics['posterior_precision'] == 'float64'
    for before, after in zip(originals, model.models):
        assert after.terms[-1] == before.terms[-1] == 1e-3
        assert after.history == before.history
        for name in ('x', 'y', 'inducing', 'length', 'amplitude', 'noise'):
            torch.testing.assert_close(getattr(after, name), getattr(before, name).double(), rtol=0, atol=0)
    np.testing.assert_allclose(model.score(x), actual, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(model.cross_covariance(x, x).diag(), actual, rtol=1e-7, atol=1e-8)
    assert model.prediction_numerics['float64_retries'] == 1


def test_persistent_bad_variance_and_unrelated_failures_are_not_hidden(monkeypatch):
    model = TargetGP(FeatureConfig(), GPConfig())
    model.projection = SimpleNamespace(transform=lambda x: x)
    model.models = [SimpleNamespace(dtype=torch.float64)]
    def invalid(*_):
        raise RuntimeError('Materially negative or nonfinite latent variance: -1.0')
    monkeypatch.setattr(model, '_score_projected', invalid)
    with pytest.raises(RuntimeError, match='negative'):
        model.score(np.zeros((2, 3)))
    model.models = [SimpleNamespace(dtype=torch.float32)]
    def oom(*_):
        raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(model, '_score_projected', oom)
    with pytest.raises(RuntimeError, match='out of memory'):
        model.score(np.zeros((2, 3)))
    assert model.prediction_numerics['float64_retries'] == 0


def test_resume_requires_exact_source_transition_and_unchanged_settings(tmp_path):
    from uq_extensions.head_study import resume_declaration
    stored = dict(source_hash='old', rounds=[0, 10], supervisor='fixed')
    requested = dict(stored, source_hash='fixed')
    with pytest.raises(RuntimeError, match='settings/source changed'):
        resume_declaration(tmp_path, stored, requested)
    atomic_json(tmp_path/'numerical_repair.json', dict(kind='kvt_float64_posterior_retry_v1',
                                                     source_before='old', source_after='fixed'))
    assert resume_declaration(tmp_path, stored, requested) == stored
    for changed in (dict(requested, source_hash='another-change'),
                    dict(requested, rounds=[0]), dict(requested, supervisor='other')):
        with pytest.raises(RuntimeError, match='settings/source changed'):
            resume_declaration(tmp_path, stored, changed)


@pytest.mark.parametrize('partial', [False, True])
def test_repair_reuses_fit_or_optimizer_cache_without_changing_scientific_inputs(tmp_path, monkeypatch, partial):
    import uq_extensions.acquisition as acquisition
    from run_progress import strip_reporting_blocks
    from pathlib import Path
    config = ExperimentConfig(arm='kv_t')
    support = SimpleNamespace(ids=['one','two'], tasks=['a','a'],
                              candidates=[SimpleNamespace(input_hash='a'),SimpleNamespace(input_hash='b')])
    cache = tmp_path/'estimator_cache'
    monkeypatch.setenv('UQ_EXT_ESTIMATOR_CACHE', str(cache))
    monkeypatch.setattr(acquisition, 'model_identity', lambda _: 'policy')
    modules = ('features','estimators','kernel_variance','target_gp','batched_gp','preprocessing','coregionalized')
    code = identity({name:strip_reporting_blocks(Path(acquisition.__file__).with_name(name+'.py').read_text()) for name in modules})
    previous = dict(method='kv_t', checkpoint='policy', support=support.ids, inputs=['a','b'],
                    features=config.features, gp=config.gp, code='previous-code')
    old_key = identity(previous); old_path = cache/old_key
    if partial:
        atomic_json(old_path/'optimizer/marker.json', dict(step=7000))
    else:
        atomic_json(old_path/'identity.json', dict(identity=old_key, **previous))
    atomic_json(tmp_path/'numerical_repair.json', dict(kind='kvt_float64_posterior_retry_v1',
        estimator_code_before='previous-code', estimator_code_after=code))
    seen = []
    class FakeEstimator:
        def __init__(self):
            self.model = SimpleNamespace(training_directory=None)
            self.diagnostics = {}
        def fit(self,*args,**kwargs):
            assert self.model.training_directory == old_path/'optimizer'
            seen.append('resume')
        def save(self, path):
            assert path == old_path
    def make(*_, **kwargs):
        if 'path' in kwargs:
            assert kwargs['path'] == old_path
            seen.append('load')
        return FakeEstimator()
    monkeypatch.setattr(acquisition,'make_estimator',make)
    monkeypatch.setattr(acquisition,'cache_features',lambda *a,**k:{
        bank:np.zeros((2,4)) for bank in (config.features.bank, config.features.target_bank)})
    acquisition.fit_estimator(config, 'policy', support, tmp_path/'head/estimator', 'cpu')
    assert seen == ['resume' if partial else 'load']
    # A different support pool must not collide with the compatible cache.
    changed = dict(previous, inputs=['changed','b'])
    assert identity(changed) != old_key
