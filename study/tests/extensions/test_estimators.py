import numpy as np
import pytest
import torch

from uq_extensions.config import FeatureConfig, GPConfig
from uq_extensions.kernel_variance import KernelVariance
from uq_extensions.target_gp import TargetGP
from uq_extensions.ivr import select_ivr
from uq_extensions.preprocessing import UninformativeBank, fit_projection, fit_output_pca


def fixture():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(36, 6)).astype(np.float32)
    y = x @ rng.normal(size=(6, 5)) + rng.normal(scale=.01, size=(36, 5))
    return x, y, [str(i % 3) for i in range(len(x))]


def test_kv_matches_direct_conditioning_and_roundtrips(tmp_path):
    x, _, _ = fixture(); q = x[:6] + .3
    fc, gc = FeatureConfig(compression="random", components=6), GPConfig(precision="float64")
    kv = KernelVariance(fc, gc).fit(x)
    z = kv.projection.transform(x).astype(float); qz = kv.projection.transform(q).astype(float)
    kernel = lambda a, b: np.exp(-((a[:, None] - b[None])**2).sum(-1) / (2 * kv.lengthscale**2))
    expected = kernel(qz, qz) - kernel(qz, z) @ np.linalg.solve(kernel(z, z) + (gc.ridge + kv.diagnostics['jitter_used']) * np.eye(len(z)), kernel(z, qz))
    np.testing.assert_allclose(kv.cross_covariance(q, q), expected, atol=1e-8)
    kv.save(tmp_path)
    restored = KernelVariance.load(tmp_path, fc, gc)
    np.testing.assert_allclose(kv.score(q), restored.score(q), atol=1e-8)
    assert np.all(kv.score(x) < .01)


def test_ivr_gain_equals_weighted_trace_reduction_and_order_is_stable():
    x, _, _ = fixture()
    kv = KernelVariance(FeatureConfig(components=4), GPConfig(precision="float64")).fit(x[:15])
    candidates, reference = x[15:23], x[23:]
    ids = [f"id_{i}" for i in range(len(candidates))]
    tasks = [str(i % 3) for i in range(len(reference))]
    selected, audit = select_ivr(kv, reference, tasks, candidates, ids, count=5)
    again, _ = select_ivr(kv, reference, tasks, candidates[::-1], ids[::-1], count=5)
    assert selected == again and len(set(selected)) == 5
    from uq_extensions.protocol import task_weights
    selected_x = candidates[[ids.index(id) for id in selected]]
    cr = kv.cross_covariance(reference, selected_x).numpy()
    cc = kv.cross_covariance(selected_x, selected_x).numpy()
    reduction = np.diag(cr @ np.linalg.solve(cc + kv.gp.ridge * np.eye(5), cr.T))
    np.testing.assert_allclose(audit['initial_integrated_variance'] - audit['final_integrated_variance'], task_weights(tasks) @ reduction, atol=1e-7)


def test_projection_training_only_pca_and_pls():
    x, y, _ = fixture()
    output, targets = fit_output_pca(y, 3)
    np.testing.assert_allclose(targets.std(0), 1, atol=1e-5)
    for method in ("pca", "random", "pls"):
        p = fit_projection(x, method=method, components=4, targets=targets)
        before = p.transform(x)
        p.transform(x + 1000)
        np.testing.assert_array_equal(before, p.transform(x))
    with pytest.raises(UninformativeBank):
        fit_projection(np.ones((10, 8)))


@pytest.mark.parametrize('inference', ['exact', 'sparse'])
def test_target_gp_latent_variance_mapping_and_serialization(tmp_path, inference):
    x, y, tasks = fixture()
    fc = FeatureConfig(components=4)
    gc = GPConfig(input_components=4, output_components=2, fit_steps=8, precision='float64',
                  inference=inference, inducing_points=12, sparse_justification='small numerical comparison' if inference == 'sparse' else None)
    model = TargetGP(fc, gc).fit(x, y, tasks=tasks)
    assert model.diagnostics['mapping_check']['performed']
    assert np.isfinite(model.score(x)).all()
    assert (model.score(x) >= 0).all()
    assert model.models[0].length.data_ptr() != model.models[1].length.data_ptr()
    model.save(tmp_path)
    restored = TargetGP.load(tmp_path, fc, gc)
    np.testing.assert_allclose(model.score(x[:4]), restored.score(x[:4]), atol=1e-7)


def test_kvt_rejects_missing_conditioning():
    x, y, tasks = fixture(); x[1] = x[0]
    with pytest.raises(ValueError, match='Repeated inputs'):
        TargetGP(FeatureConfig(), GPConfig()).fit(x, y, tasks=tasks)


def test_sparse_kv_retains_variance_far_from_support():
    x, _, _ = fixture()
    kv = KernelVariance(FeatureConfig(components=4), GPConfig(inference='sparse', inducing_points=12,
                       sparse_justification='numerical fixture')).fit(x)
    assert kv.score(x[:1] + 1000)[0] > .99


def test_float32_matches_float64_and_permuting_targets_does_not_affect_kv():
    x,y,tasks=fixture();f=FeatureConfig(components=4)
    a=KernelVariance(f,GPConfig(precision='float32')).fit(x)
    b=KernelVariance(f,GPConfig(precision='float64')).fit(x)
    np.testing.assert_allclose(a.score(x+.5),b.score(x+.5),atol=5e-5)
    # Fixed KV has no target argument or learned output-dependent kernel.
    assert not hasattr(a,'targets')
    z=a.z.clone();factor=a.factor.clone();a.score(x+100)
    assert torch.equal(z,a.z) and torch.equal(factor,a.factor)
