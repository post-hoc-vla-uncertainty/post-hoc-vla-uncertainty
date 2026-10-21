"""Adapters to the pinned authors' scoring implementations."""
from pathlib import Path
import importlib.util
import os
import numpy as np
import torch
from .common import atomic_json, identity, isolated_rng, model_identity, read_json, stream_seed
from .features import load_policy, preprocess_inputs

METHODS = ('vfd', 'gu', 'action_l2', 'ace', 'decu', 'vlm_token_entropy', 'vlm_perplexity')


def baseline_scores(method, models, view, directory, *, device='cpu', seed=42):
    if method not in METHODS: raise ValueError(method)
    needs_ensemble = method in ('vfd', 'gu', 'action_l2', 'decu')
    if needs_ensemble and len(models) < 2: raise ValueError(f'{method} requires a companion checkpoint')
    path = Path(directory) / (method + '.json')
    spec = dict(method=method, models=[model_identity(p) for p in models[:2 if needs_ensemble else 1]],
                inputs=[(c.id, c.input_hash) for c in view.candidates], seed=seed, samples=32 if method=='ace' else 5,
                reference='pinned upstream calibration_comparison.py', batching='stable per-episode RNG')
    if path.exists():
        cached = read_json(path)
        if cached['identity'] != identity(spec): raise RuntimeError('Baseline cache identity mismatch')
        return cached['scores']
    partial=Path(directory)/(method+'_partial.json')
    if partial.exists() and read_json(partial)['identity'] != identity(spec):
        raise RuntimeError('Partial baseline score identity mismatch')
    scores=read_json(partial)['scores'] if partial.exists() else {}
    source = Path(os.environ['VLA_PROJECT_ROOT']) / 'upstream/uq_vla_extensions/scripts/calibration/calibration_comparison.py'
    module_spec = importlib.util.spec_from_file_location('uq_extension_upstream_calibration', source)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    policy, pre, _ = load_policy(models[0], device)
    adapter = module.make_flow_matching_adapter_from_policy(policy=policy)
    companions = []; companion_policies = []
    if needs_ensemble:
        for model in models[1:2]:
            p, _, _ = load_policy(model, device); companion_policies.append(p)
            companions.append(module.make_flow_matching_adapter_from_policy(policy=p))
    if method in ('vfd', 'action_l2'):
        sampler = module.build_ensemble_sampler(adapter, companions, 'vfd_oneway' if method=='vfd' else method, 5)
    elif method == 'gu': sampler = module.build_ensemble_tv_sampler(adapter, companions, 5)
    elif method == 'decu': sampler = module.build_decu_sampler(adapter, companions, 5)
    elif method == 'ace':
        from lerobot.uncertainty.uncertainty_samplers.entropy_sampler import ACESampler
        from lerobot.uncertainty.uncertainty_samplers.configuration_uncertainty_sampler import ACESamplerConfig
        sampler = ACESampler(config=ACESamplerConfig(), model=adapter)
    else: sampler = module.VLMTokenSampler(adapter, method)
    from run_progress import Progress
    progress=Progress(f'paper score {method}',len(view.ids),initial=len(scores),unit='observations',every=10)
    for id in view.ids:
        if id in scores:continue
        with isolated_rng(stream_seed(seed, id)), torch.no_grad():
            batch = preprocess_inputs([view.observation(id)], pre)
            _, value = sampler.conditional_sample_with_uncertainty_batch(observation=batch)
            scores[id] = float(value[0])
        if not np.isfinite(scores[id]): raise RuntimeError(f'Nonfinite {method}: {id}')
        atomic_json(partial,dict(identity=identity(spec),scores=scores))
        progress.update(len(scores))
    atomic_json(path, dict(identity=identity(spec), spec=spec, scores=scores))
    del sampler, adapter, policy, companion_policies, companions
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    progress.finish()
    return scores
