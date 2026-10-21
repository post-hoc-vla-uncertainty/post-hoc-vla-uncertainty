"""Selected scientific estimators, with caches shared by acquisition and scoring."""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import torch

from uq_extensions.cache_lock import cache_lock
from uq_extensions.common import atomic_json, atomic_npz, file_hash, identity, isolated_rng, model_identity, read_json, stream_seed
from uq_extensions.config import FeatureConfig
from uq_followup.features import cache_bank, load_policy, preprocess_inputs
from uq_followup.estimators import Estimator
from .design import COLLABORATOR_SOURCE


def collaborator_module(project):
    name = 'uq_finalstudy_collaborator_kernel'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(project)/COLLABORATOR_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def reusable_head(declaration, model_hash, support_ids, row):
    """Reuse a fit only for identical weights, demonstrations and complete recipe."""
    followup = Path(declaration['followup'])
    if read_json(followup/'declaration.json')['manifest_hash'] != declaration['manifest_hash']:
        return None
    for record_path in sorted((followup/'policies').glob('*/model.json')):
        record = read_json(record_path)
        if record['model_hash'] != model_hash or set(record['spec']['ids']) != set(support_ids): continue
        directory = followup/'heads'/record_path.parent.name/row['id']
        result = directory/'result.json'
        if result.exists() and read_json(result).get('recipe') == row:
            return directory
    return None


@torch.no_grad()
def pooled_decoder(policy, batch):
    """Collaborator joint_late: ninth Euler evaluation, mean over action tokens.

    Their adapter's return_features=True is reproduced by a pre-projection hook.
    One observation per call fixes Gaussian draws independently of batch position.
    """
    from lerobot.policies.smolvla.smolvla_adapter import SmolVLAAdapter
    from lerobot.policies.common.flow_matching.ode_solver import make_sampling_time_grid
    adapter = SmolVLAAdapter(policy.config, policy.model)
    conditioning = adapter.prepare_conditioning(batch, num_action_samples=1)
    velocity = adapter.make_velocity_fn(conditioning)
    generator = torch.Generator(device=adapter.device).manual_seed(0)
    x = adapter.sample_prior(1, generator=generator)
    grid = make_sampling_time_grid(step_size=adapter.ode_solver_config['step_size'], device=adapter.device, dtype=adapter.dtype)
    for level, (start, stop) in enumerate(zip(grid[:-1], grid[1:]), start=1):
        hidden = []
        handle = policy.model.action_out_proj.register_forward_pre_hook(lambda module, args: hidden.append(args[0].detach())) if level == 9 else None
        try: v = velocity(t=start, x_t=x)
        finally:
            if handle is not None: handle.remove()
        if level == 9:
            if len(hidden) != 1: raise RuntimeError('Missing pooled-decoder feature hook')
            return hidden[0].mean(dim=1).float()
        x = x + (stop-start)*v
    raise RuntimeError('Policy flow solver has fewer than nine evaluations')


def collaborator_features(root, model, view, *, device):
    key = identity(dict(inputs=[(c.id, c.input_hash) for c in view.candidates], recipe='pooled_decoder_level9_seed0_batch1'))
    directory = Path(root)/key
    with cache_lock(directory/'.lock', label='pooled decoder features'):
        destination = directory/'features.npz'
        if destination.exists():
            with np.load(destination) as data: return data['features']
        directory.mkdir(parents=True, exist_ok=True)
        policy, pre, _ = load_policy(model, device)
        from run_progress import Progress
        progress = Progress('collaborator pooled-decoder features', len(view.ids), unit='observations', every=10)
        values = []
        try:
            for i, id_ in enumerate(view.ids):
                part = directory/f'{i:05d}.npz'
                if not part.exists():
                    batch = preprocess_inputs([view.observation(id_)], pre)
                    atomic_npz(part, features=pooled_decoder(policy, batch).cpu().numpy())
                with np.load(part) as data: values.append(data['features'])
                progress.update(i+1)
        finally:
            del policy
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        result = np.concatenate(values)
        atomic_npz(destination, features=result)
        for part in directory.glob('[0-9]*.npz'): part.unlink()
        progress.finish()
        return result


def score_head(root, declaration, head, model, support, queries, *, device, ivr=False, ivr_seed=42):
    """No evaluation outcome is an input to fitting, transforms or acquisition."""
    root = Path(root); row = declaration['recipes'][head]
    mh = model_identity(model)
    fit_spec = dict(model=mh, support=[(c.id, c.input_hash) for c in support.candidates], recipe=row)
    directory = root/'fits'/mh/head/identity(fit_spec)
    features_root = root/'features'/mh
    results = {}; query_features = {}
    with cache_lock(directory/'.lock', label=f'{head} fitting'):
        if head == 'h2':
            cls = collaborator_module(declaration['project']).KernelVariance
            path = directory/'kernel.pt'
            if path.exists(): estimator = cls.load(path, device=device)
            else:
                x = collaborator_features(features_root, model, support, device=device)
                estimator = cls(ridge=.01, projection='whiten', seed=0).fit(torch.as_tensor(x, device=device))
                directory.mkdir(parents=True, exist_ok=True)
                temporary = directory/'kernel.partial.pt'; estimator.save(temporary); temporary.replace(path)
            diagnostics = dict(support_episodes=len(support.ids), components=480,
                lengthscale=estimator.length_scale, ridge=.01, precision='float32', inference='exact',
                whitening='Ledoit-Wolf on standardized inputs', flow_level_one_based=9, noise_seed=0)
        else:
            config = FeatureConfig(**row['features'])
            needed = head in ('h3', 'h4')
            banks = (config.bank, config.target_bank) if needed else (config.bank,)
            reused = reusable_head(declaration, mh, support.ids, row)
            if (directory/'estimator_fit.json').exists() and (directory/'estimator.npz').exists():
                estimator = Estimator.load(row, directory, device)
            elif reused is not None and (reused/'estimator/estimator_fit.json').exists() and (reused/'estimator/estimator.npz').exists():
                estimator = Estimator.load(row, reused/'estimator', device)
                atomic_json(directory/'reuse.json', dict(source=str(reused/'estimator'),
                    checksum=file_hash(reused/'estimator/estimator.npz')))
                print(f'REUSED {head} fitted estimator: {reused}', flush=True)
            else:
                arrays = cache_bank(features_root, model, support, config, banks, role='support', device=device, batch_size=4)
                estimator = Estimator(row, device).fit(arrays[config.bank], arrays.get(config.target_bank), support.tasks, directory)
            diagnostics = estimator.diagnostics
        atomic_json(directory/'identity.json', fit_spec)
        for name, view in queries.items():
            if head == 'h2':
                x = collaborator_features(features_root, model, view, device=device)
                values = estimator.score(torch.as_tensor(x, device=device)).cpu().numpy()
            else:
                x = cache_bank(features_root, model, view, config, (config.bank,), role='query', device=device, batch_size=4)[config.bank]
                values = estimator.score(x)
            if not np.isfinite(values).all(): raise RuntimeError('Nonfinite uncertainty scores')
            results[name] = dict(zip(view.ids, map(float, values)))
            query_features[name] = x
        if ivr:
            if head != 'h1': raise ValueError('Only H1 has a declared IVR arm')
            from uq_extensions.ivr import select_ivr
            # The candidate pool is also the task-balanced integration population.
            pool = queries['pool']; x = query_features['pool']
            ids, audit = select_ivr(estimator.model, x, pool.tasks, x, pool.ids, count=5, seed=ivr_seed)
            results['ivr'] = dict(selected_ids=ids, audit=audit)
    return results, diagnostics


def paper_scores(method, models, view, directory, *, device, seed=42, acquisition=False):
    """Keep calibration's symmetric VFD distinct from SAVE's one-way VFD."""
    from uq_extensions.baselines import baseline_scores
    if method != 'vfd' or acquisition:
        return baseline_scores(method, models, view, directory, device=device, seed=seed)
    # All other baseline methods use the existing pinned adapter unchanged.
    path = Path(directory)/'vfd_symmetric.json'
    spec = dict(method='vfd_symmetric', models=[model_identity(p) for p in models],
        inputs=[(c.id, c.input_hash) for c in view.candidates], seed=seed, samples=5)
    key = identity(spec)
    if path.exists():
        data = read_json(path)
        if data['identity'] != key: raise RuntimeError('VFD score cache changed')
        return data['scores']
    partial = path.with_name('vfd_symmetric_partial.json')
    scores = {}
    if partial.exists():
        data = read_json(partial)
        if data['identity'] != key: raise RuntimeError('VFD partial identity changed')
        scores = data['scores']
    source = Path(os.environ['VLA_PROJECT_ROOT'])/'upstream/uq_vla_extensions/scripts/calibration/calibration_comparison.py'
    module_spec = importlib.util.spec_from_file_location('uq_finalstudy_calibration', source)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    loaded = [load_policy(p, device) for p in models]
    adapters = [module.make_flow_matching_adapter_from_policy(policy=p[0]) for p in loaded]
    sampler = module.build_ensemble_sampler(adapters[0], adapters[1:], 'vfd', 5)
    from run_progress import Progress
    progress = Progress('paper symmetric VFD', len(view.ids), initial=len(scores), unit='observations', every=10)
    for id_ in view.ids:
        if id_ in scores: continue
        with isolated_rng(stream_seed(seed, id_)), torch.no_grad():
            _, value = sampler.conditional_sample_with_uncertainty_batch(observation=preprocess_inputs([view.observation(id_)], loaded[0][1]))
        scores[id_] = float(value[0])
        if not np.isfinite(scores[id_]): raise RuntimeError('Nonfinite symmetric VFD')
        atomic_json(partial, dict(identity=key, scores=scores)); progress.update(len(scores))
    atomic_json(path, dict(identity=key, spec=spec, scores=scores)); progress.finish()
    return scores
