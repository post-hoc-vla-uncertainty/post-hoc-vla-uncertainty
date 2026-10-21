"""Round estimators and selection consume input-only views, never trajectories."""
from pathlib import Path
import os
import time
import numpy as np
from .common import atomic_json, atomic_npz, identity, model_identity, read_json, stream_seed
from .features import cache_features
from .cache_lock import cache_lock
from .estimators import make_estimator, PatchEstimator, probe_weights
from .config import is_target_arm, feature_uses_flow
from .protocol import select_save
from .ivr import select_ivr


def fit_estimator(config, checkpoint, support, directory, device):
    import resource,torch
    started=time.perf_counter()
    if str(device).startswith('cuda') and not os.environ.get('UQ_EXT_STUDY_MEMORY_GUARD'):
        torch.cuda.reset_peak_memory_stats(device)
    directory = Path(directory)
    kind = 'kv_t' if is_target_arm(config.arm) else 'kv'
    spec = dict(method=kind, checkpoint=model_identity(checkpoint), support=support.ids,
                inputs=[c.input_hash for c in support.candidates], features=config.features, gp=config.gp,
                code=identity({name:Path(__file__).with_name(name+'.py').read_text() for name in
                    ('features','estimators','kernel_variance','target_gp','batched_gp','preprocessing','coregionalized')}))
    # BEGIN RUN PROGRESS
    from run_progress import Progress,strip_reporting_blocks
    spec['code']=identity({name:strip_reporting_blocks(Path(__file__).with_name(name+'.py').read_text()) for name in
        ('features','estimators','kernel_variance','target_gp','batched_gp','preprocessing','coregionalized')})
    # END RUN PROGRESS
    if config.features.bank == 'visual_patches':
        from .common import plain
        spec['features']=plain(config.features)
        for field in ('patch_aggregation','patch_top_fraction','camera_aggregation'):
            spec['features'].pop(field,None)
    key = identity(spec)
    if os.environ.get('UQ_EXT_ESTIMATOR_CACHE'):
        cache_root=Path(os.environ['UQ_EXT_ESTIMATOR_CACHE'])
        shared=cache_root/key
        repair_path=cache_root.parent/'numerical_repair.json'
        recovery={}
        from .study_compatibility import legacy_estimator_codes
        # An empty directory/feature link from an interrupted startup must not
        # hide an older completed fit or its optimizer checkpoints.
        current_fit=(shared/'identity.json').exists() or any((shared/'optimizer').rglob('*.pt'))
        if not current_fit:
            for code in legacy_estimator_codes(spec):
                previous=dict(spec,code=code)
                previous_key=identity(previous);previous_path=cache_root/previous_key
                if previous_path.exists():
                    recovery=dict(compatible_extension='whitening_v1',execution_code=spec['code'])
                    spec,key,shared=previous,previous_key,previous_path
                    print(f'Reusing unchanged original-head estimator/optimizer: {shared}',flush=True)
                    break
        if repair_path.exists():
            repair=read_json(repair_path)
            if (repair.get('kind')=='kvt_float64_posterior_retry_v1'
                    and repair.get('estimator_code_after')==spec['code']):
                previous=dict(spec,code=repair['estimator_code_before'])
                previous_key=identity(previous)
                previous_path=cache_root/previous_key
                if previous_path.exists() and not shared.exists():
                    recovery=dict(numerical_repair=str(repair_path.resolve()),execution_code=spec['code'])
                    spec,key,shared=previous,previous_key,previous_path
                    print(f'Reusing unchanged GP fit/optimizer from before numerical repair: {shared}',flush=True)
        atomic_json(directory/'cache_link.json',dict(identity=key,path=str(shared.resolve()),**recovery))
        directory=shared
    with cache_lock(directory/'.estimator.lock',label='fitted estimator'):
        if (directory / 'identity.json').exists():
            if read_json(directory / 'identity.json')['identity'] != key:
                raise RuntimeError('Estimator identity changed')
            return make_estimator(config, device=device, path=directory)
        needs_targets = kind == 'kv_t' or config.features.compression.startswith('pls')
        banks = tuple(dict.fromkeys([config.features.bank] + ([config.features.target_bank] if needs_targets else [])))
        values = cache_features(checkpoint, support, config.features, directory / 'support_features',
                                device=device, batch_size=config.feature_batch_size, banks=banks,
                                role='support', support_ids=support.ids)
        # BEGIN RUN PROGRESS
        progress=Progress(f'{kind} fit: {config.features.bank} -> {config.features.target_bank}')
        print(f'GP support={len(support.ids)} episodes; inference={config.gp.inference}; outputs={config.gp.output_dependence}; '
              f'precision={config.gp.precision}; projection={config.gp.input_projection if kind=="kv_t" else config.features.compression}',flush=True)
        # END RUN PROGRESS
        model = make_estimator(config, device=device)
        if kind == 'kv_t':
            model.model.training_directory = directory / 'optimizer'
            model.fit(values[config.features.bank], values[config.features.target_bank], tasks=support.tasks)
        elif needs_targets:
            model.fit(values[config.features.bank], values[config.features.target_bank], tasks=support.tasks)
        else:
            model.fit(values[config.features.bank])
        model.diagnostics.update(total_fit_and_feature_seconds=time.perf_counter()-started,
            process_peak_host_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if str(device).startswith('cuda') else 0)
        model.save(directory)
        atomic_json(directory / 'identity.json', dict(identity=key, **spec))
        # BEGIN RUN PROGRESS
        mapping=model.diagnostics.get('mapping_check',{})
        progress.finish(f'mapping_usable={mapping.get("usable","not applicable")}; held-out GP R2={mapping.get("gp_r2")}; artifact={directory}')
        # END RUN PROGRESS
        return model


def variance_scores(config, checkpoint, support, query, directory, device, role='candidate'):
    directory = Path(directory)
    estimator = fit_estimator(config, checkpoint, support, directory / 'estimator', device)
    x = cache_features(checkpoint, query, config.features, directory / (role + '_features'), device=device,
                        batch_size=config.feature_batch_size, banks=(config.features.bank,),
                        role=role, support_ids=support.ids)[config.features.bank]
    if isinstance(estimator, PatchEstimator):
        patches=estimator.patch_scores(x)
        atomic_npz(directory/(role+'_patch_scores.npz'), ids=np.asarray(query.ids), variances=patches)
        return dict(zip(query.ids,estimator.aggregate(patches).astype(float))),estimator,x
    if x.ndim==3:
        values=estimator.model.score(x.reshape(-1,x.shape[-1])).reshape(x.shape[:2])
        atomic_npz(directory/(role+'_flow_scores.npz'),ids=np.asarray(query.ids),latent_variances=values,
            flow_indices=np.asarray(config.features.flow_indices),weights=probe_weights(x.shape[1],config.features))
        return dict(zip(query.ids,values@probe_weights(x.shape[1],config.features))),estimator,x
    return dict(zip(query.ids, estimator.score(x).astype(float))), estimator, x


def acquire(config, models, support, candidates, integration, directory, round_index, device):
    started = time.perf_counter(); directory = Path(directory)
    # BEGIN RUN PROGRESS
    from run_progress import Progress
    progress=Progress(f'acquisition {config.arm} round {round_index}')
    # END RUN PROGRESS
    seed = stream_seed(config.seed, 'acquisition', round_index)
    if config.arm == 'random':
        scores = dict.fromkeys(candidates.ids, 0.)
    elif config.arm in ('vfd', 'gu'):
        from .baselines import baseline_scores
        scores = baseline_scores(config.arm, models, candidates, directory / 'baseline', device=device, seed=seed)
    else:
        scores, estimator, cx = variance_scores(config, models[0], support, candidates, directory, device)
    if config.arm in ('kv_ivr','kv_t_ivr'):
        rx = cache_features(models[0], integration, config.features, directory / 'integration_features', device=device,
                            batch_size=config.feature_batch_size, banks=(config.features.bank,), role='integration',
                            support_ids=support.ids)[config.features.bank]
        if feature_uses_flow(config.features):
            cx=cache_features(models[0],candidates,config.features,directory/'hypothetical_features',device=device,
                batch_size=config.feature_batch_size,banks=(config.features.bank,),role='hypothetical_support',support_ids=support.ids)[config.features.bank]
        ids, audit = select_ivr(estimator, rx, integration.tasks, cx, candidates.ids, count=config.batch_queries, seed=seed)
    else:
        ids, audit = select_save(candidates, scores, count=config.batch_queries,
                                 temperature=0 if config.arm == 'random' else 1 if config.arm == 'gu' else 2.5,
                                 seed=seed, random_within_task=config.arm == 'random')
    audit.update(seconds=time.perf_counter()-started, support_ids=support.ids,
                 eligible_ids=candidates.ids, scores=scores, selected_ids=ids, arm=config.arm)
    atomic_json(directory / 'acquisition.json', audit)
    # BEGIN RUN PROGRESS
    progress.finish(f'selected {len(ids)} of {len(candidates.ids)} candidates: {ids}')
    # END RUN PROGRESS
    return ids, audit
