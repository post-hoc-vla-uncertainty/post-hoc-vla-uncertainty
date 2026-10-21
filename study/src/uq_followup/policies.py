"""One-shot policy fitting with explicit episode IDs and rolling checkpoints."""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import sys
import time

from uq_extensions.common import atomic_json, model_identity, read_json
from reproduction_storage import validate_checkpoint


def training_config(project, study, spec, workers=4):
    cfg = deepcopy(read_json(Path(project)/f'configs/reproduction/pretrain_seed_{spec["seed"]}.json'))
    # Keep the full dataset's global frame indexing. The worker installs the
    # explicit episode filter used by both the sampler and training statistics.
    cfg['dataset']['episodes'] = None
    cfg['dataset']['libero_tasks'] = {suite: list(range(10)) for suite in
        ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')}
    cfg['policy'].update(pretrained_path=None, device='cuda',
        scheduler_decay_steps=spec['steps'], scheduler_warmup_steps=spec['warmup_steps'])
    cfg.update(seed=spec.get('training_seed', spec['seed']), steps=spec['steps'], batch_size=spec['batch_size'],
        num_workers=workers, eval_freq=0, enable_val_loss=False,
        save_freq=max(1, round(spec['frames']/spec['batch_size'])),
        output_dir=str(Path(study)/'policies'/spec['id']/'training'),
        job_name='followup_'+spec['id'])
    return cfg


def original_checkpoint(project, spec):
    if spec['condition'] != 'original': return None
    path = Path(project)/f'storage/outputs/pretrain/libero_pretrain_leak3/seed_{spec["seed"]}/checkpoints/last'
    if not path.exists(): return None
    path = path.resolve()
    if validate_checkpoint(path) != 30000: raise RuntimeError(f'Incomplete original policy: {path}')
    cfg = read_json(path/'pretrained_model/train_config.json')
    wanted = read_json(Path(project)/f'configs/reproduction/pretrain_seed_{spec["seed"]}.json')
    for key in ('seed', 'batch_size', 'steps'):
        if cfg[key] != wanted[key]: raise RuntimeError(f'Original checkpoint {key} mismatch')
    for key in ('libero_tasks', 'revision', 'root'):
        if cfg['dataset'][key] != wanted['dataset'][key]: raise RuntimeError(f'Original dataset {key} mismatch')
    return str(path/'pretrained_model')


def committed_checkpoint(output):
    candidates = sorted((Path(output)/'checkpoints').glob('[0-9]'*6), reverse=True)
    for path in candidates:
        try:
            validate_checkpoint(path, resumable=True)
            return path
        except (RuntimeError, ValueError, OSError):
            continue
    return None


def train(project, study, spec, *, workers=4):
    directory = Path(study)/'policies'/spec['id']
    directory.mkdir(parents=True, exist_ok=True)
    cfg = training_config(project, study, spec, workers)
    output = Path(cfg['output_dir'])
    existing = committed_checkpoint(output)
    started = time.monotonic()
    if existing is None or validate_checkpoint(existing) != spec['steps']:
        if existing:
            config_path = existing/'pretrained_model/train_config.json'
            arguments = [f'--config_path={config_path}', '--resume=true']
        else:
            if output.exists():
                # Preserve an interrupted startup rather than erase its evidence.
                output.rename(output.with_name('interrupted_startup_'+str(time.time_ns())))
            config_path = directory/'train_config.json'
            atomic_json(config_path, cfg)
            arguments = [f'--config_path={config_path}']
        from lerobot.scripts import lerobot_train as training
        allowed = list(spec['episode_indices'])
        def explicit_filter(dataset, tasks_to_use):
            available = set(dataset.meta.episodes['episode_index'])
            if not set(allowed) <= available: raise RuntimeError('Requested training episodes are missing')
            print(f'FOLLOWUP DATA: {len(allowed)} unique demos, {spec["frames"]} frames; uniform frame sampling; '
                  f'{spec["effective_epochs"]:.6f} effective epochs; seed={spec.get("training_seed", spec["seed"])}', flush=True)
            return allowed
        # Process-local adapter: no edits to the upstream checkout or live study.
        training.filter_libero_episodes = explicit_filter
        previous_update = training.update_last_checkpoint
        def commit_and_retire(path):
            path = Path(path)
            validate_checkpoint(path, resumable=True)
            previous_update(path)
            if path.parent.resolve() != (output/'checkpoints').resolve():
                raise RuntimeError('Checkpoint retention escaped this training job')
            for older in path.parent.glob('[0-9]'*6):
                if older != path and not older.is_symlink() and older.name < path.name:
                    atomic_json(directory/'retired'/f'{older.name}.json', dict(path=str(older), replaced_by=str(path)))
                    shutil.rmtree(older)
        training.update_last_checkpoint = commit_and_retire
        sys.argv = ['followup-train', *arguments]
        training.main()
        existing = committed_checkpoint(output)
    if existing is None or validate_checkpoint(existing) != spec['steps']:
        raise RuntimeError('Policy did not complete the declared equal-epoch budget')
    model = str(existing/'pretrained_model')
    atomic_json(directory/'model.json', dict(model=model, model_hash=model_identity(model), spec=spec,
                                           seconds=time.monotonic()-started, reused=False))


def register_original(project, study, spec):
    path = original_checkpoint(project, spec)
    if path is None: return False
    atomic_json(Path(study)/'policies'/spec['id']/'model.json',
                dict(model=path, model_hash=model_identity(path), spec=spec, reused=True))
    return True
