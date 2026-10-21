"""One fixed same-architecture supervisor trained on the full permitted pool."""
from pathlib import Path
import os
import subprocess
import sys
from .common import atomic_json,identity,model_identity,read_json,preserve_environment
from .storage import recover_checkpoint,finalize_model


@preserve_environment('UQ_EXT_RUN_ROOT', 'UQ_EXT_BUDGET_GB')
def train_supervisor(directory,manifest,*,steps=30000,seed=991,device='cuda:0',budget_gb=15,smoke=False):
    if steps!=30000 and not smoke:raise ValueError('Supervisor uses 30000 preparatory updates; other budgets must be smoke')
    if not smoke:
        from .data import verify_dataset
        verify_dataset(manifest)
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    # BEGIN RUN PROGRESS
    from run_progress import Progress
    progress=Progress('full-pool supervisor')
    # END RUN PROGRESS
    source=Path(os.environ['VLA_PROJECT_ROOT'])/'configs/reproduction/pretrain_seed_0.json'
    training_directory=directory/'training'
    cfg=read_json(source);cfg['output_dir']=str(training_directory);cfg['seed']=seed
    cfg['dataset']['libero_tasks']={k:list(range(10)) for k in ('libero_spatial','libero_object','libero_goal','libero_10')}
    cfg['dataset']['root']=manifest['dataset_root'];cfg['steps']=steps;cfg['save_freq']=min(1000,steps);cfg['eval_freq']=0
    cfg['policy']['device']=device
    if smoke:cfg['batch_size']=1;cfg['num_workers']=1
    spec=dict(config=cfg,manifest=manifest['manifest_hash'],role='fixed_full_pool_supervisor',smoke=smoke)
    request=directory/'supervisor_request.json'
    if request.exists() and read_json(request)['identity']!=identity(spec):raise ValueError('Supervisor recipe changed')
    atomic_json(request,dict(identity=identity(spec),**spec))
    cfgpath=directory/'config.json';atomic_json(cfgpath,cfg)
    # BEGIN RUN PROGRESS
    os.environ['PYTHONUNBUFFERED']='1'
    # END RUN PROGRESS
    env=os.environ.copy();env['UQ_EXT_RUN_ROOT']=str(directory);env['UQ_EXT_BUDGET_GB']=str(budget_gb)
    os.environ.update({k:env[k] for k in ('UQ_EXT_RUN_ROOT','UQ_EXT_BUDGET_GB')})
    checkpoint=recover_checkpoint(training_directory/'checkpoints')
    from reproduction_storage import validate_checkpoint
    if checkpoint and validate_checkpoint(checkpoint)==steps:model=checkpoint/'pretrained_model'
    else:
        if checkpoint:
            args=[f'--config_path={checkpoint}/pretrained_model/train_config.json','--resume=true']
        else:
            if training_directory.exists():
                raise RuntimeError('Supervisor stopped before its first checkpoint; use a new output directory')
            args=[f'--config_path={cfgpath}']
        subprocess.run([sys.executable,'-m','lerobot.scripts.lerobot_train',*args],env=env,check=True)
        checkpoint=recover_checkpoint(training_directory/'checkpoints');model=checkpoint/'pretrained_model'
        if validate_checkpoint(checkpoint)!=steps:raise RuntimeError('Incomplete supervisor training')
    finalize_model(model)
    atomic_json(directory/'supervisor.json',dict(model=str(model),model_hash=model_identity(model),**spec))
    # BEGIN RUN PROGRESS
    progress.finish(f'{steps} updates; checkpoint={model}')
    # END RUN PROGRESS
    return str(model)
