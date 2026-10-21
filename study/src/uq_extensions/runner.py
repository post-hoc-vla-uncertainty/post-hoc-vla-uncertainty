"""Resumable acquisition trajectories; no scientific job starts on import."""
from pathlib import Path
import fcntl
import os
import subprocess
import time
from .common import atomic_json, file_hash, identity, model_identity, plain, read_json, preserve_environment
from .data import load_manifest, candidate_view, verify_dataset
from .protocol import AcquisitionLedger
from .acquisition import acquire
from .training import fit_policies, starting_models, retire_optimizers
from .storage import check_budget


def source_identity():
    root=Path(os.environ['VLA_PROJECT_ROOT'])
    # Presentation and stage orchestration may be refined after the output preview.
    # Numerical training/acquisition/evaluation modules remain part of run identity.
    files=sorted(p for p in (root/'src/uq_extensions').glob('*.py') if p.name not in ('reporting.py','experiments.py','matrix.py','cli.py'))
    files+=sorted((root/'upstream/uq_vla_extensions/src').rglob('*.py'))
    # BEGIN RUN PROGRESS
    from run_progress import scientific_file_hash
    return identity({str(p.relative_to(root)):scientific_file_hash(p) for p in files})
    # END RUN PROGRESS
    return identity({str(p.relative_to(root)):file_hash(p) for p in files})


def verify_bases(config):
    from reproduction_storage import validate_checkpoint
    if len(config.base_models)<config.policy_count: raise ValueError('Provide immutable pretraining models for each required member')
    paths=[]
    for model in config.base_models[:config.policy_count]:
        p=Path(model).resolve()
        step=validate_checkpoint(p.parent)
        if not config.allow_smoke_base and step!=30000: raise ValueError(f'Preparatory model has {step} steps; need completed 30000: {p}')
        paths.append(str(p))
    return paths


def validate_frozen(config):
    if config.phase!='final': return
    recipe=read_json(config.frozen_recipe)
    payload={k:v for k,v in recipe.items() if k!='recipe_hash'}
    if recipe.get('recipe_hash')!=identity(payload): raise ValueError('Frozen recipe checksum mismatch')
    selected=recipe['kv_t_features'] if config.arm in ('kv_t','kv_t_ivr') else recipe['kv_features']
    if selected!=plain(config.features) or recipe.get('kv_t_gp' if config.arm in ('kv_t','kv_t_ivr') else 'kv_gp',recipe.get('gp'))!=plain(config.gp):
        raise ValueError('Final feature/GP settings differ from development freeze')
    if config.training_mode=='pristine_refit' and config.refit_steps!=recipe['refit_steps']:
        raise ValueError('Final pristine budget differs from development freeze')
    from .config import ELIGIBILITY_BY_MODE
    protocols=recipe.get('eligibility_by_mode',ELIGIBILITY_BY_MODE)
    if config.eligibility_protocol!=protocols[config.training_mode]:
        raise ValueError('Final eligibility differs from the declared training-mode protocol')


@preserve_environment('UQ_EXT_RUN_ROOT', 'UQ_EXT_BUDGET_GB')
def run(config, *, gpu_ids, device, evaluation_limit=None):
    from .evaluation import evaluate
    from .reference import reference_curve,evaluate_reference
    config.validate();validate_frozen(config)
    if evaluation_limit is not None and config.phase!='smoke': raise ValueError('Evaluation limits are smoke-only')
    bases=verify_bases(config);manifest=load_manifest(config.data_manifest)
    if config.phase!='smoke': verify_dataset(manifest)
    root=config.run_dir().resolve();root.mkdir(parents=True,exist_ok=True)
    spec=dict(config=config.scientific_dict(),base_models=bases,base_hashes=[model_identity(p) for p in bases],
              manifest_hash=manifest['manifest_hash'], source_hash=source_identity())
    with (root/'run.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This experiment already has a running process')
        request=root/'experiment.json'
        if request.exists() and read_json(request)['identity']!=identity(spec):
            raise RuntimeError('Experiment identity changed; choose a new name/output directory')
        atomic_json(request,dict(identity=identity(spec),**spec))
        os.environ['UQ_EXT_RUN_ROOT']=str(root);os.environ['UQ_EXT_BUDGET_GB']=str(config.budget_gb)
        ledger=AcquisitionLedger(root/'acquisition_ledger.json',manifest_hash=manifest['manifest_hash'],
            protocol=config.eligibility_protocol,pretraining_ids=manifest['pretraining_ids'],candidates=manifest['target_ids'])
        all_inputs=candidate_view(manifest); integration=all_inputs.subset(manifest['integration_ids'])
        previous=bases
        def evaluate_round(model,r):
            directory=root/f'round_{r:03d}/evaluation'
            if reference_curve(config):
                return evaluate_reference(config,model,directory,r,device=device)
            return evaluate(model,manifest,directory,split=config.evaluation_split,
                device=device,max_steps=config.evaluation_max_steps,limit=evaluation_limit,
                batch_size=config.evaluation_batch_size,per_task=config.evaluation_per_task)
        evaluate_round(bases[0],0)
        for r in range(1,config.rounds+1):
            print(f'{config.arm}/{config.training_mode}/seed {config.seed}: round {r}/{config.rounds}',flush=True)
            check_budget(root,config.budget_gb)
            if len(ledger.data['rounds'])<r:
                ids,audit=acquire(config,previous,all_inputs.subset(ledger.support),all_inputs.subset(ledger.eligible()),
                                  integration,root/f'round_{r:03d}/selection',r,device)
                record=ledger.commit(r-1,ids,audit=audit,expected_count=config.batch_queries)
                print(f'Acquisition audit: {len(ids)} query slots; {len(record["prior_pretraining"])} previously in preparatory data; '
                      f'{len(record["prior_acquisition"])} previously acquired; '
                      f'{len(set(ledger.acquired)-ledger.pretraining)} cumulative unique new demonstrations',flush=True)
            # Historical resume must use only the history committed through THIS round.
            acquired=[id for row in ledger.data['rounds'][:r] for id in row['ids']]
            ledger.reveal_indices(r-1,manifest['episode_map'])
            starts=starting_models(config,bases,previous,r)
            models=fit_policies(config,manifest,starts,acquired,run_dir=root,round_index=r,gpu_ids=gpu_ids)
            evaluate_round(models[0],r)
            retire_optimizers(models)
            atomic_json(root/f'round_{r:03d}/complete.json',dict(round=r,models=models,acquired=acquired))
            from .storage import retire_feature_arrays
            retire_feature_arrays(root/f'round_{r:03d}/selection')
            previous=models
        from .reporting import report_run
        report_run(root)
        atomic_json(root/'complete.json',dict(rounds=config.rounds,models=previous))
    return root
