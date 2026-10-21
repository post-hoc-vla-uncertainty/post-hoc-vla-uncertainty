"""Reuse committed reproduction evidence and evaluate new arms with its recipe.

Imports only small metadata into the extension workspace. Model references stay
in reproduction storage; retired weights are explicitly unavailable for panels.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import atomic_json, file_hash, identity, model_identity, plain, read_json
from .protocol import AcquisitionLedger

EVALUATION_PROTOCOL = 'upstream_30x10_seed100_batch15'


def reference_curve(config):
    return config.phase == 'final' and config.eligibility_protocol == 'upstream_reference'


def converted_evaluation(data):
    """Preserve success evidence without inventing unrecorded movement metrics."""
    episodes=[]
    for task in data['per_task']:
        name=f"{task['task_group']}/{task['task_id']}"
        for index,row in enumerate(task['per_rollout_results']):
            episodes.append(dict(id=f'{name}/reference_{index:03d}',task=name,
                success=int(row['success']),steps=row['steps'],
                q_time=None,q_path=None,relative_time=None,relative_path=None))
    return dict(evaluation_protocol=EVALUATION_PROTOCOL,episodes=episodes,
                source_identity=identity(data),quality_metrics_available=False)


def publish_evaluation(data,directory):
    directory=Path(directory)
    atomic_json(directory/'evaluation.json',converted_evaluation(data))
    atomic_json(directory/'timing.json',dict(seconds=data['timings']['total']['duration_s']))


def evaluate_reference(config,model,directory,round_index,*,device):
    """Use the original evaluator, including its initial states and policy RNG."""
    from reproduction_rounds import validate_evaluation
    from run_progress import LogFollower,tail_text
    from lerobot.envs.libero import get_task_instruction
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    result=directory/'reference_evaluation.json'
    spec=dict(checkpoint=model_identity(model),protocol=EVALUATION_PROTOCOL,round_index=round_index-1)
    spec_path=directory/'reference_identity.json'
    if spec_path.exists() and read_json(spec_path)!=spec:
        raise RuntimeError('Reference evaluation identity changed')
    if result.exists():
        if not spec_path.exists():raise RuntimeError('Reference evaluation has no identity')
        validate_evaluation(result,round_index-1)
        publish_evaluation(read_json(result),directory);return
    atomic_json(spec_path,spec)
    request=dict(round_index=round_index-1,member_index=0,model_path=str(Path(model).resolve()),
        policy_type='smolvla',task_specs=[dict(task_group='libero_10',task_id=i,
            task_instruction=get_task_instruction('libero_10',i)) for i in range(10)],
        n_rollouts=30,eval_batch_size=15,seed_start=100,seed_step=1,max_steps=520,
        device=device,use_async_envs=False,env_type='libero',env_file=None,result_path=str(result.resolve()))
    request_path=directory/'reference_request.json';atomic_json(request_path,request)
    log_path=directory/'reference_evaluation.log';follower=LogFollower([log_path],label=f'reference evaluation round {round_index}')
    env=dict(os.environ,PYTHONUNBUFFERED='1')
    with log_path.open('a') as log:
        process=subprocess.Popen([sys.executable,'-m','iterative_fine_tuning.eval_worker',
            '--request_path',str(request_path.resolve())],stdout=log,stderr=subprocess.STDOUT,env=env)
    try:
        while process.poll() is None:
            follower.poll();time.sleep(1)
        follower.poll(heartbeat=False,final=True)
        if process.returncode:raise RuntimeError(f'Reference evaluation failed: {log_path}\n{tail_text(log_path)}')
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    validate_evaluation(result,round_index-1);publish_evaluation(read_json(result),directory)


def validate_source(config,source,manifest,*,through_round=None):
    """Validate numerical recipe and committed all-round evidence before reuse."""
    from reproduction_rounds import validate_evaluation
    from reproduction_storage import validate_checkpoint
    if config.training_mode!='save_warm_50_50' or config.eligibility_protocol!='upstream_reference' or config.arm not in ('random','vfd'):
        raise ValueError('Only original-eligibility SAVE Random/VFD reproduction can be reused')
    if config.policy_count!=2:raise ValueError('Reproduction reuse includes two members; Random needs its scoring companion enabled')
    source=Path(source).resolve();path=source/'iterative_fine_tuning_config.json'
    cfg=read_json(path)
    expected=read_json(Path(os.environ['VLA_PROJECT_ROOT'])/f'configs/reproduction/{config.arm}.json')
    for section in ('dataset','replay','iteration'):
        if any(cfg[section].get(k)!=v for k,v in expected[section].items()):
            raise ValueError(f'Reproduction {section} differs from the pinned recipe')
    for key in ('steps','batch_size','reinitialize_selected_layers','use_policy_training_preset'):
        if cfg[key]!=expected[key]:raise ValueError(f'Reproduction {key} differs from the pinned recipe')
    if cfg['seed']!=config.seed or config.steps!=4000 or config.train_batch_size!=32 or config.rounds!=15 or config.batch_queries!=5:
        raise ValueError('Reproduction seed or training budget mismatch')
    for key in ('type','optimizer_lr','scheduler_warmup_steps','scheduler_decay_steps','scheduler_decay_lr'):
        if cfg['policy'][key]!=expected['policy'][key]:raise ValueError(f'Reproduction policy {key} mismatch')
    for key in ('strategy','scoring_metric','num_action_samples','task_selection_temperature','task_weighted_episode_selection',
                'candidate_tasks','max_candidates_per_task','max_selected_per_task'):
        if cfg['selection'][key]!=expected['selection'][key]:raise ValueError(f'Reproduction selection {key} mismatch')
    if manifest['source_revision']!=cfg['dataset']['revision'] or Path(manifest['dataset_root']).resolve()!=Path(cfg['dataset']['root']).resolve():
        raise ValueError('Reproduction dataset differs from extension manifest')
    state=read_json(source/'run_state.json');all_rows=state['rounds']
    required=15 if through_round is None else int(through_round)
    if not 0 <= required <= 15:raise ValueError('Requested reproduction round must be between 0 and 15')
    rows=all_rows if through_round is None else all_rows[:required]
    if [r['round_index'] for r in rows]!=list(range(required)):
        raise ValueError(f'Reproduction is not complete: {source}. Finish the existing run; no duplicate training is started.')
    bases=[str(Path(p).resolve()) for p in all_rows[0]['start_model_paths']]
    if bases!=[str(Path(p).resolve()) for p in config.base_models[:2]]:
        raise ValueError('Reproduction starting policies differ from configured bases')
    for base in bases:
        if validate_checkpoint(Path(base).parent)!=30000:raise ValueError('Incomplete preparatory policy')
    evidence={str(path):file_hash(path)};models={0:bases};available=[0]
    mapping={int(v):k for k,v in manifest['episode_map'].items()}
    pretraining={int(manifest['episode_map'][id]) for id in manifest['pretraining_ids']}
    initial=source/'initial_evaluation/member_00/member_evaluation.json'
    validate_evaluation(initial,-1);evidence[str(initial)]=file_hash(initial)
    if str(Path(read_json(initial)['model_path']).resolve())!=bases[0]:raise ValueError('Initial evaluation policy mismatch')
    for row in rows:
        i=row['round_index'];directory=source/f'round_{i:03d}'
        commit=read_json(directory/'reproduction_commit.json')
        if commit['round_index']!=i:raise ValueError('Reproduction commit round mismatch')
        for relative,digest in commit['evidence'].items():
            p=directory/relative
            if file_hash(p)!=digest:raise RuntimeError(f'Reproduction evidence changed: {p}')
            evidence[str(p)]=digest
        for required in ('selection_manifest.json','training_manifest.json','training/member_00/member_evaluation.json'):
            if required not in commit['evidence']:raise ValueError('Incomplete reproduction commit')
        train=read_json(directory/'training_manifest.json');selection=read_json(directory/'selection_manifest.json')
        ids=row['selected_episode_ids']
        if len(ids)!=5 or len(set(ids))!=5 or not set(ids)<=mapping.keys():raise ValueError('Invalid reference acquisition batch')
        if ids!=[r['episode_id'] for r in selection['selected_episodes']]:raise ValueError('Reference selection history changed')
        if set(train['history_episode_ids'])!=pretraining:raise ValueError('Reference preparatory history mismatch')
        if row['final_model_paths']!=train['final_model_paths']:raise ValueError('Reference models mismatch')
        validate_evaluation(directory/'training/member_00/member_evaluation.json',i)
        evaluated=read_json(directory/'training/member_00/member_evaluation.json')['model_path']
        if Path(evaluated).resolve()!=Path(train['final_model_paths'][0]).resolve():raise ValueError('Evaluation policy mismatch')
        retained=[]
        for model in train['final_model_paths']:
            p=Path(model).resolve();retention=p.parent/'retention.json'
            if retention.exists() and read_json(retention).get('weights_retired'):
                retained=[];break
            if validate_checkpoint(p.parent)!=4000:raise ValueError('Incomplete reference policy')
            retained.append(str(p))
        if retained:models[i+1]=retained;available.append(i+1)
    return dict(source=source,rows=rows,bases=bases,models=models,available=available,evidence=evidence,mapping=mapping)


def import_reproduction(config,source,manifest,*,through_round=None):
    checked=validate_source(config,source,manifest,through_round=through_round);root=config.run_dir().resolve();source=checked['source']
    spec=dict(source_run=str(source),evidence=checked['evidence'],config=plain(config),manifest_hash=manifest['manifest_hash'])
    key=identity(spec);reuse_path=root/'reuse.json'
    if reuse_path.exists() and read_json(reuse_path)['identity']!=key:
        raise RuntimeError('Imported reproduction identity changed; use a new output directory')
    if (root/'experiment.json').exists() and not reuse_path.exists():raise RuntimeError('Refusing to overwrite a trained extension run')
    atomic_json(reuse_path,dict(identity=key,**spec,available_rounds=checked['available'],
        unavailable_rounds=sorted(set(range(16))-set(checked['available'])),evaluation_protocol=EVALUATION_PROTOCOL,
        copied_model_bytes=0,double_count_as_new_seed=False))
    atomic_json(root/'experiment.json',dict(config=plain(config),base_models=checked['bases'],
        manifest_hash=manifest['manifest_hash'],source_run=str(source),imported_reproduction=True))
    ledger=AcquisitionLedger(root/'acquisition_ledger.json',manifest_hash=manifest['manifest_hash'],
        protocol='upstream_reference',pretraining_ids=manifest['pretraining_ids'],candidates=manifest['target_ids'])
    initial=read_json(source/'initial_evaluation/member_00/member_evaluation.json')
    publish_evaluation(initial,root/'round_000/evaluation')
    for row in checked['rows']:
        i=row['round_index'];r=i+1;src=source/f'round_{i:03d}';out=root/f'round_{r:03d}'
        ids=[checked['mapping'][j] for j in row['selected_episode_ids']]
        ledger.commit(i,ids,audit=dict(source_manifest=str(src/'selection_manifest.json')))
        atomic_json(out/'selection/acquisition.json',dict(selected_ids=ids,source_manifest=str(src/'selection_manifest.json'),
            seconds=row['timings']['selection']['duration_s'],scores={}))
        models=checked['models'].get(r,[])
        atomic_json(out/'policy_fit.json',dict(models=models,weights_available=bool(models),
            source_manifest=str(src/'training_manifest.json'),seconds=row['timings']['training']['duration_s'],
            requested_steps_per_member=4000,policy_count=2))
        atomic_json(out/'complete.json',dict(round=r,models=models,acquired=ledger.acquired))
        publish_evaluation(read_json(src/'training/member_00/member_evaluation.json'),out/'evaluation')
    imported_rounds=len(checked['rows'])
    atomic_json(root/'complete.json',dict(rounds=imported_rounds,models=checked['models'].get(imported_rounds,[]),imported=True))
    print(f'Reused {config.arm} seed {config.seed}: {imported_rounds} completed rounds; retained policy rounds {checked["available"]}; no policy training.',flush=True)
    return root


def available_panel_rounds(run_dir,requested):
    path=Path(run_dir)/'reuse.json'
    if not path.exists():return list(requested)
    available=set(read_json(path)['available_rounds'])
    return [r for r in requested if r in available]
