"""Paired fixed simulator starts; outcome files stay outside acquisition inputs."""
from pathlib import Path
import numpy as np
import torch
from .common import array_hash, atomic_json, atomic_npz, identity, isolated_rng, model_identity, read_json, stream_seed
from .features import load_policy, preprocess_inputs
from .protocol import Candidate, CandidateView
from .rollout_metrics import episode_metrics
from .cache_lock import lock_directory


OBSERVATION_PREPROCESSING = 'libero_rgb_rotate180_state_axisangle_v2'
RENDERING_PROTOCOL = 'per_environment_gl_context_v1'


def initial_observation(raw, instruction):
    """Convert simulator observations exactly as upstream's environment processor.

    Dataset images already have this orientation; only simulator inputs pass here.
    The checkpoint policy preprocessor does not include the environment processor.
    """
    from lerobot.processor.env_processor import LiberoProcessorStep
    state = raw['robot_state']
    observation = {f'observation.images.{key}': torch.as_tensor(np.ascontiguousarray(raw['pixels'][key])).permute(2, 0, 1)[None]
                   for key in ('image', 'image2')}
    observation['observation.robot_state'] = {
        'eef': {key: torch.as_tensor(state['eef'][key], dtype=torch.float32)[None] for key in ('pos', 'quat')},
        'gripper': {'qpos': torch.as_tensor(state['gripper']['qpos'], dtype=torch.float32)[None]},
    }
    processed = LiberoProcessorStep().observation(observation)
    return dict(**{key: processed[f'observation.images.{key}'][0].permute(1, 2, 0).contiguous().numpy()
                  for key in ('image', 'image2')},
                state=processed['observation.state'][0].numpy(),
                instruction=np.asarray(instruction))


def evaluation_view(directory):
    return CandidateView([Candidate(**r) for r in read_json(Path(directory)/'initial_inputs.json')])


@lock_directory('directory','.evaluation.lock',label='policy evaluation')
def evaluate(checkpoint, manifest, directory, *, split='development', device='cpu', max_steps=520,
             condition=None, limit=None, batch_size=15, per_task=None):
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_task_instruction
    from .conditions import apply_condition, validate_condition
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    condition = condition or {'name': 'original', 'family': 'original'}
    validate_condition(condition, final=split.startswith('final'))
    starts = manifest['evaluation_splits'][split]
    if condition.get('tasks'): starts = [s for s in starts if s['task'] in condition['tasks']]
    if per_task is not None:
        if split.startswith('final') or per_task<1:raise ValueError('Balanced limits are development/preview only')
        counts={};balanced=[]
        for s in starts:
            if counts.get(s['task'],0)<per_task:
                balanced.append(s);counts[s['task']]=counts.get(s['task'],0)+1
        starts=balanced
    if limit is not None: starts = starts[:limit]
    spec = dict(checkpoint=model_identity(checkpoint), manifest=manifest['manifest_hash'], starts=starts,
                split=split, max_steps=max_steps, condition=condition, noise_rule='explicit per-ID Gaussian chunk noise v2', reset_order='reset then apply state then settle',
                observation_preprocessing=OBSERVATION_PREPROCESSING, rendering_protocol=RENDERING_PROTOCOL)
    key = identity(spec); result_path = directory / 'evaluation.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['identity'] != key: raise RuntimeError('Evaluation cache identity mismatch')
        return result['episodes']
    if (directory/'request.json').exists() and read_json(directory/'request.json')['identity'] != key:
        raise RuntimeError('Cannot resume evaluation with changed settings')
    atomic_json(directory/'request.json', dict(identity=key, **spec))
    policy, pre, post = load_policy(checkpoint, device)
    normalizers = read_json(Path(manifest['evaluator_directory'])/'normalizers.json')['tasks']
    suite = benchmark.get_benchmark_dict()['libero_10']()
    records = []; inputs = []
    pending=[]
    for index,start in enumerate(starts):
        episode_path=directory/'episodes'/f'{index:05d}.json'
        if episode_path.exists():
            record=read_json(episode_path)
            if record['evaluation_identity']!=key: raise RuntimeError('Episode identity mismatch')
            records.append(record);inputs.append(record['initial_input'])
        else:pending.append((index,start))
    if policy.config.n_obs_steps!=1: raise ValueError('Batched paired evaluation requires the declared single observation policy')
    import time
    started=time.perf_counter()
    # BEGIN RUN PROGRESS
    from run_progress import Progress
    progress=Progress(f'evaluation {split} {condition["name"]}',len(starts),initial=len(records),unit='episodes',every=batch_size)
    print(f'Evaluation checkpoint={checkpoint}; horizon={max_steps}; batch={batch_size}',flush=True)
    # END RUN PROGRESS
    try:
        for offset in range(0,len(pending),batch_size):
            active=[];envs=[]
            try:
                for index,start in pending[offset:offset+batch_size]:
                    task_id=int(start['task'].split('/')[1])
                    env=LiberoEnv(suite,task_id,'libero_10',obs_type='pixels_agent_pos',
                                  episode_index=start['init_state_id'],episode_length=max_steps)
                    envs.append(env)
                    if array_hash(np.asarray(env._init_states[start['init_state_id']]))!=start['init_state_hash']:
                        raise RuntimeError('Simulator initial-state checksum changed')
                    raw,_=env.reset(seed=start['reset_seed'],options={'init_state_id':start['init_state_id']})
                    instruction=get_task_instruction('libero_10',task_id)
                    obs=apply_condition(initial_observation(raw,instruction),condition,start['task'])
                    path=directory/'initial_inputs'/f'{index:05d}.npz';atomic_npz(path,**obs)
                    candidate=dict(id=start['id'],episode_index=index,task=start['task'],observation_file=str(path),
                                   input_hash=identity({k:array_hash(v) for k,v in obs.items()}))
                    active.append(dict(index=index,start=start,env=env,instruction=instruction,obs=obs,
                                       candidate=candidate,positions=[np.asarray(raw['robot_state']['eef']['pos']).copy()]))
                for step in range(max_steps):
                    if step%policy.config.n_action_steps==0:
                        batch=preprocess_inputs([a['obs'] for a in active],pre)
                        noise=[]
                        for a in active:
                            generator=torch.Generator(device=device).manual_seed(stream_seed(100,a['start']['id'],'policy_noise',step))
                            noise.append(torch.randn((1,policy.config.chunk_size,policy.config.max_action_dim),device=device,generator=generator))
                        policy.reset()
                        with torch.no_grad(): chunks=post(policy.predict_action_chunk(batch,noise=torch.cat(noise)) ).cpu().numpy()
                        for a,chunk in zip(active,chunks):a['actions']=chunk
                    remaining=[]
                    for a in active:
                        raw,_,terminated,truncated,info=a['env'].step(a['actions'][step%policy.config.n_action_steps])
                        a['positions'].append(np.asarray(raw['robot_state']['eef']['pos']).copy())
                        success=bool(info.get('is_success',False))
                        if success or terminated or truncated or step+1==max_steps:
                            hz=float(a['env']._env.env.control_freq)
                            record={**a['start'],'condition':condition['name'],'condition_hash':identity(condition),'split':split,
                                    'evaluation_identity':key,'max_steps':max_steps,'control_hz':hz,
                                    'action_steps':policy.config.n_action_steps,'flow_steps':policy.config.num_steps,
                                    'observation_spec':'workspace+wrist256/state8/instruction','initial_input':a['candidate'],
                                    'observation_preprocessing':OBSERVATION_PREPROCESSING,
                                    'rendering_protocol':RENDERING_PROTOCOL,
                                    **episode_metrics(success=success,steps=step+1,positions=a['positions'],control_hz=hz,
                                                      normalizer=normalizers.get(a['start']['task']))}
                            atomic_json(directory/'episodes'/f"{a['index']:05d}.json",record)
                            records.append(record);inputs.append(a['candidate'])
                            print(f"Evaluated {len(records)}/{len(starts)} {a['start']['id']} success={success}",flush=True)
                        else:
                            a['obs']=apply_condition(initial_observation(raw,a['instruction']),condition,a['start']['task']);remaining.append(a)
                    # BEGIN RUN PROGRESS
                    progress.update(len(records),f'success so far={sum(bool(r["success"]) for r in records)}/{len(records)}')
                    # END RUN PROGRESS
                    active=remaining
                    if not active:break
            finally:
                for env in envs:env.close()
    finally:
        del policy
        if torch.cuda.is_available():torch.cuda.empty_cache()
    records.sort(key=lambda r:r['id']);inputs.sort(key=lambda r:r['id'])
    atomic_json(directory/'timing.json',dict(seconds=time.perf_counter()-started,resumed_episodes=len(starts)-len(pending),batch_size=batch_size))
    atomic_json(directory/'initial_inputs.json', inputs)
    import pandas as pd
    pd.DataFrame([{k:v for k,v in r.items() if k!='initial_input'} for r in records]).to_parquet(directory/'episode_metrics.parquet',index=False)
    atomic_json(result_path, dict(identity=key, spec=spec, episodes=records))
    # BEGIN RUN PROGRESS
    rates=pd.DataFrame(records).groupby('task').success.mean()
    progress.finish(f'task-macro success={rates.mean():.1%}; per-task={rates.to_dict()}; output={directory}')
    # END RUN PROGRESS
    return records
