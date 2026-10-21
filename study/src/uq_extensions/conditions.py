"""Explicit observation interventions, shared by evaluation and revealed demos.

Pixel recoloring changes appearance without changing collision geometry. Conditions
must identify a verified color region and certify preservation of named attributes;
verification is a development artifact, never inferred from final outcomes.
"""
import copy
from pathlib import Path
import numpy as np
import torch
from .common import identity, read_json

FAMILIES = ('original','undertrained_language','undertrained_appearance','harmless_variation','familiar_diversity')


def validate_condition(condition, *, final=False):
    if condition.get('family') not in FAMILIES or not condition.get('name'):
        raise ValueError('Condition needs a name and declared family')
    if condition['family']=='original': return
    if not condition.get('tasks'): raise ValueError('Controlled condition must declare affected tasks')
    if condition['family']=='undertrained_language' and not condition.get('instructions'):
        raise ValueError('Language condition needs explicit goal-preserving instructions')
    if condition['family']=='undertrained_appearance' and not condition.get('recolor'):
        raise ValueError('Appearance condition needs an explicit recoloring rule')
    if condition.get('instructions') and not set(condition['instructions']).issubset(condition['tasks']):
        raise ValueError('Instruction override outside declared tasks')
    visual = condition.get('recolor')
    if visual:
        for key in ('lower_rgb','upper_rgb','replacement_rgb'):
            a=np.asarray(visual.get(key))
            if a.shape!=(3,) or (a<0).any() or (a>255).any(): raise ValueError(f'Invalid {key}')
        if np.any(np.array(visual['lower_rgb'])>visual['upper_rgb']): raise ValueError('Invalid color interval')
        if visual.get('camera','image') not in ('image','image2','both'): raise ValueError('Invalid camera')
    if final:
        verification=condition.get('development_verification')
        if not verification: raise ValueError('Final condition requires development verification')
        evidence=read_json(verification)
        payload={k:v for k,v in condition.items() if k!='development_verification'}
        if evidence.get('condition_hash')!=identity(payload): raise ValueError('Changed condition after verification')
        for key in ('meaning_preserved','physical_feasibility_preserved','named_attributes_preserved'):
            if evidence.get(key) is not True: raise ValueError(f'Condition lacks verified {key}')
        if not evidence.get('development_evaluation'): raise ValueError('Missing development outcome artifact')
        artifact=read_json(evidence['development_evaluation'])
        if artifact.get('spec',{}).get('split') not in ('development','development_gain_panel'):
            raise ValueError('Condition verification must reference a development evaluation')


def apply_condition(observation, condition, task):
    result={k:np.array(v,copy=True) for k,v in observation.items()}
    if condition.get('tasks') and task not in condition['tasks']: return result
    if task in condition.get('instructions',{}): result['instruction']=np.asarray(condition['instructions'][task])
    visual=condition.get('recolor')
    if visual:
        cameras=('image','image2') if visual.get('camera')=='both' else (visual.get('camera','image'),)
        for key in cameras:
            a=result[key];mask=np.all((a>=visual['lower_rgb'])&(a<=visual['upper_rgb']),axis=-1)
            a[mask]=np.asarray(visual['replacement_rgb'],dtype=np.uint8)
    return result


def transform_training_batch(batch, overlay):
    """Only IDs explicitly revealed in a teaching branch may receive an overlay."""
    result=dict(batch)
    for index, episode in enumerate(batch['episode_index'].reshape(-1).tolist()):
        record=overlay.get(str(int(episode)))
        if not record: continue
        task=record['task']; condition=record['condition']
        if 'task' in result:
            result['task']=list(result['task'])
            result['task'][index]=condition.get('instructions',{}).get(task,result['task'][index])
        visual=condition.get('recolor')
        if visual:
            cameras=('image','image2') if visual.get('camera')=='both' else (visual.get('camera','image'),)
            for camera in cameras:
                key='observation.images.'+camera
                result[key]=result[key].clone();a=result[key][index]
                lo=torch.tensor(visual['lower_rgb'],device=a.device,dtype=a.dtype)[:,None,None]/255
                hi=torch.tensor(visual['upper_rgb'],device=a.device,dtype=a.dtype)[:,None,None]/255
                mask=((a>=lo)&(a<=hi)).all(0)
                color=torch.tensor(visual['replacement_rgb'],device=a.device,dtype=a.dtype)[:,None]/255
                a[:,mask]=color
    return result
