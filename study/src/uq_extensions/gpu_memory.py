"""Per-process GPU budget and measurable worst-shape GP checks."""
import argparse
from dataclasses import replace
from pathlib import Path
import time

from .common import atomic_json

GIB = 1024**3


def configure_memory(device, limit_gib=24., reserve_gib=2.):
    import torch
    if not str(device).startswith('cuda'):
        return dict(device=str(device), measured=False, reason='CPU')
    total=torch.cuda.get_device_properties(device).total_memory
    allocation_limit=min(total,limit_gib*GIB)-reserve_gib*GIB
    if allocation_limit<=0:raise ValueError('GPU memory reserve exceeds the limit')
    torch.cuda.set_per_process_memory_fraction(allocation_limit/total,device)
    torch.cuda.reset_peak_memory_stats(device)
    return dict(device=torch.cuda.get_device_name(device), total_bytes=total,
        requested_limit_bytes=int(limit_gib*GIB), allocator_limit_bytes=int(allocation_limit),
        non_pytorch_reserve_bytes=int(reserve_gib*GIB))


def memory_record(device, budget):
    import torch
    if not str(device).startswith('cuda'):return budget
    record=dict(**budget, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    if record['peak_reserved_bytes'] > budget['allocator_limit_bytes']:
        raise RuntimeError('GPU allocator exceeded the configured memory budget')
    record['within_budget']=True
    return record


def benchmark(directory, *, device, support_episodes=1472, real_features=False):
    """Three optimizer updates cover Adam state and backward peak, not convergence."""
    import gc
    import numpy as np
    import torch
    from .batched_gp import fit_batched
    from .config import GPConfig
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    results=[]
    for inference,rows in [('exact',support_episodes),('sparse',support_episodes*12)]:
        budget=configure_memory(device)
        rng=np.random.default_rng(7)
        x=rng.normal(size=(rows,512)).astype('float32')
        y=rng.normal(size=(rows,16)).astype('float32')
        cfg=GPConfig(input_components=512,output_components=16,fit_steps=3,
                     inference=inference,sparse_justification='Worst study shape' if inference=='sparse' else None)
        started=time.perf_counter()
        models,_=fit_batched(x,y,cfg,device=device)
        for model in models:model.predict(x[:256])
        result=dict(kind=inference,rows=rows,input_dimensions=512,outputs=16,optimizer_steps=3,
                    seconds=time.perf_counter()-started,**memory_record(device,budget))
        results.append(result);atomic_json(directory/'gpu_memory_validation.json',results)
        del models,x,y;gc.collect();torch.cuda.empty_cache()
    if real_features:
        from .data import load_manifest,candidate_view
        from .features import FeatureExtractor,load_policy,preprocess_inputs,EXPERT_ALIASES
        from .config import FeatureConfig
        budget=configure_memory(device)
        manifest=load_manifest('storage/extensions/data/data_manifest.json')
        support=candidate_view(manifest,manifest['pretraining_ids'])
        model='storage/outputs/pretrain/libero_pretrain_leak3/seed_0/checkpoints/last/pretrained_model'
        policy,pre,_=load_policy(model,device)
        f=FeatureConfig(bank='action_expert',target_bank='velocity',flow_indices=(0,5,9),action_bins=50,flow_samples=50)
        extractor=FeatureExtractor(policy,f)
        started=time.perf_counter()
        banks=(*EXPERT_ALIASES,'velocity','action_context','language','visual_scene','contextual_language','contextual_vision','contextual_state_late','joint_late')
        values,_=extractor.extract(preprocess_inputs([support.observation(i) for i in support.ids[:4]],pre),banks)
        results.append(dict(kind='real_policy_flow_features',observations=4,gaussian_samples=50,
            dimensions={k:list(v.shape) for k,v in values.items()},seconds=time.perf_counter()-started,**memory_record(device,budget)))
        atomic_json(directory/'gpu_memory_validation.json',results)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',required=True);p.add_argument('--output',default='reports/head_study_memory_check')
    p.add_argument('--real-features',action='store_true')
    args=p.parse_args()
    from .cli import device_setup
    _,device=device_setup(args.gpu)
    for result in benchmark(args.output,device=device,real_features=args.real_features):
        print(result,flush=True)


if __name__=='__main__':main()
