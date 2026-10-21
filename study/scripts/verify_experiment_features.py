#!/usr/bin/env python3
"""Bounded real-policy feature validation; no acquisition or policy training."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from uq_extensions.cli import device_setup
from uq_extensions.common import atomic_json,isolated_rng
from uq_extensions.config import FeatureConfig
from uq_extensions.data import load_manifest,candidate_view
from uq_extensions.features import load_policy,preprocess_inputs,FeatureExtractor

p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--model',required=True)
p.add_argument('--output',default='reports/experiment_feature_verification.json');args=p.parse_args()
_,device=device_setup(args.gpus)
manifest=load_manifest('storage/extensions/data/data_manifest.json');view=candidate_view(manifest,manifest['pretraining_ids'][:2])
policy,pre,_=load_policy(args.model,device);batch=preprocess_inputs([view.observation(id) for id in view.ids],pre)
noise=torch.randn((2,policy.config.chunk_size,policy.config.max_action_dim),device=device,generator=torch.Generator(device=device).manual_seed(5))
with torch.no_grad():before=policy.predict_action_chunk(batch,noise=noise.clone()).clone()
checks={}
for distance in (0,1,3):
    f=FeatureConfig(bank='action_expert',target_bank='velocity',expert_distance=distance,flow_samples=2,support_flow_samples=2,flow_indices=(0,5,9))
    extractor=FeatureExtractor(policy,f)
    values,_=extractor.extract(batch,('action_expert','velocity'),probe_count=2)
    again,_=extractor.extract(batch,('action_expert','velocity'),probe_count=2)
    for k in values:np.testing.assert_array_equal(values[k],again[k])
    assert values['velocity'].shape==(2,6,350)
    checks[str(distance)]=extractor.spec
f=FeatureConfig(bank='visual_patches');extractor=FeatureExtractor(policy,f)
values,_=extractor.extract(batch,('visual_patches','contextual_language','contextual_vision'))
assert values['visual_patches'].ndim==4 and values['visual_patches'].shape[1]==2
checks['patches']=extractor.spec
f=FeatureConfig(bank='joint_late',target_bank='action_chunk',flow_indices=(0,5,9))
values,_=FeatureExtractor(policy,f).extract(batch,('joint_late','action_chunk'),probe_count=1)
probe=torch.randn((1,policy.config.chunk_size,policy.config.max_action_dim),device=device,
    generator=torch.Generator(device=device).manual_seed(f.flow_seed)).expand(2,-1,-1).clone()
with torch.no_grad():generated=policy.predict_action_chunk(batch,noise=probe).cpu().numpy().reshape(2,-1)
np.testing.assert_allclose(values['action_chunk'][:,0],generated,atol=1e-5,rtol=1e-5)
checks['action_chunk_matches_policy']=True
with torch.no_grad():after=policy.predict_action_chunk(batch,noise=noise.clone())
torch.testing.assert_close(before,after,atol=0,rtol=0)
atomic_json(args.output,dict(checks=checks,predictions_unchanged=True,repeated_probes_identical=True,policy_updates=0))
print('Feature hooks and flow probes passed; policy predictions unchanged.')
