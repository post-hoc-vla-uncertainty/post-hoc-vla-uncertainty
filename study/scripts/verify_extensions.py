#!/usr/bin/env python3
"""Short real-model checks; does not run scientific acquisition trajectories."""
from pathlib import Path
import argparse
import numpy as np
from uq_extensions.cli import device_setup
from uq_extensions.common import atomic_json,read_json
from uq_extensions.config import FeatureConfig,GPConfig
from uq_extensions.data import load_manifest,candidate_view
from uq_extensions.features import cache_features
from uq_extensions.kernel_variance import KernelVariance
from uq_extensions.target_gp import TargetGP
from uq_extensions.ivr import select_ivr
from uq_extensions.baselines import METHODS,baseline_scores
from uq_extensions.evaluation import evaluate

p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--output',default='storage/extensions/verification')
a=p.parse_args();gpus,device=device_setup(a.gpus);output=Path(a.output)
m=load_manifest('storage/extensions/data/data_manifest.json')
models=['outputs/smoke/pretrain/checkpoints/000002/pretrained_model','outputs/smoke/pretrain_member1/checkpoints/000002/pretrained_model']
# Use small but real banks from several tasks; the smoke budget is explicitly recorded.
ids=[]
for task in sorted({r['task'] for r in m['observations']})[:4]:
 ids += [r['id'] for r in m['observations'] if r['task']==task][:8]
v=candidate_view(m,ids);fc=FeatureConfig(components=8);gc=GPConfig(input_components=4,output_components=2,fit_steps=4,precision='float64')
f=cache_features(models[0],v,fc,output/'features',device=device,batch_size=4,banks=('joint_intermediate','joint_late'),role='support',support_ids=v.ids)
x=f['joint_intermediate'];y=f['joint_late'];support=x[:24];query=x[24:]
kv=KernelVariance(fc,gc,device=device).fit(support);kv.save(output/'kv')
kvt=TargetGP(fc,gc,device=device).fit(support,y[:24],tasks=v.tasks[:24]);kvt.save(output/'kv_t')
selected,audit=select_ivr(kv,query,v.tasks[24:],query,v.ids[24:],count=5)
result=dict(smoke=True,kv_scores=kv.score(query).tolist(),kv_t_scores=kvt.score(query).tolist(),ivr_selected=selected,ivr=audit)
small=v.subset(v.ids[:1]);result['baselines']={}
for method in METHODS:
 result['baselines'][method]=baseline_scores(method,models,small,output/'baseline_scores',device=device)
 print('BASELINE PASS',method,flush=True)
# Two simulator starts in one batch: exercise pairing, physical metrics and noise.
first=evaluate(models[0],m,output/'paired_batch',device=device,max_steps=2,limit=2,batch_size=2)
second=evaluate(models[0],m,output/'paired_serial',device=device,max_steps=2,limit=2,batch_size=1)
for a,b in zip(first,second):
 assert a['id']==b['id'] and a['initial_input']['input_hash']==b['initial_input']['input_hash']
 assert abs(a['path_meters']-b['path_meters'])<1e-4
result['paired_initial_inputs_identical']=True;result['batch_serial_path_tolerance_meters']=1e-4
result['max_batch_serial_path_difference_meters']=max(abs(a['path_meters']-b['path_meters']) for a,b in zip(first,second))
atomic_json(output/'verification.json',result)
atomic_json('reports/extension_runtime_verification.json',result)
print('ALL EXTENSION RUNTIME CHECKS PASSED',flush=True)
