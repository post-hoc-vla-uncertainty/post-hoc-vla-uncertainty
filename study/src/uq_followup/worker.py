"""Fresh-process jobs; each owns exactly one selected CUDA/EGL device."""
import argparse
import os
from pathlib import Path
import time
from uq_extensions.common import atomic_json, identity, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    request = read_json(args.request); spec = request['spec']
    if identity(spec) != request['identity']: raise RuntimeError('Job request checksum mismatch')
    from .launcher import source_hash
    if source_hash(Path(spec['project'])) != spec['source_hash']:
        raise RuntimeError('Scientific code changed since the study was declared; refusing mixed results')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('This scientific worker requires its selected GPU')
    if torch.cuda.device_count() != 1: raise RuntimeError('Workers must see exactly one GPU')
    device = 'cuda:0'
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    kind = spec['kind']; started = time.monotonic()
    if kind == 'head' and spec['recipe']['method'] == 'kv_t' and 'A100' not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError('Double-precision KV-T must run on an A100')
    print(f'FOLLOWUP START {spec["id"]}: GPU={torch.cuda.get_device_name(0)}; visible={os.environ.get("CUDA_VISIBLE_DEVICES")}; '
          f'EGL={os.environ.get("MUJOCO_EGL_DEVICE_ID")}', flush=True)
    if kind == 'train':
        from .policies import train
        train(spec['project'], spec['study'], spec['policy'], workers=spec['data_workers'])
    elif kind == 'evaluate':
        from .evaluation import execute
        execute(spec['study'], spec['policy_id'], spec['repetition'], device=device, batch_size=spec['evaluation_batch_size'])
    elif kind == 'head':
        from uq_extensions.gpu_memory import configure_memory, memory_record
        memory_budget = configure_memory(device)
        from .scoring import score
        os.environ['UQ_EXT_PROJECTION_CACHE'] = str(Path(spec['study'])/'projection_cache')
        score(spec['study'], spec, device=device, feature_batch_size=spec['feature_batch_size'])
        atomic_json(Path(spec['directory'])/'gpu_memory.json', memory_record(device, memory_budget))
    else:
        raise ValueError('Unknown follow-up job')
    atomic_json(args.request.with_name('complete.json'), dict(identity=request['identity'], seconds=time.monotonic()-started))
    print(f'FOLLOWUP DONE {spec["id"]}; wall={(time.monotonic()-started)/60:.1f} min', flush=True)


if __name__ == '__main__': main()
