"""Small validation-only diagnostic of stochastic policy execution.

Run after sourcing scripts/extensions_env.sh. This never trains policies or
modifies the head study. A saved request makes repeated invocations resumable.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import subprocess


def repeated_starts(manifest, per_task, repetitions, seed):
    groups = {}
    for start in manifest['evaluation_splits']['development']:
        groups.setdefault(start['task'], []).append(start)
    rng = random.Random(seed)
    result = []
    for task, starts in sorted(groups.items()):
        if per_task > len(starts):
            raise ValueError(f'Only {len(starts)} validation starts for {task}')
        # Select without consulting outcomes, then keep the scene/reset seed
        # identical. Only the ID-dependent policy-noise stream changes.
        for start in rng.sample(sorted(starts, key=lambda s: s['id']), per_task):
            for rep in range(repetitions):
                row = dict(start, repetition=rep)
                row['id'] = start['id'].rsplit('/rep_', 1)[0] + f'/rep_{rep}'
                result.append(row)
    return result


def summarize(records):
    groups = {}
    for row in records:
        key = (row['task'], row['init_state_hash'])
        groups.setdefault(key, []).append(row)
    cases = []
    for (task, state), rows in sorted(groups.items()):
        if len({r['initial_input']['input_hash'] for r in rows}) != 1:
            raise RuntimeError('Repeated executions did not start from identical observations')
        rows.sort(key=lambda r: r['repetition'])
        successes = [int(r['success']) for r in rows]
        cases.append(dict(task=task, init_state_hash=state,
                          init_state_id=rows[0]['init_state_id'], successes=successes,
                          steps=[r['steps'] for r in rows],
                          mixed=len(set(successes)) > 1))
    mixed = sum(c['mixed'] for c in cases)
    return dict(conditions=len(cases), executions=len(records),
                mixed_outcome_conditions=mixed, mixed_fraction=mixed / len(cases),
                all_success_conditions=sum(all(c['successes']) for c in cases),
                all_failure_conditions=sum(not any(c['successes']) for c in cases),
                cases=cases,
                interpretation='Small diagnostic for this checkpoint; consistent outcomes do not prove deterministic success.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpu', required=True, help='Physical GPU index or UUID')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('storage/extensions/data/data_manifest.json'))
    parser.add_argument('--per-task', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--batch-size', type=int, default=15)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.per_task < 1 or args.repetitions < 2 or args.batch_size < 1:
        parser.error('Positive per-task/batch size and at least two repetitions required')
    manifest = json.loads(args.manifest.read_text())
    starts = repeated_starts(manifest, args.per_task, args.repetitions, args.seed)
    if args.prepare_only:
        print(json.dumps(dict(model=str(args.model), conditions=len(starts)//args.repetitions,
                              executions=len(starts), starts=starts), indent=2))
        return
    result = args.output / 'evaluation/evaluation.json'
    if not result.exists():
        listing = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'], text=True)
        gpu = next((row for line in listing.splitlines()
                    if (row := [x.strip() for x in line.split(',')])
                    and args.gpu.lower() in (row[0].lower(), row[1].lower())), None)
        if gpu is None or int(gpu[2]) > 1000 or int(gpu[3]) > 10:
            raise RuntimeError('Requested GPU is unavailable or busy; no diagnostic started')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu[1]
        from lerobot.utils.gpu_utils import configure_egl_for_visible_cuda
        print('Render GPU mapping:', configure_egl_for_visible_cuda(os.environ), flush=True)
    from uq_extensions.common import atomic_json, identity
    from uq_extensions.evaluation import evaluate
    diagnostic = copy.deepcopy(manifest)
    diagnostic['evaluation_splits']['development'] = starts
    diagnostic['manifest_hash'] = identity(dict(source=manifest['manifest_hash'], starts=starts))
    args.output.mkdir(parents=True, exist_ok=True)
    spec = dict(source_manifest=manifest['manifest_hash'], starts=starts,
                model=str(args.model.resolve()), seed=args.seed, horizon=520)
    request = args.output / 'diagnostic.json'
    if request.exists() and json.loads(request.read_text()) != spec:
        raise RuntimeError('Diagnostic recipe changed; choose a new output directory')
    atomic_json(request, spec)
    records = evaluate(str(args.model), diagnostic, args.output / 'evaluation',
                       device='cuda:0', batch_size=args.batch_size)
    summary = summarize(records)
    atomic_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
