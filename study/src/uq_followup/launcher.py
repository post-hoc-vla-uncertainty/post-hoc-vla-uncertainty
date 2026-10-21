"""One-command resumable policies → evaluations → quality check → head matrix."""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from uq_extensions.common import atomic_json, file_hash, identity, model_identity, read_json
from uq_extensions.data import load_manifest
from uq_extensions.head_queue import parse_gpus, resolve_slots
from .design import recipes
from .protocol import policy_specs, evaluation_manifest, SEEDS


def source_hash(project):
    from uq_extensions.runner import source_identity
    # Reporting can be refined from immutable raw scores without invalidating
    # days of policy/GP work. Numerical and protocol code remains pinned.
    files = sorted(p for p in (Path(project)/'src/uq_followup').glob('*.py') if p.name != 'reporting.py')
    return identity(dict(library=source_identity(), followup={p.name: file_hash(p) for p in files}))


def prepare(args):
    project = Path(os.environ['VLA_PROJECT_ROOT']).resolve()
    root = args.output.resolve(); manifest = load_manifest(args.manifest)
    policies = policy_specs(manifest)
    if args.companions == 'matched_data':
        for policy in list(policies):
            if policy['condition'] != 'plus35': continue
            companion = deepcopy(policy)
            companion.update(id=f'companion_seed{policy["seed"]}', condition='companion', training_seed=policy['seed']+3)
            policies.append(companion)
    declaration = dict(version=1, project=str(project), manifest=str(args.manifest.resolve()),
        manifest_hash=manifest['manifest_hash'], source_hash=source_hash(project), seeds=list(SEEDS),
        policies=policies, recipes=recipes(conditioning_mode=args.conditioning_mode), evaluation_conditions=500, repetitions=3,
        conditioning_mode=args.conditioning_mode,
        companion_protocol=args.companions, bootstrap_repetitions=2000,
        policy_quality_rule='Review overall spread >25pp, zero success, or per-task spread >60pp; never auto-exclude',
        main_paper=dict(original=['global_referral', 'task_correlations'], plus35=['global_referral', 'within_task_referral', 'task_correlations']))
    root.mkdir(parents=True, exist_ok=True)
    path = root/'declaration.json'
    if path.exists() and read_json(path) != declaration and (root/'started.json').exists():
        raise RuntimeError('Started study declaration differs: source, data, or companion protocol changed. '
            'For the completed-rollout render-roundoff repair, run scripts/repair_followup_observation_pairing.py '
            'in the project environment first. Other protocol changes require a new output directory.')
    atomic_json(path, declaration)
    atomic_json(root/'evaluation_manifest.json', evaluation_manifest(manifest))
    from .policies import register_original
    for policy in policies:
        path = root/'policies'/policy['id']/'model.json'
        if path.exists():
            record = read_json(path)
            if record['spec'] != policy: raise RuntimeError('Policy dataset/budget differs')
            if model_identity(record['model']) != record['model_hash']:
                raise RuntimeError(f'Saved policy weights changed: {policy["id"]}')
        elif policy['condition'] == 'original':
            register_original(project, root, policy)
    atomic_json(root/'configuration_matrix.json', declaration['recipes'])
    print(f'Study: {root}\n3 policy seeds × 2 datasets; {len(declaration["recipes"])} heads/policy = 798 head evaluations.\n'
          f'Policies: {len(policies)} including supervisors; evaluation: 13,500 executions, cached once.\n'
          f'Companion protocol: {args.companions}.', flush=True)
    return declaration


def job(root, declaration, kind, id_, **extra):
    directory = root/'jobs'/id_
    spec = dict(id=id_, kind=kind, project=declaration['project'], study=str(root),
                source_hash=declaration['source_hash'], **extra)
    request = dict(identity=identity(spec), spec=spec)
    path = directory/'request.json'
    reused_source = False
    if path.exists():
        previous = read_json(path)
        if previous != request:
            from .resume_compatibility import reusable_completed_request
            if not reusable_completed_request(root, previous, request):
                raise RuntimeError(f'Job request changed: {id_}')
            # Keep the original request/completion identities as historical facts.
            # Such requests may only be skipped, never run under new source code.
            request = previous
            reused_source = True
    done = directory/'complete.json'
    if done.exists():
        if read_json(done)['identity'] != request['identity']: raise RuntimeError(f'Completion identity mismatch: {id_}')
        if kind == 'head':
            result = Path(extra['directory'])/'result.json'
            if result.exists():
                record = read_json(result)
                if record.get('recipe') != extra['recipe']: raise RuntimeError('Completed head recipe changed')
                if record['status'] == 'uninformative' or (result.parent/'report.html').exists(): return None
        elif kind == 'evaluate':
            evaluation = root/'evaluations'/extra['policy_id']/f'rep_{extra["repetition"]}'/'evaluation.json'
            if evaluation.exists() and len(read_json(evaluation)['episodes']) == 500: return None
        elif kind == 'train' and (root/'policies'/extra['policy']['id']/'model.json').exists(): return None
        if reused_source:
            raise RuntimeError(f'Missing artifact from audited completed job: {id_}')
        done.unlink()
    if reused_source:
        raise RuntimeError(f'Missing completion from audited job: {id_}')
    atomic_json(path, request)
    return dict(request=request, path=path)


def accepts(spec, slot):
    if spec['kind'] != 'head': return True
    return ('A100' in slot['name'].upper()) if spec['recipe']['method'] == 'kv_t' else ('3090' in slot['name'])


def drain(root, jobs, slots, stage):
    from run_progress import LogFollower, emit, tail_text
    from .reporting import refresh_index
    if not slots: raise ValueError('Select at least one GPU')
    outstanding = [j for j in jobs if j is not None]
    pending = [j for j in outstanding if any(accepts(j['request']['spec'], s) for s in slots)]
    deferred = [j['request']['spec']['id'] for j in outstanding
                if not any(accepts(j['request']['spec'], s) for s in slots)]
    running = {}; failures = []; completed = 0
    if deferred:
        emit(stage, f'DEFER {len(deferred)} jobs: no compatible selected GPU; '
             'rerun with A100 for KV-T and RTX 3090 for other heads. Completed work will be reused.')
    total = len(outstanding); last_report = time.monotonic()
    def update():
        atomic_json(root/'queue.json', dict(stage=stage, completed_this_stage=completed, total_this_stage=total,
            pending=[j['request']['spec']['id'] for j in pending], deferred=deferred, failed=failures,
            running=[dict(job=j['request']['spec']['id'], gpu=label, pid=p.pid,
                          seconds=time.monotonic()-started) for label, (p, j, _, started) in running.items()],
            updated_at=time.time()))
    try:
        while pending or running:
            changed = False
            for label, (process, j, follower, started) in list(running.items()):
                code = process.poll(); follower.poll(heartbeat=code is None, final=code is not None)
                if code is None: continue
                completion = j['path'].with_name('complete.json')
                if code or not completion.exists():
                    failure = dict(id=j['request']['spec']['id'], exit_code=code,
                        log=str(j['path'].with_name('run.log')), tail=tail_text(j['path'].with_name('run.log')))
                    atomic_json(j['path'].with_name('failure.json'), failure); failures.append(failure)
                    emit(stage, f'FAILED {failure["id"]}; continuing independent jobs; log={failure["log"]}')
                else:
                    j['path'].with_name('failure.json').unlink(missing_ok=True)
                    completed += 1
                    emit(stage, f'DONE {j["request"]["spec"]["id"]} on GPU {label}; wall={(time.monotonic()-started)/60:.1f} min')
                    if j['request']['spec']['kind'] == 'head': refresh_index(root)
                del running[label]; changed = True
            for slot in slots:
                if slot['label'] in running: continue
                selected = next((j for j in pending if accepts(j['request']['spec'], slot)), None)
                if selected is None: continue
                pending.remove(selected)
                path = selected['path']; env = dict(os.environ, **slot['environment'], PYTHONUNBUFFERED='1')
                for key in ('VLA_REPRODUCTION_POLICY', 'UQ_EXT_RUN_ROOT', 'UQ_EXT_BUDGET_GB', 'UQ_EXT_TRAINING_OVERLAY'):
                    env.pop(key, None)
                env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                follower = LogFollower([path.with_name('run.log')], label=f'{stage}/GPU {slot["label"]}/{selected["request"]["spec"]["id"]}')
                with path.with_name('run.log').open('a') as log:
                    process = subprocess.Popen([sys.executable, '-u', '-m', 'uq_followup.worker', str(path)],
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[slot['label']] = (process, selected, follower, time.monotonic())
                emit(stage, f'START {selected["request"]["spec"]["id"]} on GPU {slot["label"]}; log={path.with_name("run.log")}')
                changed = True
            if changed or time.monotonic()-last_report > 30:
                update(); last_report = time.monotonic()
            if running: time.sleep(.5)
    finally:
        for process, _, _, _ in running.values():
            if process.poll() is None: os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic()+20
        while any(p.poll() is None for p, _, _, _ in running.values()) and time.monotonic()<deadline: time.sleep(.1)
        for process, _, _, _ in running.values():
            if process.poll() is None: os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        update()
    return dict(failures=failures, deferred=deferred)


def companion_id(policy, declaration):
    if declaration['companion_protocol'] == 'matched_data':
        if policy['condition'] == 'plus35': return f'companion_seed{policy["seed"]}'
        return f'original_companion_seed{policy["seed"]}'
    index = declaration['seeds'].index(policy['seed'])
    other = declaration['seeds'][(index+1) % len(declaration['seeds'])]
    return f'{policy["condition"]}_seed{other}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='4,5,6,7', help='Physical NVIDIA GPU IDs; one job per GPU')
    parser.add_argument('--output', type=Path, default=Path('storage/extensions/followup_v1'))
    parser.add_argument('--manifest', type=Path, default=Path('storage/extensions/data/data_manifest.json'))
    parser.add_argument('--companions', choices=['study_seed', 'matched_data'], default='study_seed')
    parser.add_argument('--conditioning-mode', choices=['append', 'compressed'], default='append',
                        help='Append robot-state/action/time intact after compressing the early/middle embedding')
    parser.add_argument('--data-workers', type=int, default=4)
    parser.add_argument('--evaluation-batch-size', type=int, default=15)
    parser.add_argument('--feature-batch-size', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true', help='Prepare and validate the manifest; launch no GPU work')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--acknowledge-quality', action='store_true', help='Record that flagged policy differences were reviewed; exclude no policies')
    args = parser.parse_args(); root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    if min(args.data_workers, args.evaluation_batch_size, args.feature_batch_size) < 1: parser.error('Batch/worker counts must be positive')
    if args.report_only:
        from .reporting import aggregate_report
        print(aggregate_report(root)); return
    with (root/'study.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('This follow-up study already has a launcher')
        declaration = prepare(args)
        if args.dry_run:
            print('DRY RUN complete: no GPU work launched.'); return
        slots = resolve_slots(parse_gpus(args.gpus))
        import csv
        names = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name', '--format=csv,noheader,nounits'], text=True)
        inventory = {u.strip().lower(): n.strip() for u, n in csv.reader(names.splitlines())}
        for slot in slots: slot['name'] = inventory[slot['device'].lower()]
        print('Selected GPUs: '+', '.join(f'{s["label"]} ({s["name"]})' for s in slots)+
              '. All can train/evaluate policies. Heads without a compatible GPU will remain pending.', flush=True)
        usage = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
        used = {u.strip().lower(): int(m.strip()) for u, m in csv.reader(usage.splitlines())}
        busy = [s['label'] for s in slots if used[s['device'].lower()] > 1500]
        if busy:
            raise RuntimeError(f'Selected GPUs {busy} already have substantial allocations. Let the current study finish or select idle GPUs; no new jobs launched.')
        if not (root/'started.json').exists():
            atomic_json(root/'started.json', dict(declaration_hash=identity(declaration), started_at=time.time()))
        policies = declaration['policies']
        if args.companions == 'matched_data':
            # The remaining three original policies already supply matched-data companions.
            from .policies import original_checkpoint
            for seed in SEEDS:
                source = next(p for p in policies if p['id'] == f'original_seed{seed}')
                model = original_checkpoint(declaration['project'], dict(source, seed=seed+3))
                if model is None: raise RuntimeError('Missing matched original-data companion')
                atomic_json(root/'policies'/f'original_companion_seed{seed}'/'model.json', dict(model=model))
        training = [job(root, declaration, 'train', 'train_'+p['id'], policy=p, data_workers=args.data_workers)
                    for p in policies if not (root/'policies'/p['id']/'model.json').exists()]
        outcome = drain(root, training, slots, 'policy-training')
        if outcome['failures']: raise RuntimeError('Policy jobs failed; inspect logs and rerun the same command. Heads have not started.')
        evaluations = [job(root, declaration, 'evaluate', f'evaluate_{p["id"]}_rep{rep}', policy_id=p['id'],
            repetition=rep, evaluation_batch_size=args.evaluation_batch_size)
            for p in policies if p['condition'] != 'companion' for rep in range(3)]
        outcome = drain(root, evaluations, slots, 'policy-evaluation')
        if outcome['failures']: raise RuntimeError('Evaluation jobs failed; rerun to resume saved episodes. Heads have not started.')
        from .evaluation import collect
        quality = collect(root, policies)
        for name, value in quality['policies'].items(): print(f'QUALITY {name}: success={value["success"]:.1%}; tasks={value["per_task"]}', flush=True)
        quality_key = identity(quality)
        review = root/'quality_review.json'
        if quality['warnings']:
            reviewed = review.exists() and read_json(review)['quality_hash'] == quality_key
            if not reviewed and not args.acknowledge_quality:
                raise RuntimeError('Policy outliers need investigation before heads: '+ '; '.join(quality['warnings'])+
                    '. Inspect policy_quality.md/logs, then rerun with --acknowledge-quality if these are valid differences.')
            if args.acknowledge_quality: atomic_json(review, dict(quality_hash=quality_key, reviewed_at=time.time(), excluded_policies=[]))
        atomic_json(root/'quality_gate.json', dict(quality_hash=quality_key, passed=True))
        heads = []
        for policy in policies:
            if policy['condition'] not in ('original', 'plus35'): continue
            for recipe in declaration['recipes']:
                heads.append(job(root, declaration, 'head', policy['id']+'__'+recipe['id'],
                    policy_id=policy['id'], companion_id=companion_id(policy, declaration), recipe=recipe,
                    directory=str(root/'heads'/policy['id']/recipe['id']), feature_batch_size=args.feature_batch_size))
        from .reporting import aggregate_report
        failures = []; deferred = []
        for action in (False, True):
            phase = [j for j in heads if j and (j['request']['spec']['recipe']['scope'] == 'action') == action]
            outcome = drain(root, phase, slots, 'action-heads' if action else 'observation-heads')
            failures.extend(outcome['failures']); deferred.extend(outcome['deferred'])
            print('Updated report:', aggregate_report(root), flush=True)
        atomic_json(root/'study_status.json', dict(
            status='failed' if failures else ('waiting_for_gpus' if deferred else 'complete'),
            failed=failures, deferred=deferred, updated_at=time.time()))
        if failures: raise RuntimeError(f'{len(failures)} head jobs failed; others completed. Rerun the same command to retry unfinished work.')
        if deferred:
            print(f'FOLLOWUP WAITING FOR GPU TYPES: {len(deferred)} head jobs remain. '
                  'Rerun with A100 for KV-T and RTX 3090 for other heads; completed jobs are skipped. '
                  f'Report: {root/"report/report.html"}', flush=True)
        else:
            print('FOLLOWUP COMPLETE:', root/'report/report.html', flush=True)


if __name__ == '__main__': main()
