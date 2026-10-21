"""Dependency-aware multi-GPU queue; failures block descendants, not other arms."""
import argparse
from collections import Counter
import csv
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from uq_extensions.common import atomic_json, file_hash, identity, read_json
from .design import evaluation_rounds, jobs, prepare


def accepts(job, slot):
    return not job.get('a100') or 'A100' in slot['name'].upper()


def validate_graph(queue):
    mapping = {j['id']: j for j in queue}
    if len(mapping) != len(queue): raise ValueError('Duplicate job IDs')
    visited = set(); active = set()
    def visit(id_):
        if id_ in active: raise ValueError('Dependency cycle')
        if id_ in visited: return
        if id_ not in mapping: raise ValueError(f'Unknown dependency {id_}')
        active.add(id_)
        for parent in mapping[id_]['dependencies']: visit(parent)
        active.remove(id_); visited.add(id_)
    for j in queue: visit(j['id'])


def completed(job, root=None):
    path = Path(job['output'])
    if not path.exists(): return False
    data = read_json(path)
    if job['kind'] == 'train':
        if not (Path(data['model'])/'model.safetensors').is_file():
            raise RuntimeError(f'Missing retained policy: {data["model"]}')
    elif job['kind'] == 'reference':
        if file_hash(data['source']) != data['source_hash']: raise RuntimeError('Reference outcomes changed')
    elif job['kind'] in ('evaluate', 'collect'):
        if len(data['episodes'] if job['kind']=='evaluate' else data) != 500: raise ValueError('Incomplete outcome artifact')
    elif job['kind'] == 'acquire':
        if len(data['selected_ids']) != 5 or len(set(data['selected_ids'])) != 5: raise ValueError('Invalid committed batch')
    elif job['kind'] == 'score':
        if not path.with_name('scores.json').exists() or 'metrics' not in data: return False
    if root is not None:
        marker = Path(root)/'jobs'/job['id']/'complete.json'
        if marker.exists():
            if read_json(marker)['output_hash'] != file_hash(path): raise RuntimeError('Completed job output changed')
        else:
            provenance = data.get('provenance') if isinstance(data, dict) else None
            imported = (provenance in ('reused_reproduction', 'original_reproduction_outcomes') if isinstance(provenance,str) else
                        isinstance(provenance,dict) and provenance.get('kind') in ('reused_reproduction','reused_completed_member_of_interrupted_round'))
            if not imported: return False
    return True


def request(root, declaration, job):
    path = root/'jobs'/job['id']/'request.json'
    spec = dict(study=str(root), declaration=identity(declaration), job=job)
    data = dict(identity=identity(spec), spec=spec)
    if path.exists() and read_json(path) != data: raise RuntimeError(f'Job request changed: {job["id"]}')
    if path.with_name('complete.json').exists():
        done = read_json(path.with_name('complete.json'))
        if done['identity'] != data['identity']: raise RuntimeError('Completion identity mismatch')
        if completed(job, root) and done['output_hash'] != file_hash(job['output']): raise RuntimeError('Completed job output changed')
    atomic_json(path, data)
    return path


def priority(job):
    # Acquisition and training advance all arms first; short scoring can fill otherwise idle slots.
    kind = job['kind']
    return (-1 if kind == 'cleanup' else 0 if kind in ('acquire', 'train') else 1 if kind == 'reference' else 2 if kind in ('evaluate', 'collect', 'quality') else 3,
            job.get('round', 0), job.get('head', ''), job.get('seed', 0), job['id'])


def ready_jobs(queue, done, failed):
    return [j for j in queue if j['id'] not in done | failed and set(j['dependencies']) <= done]


def run_queue(root, declaration, queue, slots):
    from run_progress import LogFollower, tail_text
    validate_graph(queue)
    done = {j['id'] for j in queue if completed(j, root)}
    pending = {j['id']:j for j in queue if j['id'] not in done}
    failed = {}; running = {}; paths = {}
    for j in queue: paths[j['id']] = request(root, declaration, j)
    print(f'{len(done)}/{len(queue)} jobs already complete or reused; {len(pending)} pending.', flush=True)
    last = 0.
    def update():
        atomic_json(root/'queue.json', dict(total=len(queue), complete=len(done), failed=failed,
            running=[dict(id=j['id'], gpu=label, pid=p.pid, seconds=time.monotonic()-start)
                     for label, (p,j,_,start) in running.items()], pending=list(pending), updated_at=time.time()))
    try:
        while pending or running:
            for label, (process, job, follower, started) in list(running.items()):
                code = process.poll(); follower.poll(heartbeat=code is None, final=code is not None)
                if code is None: continue
                if code == 0 and completed(job, root) and paths[job['id']].with_name('complete.json').exists():
                    done.add(job['id']); paths[job['id']].with_name('failure.json').unlink(missing_ok=True)
                    print(f'DONE {job["id"]} on GPU {label}; wall={(time.monotonic()-started)/60:.1f} min; '
                          f'{len(done)}/{len(queue)} jobs complete.', flush=True)
                else:
                    failure = dict(exit_code=code, log=str(paths[job['id']].with_name('run.log')),
                                   tail=tail_text(paths[job['id']].with_name('run.log')))
                    failed[job['id']] = failure
                    atomic_json(paths[job['id']].with_name('failure.json'), failure)
                    print(f'FAILED {job["id"]}; other independent jobs continue. Log: {failure["log"]}', flush=True)
                del running[label]
            for slot in slots:
                if slot['label'] in running: continue
                candidates = sorted((j for j in pending.values() if set(j['dependencies']) <= done and accepts(j, slot)),
                    key=lambda j:(0 if j['kind']=='cleanup' else 1 if j.get('a100') else 2, priority(j)))
                if not candidates: continue
                job = candidates[0]; del pending[job['id']]
                path = paths[job['id']]
                env = dict(os.environ, **slot['environment'], PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                for key in ('VLA_REPRODUCTION_POLICY', 'UQ_EXT_RUN_ROOT', 'UQ_EXT_BUDGET_GB', 'UQ_EXT_TRAINING_OVERLAY'):
                    env.pop(key, None)
                follower = LogFollower([path.with_name('run.log')], label=f'GPU {slot["label"]}/{job["id"]}')
                with path.with_name('run.log').open('a') as log:
                    process = subprocess.Popen([sys.executable, '-u', '-m', 'uq_finalstudy.worker', str(path)],
                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[slot['label']] = process, job, follower, time.monotonic()
                print(f'START {job["id"]} on GPU {slot["label"]}; log={path.with_name("run.log")}', flush=True)
            if time.monotonic()-last > 30:
                update(); last = time.monotonic()
                print(f'PROGRESS {len(done)}/{len(queue)} complete; running={len(running)}; queued={len(pending)}; failed={len(failed)}', flush=True)
            if not running:
                if pending: print(f'{len(pending)} jobs deferred: missing successful dependencies or a selected A100. Restart retries unfinished work.', flush=True)
                break
            time.sleep(.5)
    finally:
        for process, _, _, _ in running.values():
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        deadline = time.monotonic()+20
        while any(p.poll() is None for p,_,_,_ in running.values()) and time.monotonic()<deadline: time.sleep(.1)
        for process, _, _, _ in running.values():
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            process.wait()
        update()
    return len(failed)+len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='4,5,6,7', help='Physical nvidia-smi indices or full UUIDs; one job per GPU')
    parser.add_argument('--output', type=Path, default=Path('storage/extensions/final_save_v1'))
    parser.add_argument('--manifest', type=Path, default=Path('storage/extensions/data/data_manifest.json'))
    parser.add_argument('--budget-gb', type=float, default=1800, help='Storage guard for this new study, frozen on first preparation')
    parser.add_argument('--dry-run', action='store_true', help='Validate artifacts and write the queue; do not use a GPU')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--report-only', action='store_true', help='Recompute the combined report; no GPU work')
    args = parser.parse_args(); root = args.output.resolve()
    if args.report_only:
        from .reporting import generate
        print(generate(root)); return
    if args.status:
        declaration = read_json(root/'declaration.json'); queue = jobs(root, declaration)
        counts = Counter((j['kind'], completed(j, root)) for j in queue)
        for kind in sorted({j['kind'] for j in queue}): print(f'{kind}: {counts[kind,True]}/{counts[kind,True]+counts[kind,False]}')
        if (root/'queue.json').exists():
            q = read_json(root/'queue.json'); print('Running:', q['running']); print('Failed:', list(q['failed']))
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.launcher.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another launcher owns this study; stop it before restarting')
        declaration = prepare(root, Path(os.environ['VLA_PROJECT_ROOT']), args.manifest, budget_gb=args.budget_gb)
        queue = jobs(root, declaration); validate_graph(queue)
        atomic_json(root/'job_graph.json', queue)
        counts = Counter(j['kind'] for j in queue)
        reused = Counter(j['kind'] for j in queue if completed(j, root))
        print('Final study: 8 acquisition methods × 3 independent policy pairs; 15 rounds × 5 queries × 4,000 updates.', flush=True)
        print('Policy pairs: (0,1), (2,3), (4,5); fixed supervisors: 0,1,2.', flush=True)
        print(f'Acquisition-performance evaluation rounds: {evaluation_rounds(declaration, "acquisition")}; '
              f'correlation/referral: Random rounds {evaluation_rounds(declaration, "common")}.', flush=True)
        print(f'Storage guard: {declaration["budget_gb"]:g} GB for this study; all completed-round weights retained.', flush=True)
        print(f'Jobs by kind: {dict(counts)}; already reusable: {dict(reused)}', flush=True)
        if args.dry_run:
            print(f'Prepared {root}; no training, simulation or GP fitting launched.'); return
        from uq_extensions.head_queue import parse_gpus, resolve_slots
        slots = resolve_slots(parse_gpus(args.gpus))
        output = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader,nounits'], text=True, timeout=15)
        names = {uuid.strip().lower():name.strip() for uuid, name in csv.reader(output.splitlines())}
        for slot in slots: slot['name'] = names[slot['device'].lower()]
        def interrupted(signum, frame): raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        remaining = run_queue(root, declaration, queue, slots)
        print('Combined report is on request: rerun this command with --report-only.', flush=True)
        if remaining: raise SystemExit(1)


if __name__ == '__main__': main()
