#!/usr/bin/env python3
"""Replace a live dispatcher, adopt its workers, and retire GPUs by a deadline.

Operational only: all scientific jobs, identities and worker commands are kept.
After stopping this dispatcher, the ordinary 11_final_study.sh command can resume.
"""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from uq_extensions.common import atomic_json, read_json
from uq_extensions.head_queue import parse_gpus, resolve_slots
from uq_finalstudy import launcher
from uq_finalstudy.design import prepare
from run_progress import LogFollower, tail_text


def process(pid):
    p = Path('/proc') / str(pid)
    try:
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start=fields[19], state=fields[0], parent=int(fields[1]),
                    uid=p.stat().st_uid, args=(p/'cmdline').read_bytes().decode().split('\0')[:-1])
    except (FileNotFoundError, ProcessLookupError):
        return None


def same(saved):
    current = process(saved['pid'])
    return current if current and (current['start'], current['uid']) == (saved['start'], saved['uid']) else None


def send(saved, sig, group=False):
    current = same(saved)
    if not current or current['state'] == 'Z':
        return
    if group:
        if os.getpgid(saved['pid']) != saved['pid']:
            raise RuntimeError('Worker is not its own process-group leader')
        os.killpg(saved['pid'], sig)
    else:
        fd = os.pidfd_open(saved['pid'])
        try:
            if same(saved):
                signal.pidfd_send_signal(fd, sig)
        finally:
            os.close(fd)


def stop_workers(entries):
    for entry in entries:
        send(entry['process'], signal.SIGTERM, group=True)
    limit = time.monotonic() + 15
    while time.monotonic() < limit:
        if all(not same(e['process']) or same(e['process'])['state'] == 'Z' for e in entries):
            break
        time.sleep(.2)
    for entry in entries:
        send(entry['process'], signal.SIGKILL, group=True)


def acquire_lock(lock):
    limit=time.monotonic()+30
    while True:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return
        except BlockingIOError:
            if time.monotonic()>=limit:raise
            time.sleep(.2)


def resume_handoff(args, root, project, deadline):
    audit=args.resume_handoff.resolve()
    saved=read_json(audit/'handoff.json')
    if same(saved['old_launcher']):raise RuntimeError('Old dispatcher still exists')
    declaration=read_json(root/'declaration.json')
    prepare(root,project,Path(declaration['manifest']),budget_gb=declaration['budget_gb'])
    queue=launcher.jobs(root,declaration);by_id={j['id']:j for j in queue}
    slots=resolve_slots(parse_gpus(args.gpus))
    inventory={uuid.strip().lower():name.strip() for uuid,name in csv.reader(subprocess.check_output(
        ['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines())}
    for slot in slots:slot['name']=inventory[slot['device'].lower()]
    q=read_json(root/'queue.json');elapsed={e['pid']:e['seconds']+max(0,time.time()-q['updated_at']) for e in q['running']}
    adopted={}
    for e in saved['workers']:
        current=same(e['process'])
        if not current or current['state']=='Z':continue
        if current['uid']!=os.getuid():raise RuntimeError('Worker owner changed')
        job=by_id[e['job']];path=root/'jobs'/job['id']/'request.json'
        if Path(current['args'][-1]).resolve()!=path:raise RuntimeError('Worker request changed')
        adopted[e['gpu']]=dict(process=e['process'],job=job,started=time.monotonic()-elapsed.get(current['pid'],0),
            follower=LogFollower([path.with_name('run.log')],label=f'GPU {e["gpu"]}/{job["id"]}'))
    with (root/'.launcher.lock').open('a') as lock:
        acquire_lock(lock)
        def interrupt(*_):raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,interrupt)
        atomic_json(audit/'resumed.json',dict(dispatcher_pid=os.getpid(),at=time.time()))
        print(f'HANDOFF resumed: adopted {len(adopted)} live workers; new jobs only on {args.gpus}.',flush=True)
        return run(root,declaration,queue,slots,adopted,deadline,audit)


def run(root, declaration, queue, slots, adopted, release_by, audit):
    launcher.validate_graph(queue)
    done = {j['id'] for j in queue if launcher.completed(j, root)}
    pending = {j['id']: j for j in queue if j['id'] not in done}
    paths = {j['id']: launcher.request(root, declaration, j) for j in queue}
    running = adopted
    for e in running.values():
        pending.pop(e['job']['id'], None)
    failed = {}; last = 0.; released = False
    allowed = {s['label'] for s in slots}

    def update():
        atomic_json(root/'queue.json', dict(total=len(queue), complete=len(done), failed=failed,
            pending=list(pending), updated_at=time.time(), dispatcher_pid=os.getpid(), allowed_gpus=sorted(allowed),
            running=[dict(id=e['job']['id'], gpu=label, pid=e['process']['pid'],
                          seconds=time.monotonic()-e['started']) for label, e in running.items()]))

    try:
        while pending or running:
            # SIGTERM 20 seconds before the hard deadline leaves room for cleanup.
            if not released and time.time() >= release_by-20:
                retiring = {g:e for g,e in running.items() if g not in allowed}
                stop_workers(list(retiring.values()))
                for g,e in retiring.items():
                    job=e['job']
                    if launcher.completed(job, root):
                        done.add(job['id'])
                    else:
                        pending[job['id']]=job
                    del running[g]
                released=True
                atomic_json(audit/'released.json', dict(at=time.time(), allowed_gpus=sorted(allowed),
                            interrupted=[e['job']['id'] for e in retiring.values()]))
                print(f'RELEASED GPUs outside {sorted(allowed)}; unfinished jobs requeued.', flush=True)
                update()
            for label,e in list(running.items()):
                child=e.get('child'); current=same(e['process'])
                code=child.poll() if child is not None else (None if current and current['state']!='Z' else 0)
                e['follower'].poll(heartbeat=code is None, final=code is not None)
                if code is None:
                    continue
                job=e['job']; path=paths[job['id']]
                if code==0 and launcher.completed(job, root) and path.with_name('complete.json').exists():
                    done.add(job['id']); path.with_name('failure.json').unlink(missing_ok=True)
                    print(f'DONE {job["id"]} on GPU {label}; wall={(time.monotonic()-e["started"])/60:.1f} min', flush=True)
                else:
                    failure=dict(exit_code=code, log=str(path.with_name('run.log')), tail=tail_text(path.with_name('run.log')))
                    failed[job['id']]=failure; atomic_json(path.with_name('failure.json'),failure)
                    print(f'FAILED {job["id"]}; independent work continues.',flush=True)
                del running[label]
            for slot in slots:
                label=slot['label']
                if label in running:
                    continue
                ready=sorted((j for j in pending.values() if set(j['dependencies'])<=done and launcher.accepts(j,slot)),
                    key=lambda j:(0 if j['kind']=='cleanup' else 1 if j.get('a100') else 2, launcher.priority(j)))
                if not ready:
                    continue
                job=ready[0]; del pending[job['id']]; path=paths[job['id']]
                env=dict(os.environ,**slot['environment'],PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                for key in ('VLA_REPRODUCTION_POLICY','UQ_EXT_RUN_ROOT','UQ_EXT_BUDGET_GB','UQ_EXT_TRAINING_OVERLAY'):
                    env.pop(key,None)
                with path.with_name('run.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u','-m','uq_finalstudy.worker',str(path)],env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[label]=dict(job=job,process=process(child.pid),child=child,started=time.monotonic(),
                    follower=LogFollower([path.with_name('run.log')],label=f'GPU {label}/{job["id"]}'))
                print(f'START {job["id"]} on GPU {label}; log={path.with_name("run.log")}',flush=True)
            if time.monotonic()-last>15:
                update(); last=time.monotonic()
                print(f'PROGRESS {len(done)}/{len(queue)} complete; running={len(running)}; queued={len(pending)}; failed={len(failed)}',flush=True)
            if not running:
                break
            time.sleep(.5)
    finally:
        stop_workers(list(running.values()))
        update()
    return len(failed)+len(pending)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher-pid',type=int)
    parser.add_argument('--resume-handoff',type=Path)
    parser.add_argument('--gpus',default='4,5,6,7')
    parser.add_argument('--release-by',required=True,help='ISO timestamp including timezone')
    parser.add_argument('--output',type=Path,default=Path('storage/extensions/final_save_v1'))
    args=parser.parse_args();root=args.output.resolve();project=Path(os.environ['VLA_PROJECT_ROOT'])
    deadline=datetime.datetime.fromisoformat(args.release_by).timestamp()
    if deadline-time.time()<30:
        raise RuntimeError('Insufficient time to perform the handoff')
    if args.resume_handoff:
        return resume_handoff(args,root,project,deadline)
    if args.launcher_pid is None:raise ValueError('Supply --launcher-pid or --resume-handoff')
    old=process(args.launcher_pid)
    if not old or old['uid']!=os.getuid() or str(project/'scripts/final_study.py') not in old['args']:
        raise RuntimeError('Expected this user\'s original final-study dispatcher')
    links=[os.readlink(p) for p in (Path('/proc')/str(old['pid'])/'fd').iterdir()]
    if str(root/'.launcher.lock') not in links:
        raise RuntimeError('Dispatcher does not hold this study lock')
    declaration=read_json(root/'declaration.json')
    # Check identities before modifying process state; the numerical code stays unchanged.
    prepare(root,project,Path(declaration['manifest']),budget_gb=declaration['budget_gb'])
    queue=launcher.jobs(root,declaration);by_id={j['id']:j for j in queue}
    slots=resolve_slots(parse_gpus(args.gpus))
    rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines()))
    inventory={uuid.strip().lower():(index.strip(),name.strip()) for index,uuid,name in rows}
    for slot in slots:
        slot['name']=inventory[slot['device'].lower()][1]
    audit=root/'handoffs'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    audit.mkdir(parents=True)
    frozen=False; replaced=False
    with (root/'.handoff.lock').open('a') as handoff_lock:
        fcntl.flock(handoff_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            send(old,signal.SIGSTOP); frozen=True
            for _ in range(100):
                if same(old)['state'] in ('T','t'):break
                time.sleep(.02)
            else:raise RuntimeError('Dispatcher did not pause')
            elapsed={x['pid']:x['seconds']+max(0,time.time()-read_json(root/'queue.json')['updated_at']) for x in read_json(root/'queue.json')['running']}
            children=(Path('/proc')/str(old['pid'])/'task'/str(old['pid'])/'children').read_text().split()
            adopted={};snapshot=[]
            for pid in children:
                child=process(int(pid))
                if not child or child['state']=='Z':continue
                if child['uid']!=os.getuid() or child['parent']!=old['pid'] or 'uq_finalstudy.worker' not in child['args']:
                    raise RuntimeError('Unexpected dispatcher child; refusing handoff')
                path=Path(child['args'][-1]).resolve()
                if path.parent.parent!=root/'jobs':raise RuntimeError('Worker belongs to another study')
                job=read_json(path)['spec']['job']
                if by_id.get(job['id'])!=job:raise RuntimeError('Worker job differs from declared queue')
                env=dict(item.split('=',1) for item in (Path('/proc')/pid/'environ').read_bytes().decode().split('\0') if '=' in item)
                gpu=inventory[env['CUDA_VISIBLE_DEVICES'].lower()][0]
                if gpu in adopted:raise RuntimeError('Multiple workers assigned to one GPU')
                adopted[gpu]=dict(process=child,job=job,started=time.monotonic()-elapsed.get(int(pid),0),
                    follower=LogFollower([path.with_name('run.log')],label=f'GPU {gpu}/{job["id"]}'))
                snapshot.append(dict(gpu=gpu,process=child,job=job['id']))
            atomic_json(audit/'handoff.json',dict(old_launcher=old,new_launcher=os.getpid(),workers=snapshot,
                allowed_gpus=args.gpus,release_by=deadline,source_hash=declaration['source_hash']))
            # Kill only the frozen dispatcher. Its session-separated workers remain alive.
            send(old,signal.SIGKILL); replaced=True
            for _ in range(100):
                if not same(old) or same(old)['state']=='Z':break
                time.sleep(.05)
            with (root/'.launcher.lock').open('a') as lock:
                acquire_lock(lock)
                def interrupt(*_):raise KeyboardInterrupt()
                signal.signal(signal.SIGTERM,interrupt)
                print(f'HANDOFF: adopted {len(adopted)} live workers; new jobs only on {args.gpus}; release deadline {args.release_by}.',flush=True)
                return run(root,declaration,queue,slots,adopted,deadline,audit)
        except BaseException as error:
            if frozen and not replaced:send(old,signal.SIGCONT)
            atomic_json(audit/'error.json',dict(error=repr(error),old_dispatcher_replaced=replaced))
            raise


if __name__=='__main__':
    sys.exit(main())
