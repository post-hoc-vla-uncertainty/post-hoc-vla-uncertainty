#!/usr/bin/env python3
"""Drain two authorized workers, replace their dispatcher, and supervise the finish queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from finish_selected_heads import VELOCITY, FINAL_ACTION


def process(pid):
    directory = Path('/proc')/str(pid)
    try:
        # comm can contain spaces or parentheses; fields after its last ')' are stable.
        fields = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        args = [a.decode() for a in (directory/'cmdline').read_bytes().split(b'\0') if a]
        return dict(pid=pid, start=fields[19], state=fields[0], parent=int(fields[1]),
                    uid=directory.stat().st_uid, args=args)
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(saved):
    current = process(saved['pid'])
    return current if current and current['start'] == saved['start'] and current['uid'] == saved['uid'] else None


def signal_process(saved, signum):
    # Pin the kernel process object so PID recycling cannot redirect the signal.
    fd = os.pidfd_open(saved['pid'])
    try:
        if not same_process(saved):
            raise RuntimeError(f'Process identity changed: {saved["pid"]}')
        signal.pidfd_send_signal(fd, signum)
    finally:
        os.close(fd)


def write_status(path, **record):
    record.update(monitor_pid=os.getpid(), updated_at=time.time())
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(record, indent=2)+'\n')
    os.replace(temporary, path)


def validate(launcher, workers, root):
    if not launcher or launcher['uid'] != os.getuid() or launcher['state'] in ('T', 't', 'Z'):
        raise RuntimeError('Expected this user\'s active launcher')
    args = launcher['args']
    if 'uq_extensions.head_study' not in args or '--worker-request' in args:
        raise RuntimeError('PID is not the broad head-study dispatcher')
    lock = str((root/'study.lock').resolve())
    if not any(os.readlink(p) == lock for p in (Path('/proc')/str(launcher['pid'])/'fd').iterdir()):
        raise RuntimeError('Dispatcher does not own this study lock file')
    rounds = set()
    for worker in workers:
        if not worker or worker['uid'] != os.getuid() or worker['parent'] != launcher['pid']:
            raise RuntimeError('Worker ownership or parent differs')
        args = worker['args']
        if 'uq_extensions.head_study' not in args or '--worker-request' not in args:
            raise RuntimeError('PID is not a head-study worker')
        request_path = Path(args[args.index('--worker-request')+1]).resolve()
        request = json.loads(request_path.read_text())
        spec = request['spec']; rd = spec['round']
        if (Path(spec['root']).resolve() != root or spec['recipe']['id'] != VELOCITY
                or request_path != root/f'round_{rd:03d}'/VELOCITY/'request.json'):
            raise RuntimeError('Worker is not one of the two authorized velocity fits')
        rounds.add(rd)
    if len(workers) != 2 or rounds != {0, 10}:
        raise RuntimeError('Expected exactly the round-0 and round-10 fits')


def artifacts(root, name):
    records = []
    for rd in (0, 10):
        directory = root/f'round_{rd:03d}'/name
        path = directory/'result.json'
        record = json.loads(path.read_text()) if path.exists() else {}
        request = json.loads((directory/'request.json').read_text())
        valid = bool(record and record.get('request_identity') == request['identity'])
        records.append(dict(round=rd, status=record.get('status', 'unfinished'),
                            saved=valid, report_saved=(directory/'report/complete.json').exists()))
    return records


def run(args):
    root = args.root.resolve(); status_path = root/'selected_handoff.json'
    launcher = process(args.launcher_pid); workers = [process(pid) for pid in args.workers]
    validate(launcher, workers, root)
    paused = False
    def record(stage, **extra):
        write_status(status_path, stage=stage, launcher_pid=launcher['pid'],
                     worker_pids=[w['pid'] for w in workers], **extra)
    try:
        signal_process(launcher, signal.SIGSTOP); paused = True
        # Stop only the dispatcher PID. Child process groups continue computing.
        for _ in range(50):
            current = same_process(launcher)
            if current and current['state'] in ('T', 't'): break
            time.sleep(.1)
        else: raise RuntimeError('Dispatcher did not pause')
        children_path = Path('/proc')/str(launcher['pid'])/'task'/str(launcher['pid'])/'children'
        children = {int(pid) for pid in children_path.read_text().split()}
        if children != {w['pid'] for w in workers}:
            raise RuntimeError('Dispatcher children changed; refusing an ambiguous handoff')
        print('Dispatcher paused; both existing GPU workers continue uninterrupted.', flush=True)
        while True:
            states = [same_process(w) for w in workers]
            active = [w['pid'] for w in states if w and w['state'] != 'Z']
            current_artifacts = artifacts(root, VELOCITY)
            record('waiting_for_velocity', active_workers=active, velocity=current_artifacts)
            if not active: break
            time.sleep(10)
        print('Both velocity workers exited. Closing the broad dispatcher.', flush=True)
        # Queue termination before resuming it, so it cannot dispatch another job.
        signal_process(launcher, signal.SIGTERM)
        signal_process(launcher, signal.SIGCONT); paused = False
        deadline = time.monotonic()+180
        while True:
            current = same_process(launcher)
            if not current or current['state'] == 'Z': break
            if time.monotonic() > deadline:
                raise RuntimeError('Broad dispatcher has not exited; refusing overlapping launch')
            time.sleep(1)
        with (root/'study.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        project = Path(__file__).resolve().parents[1]
        log_path = root/'selected_finish.log'
        command = ['bash', str(project/'scripts/experiments/08_finish_selected_heads.sh'),
                   '--gpus', args.gpus, '--root', str(root)]
        with log_path.open('a') as log:
            child = subprocess.Popen(command, cwd=project, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Restricted finish queue started: PID {child.pid}; log={log_path}', flush=True)
        while child.poll() is None:
            record('running_selected', selected_pid=child.pid, log=str(log_path),
                   final_action=artifacts(root, FINAL_ACTION))
            time.sleep(10)
        final = artifacts(root, FINAL_ACTION)
        complete = child.returncode == 0 and all(r['saved'] and r['report_saved'] for r in final)
        record('complete' if complete else 'selected_failed', selected_pid=child.pid,
               exit_code=child.returncode, log=str(log_path), final_action=final)
        print(f'Selected queue exited {child.returncode}; handoff complete={complete}.', flush=True)
        return 0 if complete else 1
    except BaseException as error:
        if paused and same_process(launcher):
            signal_process(launcher, signal.SIGCONT)
        record('monitor_failed', error=f'{type(error).__name__}: {error}', dispatcher_resumed=paused)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('storage/extensions/head_ablation_v1'))
    parser.add_argument('--launcher-pid', type=int, required=True)
    parser.add_argument('--workers', type=int, nargs=2, required=True)
    parser.add_argument('--gpus', default='4,5')
    args = parser.parse_args()
    def interrupted(*_): raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    with (args.root/'selected_handoff.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == '__main__':
    sys.exit(main())
