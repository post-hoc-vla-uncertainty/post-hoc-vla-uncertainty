"""One subprocess per selected GPU, with dynamic dispatch and dependency checks."""
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def parse_gpus(value):
    ids=[part.strip() for part in value.split(',')]
    ids=[str(int(part)) if part.isdigit() else part for part in ids]
    if not all(ids):raise ValueError('GPU selections cannot be empty')
    if len({part.lower() for part in ids})!=len(ids):raise ValueError('Duplicate GPU selections')
    if 'cpu' in ids and ids!=['cpu']:raise ValueError('CPU cannot be mixed with GPU selections')
    if any(part!='cpu' and not part.isdigit() and not part.lower().startswith('gpu-') for part in ids):
        raise ValueError('Use physical NVIDIA GPU indices or GPU UUIDs')
    return ids


def resolve_slots(ids):
    if ids==['cpu']:return [dict(label='cpu',device='cpu',environment={'CUDA_VISIBLE_DEVICES':''})]
    from lerobot.utils.gpu_utils import resolve_cuda_uuids,configure_egl_for_visible_cuda
    # Resolve numeric selections once, before launching any work. Children get full UUIDs.
    uuids=resolve_cuda_uuids(','.join(ids),'PCI_BUS_ID')
    slots=[]
    for label,uuid in zip(ids,uuids):
        env=dict(CUDA_VISIBLE_DEVICES=uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID')
        mapping=configure_egl_for_visible_cuda(env)
        slots.append(dict(label=label,device=env['CUDA_VISIBLE_DEVICES'],environment=env))
        print(f'GPU {label}: compute={env["CUDA_VISIBLE_DEVICES"]}; EGL={mapping["egl_device_index"]}',flush=True)
    return slots


def assign_model_roles(slots):
    """Restrict only the explicitly selected GPUs, without opening CUDA contexts."""
    import csv
    output=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    inventory={uuid.strip().lower():name.strip() for uuid,name in csv.reader(output.splitlines())}
    assigned=[]
    for slot in slots:
        name=inventory.get(slot['device'].lower())
        if not name:raise ValueError(f'Cannot identify selected GPU {slot["label"]}; refusing a routing fallback')
        role='kv_t' if 'A100' in name.upper() else 'other' if 'RTX 3090' in name.upper() else None
        if role is None:raise ValueError(f'Model routing supports A100 and RTX 3090, got {name}')
        assigned.append(dict(slot,name=name,role=role))
        print(f'GPU {slot["label"]}: {name}; assigned to {"KV-T only" if role=="kv_t" else "KV and paper baselines only"}',flush=True)
    return assigned


def slot_accepts(job,slot):
    required='kv_t' if job['spec']['recipe'].get('method')=='kv_t' else 'other'
    return slot.get('role',required)==required


def validate_slots(jobs,slots):
    for job in jobs:
        if not any(slot_accepts(job,slot) for slot in slots):
            kind='A100 for KV-T' if job['spec']['recipe'].get('method')=='kv_t' else 'RTX 3090 for KV/paper heads'
            raise ValueError(f'No eligible selected GPU: need {kind}. Change --gpus; no work was launched.')


@dataclass
class RunningHead:
    process: object
    follower: object
    job: dict
    slot: dict
    log_path: Path
    started: float

    def poll(self):
        status=self.process.poll()
        if self.follower:self.follower.poll(heartbeat=status is None,final=status is not None)
        return status

    def send_signal(self,signum):
        if self.process.poll() is None:
            try:os.killpg(self.process.pid,signum)
            except ProcessLookupError:pass


def start_child(job,slot):
    if not slot_accepts(job,slot):raise ValueError('GPU model does not match the head routing rule')
    from run_progress import LogFollower
    spec=job['spec'];out=Path(spec['directory']);log_path=out/'run.log'
    log_path.parent.mkdir(parents=True,exist_ok=True)
    label=f'GPU {slot["label"]} / round {spec["round"]} / {spec["recipe"]["id"]}'
    follower=LogFollower([log_path],label=label)
    env=dict(os.environ,**slot['environment'],PYTHONUNBUFFERED='1')
    # Children own isolated process groups, so Ctrl-C in the launcher can checkpoint
    # and stop every worker before releasing the study lock.
    with log_path.open('a') as log:
        process=subprocess.Popen([sys.executable,'-u','-m','uq_extensions.head_study',
            '--worker-request',str(out/'request.json'),'--gpu',slot['device']],
            stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    print(f'START {label}; pid={process.pid}; log={log_path}',flush=True)
    return RunningHead(process,follower,job,slot,log_path,time.monotonic())


def stop_workers(workers,timeout=30):
    workers=list(workers)
    for worker in workers:worker.send_signal(signal.SIGTERM)
    deadline=time.monotonic()+timeout
    while any(worker.poll() is None for worker in workers) and time.monotonic()<deadline:
        time.sleep(.1)
    for worker in workers:
        if worker.process.poll() is None:worker.send_signal(signal.SIGKILL)
    for worker in workers:worker.process.wait()


def execution_phase(job):
    """Scheduling only: keep scientific recipe/request identities unchanged."""
    row=job['spec']['recipe']
    if row.get('scope')!='action':return 0
    return 2 if row.get('method')=='kv_t' else 1


def run_queue(jobs,slots,*,ready,on_complete,on_update,on_failure=None,start=start_child,poll_seconds=.5):
    """Drain every phase despite individual failures; cancellation still stops all workers.

    A failed head is attempted once per invocation. Conditional heads whose
    predecessor failed are blocked, never recorded as successfully completed.
    """
    pending=sorted(jobs,key=execution_phase);running={};failures=[];failed_keys=set()
    if pending and not slots:raise ValueError('Pending heads require at least one worker slot')
    validate_slots(pending,slots)
    def failed(job,reason,*,worker=None,slot=None,exit_code=None,status='failed'):
        slot=worker.slot if worker is not None else slot
        event=dict(status=status,reason=reason,exit_code=exit_code,
            gpu=slot['label'] if slot else None,
            log=str(worker.log_path) if worker is not None else str(Path(job['spec']['directory'])/'run.log'),
            seconds=time.monotonic()-worker.started if worker is not None else 0.)
        failures.append(dict(job=job,**event))
        failed_keys.add((job['spec']['round'],job['spec']['recipe']['id']))
        if on_failure:on_failure(job,event)
        print(f'{status.upper()} round {job["spec"]["round"]}, {job["spec"]["recipe"]["id"]}: '
              f'{reason}\nContinuing with other heads. Log: {event["log"]}',flush=True)
    try:
        while pending or running:
            changed=False
            for label,worker in list(running.items()):
                status=worker.poll()
                if status is None:continue
                if status:
                    from run_progress import tail_text
                    try:detail=tail_text(worker.log_path)
                    except OSError:detail=''
                    failed(worker.job,f'Worker exited with status {status}.\n{detail}',worker=worker,exit_code=status)
                else:
                    try:on_complete(worker.job)
                    except Exception as error:
                        failed(worker.job,f'Completion/report error: {type(error).__name__}: {error}',worker=worker)
                    else:
                        print(f'DONE GPU {label}: round {worker.job["spec"]["round"]}, '
                            f'{worker.job["spec"]["recipe"]["id"]}; wall={time.monotonic()-worker.started:.1f}s',flush=True)
                del running[label];changed=True
            for slot in slots:
                if slot['label'] in running:continue
                phase=min([execution_phase(j) for j in pending]+
                          [execution_phase(w.job) for w in running.values()],default=0)
                for job in list(pending):
                    if execution_phase(job)!=phase:continue
                    if not slot_accepts(job,slot):continue
                    try:
                        state=ready(job)
                        if state not in ('ready','wait','skip'):raise ValueError(f'Invalid queue readiness: {state}')
                    except Exception as error:
                        pending.remove(job);changed=True
                        failed(job,f'Readiness/report error: {type(error).__name__}: {error}',slot=slot)
                        continue
                    if state=='wait':
                        predecessor=job['spec']['recipe'].get('conditional_on')
                        if (job['spec']['round'],predecessor) in failed_keys:
                            pending.remove(job);changed=True
                            failed(job,f'Prerequisite {predecessor} failed; retry after it succeeds.',status='blocked')
                        continue
                    pending.remove(job);changed=True
                    if state=='skip':continue
                    try:running[slot['label']]=start(job,slot)
                    except Exception as error:
                        failed(job,f'Worker startup error: {type(error).__name__}: {error}',slot=slot)
                        continue
                    break
            if pending and not running:
                # Only block unresolved jobs in the current phase. Later phases
                # remain runnable once these dependencies have been accounted for.
                phase=min(execution_phase(j) for j in pending)
                stalled=[j for j in pending if execution_phase(j)==phase]
                # A phase may just have completed via skips/startup failures.
                # In that case the next phase has not yet had a dispatch attempt.
                if not changed:
                    for job in stalled:
                        pending.remove(job)
                        failed(job,'Required predecessor result is unavailable; retry with its prerequisite.',status='blocked')
                    changed=True
            if changed:on_update(list(running.values()),pending)
            if running:time.sleep(poll_seconds)
    finally:
        stop_workers(running.values())
    return failures
