"""Stdlib-only progress reporting; no model, optimizer, sampler or RNG changes."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import re
import time


def duration(seconds):
    seconds=max(0,int(seconds));hours,seconds=divmod(seconds,3600);minutes,seconds=divmod(seconds,60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def emit(label,message):
    stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f'[{stamp}] [progress] [{label}] {message}',flush=True)


def strip_reporting_blocks(source):
    """Keep existing scientific/cache identities across marked reporting additions."""
    return re.sub(r'(?ms)^[ \t]*# BEGIN RUN PROGRESS\n.*?^[ \t]*# END RUN PROGRESS\n','',source)


def scientific_file_hash(path):
    return hashlib.sha256(strip_reporting_blocks(Path(path).read_text()).encode()).hexdigest()


class Progress:
    def __init__(self,label,total=None,*,initial=0,unit='steps',every=None,seconds=None):
        self.label,self.total,self.initial,self.unit=label,total,initial,unit
        self.every=int(os.environ.get('VLA_PROGRESS_STEPS','100')) if every is None else every
        self.seconds=float(os.environ.get('VLA_PROGRESS_SECONDS','60')) if seconds is None else seconds
        self.started=self.last=time.monotonic();self.count=initial;self.last_count=initial;self.finished=False
        self.latest=''
        emit(label,f'START {initial}/{total} {unit}' if total is not None else 'START')

    def update(self,count=None,detail='',*,force=False):
        self.count=self.count+1 if count is None else count
        now=time.monotonic()
        advanced=self.count!=self.last_count
        if force or (advanced and (self.count==self.initial+1 or (self.total is not None and self.count>=self.total) or self.count-self.last_count>=self.every)) or now-self.last>=self.seconds:
            elapsed=now-self.started;done=self.count-self.initial
            eta=duration(elapsed*(self.total-self.count)/done) if self.total is not None and done>0 else '?'
            position=f'{self.count}/{self.total} {self.unit}' if self.total is not None else f'{self.count} {self.unit}'
            if self.total:position+=f' ({100*self.count/self.total:.1f}%)'
            self.latest=detail
            emit(self.label,f'{position} | wall={duration(elapsed)} ETA={eta}'+(f' | {detail}' if detail else ''))
            self.last=now;self.last_count=self.count
            return True
        return False

    def finish(self,detail=''):
        if not self.finished:
            emit(self.label,f'DONE wall={duration(time.monotonic()-self.started)} | {detail or self.latest}')
            self.finished=True


class TrainingProgress(Progress):
    """Independent accumulators: never reset LeRobot/W&B's own metric windows."""
    def __init__(self,label,total,tracker):
        super().__init__(label,total,initial=tracker.steps)
        self.epoch=math.floor(tracker.epochs);self.partial_epoch=tracker.epochs%1>1e-9
        self.window=defaultdict(list);self.epoch_values=defaultdict(list);self.all_values=defaultdict(list)
        self.epoch_started=self.started;self.last_metrics={}
        emit(label,f'epoch means sampled-frame equivalents; resume step={tracker.steps}, epoch={tracker.epochs:.3f}')

    @staticmethod
    def metrics_text(values):
        names=(('loss','loss'),('grad_norm','grad'),('lr','lr'),('update_s','update_s'),('dataloading_s','data_s'))
        return ' '.join(f'{label}={(items[-1] if name=="lr" else sum(items)/len(items)):.5g}'
            for name,label in names if (items:=values.get(name)))

    def record(self,tracker):
        metrics=tracker.to_dict(use_avg=False);self.last_metrics=metrics
        for name in ('loss','grad_norm','lr','update_s','dataloading_s'):
            value=float(metrics[name])
            for target in (self.window,self.epoch_values,self.all_values):target[name].append(value)
        if self.update(tracker.steps,f'epoch={tracker.epochs:.3f} | '+self.metrics_text(self.window)):
            self.window.clear()
        epoch=math.floor(tracker.epochs)
        if epoch>self.epoch:
            label=f'{self.epoch+1}' if epoch==self.epoch+1 else f'{self.epoch+1}..{epoch} (one batch crossed multiple boundaries)'
            note='partial since resume; ' if self.partial_epoch else ''
            emit(self.label,f'EPOCH {label} DONE | {note}wall={duration(time.monotonic()-self.epoch_started)} | '+self.metrics_text(self.epoch_values))
            self.epoch=epoch;self.partial_epoch=False;self.epoch_values.clear();self.epoch_started=time.monotonic()

    def finish(self,detail=''):
        if self.epoch_values:
            emit(self.label,f'FINAL PARTIAL EPOCH {self.last_metrics.get("epochs",0):.3f} | '+self.metrics_text(self.epoch_values))
            self.epoch_values.clear()
        super().finish(f'{self.count}/{self.total} updates | segment means: {self.metrics_text(self.all_values)} | {detail}')


ANSI=re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


class LogFollower:
    """Follow append/truncate/replace and newly created nested worker logs."""
    def __init__(self,paths=(),roots=(),*,label='job',replay=0,seconds=None):
        self.paths=[Path(p) for p in paths];self.roots=[Path(p) for p in roots]
        self.label=label;self.states={};self.started=self.last_heartbeat=time.monotonic();self.last_activity=self.started
        self.seconds=float(os.environ.get('VLA_PROGRESS_SECONDS','60')) if seconds is None else seconds
        self.initial_replay=replay;self.latest='waiting for worker output';self.last_legacy={};self.structured=set()
        self.tasks={}
        files=self.files()
        recent=set(sorted(files,key=lambda p:p.stat().st_mtime if p.exists() else 0)[-6:]) if replay else set()
        for path in files:self.register(path,replay if path in recent else 0)

    def files(self):
        paths=set(self.paths)
        for root in self.roots:
            if root.exists():paths.update(root.rglob('*.log'))
        return sorted(paths)

    def name(self,path):
        for root in self.roots:
            try:return str(path.relative_to(root))
            except ValueError:pass
        return path.name

    def display(self,path,line):
        line=ANSI.sub('',line).strip()
        if not line:return
        label=self.name(path)
        task=re.match(r'(libero_\w+/\d+):\s*\d+%',line)
        if task:
            if self.tasks.get(path)!=task[1]:
                self.tasks[path]=task[1];emit(f'{self.label}/{label}',f'Task {task[1]} started (legacy log)');self.latest=f'{label}: task {task[1]}'
            return
        if '[progress]' in line:
            self.structured.add(path)
            message=line.split('[progress]',1)[-1].strip()
            emit(f'{self.label}/{label}',message);self.latest=f'{label}: {message}';return
        legacy=bool(re.search(r'\bstep:\s*\d',line))
        if legacy:
            if path in self.structured:return
            now=time.monotonic()
            if now-self.last_legacy.get(path,-float('inf'))<min(30,self.seconds):return
            self.last_legacy[path]=now
        useful=legacy or bool(re.search(r'\b(ERROR|WARNING|Traceback|Exception|Error:)|Creating (training )?dataset|Loading round-start|Launching |Selecting episodes|Training round|Iterative round training|Frames per epoch|Sampling weights|Saving checkpoint|Checkpoint policy|End of training|Finished |Evaluating |evaluation.*(success|rollout)|success_rate|Saved .*summary|Start offline',line,re.I))
        if useful and not line.startswith(("'",'"')):
            emit(f'{self.label}/{label}',line);self.latest=f'{label}: {line[-200:]}'

    def register(self,path,replay=0):
        if not path.exists():return
        with path.open('rb') as f:
            stat=os.fstat(f.fileno());f.seek(max(0,stat.st_size-65536))
            tail=f.read().decode('utf-8',errors='replace');offset=f.tell()
        self.states[path]={'inode':stat.st_ino,'offset':offset,'buffer':''}
        if replay:
            for line in tail.replace('\r','\n').splitlines()[-replay:]:self.display(path,line)

    def poll(self,*,heartbeat=True,final=False):
        for path in self.files():
            if not path.exists():continue
            if path not in self.states:
                self.states[path]={'inode':path.stat().st_ino,'offset':0,'buffer':''}
                emit(self.label,f'Worker log: {self.name(path)}')
            state=self.states[path]
            try:
                with path.open('rb') as f:
                    stat=os.fstat(f.fileno())
                    if stat.st_ino!=state['inode'] or stat.st_size<state['offset']:
                        state.update(inode=stat.st_ino,offset=0,buffer='');self.structured.discard(path)
                    f.seek(state['offset']);data=f.read(1024*1024);state['offset']=f.tell()
            except FileNotFoundError:continue
            if data:self.last_activity=time.monotonic()
            lines=(state['buffer']+data.decode('utf-8',errors='replace').replace('\r','\n')).split('\n')
            state['buffer']=lines.pop()
            if final and state['buffer']:lines.append(state['buffer']);state['buffer']=''
            for line in lines:self.display(path,line)
        if heartbeat and time.monotonic()-self.last_heartbeat>=self.seconds:
            emit(self.label,f'RUNNING wall={duration(time.monotonic()-self.started)} | last log write {duration(time.monotonic()-self.last_activity)} ago | {self.latest}')
            self.last_heartbeat=time.monotonic()


def tail_text(path,lines=30):
    path=Path(path)
    if not path.exists():return '(no log file)'
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-65536));return '\n'.join(f.read().decode('utf-8',errors='replace').splitlines()[-lines:])
