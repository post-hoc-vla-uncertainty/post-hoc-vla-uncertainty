"""The handoff must never interrupt the two workers or signal a recycled PID."""
from copy import deepcopy
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import monitor_selected_head_handoff as handoff


def test_process_identity_includes_kernel_start_time():
    current = handoff.process(os.getpid())
    assert current['uid'] == os.getuid()
    assert handoff.same_process(current)['pid'] == os.getpid()
    assert handoff.same_process(dict(current, start='not-the-same-process')) is None


def test_signal_uses_pidfd_and_rejects_changed_identity(monkeypatch):
    events = []
    monkeypatch.setattr(handoff.os, 'pidfd_open', lambda pid: 123)
    monkeypatch.setattr(handoff.os, 'close', lambda fd: events.append(('close', fd)))
    monkeypatch.setattr(handoff.signal, 'pidfd_send_signal', lambda *args: events.append(args))
    monkeypatch.setattr(handoff, 'same_process', lambda _: None)
    with pytest.raises(RuntimeError, match='identity changed'):
        handoff.signal_process(dict(pid=456), signal.SIGTERM)
    assert events == [('close', 123)]


@pytest.mark.parametrize('unexpected_child', [False, True])
def test_handoff_drains_workers_before_terminating_parent_and_launching(tmp_path, monkeypatch, unexpected_child):
    launcher = dict(pid=101, state='S', start='one', uid=os.getuid())
    workers = [dict(pid=102, state='R', start='two', uid=os.getuid()),
               dict(pid=103, state='R', start='three', uid=os.getuid())]
    processes = {p['pid']: p for p in [launcher, *workers]}
    events = []; polling = [0]
    monkeypatch.setattr(handoff, 'process', lambda pid: deepcopy(processes.get(pid)))
    monkeypatch.setattr(handoff, 'same_process', lambda p: deepcopy(processes.get(p['pid'])))
    monkeypatch.setattr(handoff, 'validate', lambda *args: None)
    def send(p, sig):
        assert p['pid'] == launcher['pid'], 'Never signal either worker'
        if sig == signal.SIGTERM:
            assert all(w['state'] == 'Z' for w in workers)
        launcher['state'] = {signal.SIGSTOP: 'T', signal.SIGTERM: 'Z', signal.SIGCONT: 'S'}[sig]
        if sig == signal.SIGCONT and any(e == ('signal', signal.SIGTERM) for e in events):
            processes.pop(launcher['pid'])
        events.append(('signal', sig))
    monkeypatch.setattr(handoff, 'signal_process', send)
    def sleep(_):
        polling[0] += 1
        if polling[0] == 1: workers[0]['state'] = 'Z'
        elif polling[0] == 2: workers[1]['state'] = 'Z'
    monkeypatch.setattr(handoff.time, 'sleep', sleep)
    original_read = Path.read_text
    def read(path, *args, **kwargs):
        if str(path) == '/proc/101/task/101/children':
            return '102 103 104' if unexpected_child else '102 103'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(handoff, 'artifacts', lambda *args: [dict(saved=True, report_saved=True)])
    class Child:
        pid = 104
        returncode = 0
        def __init__(self, command, **kwargs):
            assert launcher['pid'] not in processes
            assert command[-4:] == ['--gpus', '4,5', '--root', str(tmp_path)]
            events.append(('launch', command))
        def poll(self): return 0
    monkeypatch.setattr(handoff.subprocess, 'Popen', Child)
    args = SimpleNamespace(root=tmp_path, launcher_pid=101, workers=[102, 103], gpus='4,5')
    if unexpected_child:
        with pytest.raises(RuntimeError, match='children changed'):
            handoff.run(args)
        assert events == [('signal', signal.SIGSTOP), ('signal', signal.SIGCONT)]
    else:
        assert handoff.run(args) == 0
        assert events[:3] == [('signal', signal.SIGSTOP), ('signal', signal.SIGTERM), ('signal', signal.SIGCONT)]
        assert events[3][0] == 'launch' and polling[0] == 2
