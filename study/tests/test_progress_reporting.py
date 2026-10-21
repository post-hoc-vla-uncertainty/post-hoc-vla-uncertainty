"""Reporting must be live, bounded, resumable and independent of training state."""
from pathlib import Path
from types import SimpleNamespace
import os
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_progress import Progress,TrainingProgress,LogFollower,strip_reporting_blocks


def test_training_epoch_averages_do_not_reset_tracker_or_change_rng(capsys,monkeypatch):
    from lerobot.utils.logging_utils import AverageMeter,MetricsTracker
    monkeypatch.setenv('VLA_PROGRESS_STEPS','100')
    tracker=MetricsTracker(2,4,2,{k:AverageMeter(k) for k in ('loss','grad_norm','lr','update_s','dataloading_s')})
    before=random.getstate();p=TrainingProgress('member 1',3,tracker)
    for loss in (2.,4.,8.):
        for k,v in dict(loss=loss,grad_norm=.2,lr=.001,update_s=.1,dataloading_s=.01).items():setattr(tracker,k,v)
        tracker.step();p.record(tracker)
    p.finish('saved')
    output=capsys.readouterr().out
    assert 'EPOCH 1 DONE' in output and 'loss=3 ' in output
    assert 'FINAL PARTIAL EPOCH 1.500' in output and '3/3 updates' in output
    assert tracker.loss.count==3 and tracker.loss.avg==14/3
    assert random.getstate()==before


def test_resume_eta_excludes_work_from_previous_process(capsys,monkeypatch):
    clock=[100.];monkeypatch.setattr('run_progress.time.monotonic',lambda:clock[0])
    p=Progress('resume',20,initial=10,every=1);clock[0]=120.
    p.update(12,'loss=.5')
    assert 'wall=00:00:20 ETA=00:01:20' in capsys.readouterr().out
    p.update(12)
    assert capsys.readouterr().out==''


def test_follow_partial_lines_new_nested_logs_truncation_and_heartbeat(tmp_path,capsys,monkeypatch):
    clock=[0.];monkeypatch.setattr('run_progress.time.monotonic',lambda:clock[0])
    log=tmp_path/'parent.log';log.write_text('old output\n')
    follower=LogFollower([log],[tmp_path/'rounds'],seconds=5)
    with log.open('a') as f:f.write('[progress] half')
    follower.poll();assert capsys.readouterr().out==''
    with log.open('a') as f:f.write(' line\n')
    nested=tmp_path/'rounds/round_001/member_01/training.log';nested.parent.mkdir(parents=True)
    nested.write_text('[progress] loss=0.42\n')
    follower.poll();text=capsys.readouterr().out
    assert 'half line' in text and 'loss=0.42' in text and 'round_001/member_01/training.log' in text
    log.write_text('[progress] restart\n');follower.poll()
    assert 'restart' in capsys.readouterr().out
    clock[0]=6;follower.poll();assert 'RUNNING wall=00:00:06' in capsys.readouterr().out


def test_launcher_relays_output_before_process_exit_and_keeps_complete_log(tmp_path,capsys,monkeypatch):
    import reproduce
    log=tmp_path/'job.log'
    release=tmp_path/'release'
    process=reproduce.launch([sys.executable,'-u','-c',
        'import time,sys;from pathlib import Path\nprint("[progress] loss=0.25",flush=True)\nwhile not Path(sys.argv[1]).exists():time.sleep(.01)\nprint("[progress] finished")',str(release)],log)
    try:
        deadline=time.monotonic()+5
        while '[progress] loss' not in log.read_text() and time.monotonic()<deadline:time.sleep(.01)
        process.progress_follower.poll()
        assert 'loss=0.25' in capsys.readouterr().out and process.poll() is None
    finally:release.touch()
    reproduce.wait_all([(process,log)])
    assert 'finished' in capsys.readouterr().out
    assert 'loss=0.25' in log.read_text() and '[progress] finished' in log.read_text()


def test_reporting_normalization_keeps_algorithm_changes_visible():
    original='def f():\n    train()\n'
    logged='def f():\n    # BEGIN RUN PROGRESS\n    print("progress")\n    # END RUN PROGRESS\n    train()\n'
    assert strip_reporting_blocks(logged)==original
    assert strip_reporting_blocks(logged.replace('train()','changed_train()'))!=original
