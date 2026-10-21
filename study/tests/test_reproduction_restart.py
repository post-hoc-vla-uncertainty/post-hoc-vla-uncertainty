"""Continue completed pretraining pairs with UUID-pinned worker environments."""
from pathlib import Path
import sys
from types import SimpleNamespace
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import reproduce
from lerobot.utils import gpu_utils


def test_pretraining_skips_completed_pair_and_uses_uuid_workers(tmp_path, monkeypatch):
    configs = tmp_path / 'configs'
    configs.mkdir()
    for seed in range(6):
        output = tmp_path / f'seed_{seed}'
        (configs / f'pretrain_seed_{seed}.json').write_text(json.dumps(dict(output_dir=str(output), steps=30000)))
        if seed < 2:
            state = output / 'checkpoints/last/training_state'
            state.mkdir(parents=True)
            (state / 'training_step.json').write_text('{"step": 30000}')
    monkeypatch.setattr(reproduce, 'CONFIGS', configs)
    uuids = ['GPU-30900006-0000-0000-0000-000000000006', 'GPU-30900007-0000-0000-0000-000000000007']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', ','.join(uuids))
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    monkeypatch.setattr(gpu_utils, 'egl_gpu_uuids', lambda: {5: uuids[0].lower(), 4: uuids[1].lower()})
    def no_rediscovery():
        raise AssertionError('Numeric discovery must not run between training pairs')
    monkeypatch.setattr(gpu_utils, 'nvidia_gpu_uuids', no_rediscovery)
    launched = []
    def launch(cmd, log, **kwargs):
        launched.append((log.name, kwargs['env'].copy()))
        return SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(reproduce, 'launch', launch)
    policy = SimpleNamespace(maintain_pretrain=lambda *a, **kw: None, check_budget=lambda: None)
    args = SimpleNamespace(seeds=[1000,1001,1002], dry_run=False, gpu_ids=uuids, storage_policy=policy)
    reproduce.pretrain(args, tmp_path, tmp_path / 'logs')
    assert [name for name, _ in launched] == [f'pretrain_seed_{i}.log' for i in range(2,6)]
    assert [env['CUDA_VISIBLE_DEVICES'] for _, env in launched] == uuids * 2
    assert [env['MUJOCO_EGL_DEVICE_ID'] for _, env in launched] == ['5','4','5','4']
