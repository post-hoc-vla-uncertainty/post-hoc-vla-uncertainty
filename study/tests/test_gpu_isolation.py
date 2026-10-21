"""Protect physical device selection against different CUDA/EGL enumeration orders."""
import pytest

from lerobot.utils import gpu_utils

NVIDIA = {"4": "gpu-a1000004-0000-0000-0000-000000000004", "5": "gpu-a1000005-0000-0000-0000-000000000005", "6": "gpu-30900006-0000-0000-0000-000000000006", "7": "gpu-30900007-0000-0000-0000-000000000007"}
EGL = {4: "gpu-30900007-0000-0000-0000-000000000007", 5: "gpu-30900006-0000-0000-0000-000000000006", 6: "gpu-a1000004-0000-0000-0000-000000000004", 7: "gpu-a1000005-0000-0000-0000-000000000005"}


@pytest.mark.parametrize("visible,expected", [("6,7", 5), ("7", 4), ("GPU-30900007", 4)])
def test_render_device_matches_identity_not_position(visible, expected):
    index, device_uuid = gpu_utils.match_egl_device(visible, "PCI_BUS_ID", NVIDIA, EGL)
    assert index == expected
    assert device_uuid.startswith("gpu-3090")


def test_child_gpu_overrides_stale_parent_renderer(monkeypatch):
    monkeypatch.setattr(gpu_utils, "nvidia_gpu_uuids", lambda: NVIDIA)
    monkeypatch.setattr(gpu_utils, "egl_gpu_uuids", lambda: EGL)
    env = {"CUDA_VISIBLE_DEVICES": "7", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "MUJOCO_EGL_DEVICE_ID": "6"}
    mapping = gpu_utils.configure_egl_for_visible_cuda(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-" + NVIDIA["7"][4:]
    assert env["MUJOCO_EGL_DEVICE_ID"] == "4"
    assert mapping["gpu_uuid"] == "gpu-30900007-0000-0000-0000-000000000007"


@pytest.mark.parametrize("visible,order,egl", [
    ("6", "FASTEST_FIRST", EGL), ("", "PCI_BUS_ID", EGL),
    ("9", "PCI_BUS_ID", EGL), ("GPU-3090", "PCI_BUS_ID", EGL),
    ("6", "PCI_BUS_ID", {6: "gpu-a1000004-0000-0000-0000-000000000004"}),
    ("6", "PCI_BUS_ID", {0: "gpu-30900006-0000-0000-0000-000000000006", 5: "gpu-30900006-0000-0000-0000-000000000006"}),
])
def test_unresolved_devices_fail_without_falling_back(visible, order, egl):
    with pytest.raises(RuntimeError):
        gpu_utils.match_egl_device(visible, order, NVIDIA, egl)


def test_full_uuid_workers_survive_later_nvidia_smi_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_utils, 'nvidia_gpu_uuids', lambda: calls.append('query') or NVIDIA)
    monkeypatch.setattr(gpu_utils, 'egl_gpu_uuids', lambda: EGL)
    parent = {'CUDA_VISIBLE_DEVICES': '6,7', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    gpu_utils.configure_egl_for_visible_cuda(parent)
    assert calls == ['query']
    assert parent['CUDA_VISIBLE_DEVICES'].lower().split(',') == [NVIDIA['6'], NVIDIA['7']]

    def unavailable():
        pytest.fail('A UUID-pinned worker must not rediscover numeric GPU indices')
    monkeypatch.setattr(gpu_utils, 'nvidia_gpu_uuids', unavailable)
    # EGL may re-enumerate: recheck identities instead of retaining old indices.
    monkeypatch.setattr(gpu_utils, 'egl_gpu_uuids', lambda: {0: NVIDIA['7'], 3: NVIDIA['6']})
    for value, expected in zip(parent['CUDA_VISIBLE_DEVICES'].split(','), [3, 0]):
        child = dict(parent, CUDA_VISIBLE_DEVICES=value)
        gpu_utils.configure_egl_for_visible_cuda(child)
        assert int(child['MUJOCO_EGL_DEVICE_ID']) == expected


def test_all_requested_gpus_validated_before_environment_changes(monkeypatch):
    monkeypatch.setattr(gpu_utils, 'nvidia_gpu_uuids', lambda: NVIDIA)
    monkeypatch.setattr(gpu_utils, 'egl_gpu_uuids', lambda: {5: NVIDIA['6']})
    env = {'CUDA_VISIBLE_DEVICES': '6,7', 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
    before = env.copy()
    with pytest.raises(RuntimeError, match='refusing a renderer fallback'):
        gpu_utils.configure_egl_for_visible_cuda(env)
    assert env == before


def test_startup_discovery_retries_exit_six(monkeypatch):
    import subprocess
    attempts = []
    sleeps = []
    def query(*args, **kwargs):
        attempts.append(args)
        if len(attempts) < 3:
            raise subprocess.CalledProcessError(6, args[0], output='No devices were found')
        return '6, ' + NVIDIA['6'] + '\n7, ' + NVIDIA['7'] + '\n'
    monkeypatch.setattr(gpu_utils.subprocess, 'check_output', query)
    monkeypatch.setattr(gpu_utils.time, 'sleep', sleeps.append)
    assert gpu_utils.nvidia_gpu_uuids() == {'6': NVIDIA['6'], '7': NVIDIA['7']}
    assert len(attempts) == 3 and sleeps == [1, 1]


def test_persistent_discovery_failure_reports_access_and_preserves_checkpoints(monkeypatch):
    import subprocess
    def query(*args, **kwargs):
        raise subprocess.CalledProcessError(6, args[0], output='No devices were found', stderr='driver diagnostic')
    monkeypatch.setattr(gpu_utils.subprocess, 'check_output', query)
    monkeypatch.setattr(gpu_utils.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError, match='after 3 attempts') as error:
        gpu_utils.nvidia_gpu_uuids()
    assert 'No devices were found' in str(error.value)
    assert 'driver diagnostic' in str(error.value)
    assert 'Existing checkpoints are preserved' in str(error.value)


def test_duplicate_uuid_aliases_rejected(monkeypatch):
    monkeypatch.setattr(gpu_utils, 'nvidia_gpu_uuids', lambda: NVIDIA)
    with pytest.raises(RuntimeError, match='duplicate'):
        gpu_utils.resolve_cuda_uuids('6,' + NVIDIA['6'], 'PCI_BUS_ID')
