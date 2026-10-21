from pathlib import Path
import os
import pytest
from uq_extensions.storage import owned_checkpoint,check_budget
from uq_extensions.common import atomic_json


def test_storage_never_owns_reproduction_or_symlinks(tmp_path,monkeypatch):
    root=tmp_path/'extension';root.mkdir();monkeypatch.setenv('UQ_EXT_RUN_ROOT',str(root))
    accepted=root/'round_001/training/member_00/checkpoints/000001'
    accepted.mkdir(parents=True)
    assert owned_checkpoint(accepted)==accepted
    with pytest.raises(RuntimeError):owned_checkpoint(tmp_path/'reproduction/checkpoints/000001')
    alias=root/'alias';alias.symlink_to(accepted.parent,target_is_directory=True)
    with pytest.raises(RuntimeError):owned_checkpoint(alias/'000001')
    with pytest.raises(RuntimeError):owned_checkpoint(accepted.parent/'last')


def test_budget_fails_before_writing_large_checkpoint(tmp_path):
    with pytest.raises(RuntimeError,match='storage guard'):check_budget(tmp_path,.01,reserve_gb=1)
def test_feature_retirement_preserves_spec_and_rejects_external_links(tmp_path,monkeypatch):
    import numpy as np
    import pytest
    from uq_extensions.common import atomic_json,read_json
    from uq_extensions.storage import retire_feature_arrays
    cache=tmp_path/'cache';owned=cache/'identity';owned.mkdir(parents=True)
    np.save(owned/'bank.npy',np.ones((4,2)))
    atomic_json(owned/'feature_spec.json',dict(identity='identity',array_files={'bank':'bank.npy'}))
    report=tmp_path/'panel';atomic_json(report/'inputs/cache_link.json',dict(path=str(owned),identity='identity'))
    monkeypatch.setenv('UQ_EXT_FEATURE_CACHE',str(cache));monkeypatch.setenv('UQ_EXT_RETIRE_FEATURE_CACHE','1')
    retire_feature_arrays(report)
    assert not (owned/'bank.npy').exists() and read_json(owned/'feature_spec.json')['arrays_retired']
    assert read_json(report/'feature_retirement.json')['retired_bytes']>0
    atomic_json(report/'bad/cache_link.json',dict(path=str(tmp_path/'elsewhere'),identity='elsewhere'))
    with pytest.raises(ValueError,match='Unowned'):retire_feature_arrays(report)
