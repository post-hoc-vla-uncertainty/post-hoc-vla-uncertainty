"""Protect GPU isolation and the declared paper configurations."""
import json
from pathlib import Path

import pytest

from iterative_fine_tuning.evaluation import normalize_worker_device
from iterative_fine_tuning.training import _normalize_worker_device

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("normalize", [_normalize_worker_device, normalize_worker_device])
def test_worker_respects_remapped_gpus(monkeypatch, normalize):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,6")
    assert normalize("cuda:0") == ("cuda", "3")
    assert normalize("cuda:1") == ("cuda", "6")
    assert normalize("cpu") == ("cpu", None)
    with pytest.raises(IndexError): normalize("cuda:2")

@pytest.mark.parametrize("normalize", [_normalize_worker_device, normalize_worker_device])
def test_worker_respects_uuid_visibility(monkeypatch, normalize):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    assert normalize("cuda:1") == ("cuda", "GPU-b")

def test_paper_settings_and_distinct_calibration_recipe():
    def load(name): return json.loads((ROOT / f"configs/reproduction/{name}.json").read_text())
    pretrain = load("pretrain_seed_0")
    assert pretrain["steps"] == 30000
    assert pretrain["batch_size"] == 32
    assert sum(map(len, pretrain["dataset"]["libero_tasks"].values())) == 33
    assert pretrain["dataset"]["libero_tasks"]["libero_10"] == [0,1,2]
    for method in ("random", "vfd", "gu", "calibration_random"):
        cfg = load(method)
        assert cfg["iteration"]["total_rounds"] == 15
        assert cfg["iteration"]["episodes_per_round"] == 5
        assert cfg["batch_size"] == 32
        assert cfg["replay"]["history_fraction"] == cfg["replay"]["new_fraction"] == 0.5
        assert cfg["steps"] == (2000 if method == "calibration_random" else 4000)
    assert load("random")["selection"]["task_weighted_episode_selection"] == "uniform_random"
    assert load("random")["selection"]["task_selection_temperature"] == 0
    assert load("vfd")["selection"]["task_selection_temperature"] == 2.5
    assert load("gu")["selection"]["task_selection_temperature"] == 1

def test_calibration_preserves_released_training_and_acquisition_seeds():
    for index in range(3):
        cfg = json.loads((ROOT / f"configs/reproduction/calibration_random_seed_{1000+index}.json").read_text())
        assert cfg["seed"] == 1000
        assert cfg["selection"]["seed"] == 2*index
        assert cfg["paths"]["run_name"] == f"seed_{1000+index}"
        assert cfg["multi_seed"] == {}
        assert f"/seed_{2*index}/" in cfg["policy"]["pretrained_path"]
