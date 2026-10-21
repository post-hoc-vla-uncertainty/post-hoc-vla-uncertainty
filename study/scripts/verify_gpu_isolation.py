"""Check actual CUDA and EGL process placement via nvidia-smi, with one LIBERO env."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def worker(gpu, output):
    import numpy as np
    import torch
    from lerobot.utils.gpu_utils import nvidia_gpu_uuids
    assert torch.cuda.device_count() == 1
    tensor = torch.ones(16, device="cuda:0")
    assert tensor.sum().item() == 16
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action
    env = LiberoEnv(benchmark.get_benchmark_dict()["libero_10"](), 0, "libero_10", obs_type="pixels_agent_pos")
    try:
        obs, _ = env.reset(seed=100)
        assert all(frame.std() > 0 for frame in obs["pixels"].values())
        env.step(np.asarray(get_libero_dummy_action(), dtype=np.float32))
        torch.cuda.synchronize()
        expected = nvidia_gpu_uuids()[gpu]
        document = ET.fromstring(subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True))
        placements = []
        for card in document.findall("gpu"):
            for process in card.findall("processes/process_info"):
                if process.findtext("pid") == str(os.getpid()):
                    placements.append({"gpu_uuid": card.findtext("uuid").lower(),
                                       "type": process.findtext("type"),
                                       "used_memory": process.findtext("used_memory")})
        assert placements and {p["gpu_uuid"] for p in placements} == {expected}, placements
        assert any("C" in p["type"] for p in placements), placements
        assert any("G" in p["type"] for p in placements), placements
        result = {"physical_gpu": int(gpu), "pid": os.getpid(), "cuda_name": torch.cuda.get_device_name(),
                  "egl_device_index": int(os.environ["MUJOCO_EGL_DEVICE_ID"]),
                  "placements": placements, "passed": True}
        output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--worker")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker, args.output)
        return
    gpus = args.gpus.split(",")
    if not all(gpu.isdigit() for gpu in gpus):
        parser.error("--gpus requires numeric physical device indices")
    results = []
    # Sequential small checks avoid competing with concurrent training on this machine.
    with tempfile.TemporaryDirectory(prefix="vla-gpu-check-") as directory:
        for gpu in gpus:
            output = Path(directory) / f"gpu_{gpu}.json"
            env = os.environ.copy()
            env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu,
                       MUJOCO_EGL_DEVICE_ID=gpu, VLA_PROJECT_ROOT=str(ROOT),
                       MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
            # Deliberately supply the old, wrong index: the simulator fix must resolve it.
            subprocess.run([sys.executable, __file__, "--worker", gpu, "--output", str(output)],
                           env=env, check=True)
            results.append(json.loads(output.read_text()))
    report = ROOT / "reports/gpu_isolation.json"
    report.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
