"""Real CUDA, GP fitting, video decoding, and headless LIBERO checks."""
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import gpytorch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

class TinyGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

def main():
    assert torch.cuda.is_available(), "CUDA unavailable; run this check from a GPU-enabled terminal."
    versions = {p: importlib.metadata.version(p) for p in
                ("torch", "torchvision", "torchcodec", "gpytorch", "transformers", "hf-libero", "mujoco", "robosuite")}
    result = {"versions": versions, "cuda_runtime": torch.version.cuda, "gpus": [], "checks": []}
    for index in range(torch.cuda.device_count()):
        device = f"cuda:{index}"
        x = torch.linspace(0, 1, 32, device=device)
        y = torch.sin(6*x)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        model = TinyGP(x, y, likelihood).to(device)
        model.train(); likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(3):
            optimizer.zero_grad()
            loss = -mll(model(x), y)
            assert torch.isfinite(loss)
            loss.backward(); optimizer.step()
        model.eval(); likelihood.eval()
        with torch.no_grad():
            posterior = model(torch.linspace(-0.2, 1.2, 20, device=device))
            assert torch.isfinite(posterior.variance).all()
            assert (posterior.variance >= 0).all()
            a = torch.randn(32, 32, device=device)
            torch.linalg.cholesky(a @ a.T + torch.eye(32, device=device))
        torch.cuda.synchronize(index)
        result["gpus"].append({"logical_index":index, "name":torch.cuda.get_device_name(index), "gp_loss":float(loss)})
    result["checks"].append("CUDA tensors, Cholesky, exact GP optimization and latent variance on every selected GPU")
    import av
    from torchcodec.decoders import VideoDecoder
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.mp4"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("libx264", rate=10)
            stream.width = 64; stream.height = 64; stream.pix_fmt = "yuv420p"
            for i in range(4):
                frame = av.VideoFrame.from_ndarray(np.full((64,64,3), i*40, dtype=np.uint8), format="rgb24")
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
        frames = VideoDecoder(str(path))[:]
        assert frames.shape == (4,3,64,64)
    result["checks"].append("PyAV H.264 encoding and TorchCodec/FFmpeg decoding")
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action
    env = LiberoEnv(benchmark.get_benchmark_dict()["libero_10"](), 0, "libero_10", obs_type="pixels_agent_pos")
    try:
        obs, _ = env.reset(seed=100)
        assert set(obs["pixels"]) == {"image", "image2"}
        for frame in obs["pixels"].values():
            assert frame.shape == (256,256,3) and frame.std() > 0
        env.step(np.asarray(get_libero_dummy_action(), dtype=np.float32))
    finally:
        env.close()
    result["checks"].append("LIBERO reset, both 256x256 cameras via EGL, and simulation step")
    result["render_device"] = {"egl_device_index": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
                               "gpu_uuid": os.environ.get("VLA_EGL_GPU_UUID")}
    import laplace
    import iterative_fine_tuning.main
    result["checks"].append("Upstream trainer, uncertainty dependencies, and Laplace imports")
    destination = ROOT / "reports/environment_check.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)

if __name__ == "__main__":
    main()
