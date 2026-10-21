"""Post-hoc Mahalanobis support detector for frozen policy features."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class PostHocMahalanobis(nn.Module):
    """Regularized full-covariance Mahalanobis distance.

    No regression targets are used. Fitting estimates only the center and feature
    covariance of in-distribution activations.
    """

    checkpoint_version = 1

    def __init__(
        self,
        input_dim: int,
        shrinkage: float = 0.1,
        ridge: float = 1e-4,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not 0 <= shrinkage <= 1:
            raise ValueError("shrinkage must satisfy 0 ≤ shrinkage ≤ 1.")
        if ridge <= 0:
            raise ValueError("ridge must be positive.")
        self.input_dim = int(input_dim)
        self.shrinkage = float(shrinkage)
        self.ridge = float(ridge)
        self.register_buffer("mean", torch.zeros(input_dim, device=device, dtype=dtype))
        self.register_buffer("covariance_cholesky", torch.eye(input_dim, device=device, dtype=dtype))
        self.register_buffer("num_fit_examples", torch.tensor(0, device=device, dtype=torch.long))

    @torch.no_grad()
    def fit(self, features: Tensor) -> None:
        flat = features.reshape(-1, features.shape[-1])
        if flat.shape[-1] != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, got {flat.shape[-1]}.")
        if flat.shape[0] < 2:
            raise ValueError("At least two fitting examples are required.")
        # Accumulate in float64; the 480-wide expert representation can otherwise
        # produce a slightly indefinite empirical covariance in float32.
        work = flat.to(device=self.mean.device, dtype=torch.float64)
        mean = work.mean(dim=0)
        centered = work - mean
        covariance = centered.T @ centered / (len(work) - 1)
        diagonal = torch.diag(torch.diagonal(covariance))
        covariance = (1.0 - self.shrinkage) * covariance + self.shrinkage * diagonal
        scale = torch.diagonal(covariance).mean().clamp_min(torch.finfo(covariance.dtype).eps)
        covariance = covariance + self.ridge * scale * torch.eye(
            self.input_dim, device=work.device, dtype=work.dtype
        )
        self.mean.copy_(mean.to(self.mean))
        self.covariance_cholesky.copy_(torch.linalg.cholesky(covariance).to(self.covariance_cholesky))
        self.num_fit_examples.fill_(len(work))

    @torch.no_grad()
    def squared_distance(self, features: Tensor, normalize: bool = True) -> Tensor:
        if self.num_fit_examples.item() == 0:
            raise RuntimeError("Mahalanobis detector has not been fit.")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, got {features.shape[-1]}.")
        leading = features.shape[:-1]
        centered = features.reshape(-1, self.input_dim).to(self.mean) - self.mean
        whitened = torch.linalg.solve_triangular(
            self.covariance_cholesky, centered.T, upper=False
        )
        distance = whitened.square().sum(dim=0)
        if normalize:
            distance = distance / self.input_dim
        return distance.reshape(leading)

    def save(self, path: str | Path) -> None:
        torch.save({
            "version": self.checkpoint_version,
            "config": {"input_dim": self.input_dim, "shrinkage": self.shrinkage, "ridge": self.ridge},
            "state_dict": self.state_dict(),
        }, Path(path))

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "PostHocMahalanobis":
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        if payload.get("version") != cls.checkpoint_version:
            raise ValueError(f"Unsupported Mahalanobis checkpoint version: {payload.get('version')!r}.")
        model = cls(**payload["config"], device=device)
        model.load_state_dict(payload["state_dict"])
        return model.eval()
