"""Post-hoc Fourier-feature Gaussian process for frozen policy representations.

This is the GP-output component of SNGP used as an external probe.  It deliberately
does not apply spectral normalization to, or modify, the policy that produced the
features.  The posterior covariance therefore measures support in the frozen feature
space; it does not guarantee distance preservation in observation space.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class PostHocFTTGP(nn.Module):
    """RBF GP approximation using fixed random Fourier features.

    The implementation is Bayesian linear regression on top of random Fourier
    features.  ``fit`` accepts arbitrary leading dimensions and treats the final
    dimension as the feature dimension.  Targets are optional because posterior
    covariance—and thus the external uncertainty score—depends only on ID inputs.
    """

    checkpoint_version = 1

    def __init__(
        self,
        input_dim: int,
        num_random_features: int = 1024,
        output_dim: int = 7,
        length_scale: float = 1.0,
        ridge: float = 1.0,
        observation_noise: float = 1.0,
        seed: int = 0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_random_features <= 0 or output_dim <= 0:
            raise ValueError("input_dim, num_random_features, and output_dim must be positive.")
        if length_scale <= 0 or ridge <= 0 or observation_noise <= 0:
            raise ValueError("length_scale, ridge, and observation_noise must be positive.")

        self.input_dim = int(input_dim)
        self.num_random_features = int(num_random_features)
        self.output_dim = int(output_dim)
        self.length_scale = float(length_scale)
        self.ridge = float(ridge)
        self.observation_noise = float(observation_noise)
        self.seed = int(seed)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        frequency = torch.randn(
            num_random_features, input_dim, generator=generator, dtype=dtype
        ) / length_scale
        phase = 2 * torch.pi * torch.rand(num_random_features, generator=generator, dtype=dtype)

        self.register_buffer("frequency", frequency.to(device))
        self.register_buffer("phase", phase.to(device))
        self.register_buffer("input_mean", torch.zeros(input_dim, device=device, dtype=dtype))
        self.register_buffer("input_scale", torch.ones(input_dim, device=device, dtype=dtype))
        self.register_buffer(
            "precision_cholesky",
            torch.sqrt(torch.tensor(ridge, device=device, dtype=dtype))
            * torch.eye(num_random_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "posterior_mean",
            torch.zeros(num_random_features, output_dim, device=device, dtype=dtype),
        )
        self.register_buffer("num_fit_examples", torch.tensor(0, device=device, dtype=torch.long))

    def _validate_features(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, got {features.shape[-1]}."
            )
        return features.reshape(-1, self.input_dim).to(self.frequency)

    def random_features(self, features: Tensor) -> Tensor:
        flat = self._validate_features(features)
        standardized = (flat - self.input_mean) / self.input_scale
        projection = standardized @ self.frequency.T + self.phase
        return (2.0 / self.num_random_features) ** 0.5 * torch.cos(projection)

    @torch.no_grad()
    def fit(
        self,
        features: Tensor,
        targets: Tensor | None = None,
        chunk_size: int = 8192,
        standardize: bool = True,
    ) -> None:
        """Fit the closed-form posterior from ID features.

        ``targets`` may have any leading shape matching ``features`` and a final
        dimension equal to ``output_dim``.  They affect the posterior mean only,
        not its covariance.
        """
        flat = self._validate_features(features)
        if flat.shape[0] == 0:
            raise ValueError("Cannot fit FTTGP with no ID examples.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        if standardize:
            self.input_mean.copy_(flat.mean(dim=0))
            scale = flat.std(dim=0, unbiased=False).clamp_min(1e-6)
            self.input_scale.copy_(scale)
        else:
            self.input_mean.zero_()
            self.input_scale.fill_(1.0)

        flat_targets = None
        if targets is not None:
            if targets.shape[:-1] != features.shape[:-1] or targets.shape[-1] != self.output_dim:
                raise ValueError(
                    "targets must match the leading feature dimensions and have "
                    f"output_dim={self.output_dim}; got {tuple(targets.shape)}."
                )
            flat_targets = targets.reshape(-1, self.output_dim).to(self.frequency)

        precision = self.ridge * torch.eye(
            self.num_random_features, device=self.frequency.device, dtype=self.frequency.dtype
        )
        phi_y = torch.zeros_like(self.posterior_mean)
        inv_noise = 1.0 / self.observation_noise
        for start in range(0, flat.shape[0], chunk_size):
            stop = min(start + chunk_size, flat.shape[0])
            phi = self.random_features(flat[start:stop])
            precision.addmm_(phi.T, phi, beta=1.0, alpha=inv_noise)
            if flat_targets is not None:
                phi_y.addmm_(phi.T, flat_targets[start:stop], beta=1.0, alpha=inv_noise)

        chol = torch.linalg.cholesky(precision)
        self.precision_cholesky.copy_(chol)
        if flat_targets is None:
            self.posterior_mean.zero_()
        else:
            self.posterior_mean.copy_(torch.cholesky_solve(phi_y, chol))
        self.num_fit_examples.fill_(flat.shape[0])

    def _require_fitted(self) -> None:
        if self.num_fit_examples.item() == 0:
            raise RuntimeError("FTTGP has not been fit on ID features.")

    @torch.no_grad()
    def predictive_variance(self, features: Tensor, include_noise: bool = False) -> Tensor:
        """Return epistemic posterior variance with the input leading shape."""
        self._require_fitted()
        leading_shape = features.shape[:-1]
        phi = self.random_features(features)
        solved = torch.linalg.solve_triangular(self.precision_cholesky, phi.T, upper=False)
        variance = solved.square().sum(dim=0)
        if include_noise:
            variance = variance + self.observation_noise
        return variance.reshape(leading_shape)

    @torch.no_grad()
    def predict(self, features: Tensor) -> tuple[Tensor, Tensor]:
        self._require_fitted()
        leading_shape = features.shape[:-1]
        phi = self.random_features(features)
        mean = (phi @ self.posterior_mean).reshape(*leading_shape, self.output_dim)
        variance = self.predictive_variance(features)
        return mean, variance

    def save(self, path: str | Path) -> None:
        payload = {
            "version": self.checkpoint_version,
            "config": {
                "input_dim": self.input_dim,
                "num_random_features": self.num_random_features,
                "output_dim": self.output_dim,
                "length_scale": self.length_scale,
                "ridge": self.ridge,
                "observation_noise": self.observation_noise,
                "seed": self.seed,
            },
            "state_dict": self.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
    ) -> "PostHocFTTGP":
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        if payload.get("version") != cls.checkpoint_version:
            raise ValueError(f"Unsupported FTTGP checkpoint version: {payload.get('version')!r}.")
        model = cls(**payload["config"], device=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def append_flow_time(features: Tensor, flow_time: Tensor | float) -> Tensor:
    """Append one normalized flow-time coordinate to every action-token feature."""
    time = torch.as_tensor(flow_time, device=features.device, dtype=features.dtype)
    while time.ndim < features.ndim - 1:
        time = time.unsqueeze(-1)
    time = torch.broadcast_to(time.unsqueeze(-1), (*features.shape[:-1], 1))
    return torch.cat([features, time], dim=-1)

