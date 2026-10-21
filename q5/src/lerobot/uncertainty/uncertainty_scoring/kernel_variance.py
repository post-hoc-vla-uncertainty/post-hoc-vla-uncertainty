#!/usr/bin/env python
"""KV — Kernel Variance, as specified in the research plan.

Gaussian conditional variance from an **exact** RBF kernel on frozen policy
features, conditioned on demonstrated examples, with the kernel geometry fitted
from inputs alone — no prediction targets anywhere.

For a preprocessed representation z and support inputs Z:

    k(z,z')  = exp(−‖z−z'‖² / 2ℓ²)
    C(z,z')  = k(z,z') − k(z,Z) (K_ZZ + λI)⁻¹ k(Z,z')
    u_KV(z)  = C(z,z)

Because k(z,z) = 1 for this kernel, the score lives in [0, 1]: 1 means no support
at all, 0 means the point is reproduced by the support set. That bounded, absolute
scale is the main practical difference from the shipped random-feature head, whose
ceiling is an arbitrary ‖φ‖²/λ_min.

Two things are fitted, and both use inputs only, as the plan requires:

  * the length scale, by the median pairwise distance of the support set;
  * optionally a projection — PCA, or a random feature subset, the two options the
    plan asks to compare — estimated from the support inputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

# Feature locations from the plan's Table 2, and the diagnostic each is meant to
# provide. The keys are what `extract_locations` returns.
LOCATIONS = {
    "vision_external": "Unfamiliar scene structure or layout (workspace camera).",
    "vision_wrist": "Local appearance or object anomalies (wrist camera).",
    "language": "Unfamiliar instructions or linguistic representations.",
    "vl_fused": "Unfamiliar combinations of scene and instruction (fused VLM prefix output).",
    "state": "Underrepresented physical configurations.",
    "joint_early": "Unfamiliar combinations of otherwise familiar inputs.",
    "joint_late": "Overall contextual support.",
}

PROJECTIONS = ("none", "pca", "random", "whiten")


def ledoit_wolf(rows: Tensor) -> Tensor:
    """Ledoit-Wolf shrinkage of the sample covariance towards a scaled identity."""
    n, d = rows.shape
    x = rows - rows.mean(dim=0)
    s = x.T @ x / n
    mu = torch.trace(s) / d
    delta = (s - mu * torch.eye(d, dtype=s.dtype, device=s.device)).square().sum() / d
    # Mean squared deviation of the per-sample outer products from s, computed without
    # materializing them: sum_i ||x_i x_i^T - s||^2 = sum_i ||x_i||^4 - n ||s||^2.
    beta = ((x.square().sum(dim=1).square().sum() - n * s.square().sum()) / (n * n * d))
    shrink = torch.clamp(beta / delta, max=1.0) if delta > 0 else torch.tensor(1.0)
    return shrink * mu * torch.eye(d, dtype=s.dtype, device=s.device) + (1 - shrink) * s


@dataclass
class KernelVariance:
    """Exact-GP conditional variance on a frozen representation."""

    ridge: float = 1e-2
    projection: str = "none"
    components: int = 64
    length_scale: float | None = None
    seed: int = 0

    support: Tensor | None = field(default=None, repr=False)
    cholesky: Tensor | None = field(default=None, repr=False)
    mean: Tensor | None = field(default=None, repr=False)
    scale: Tensor | None = field(default=None, repr=False)
    basis: Tensor | None = field(default=None, repr=False)

    # ---- geometry, fitted from inputs only -------------------------------------

    def _fit_projection(self, rows: Tensor) -> None:
        if self.projection == "none":
            self.basis = None
            return
        width = rows.shape[-1]
        components = min(self.components, width, max(rows.shape[0] - 1, 1))
        if self.projection == "pca":
            # Principal directions of the support inputs; no targets involved.
            centered = rows - rows.mean(dim=0)
            _, _, right = torch.linalg.svd(centered, full_matrices=False)
            self.basis = right[:components].T.contiguous()
        elif self.projection == "whiten":
            # Mahalanobis metric: whiten by the shrunk support covariance, so the RBF
            # kernel measures Mahalanobis rather than Euclidean distance.
            values, vectors = torch.linalg.eigh(ledoit_wolf(rows.double()))
            self.basis = (vectors * values.clamp_min(1e-10).rsqrt()) @ vectors.T
            self.basis = self.basis.to(rows.dtype)
        elif self.projection == "random":
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            index = torch.randperm(width, generator=generator)[:components]
            basis = torch.zeros(width, components, device=rows.device, dtype=rows.dtype)
            basis[index, torch.arange(components)] = 1.0
            self.basis = basis
        else:
            raise ValueError(f"unknown projection {self.projection!r}")

    def _prepare(self, rows: Tensor) -> Tensor:
        standardized = (rows - self.mean) / self.scale
        return standardized if self.basis is None else standardized @ self.basis

    @staticmethod
    def median_distance(rows: Tensor, limit: int = 4096, seed: int = 0) -> float:
        count = min(limit, rows.shape[0])
        if count < 2:
            return 1.0
        generator = torch.Generator(device="cpu").manual_seed(seed)
        index = torch.randperm(rows.shape[0], generator=generator)[:count].to(rows.device)
        distances = torch.cdist(rows[index], rows[index])
        mask = ~torch.eye(count, dtype=torch.bool, device=rows.device)
        return float(distances[mask].median().clamp_min(1e-6))

    # ---- fit and score ---------------------------------------------------------

    @torch.no_grad()
    def fit(self, support: Tensor) -> "KernelVariance":
        """Condition on the demonstrated support inputs. No targets are used."""
        if support.ndim != 2:
            raise ValueError(f"support must be (n, dim); got {tuple(support.shape)}")
        self.mean = support.mean(dim=0)
        self.scale = support.std(dim=0, unbiased=False).clamp_min(1e-6)
        self._fit_projection((support - self.mean) / self.scale)
        prepared = self._prepare(support)
        if self.length_scale is None:
            self.length_scale = self.median_distance(prepared, seed=self.seed)
        gram = self._kernel(prepared, prepared)
        gram.diagonal().add_(self.ridge)
        self.cholesky = torch.linalg.cholesky(gram)
        self.support = prepared
        return self

    def _kernel(self, a: Tensor, b: Tensor, chunk: int = 4096) -> Tensor:
        out = torch.empty(a.shape[0], b.shape[0], device=a.device, dtype=a.dtype)
        for start in range(0, a.shape[0], chunk):
            block = a[start : start + chunk]
            out[start : start + chunk] = torch.exp(
                -torch.cdist(block, b).square() / (2.0 * self.length_scale**2)
            )
        return out

    @torch.no_grad()
    def score(self, rows: Tensor, chunk: int = 8192) -> Tensor:
        """u_KV(z) = 1 − k(z,Z) (K_ZZ + λI)⁻¹ k(Z,z), bounded to [0, 1]."""
        if self.cholesky is None:
            raise RuntimeError("KernelVariance has not been fitted.")
        prepared = self._prepare(rows)
        out = []
        for start in range(0, prepared.shape[0], chunk):
            cross = self._kernel(prepared[start : start + chunk], self.support)
            solved = torch.linalg.solve_triangular(self.cholesky, cross.T, upper=False)
            out.append((1.0 - solved.square().sum(dim=0)).clamp(0.0, 1.0))
        return torch.cat(out)

    # ---- persistence -----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        torch.save({
            "config": {"ridge": self.ridge, "projection": self.projection,
                       "components": self.components, "length_scale": self.length_scale,
                       "seed": self.seed},
            "support": self.support, "cholesky": self.cholesky,
            "mean": self.mean, "scale": self.scale, "basis": self.basis,
        }, Path(path))

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "KernelVariance":
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        model = cls(**payload["config"])
        for name in ("support", "cholesky", "mean", "scale", "basis"):
            value = payload[name]
            setattr(model, name, value.to(device) if isinstance(value, Tensor) else value)
        return model


# ---- feature locations ---------------------------------------------------------


@torch.no_grad()
def extract_locations(adapter, batch: dict, flow_levels=(1, 9)) -> dict[str, Tensor]:
    """One representation vector per observation, at each of the plan's locations.

    The modality embeddings mirror the masked-mean pooling the repository's own
    FTTGP sampler uses, so a KV score at `vision_external` is comparable with the
    existing structured head. The joint locations pool the action-expert hidden
    state over action tokens at an early and a late Euler level.
    """
    model = adapter.model
    images, masks = model.prepare_images(batch)
    cameras = []
    for image, mask in zip(images, masks, strict=True):
        tokens = model.vlm_with_expert.embed_image(image)
        tokens = tokens * math.sqrt(tokens.shape[-1])
        valid = mask[:, None, None].to(tokens)
        cameras.append((tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0))

    language = model.vlm_with_expert.embed_language_tokens(batch["observation.language.tokens"])
    language = language * math.sqrt(language.shape[-1])
    language_mask = batch["observation.language.attention_mask"].to(language).unsqueeze(-1)
    language = (language * language_mask).sum(dim=1) / language_mask.sum(dim=1).clamp_min(1.0)

    state = model.state_proj(model.prepare_state(batch))
    if state.ndim == 3:
        state = state.mean(dim=1)

    # Fused vision-language prefix: the VLM's own final-layer output over the prefix
    # (both cameras, the instruction and the state token), after its transformer
    # layers have mixed them. The policy computes this for its KV cache and discards it.
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(
        images=images, img_masks=masks,
        lang_tokens=batch["observation.language.tokens"],
        lang_masks=batch["observation.language.attention_mask"],
        state=model.prepare_state(batch),
    )
    outputs, _ = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(prefix_pad, prefix_att),
        position_ids=torch.cumsum(prefix_pad, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=False,
        fill_kv_cache=True,
    )
    valid = prefix_pad.unsqueeze(-1).to(outputs[0])
    fused = (outputs[0] * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    features = {
        "vl_fused": fused,
        "vision_external": cameras[0],
        "vision_wrist": cameras[1] if len(cameras) > 1 else cameras[0],
        "language": language,
        "state": state,
    }

    # Joint features: walk the flow and pool the hidden state over action tokens.
    from lerobot.policies.common.flow_matching.ode_solver import make_sampling_time_grid

    batch_size = next(v.shape[0] for v in batch.values() if isinstance(v, Tensor))
    conditioning = adapter.prepare_conditioning(batch, num_action_samples=1)
    velocity_and_features = adapter.make_velocity_and_features_fn(conditioning)
    generator = torch.Generator(device=adapter.device).manual_seed(0)
    x = adapter.sample_prior(batch_size, generator=generator)
    grid = make_sampling_time_grid(step_size=adapter.ode_solver_config["step_size"],
                                   device=adapter.device, dtype=adapter.dtype)
    early, late = flow_levels
    for level, (start, stop) in enumerate(zip(grid[:-1], grid[1:], strict=True), start=1):
        velocity, hidden = velocity_and_features(x_t=x, t=start)
        if level == early:
            features["joint_early"] = hidden.mean(dim=1)
        if level == late:
            features["joint_late"] = hidden.mean(dim=1)
        x = x + (stop - start) * velocity
    return features


def location_report(scores: dict[str, float]) -> str:
    return json.dumps({name: {"score": value, "diagnostic": LOCATIONS.get(name, "")}
                       for name, value in scores.items()}, indent=2)


class KVSampler:
    """KV as a drop-in uncertainty sampler over initial observations.

    Conditions an exact RBF kernel on one frozen feature of the policy at the initial
    observation of each support episode, i.e. every demonstration the learner has
    received. Used identically by SAVE selection and by the calibration comparison,
    so both score with the same support bank.
    """

    def __init__(self, adapter, preprocessor, dataset, support_episode_ids, *, location: str,
                 ridge: float, device, batch_size: int, collate, progress=None,
                 projection: str = "none") -> None:
        import time

        if location not in LOCATIONS:
            raise ValueError(f"Unknown KV location {location!r}.")
        self.adapter = adapter
        self.location = location
        episodes = dataset.meta.episodes
        first_frame = {int(e): int(s) for e, s in
                       zip(episodes["episode_index"], episodes["dataset_from_index"], strict=True)}
        instruction = {int(e): str(t[0]) for e, t in
                       zip(episodes["episode_index"], episodes["tasks"], strict=True)}
        # Instructions tokenize to different lengths, so a batch never mixes tasks.
        groups: dict[str, list[int]] = {}
        for episode_id in sorted({int(e) for e in support_episode_ids}):
            groups.setdefault(instruction[episode_id], []).append(first_frame[episode_id])

        started = time.perf_counter()
        rows = []
        for frames in (progress(groups.values()) if progress else groups.values()):
            for start in range(0, len(frames), batch_size):
                processed = [preprocessor(dataset[f]) for f in frames[start : start + batch_size]]
                rows.append(self.features(collate(processed, device)))
        support = torch.cat(rows)
        self.ridge, self.projection = ridge, projection
        self.head = self._make_head(support)
        self.summary = {
            "location": location,
            "projection": projection,
            "ridge": float(ridge),
            "support_size": int(support.shape[0]),
            "length_scale": float(self.head.length_scale),
            "support_seconds": time.perf_counter() - started,
        }

    def _make_head(self, support: Tensor):
        """The estimator conditioned on the support features; subclasses swap it."""
        return KernelVariance(ridge=self.ridge, projection=self.projection).fit(support)

    @torch.no_grad()
    def features(self, observation: dict) -> Tensor:
        return extract_locations(self.adapter, observation)[self.location].float()

    @torch.no_grad()
    def conditional_sample_with_uncertainty_batch(self, observation, generator=None):
        del generator
        return None, self.head.score(self.features(observation)).cpu()


# ---- KV-T: target-informed kernel variance -------------------------------------


def _pls_directions(x: Tensor, y: Tensor, components: int, iterations: int = 200) -> Tensor:
    """PLS2 (NIPALS) weight matrix mapping centred x to its y-predictive scores."""
    x, y = x.clone(), y.clone()
    weights, loadings = [], []
    for _ in range(components):
        u = y[:, [int(y.var(dim=0).argmax())]]
        w = None
        for _ in range(iterations):
            w_new = x.T @ u
            w_new = w_new / w_new.norm().clamp_min(1e-12)
            t = x @ w_new
            c = y.T @ t / (t.T @ t).clamp_min(1e-12)
            u = y @ c / (c.T @ c).clamp_min(1e-12)
            if w is not None and (w_new - w).norm() < 1e-8:
                w = w_new
                break
            w = w_new
        t = x @ w
        p = x.T @ t / (t.T @ t).clamp_min(1e-12)
        x = x - t @ p.T
        y = y - t @ (y.T @ t / (t.T @ t).clamp_min(1e-12)).T
        weights.append(w)
        loadings.append(p)
    w, p = torch.cat(weights, dim=1), torch.cat(loadings, dim=1)
    # Rotation so that scores = x_centred @ R, valid for new points too.
    return w @ torch.linalg.inv(p.T @ w)


@dataclass
class TargetInformedKV:
    """KV-T: separate exact GP regressions from an earlier to a later policy feature.

    Targets (later activations) are standardized and compressed by PCA into
    ``targets`` outputs; inputs are compressed by PLS against those outputs. Each
    output j then gets its own RBF kernel whose length scale, signal variance and
    noise are learned by marginal likelihood. The score is the average of the
    standardized latent posterior variances C_j(z,z)/s_j², which excludes
    observation noise and lives in [0, 1] like KV.
    """

    targets: int = 8
    components: int = 16
    steps: int = 300
    seed: int = 0

    x_mean: Tensor | None = field(default=None, repr=False)
    x_scale: Tensor | None = field(default=None, repr=False)
    rotation: Tensor | None = field(default=None, repr=False)
    support: Tensor | None = field(default=None, repr=False)
    log_length: Tensor | None = field(default=None, repr=False)
    log_signal: Tensor | None = field(default=None, repr=False)
    log_noise: Tensor | None = field(default=None, repr=False)
    cholesky: Tensor | None = field(default=None, repr=False)
    fit_report: dict = field(default_factory=dict)

    def _prepare(self, rows: Tensor) -> Tensor:
        return ((rows - self.x_mean) / self.x_scale) @ self.rotation

    def fit(self, inputs: Tensor, later: Tensor) -> "TargetInformedKV":
        inputs, later = inputs.double(), later.double()
        self.x_mean = inputs.mean(dim=0)
        self.x_scale = inputs.std(dim=0, unbiased=False).clamp_min(1e-6)
        x = (inputs - self.x_mean) / self.x_scale
        # Compress the downstream targets: standardized PCA scores.
        y = (later - later.mean(dim=0)) / later.std(dim=0, unbiased=False).clamp_min(1e-6)
        _, _, right = torch.linalg.svd(y, full_matrices=False)
        t = y @ right[: self.targets].T
        t = (t - t.mean(dim=0)) / t.std(dim=0, unbiased=False).clamp_min(1e-6)
        # Target-informed input projection.
        components = min(self.components, x.shape[1], x.shape[0] - 1)
        self.rotation = _pls_directions(x, t, components)
        z = x @ self.rotation
        self.support = z

        # Per-output kernel learning by exact marginal likelihood.
        n, m = z.shape[0], t.shape[1]
        start = KernelVariance.median_distance(z.float(), seed=self.seed)
        log_length = torch.full((m,), math.log(start), dtype=z.dtype, requires_grad=True)
        log_signal = torch.zeros(m, dtype=z.dtype, requires_grad=True)
        log_noise = torch.full((m,), math.log(0.1), dtype=z.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([log_length, log_signal, log_noise], lr=0.05)
        squared = torch.cdist(z, z).square()
        eye = torch.eye(n, dtype=z.dtype)
        for _ in range(self.steps):
            optimizer.zero_grad()
            gram = log_signal.exp()[:, None, None] * torch.exp(
                -squared[None] / (2 * log_length.exp()[:, None, None] ** 2))
            a = gram + (log_noise.exp()[:, None, None] + 1e-6) * eye
            chol = torch.linalg.cholesky(a)
            alpha = torch.cholesky_solve(t.T.unsqueeze(-1), chol)
            nll = 0.5 * (t.T.unsqueeze(-1) * alpha).sum() + torch.log(
                torch.diagonal(chol, dim1=-2, dim2=-1)).sum()
            nll.backward()
            optimizer.step()
        with torch.no_grad():
            self.log_length, self.log_signal, self.log_noise = (
                log_length.detach(), log_signal.detach(), log_noise.detach())
            gram = self._gram(z, z)
            self.cholesky = torch.linalg.cholesky(gram + (self.log_noise.exp()[:, None, None] + 1e-6) * eye)
            alpha = torch.cholesky_solve(t.T.unsqueeze(-1), self.cholesky)
            fitted = (gram @ alpha).squeeze(-1).T
            self.fit_report = {
                "support": n, "targets": m, "components": components,
                "target_variance_kept": float((right[:m] @ y.T).square().sum() / y.square().sum()),
                "nll_per_point": float(nll) / (n * m),
                "length_scale": self.log_length.exp().tolist(),
                "signal_variance": self.log_signal.exp().tolist(),
                "noise_variance": self.log_noise.exp().tolist(),
                "train_r2": float(1 - (fitted - t).square().mean() / t.var(unbiased=False)),
            }
        return self

    def _gram(self, a: Tensor, b: Tensor) -> Tensor:
        return self.log_signal.exp()[:, None, None] * torch.exp(
            -torch.cdist(a, b).square()[None] / (2 * self.log_length.exp()[:, None, None] ** 2))

    @torch.no_grad()
    def score(self, rows: Tensor, chunk: int = 4096) -> Tensor:
        device, dtype = rows.device, rows.dtype
        out = []
        for start in range(0, rows.shape[0], chunk):
            z = self._prepare(rows[start : start + chunk].double().cpu())
            cross = self._gram(z, self.support)                      # (m, b, n)
            solved = torch.linalg.solve_triangular(self.cholesky, cross.transpose(1, 2), upper=False)
            latent = self.log_signal.exp()[:, None] - solved.square().sum(dim=1)   # (m, b)
            out.append((latent / self.log_signal.exp()[:, None]).clamp(0, 1).mean(dim=0))
        return torch.cat(out).to(device=device, dtype=dtype)

    def save(self, path: str | Path) -> None:
        torch.save({"kind": "kv_t", "config": {"targets": self.targets, "components": self.components,
                                               "steps": self.steps, "seed": self.seed},
                    **{k: getattr(self, k) for k in ("x_mean", "x_scale", "rotation", "support",
                                                     "log_length", "log_signal", "log_noise",
                                                     "cholesky", "fit_report")}}, Path(path))

    @classmethod
    def load(cls, path: str | Path, device="cpu") -> "TargetInformedKV":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(**payload["config"])
        for key in ("x_mean", "x_scale", "rotation", "support", "log_length", "log_signal",
                    "log_noise", "cholesky", "fit_report"):
            setattr(model, key, payload[key])
        return model


def load_head(path: str | Path, device="cpu"):
    """Load either a KV or a KV-T head from disk."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("kind") == "kv_t":
        return TargetInformedKV.load(path, device=device)
    return KernelVariance.load(path, device=device)
