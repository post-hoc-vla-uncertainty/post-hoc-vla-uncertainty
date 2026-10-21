"""Context KV: a reimplementation of the kernel-variance head of extern/vla-uncertainty.

That repository (not modified here) scores SmolVLA with an exact fixed-kernel GP on frozen
features, like our KV, but with a different feature and a different estimator. Its best
reported heads use the "final context": the final VLM prefix layer, masked-mean pooled
separately over each camera's tokens and over the instruction tokens, concatenated with the
policy's (normalized, padded) state truncated to 8 dims, i.e. 2*960 + 960 + 8 = 2888 values
for LIBERO (their features.py, `joint(output[0])`).

Estimator, following their kernel_variance.py and preprocessing.py:
  * per-coordinate standardization on the bank (constant coordinates excluded from projections)
  * projection: "random" (512 actual coordinates, seed 42), "pca" (randomized PCA to 512, unwhitened)
    or "pca_whiten" (PCA, then Ledoit-Wolf rescaling along the component axes)
  * duplicate support rows removed
  * length scale: k = ceil(sqrt(N_unique)); median over points of the k-th neighbour distance m;
    l = m / sqrt(2 ln 2), so the kernel is 0.5 at the median k-th neighbour distance
  * exact GP, RBF with unit amplitude, ridge 1e-3, jitter escalation; score = posterior variance
"""

from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

STATE_DIMS = 8
COMPONENTS, SEED, RIDGE, FLOOR = 512, 42, 1e-3, 1e-8


@torch.no_grad()
def final_context_features(adapter, batch: dict, state_dims: int = STATE_DIMS) -> torch.Tensor:
    """Their `joint_late`: per-camera and instruction means of the final prefix layer, plus state."""
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    model = adapter.model
    if model.add_image_special_tokens:
        raise NotImplementedError("token groups below assume no image special tokens")
    images, masks = model.prepare_images(batch)
    lang = batch["observation.language.tokens"]
    state = model.prepare_state(batch)
    prefix, pad, att = model.embed_prefix(images=images, img_masks=masks, lang_tokens=lang,
                                          lang_masks=batch["observation.language.attention_mask"], state=state)
    outputs, _ = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, dim=1) - 1,
        past_key_values=None, inputs_embeds=[prefix, None], use_cache=False, fill_kv_cache=True)
    hidden = outputs[0]
    # Layout: [camera 1 | camera 2 | ... | instruction | state token(s)], equal tokens per camera.
    per_camera = (prefix.shape[1] - lang.shape[1] - 1) // len(images)
    if per_camera * len(images) + lang.shape[1] + 1 != prefix.shape[1]:
        raise RuntimeError(f"unexpected prefix layout: {prefix.shape[1]} tokens, {len(images)} cameras, "
                           f"{lang.shape[1]} language tokens")
    groups = [(i * per_camera, (i + 1) * per_camera) for i in range(len(images))]
    groups.append((len(images) * per_camera, len(images) * per_camera + lang.shape[1]))

    def masked_mean(a: int, b: int) -> torch.Tensor:
        m = pad[:, a:b].to(hidden.dtype)[..., None]
        return (hidden[:, a:b] * m).sum(1) / m.sum(1).clamp_min(1)

    return torch.cat([*(masked_mean(a, b) for a, b in groups), state[:, :state_dims].to(hidden.dtype)], 1).float()


def _shrunk_covariance(z: np.ndarray) -> np.ndarray:
    return LedoitWolf(store_precision=False).fit(z).covariance_


def geometry_lengthscale(z: torch.Tensor) -> float:
    if len(z) > 4096:
        z = z[np.sort(np.random.default_rng(0).choice(len(z), 4096, replace=False))]
    k = min(len(z) - 1, math.ceil(math.sqrt(len(z))))
    d = torch.cdist(z, z)
    d.fill_diagonal_(float("inf"))
    median = torch.quantile(d.kthvalue(k, dim=1).values, 0.5).item()
    return median / math.sqrt(2 * math.log(2))


def _cholesky(a: torch.Tensor, jitter: float = 1e-6) -> torch.Tensor:
    eye = torch.eye(len(a), device=a.device, dtype=a.dtype)
    current = jitter
    for _ in range(7):
        factor, info = torch.linalg.cholesky_ex((a + a.T) / 2 + current * eye)
        if not info.any() and torch.isfinite(factor).all():
            return factor
        current *= 10
    if a.dtype != torch.float64:
        return _cholesky(a.double(), jitter)
    raise RuntimeError("Cholesky failed after jitter escalation and float64 fallback")


class ContextKV:
    """Their fixed-kernel KV estimator on any feature matrix (projection: random | pca | pca_whiten)."""

    def __init__(self, projection: str = "random", components: int = COMPONENTS, seed: int = SEED,
                 ridge: float = RIDGE, device="cpu") -> None:
        self.projection, self.components, self.seed, self.ridge = projection, components, seed, ridge
        self.device = torch.device(device)

    def _fit_projection(self, x: np.ndarray) -> None:
        self.mean = x.mean(0)
        std = x.std(0)
        active = np.flatnonzero(std > FLOOR)
        self.scale = np.maximum(std, FLOOR)
        standardized = (x - self.mean) / self.scale
        self.center = np.zeros(x.shape[1])
        n = min(self.components, len(x) - 1, len(active))
        base = self.projection.removesuffix("_whiten")
        if base == "random":
            coords = np.sort(np.random.default_rng(self.seed).choice(active, min(self.components, len(active)), replace=False))
            self.weights = np.zeros((x.shape[1], len(coords)))
            self.weights[coords, np.arange(len(coords))] = 1
        elif base == "pca":
            if n > 32:
                n = min(n, len(np.unique(x, axis=0)) - 1)
            pca = PCA(n_components=n, whiten=False, svd_solver="randomized", random_state=self.seed)
            pca.fit(standardized[:, active])
            valid = pca.explained_variance_ > max(pca.explained_variance_[0] * 1e-10, FLOOR)
            self.weights = np.zeros((x.shape[1], int(valid.sum())))
            self.weights[active] = pca.components_[valid].T
            self.center[active] = pca.mean_
        else:
            raise ValueError(f"unknown projection {self.projection!r}")
        if self.projection.endswith("_whiten"):
            z = (standardized - self.center) @ self.weights
            cov = _shrunk_covariance(z)
            if base == "random":
                values, axes = np.linalg.eigh(cov)
                values, axes = values[::-1], axes[:, ::-1]
            else:  # PCA scores are already decorrelated: keep their axes
                values, axes = cov.diagonal().copy(), np.eye(z.shape[1])
            floor = max(FLOOR, float(values.max()) * 1e-10)
            self.weights = self.weights @ (axes / np.sqrt(np.maximum(values, floor)))

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (((x - self.mean) / self.scale - self.center) @ self.weights).astype(np.float32)

    def fit(self, x) -> "ContextKV":
        x = np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x, dtype=np.float64)
        self._fit_projection(x)
        unique = np.unique(self.transform(x), axis=0)
        self.z = torch.as_tensor(unique, device=self.device)
        self.length_scale = geometry_lengthscale(self.z)
        k = self._rbf(self.z, self.z)
        self.factor = _cholesky(k + self.ridge * torch.eye(len(k), device=self.device, dtype=k.dtype))
        self.z = self.z.to(self.factor.dtype)
        self.summary = {"support_rows": len(x), "unique_rows": len(unique), "components": unique.shape[1],
                        "length_scale": self.length_scale, "projection": self.projection}
        return self

    def _rbf(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * torch.cdist(a / self.length_scale, b / self.length_scale).square())

    @torch.no_grad()
    def score(self, x, batch_size: int = 256) -> np.ndarray:
        x = np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x, dtype=np.float64)
        q = torch.as_tensor(self.transform(x), device=self.device, dtype=self.factor.dtype)
        out = []
        for part in q.split(batch_size):
            v = torch.linalg.solve_triangular(self.factor, self._rbf(self.z, part), upper=False)
            out.append((1.0 - v.square().sum(0)).clamp_min(0))
        return torch.cat(out).float().cpu().numpy()


class ContextKVSampler:
    """KVSampler with the extern/vla-uncertainty head: final-context feature and ContextKV estimator.

    Same support rule as KV (first frame of every demonstration the checkpoint was trained on)."""

    def __new__(cls, *args, context_projection: str = "random", **kwargs):
        from lerobot.uncertainty.uncertainty_scoring.kernel_variance import KVSampler

        class _Sampler(KVSampler):
            def _make_head(self, support):
                return ContextKV(projection=context_projection, device=support.device).fit(support)

            @torch.no_grad()
            def features(self, observation):
                return final_context_features(self.adapter, observation)

            @torch.no_grad()
            def conditional_sample_with_uncertainty_batch(self, observation, generator=None):
                del generator
                return None, torch.as_tensor(self.head.score(self.features(observation)))

        sampler = _Sampler(*args, **kwargs)
        sampler.summary["head"] = f"context_kv/{context_projection}512"
        return sampler
