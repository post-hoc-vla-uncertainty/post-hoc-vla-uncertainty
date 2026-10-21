"""Exact fixed RBF latent covariance and a declared inducing-point fallback."""
import math
from pathlib import Path
import time
import numpy as np
import torch

from .common import atomic_json, atomic_npz, read_json
from .preprocessing import Projection, UninformativeBank, fit_projection, fit_output_projection, shrunk_covariance


def rbf(x, y, lengthscale, amplitude=1.):
    return amplitude * torch.exp(-0.5 * torch.cdist(x / lengthscale, y / lengthscale).square())


def stable_cholesky(a, jitter=1e-6):
    eye = torch.eye(len(a), device=a.device, dtype=a.dtype)
    current = jitter
    for _ in range(7):
        factor, info = torch.linalg.cholesky_ex((a + a.T) / 2 + current * eye)
        if not info.any() and torch.isfinite(factor).all():
            return factor, current
        current *= 10
    if a.dtype != torch.float64:
        factor, used = stable_cholesky(a.double(), jitter)
        return factor, used
    raise RuntimeError("Cholesky failed after jitter escalation and float64 fallback")


def nonnegative_variance(v, amplitude=1.):
    tolerance = 2e-5 * amplitude if v.dtype == torch.float32 else 1e-9 * amplitude
    if not torch.isfinite(v).all() or torch.any(v < -tolerance):
        raise RuntimeError(f"Materially negative or nonfinite latent variance: {v.min().item()}")
    return v.clamp_min(0)


def geometry_lengthscale(z):
    if len(z)>4096:
        indices=np.sort(np.random.default_rng(0).choice(len(z),4096,replace=False))
        z=z[indices]
    if len(z) < 2:
        raise UninformativeBank("Fewer than two distinct support locations")
    k = min(len(z) - 1, math.ceil(math.sqrt(len(z))))
    distances = torch.cdist(z, z)
    distances.fill_diagonal_(float("inf"))
    median = torch.quantile(distances.kthvalue(k, dim=1).values, 0.5).item()
    if not math.isfinite(median) or median <= 0:
        raise UninformativeBank("No positive neighborhood distance")
    return median / math.sqrt(2 * math.log(2)), {"neighbor_count": k, "median_neighbor_distance": median}


def initial_log_parameters(x, config):
    """Preserve legacy initialization unless the variance control is requested."""
    initial, _ = geometry_lengthscale(torch.unique(x, dim=0))
    lengths = np.full(x.shape[1], initial)
    diagnostics = {}
    if config.lengthscale_initialization == 'variance':
        covariance, diagnostics = shrunk_covariance(x.detach().cpu().numpy())
        variances = covariance.diagonal()
        floor = max(1e-8, float(variances.max())*1e-10)
        scales = torch.as_tensor(np.sqrt(np.maximum(variances, floor)), device=x.device, dtype=x.dtype)
        initial, _ = geometry_lengthscale(torch.unique(x/scales, dim=0))
        lengths = (scales*initial).detach().cpu().numpy().astype(np.float64)
        diagnostics.update(kind='variance', variance_floor=floor, initial_lengthscales=lengths.tolist(),
                           learned_after_initialization=True)
    return np.r_[np.log(lengths), 0., np.log(1e-3)], diagnostics


class KernelVariance:
    def __init__(self, feature_config, gp_config, *, device="cpu"):
        self.features, self.gp = feature_config, gp_config
        self.device = torch.device(device)
        self.dtype = getattr(torch, gp_config.precision)

    def tensor(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def fit(self, x, y=None):
        started = time.perf_counter()
        geometry_seed=self.features.seed if self.features.geometry_seed is None else self.features.geometry_seed
        fit_x=x
        if self.gp.inference=='sparse' and len(x)>self.features.geometry_samples:
            fit_x=x[np.sort(np.random.default_rng(geometry_seed).choice(len(x),self.features.geometry_samples,replace=False))]
        targets = None
        target_diagnostics = {}
        if self.features.compression.startswith('pls'):
            if y is None or len(y) != len(x) or len(fit_x) != len(x):
                raise ValueError('PLS-KV requires aligned full observation training inputs and targets')
            output, targets = fit_output_projection(y, method=self.gp.output_compression,
                components=self.gp.output_components,
                seed=self.features.seed if self.gp.output_seed is None else self.gp.output_seed,
                variance_floor=self.features.variance_floor)
            target_diagnostics = dict(target_informed_projection=True, target_bank=self.features.target_bank,
                target_components=targets.shape[1], output_compression=self.gp.output_compression,
                output_pca_variance_retained=float(output.explained_variance.sum()))
        self.projection = fit_projection(fit_x, method=self.features.compression, components=self.features.components,
                                         seed=self.features.seed, variance_floor=self.features.variance_floor, targets=targets)
        projected = self.projection.transform(x)
        unique, counts = np.unique(projected, axis=0, return_counts=True)
        if self.features.compression.endswith('_whiten'):
            # Match support ordering and inducing examples to the unwhitened control.
            base = fit_projection(fit_x, method=self.features.compression.removesuffix('_whiten'),
                components=self.features.components, seed=self.features.seed,
                variance_floor=self.features.variance_floor, targets=targets)
            _, indices, counts = np.unique(base.transform(x), axis=0, return_index=True, return_counts=True)
            unique = projected[indices]
        self.z = self.tensor(unique)
        self.lengthscale, geometry = geometry_lengthscale(self.z)
        self.sparse = self.gp.inference == "sparse"
        if self.sparse and not self.gp.sparse_justification:
            raise ValueError("Sparse KV requires a measured scaling justification")
        if self.sparse:
            n = min(self.gp.inducing_points, len(unique))
            indices = np.sort(np.random.default_rng(geometry_seed).choice(len(unique), n, replace=False))
            self.inducing = self.z[indices]
            self.lu, used = stable_cholesky(rbf(self.inducing, self.inducing, self.lengthscale), self.gp.jitter)
            self.dtype = self.lu.dtype; self.z = self.z.to(self.dtype); self.inducing = self.inducing.to(self.dtype)
            sufficient=torch.eye(n,device=self.device,dtype=self.dtype)
            for part in self.z.split(self.gp.algebra_batch_size):
                a = torch.linalg.solve_triangular(self.lu, rbf(self.inducing, part, self.lengthscale), upper=False)
                sufficient+=a@a.T/self.gp.ridge
            self.ls, _ = stable_cholesky(sufficient, self.gp.jitter)
        else:
            k = rbf(self.z, self.z, self.lengthscale)
            self.factor, used = stable_cholesky(k + self.gp.ridge * torch.eye(len(k), device=self.device, dtype=self.dtype), self.gp.jitter)
            self.dtype = self.factor.dtype; self.z = self.z.to(self.dtype)
        self.diagnostics = {**geometry, "support_rows": len(x), "unique_rows": len(unique),
                            "multiplicities": counts.tolist(), "lengthscale": self.lengthscale,
                            "ridge": self.gp.ridge, "jitter_used": used, "precision": str(self.dtype),
                            "inference": self.gp.inference, "fit_seconds": time.perf_counter() - started,
                            "components": projected.shape[1],"geometry_fit_rows":len(fit_x),"lengthscale_rows":min(len(unique),4096)}
        self.diagnostics.update(self.projection.diagnostics, **target_diagnostics)
        return self

    def _covariance(self, x, y):
        prior = rbf(x, y, self.lengthscale)
        if self.sparse:
            ax = torch.linalg.solve_triangular(self.lu, rbf(self.inducing, x, self.lengthscale), upper=False)
            ay = torch.linalg.solve_triangular(self.lu, rbf(self.inducing, y, self.lengthscale), upper=False)
            return prior - ax.T @ ay + ax.T @ torch.cholesky_solve(ay, self.ls)
        vx = torch.linalg.solve_triangular(self.factor, rbf(self.z, x, self.lengthscale), upper=False)
        vy = torch.linalg.solve_triangular(self.factor, rbf(self.z, y, self.lengthscale), upper=False)
        return prior - vx.T @ vy

    @torch.no_grad()
    def cross_covariance(self, x, y):
        return self._covariance(self.tensor(self.projection.transform(x)), self.tensor(self.projection.transform(y)))

    @torch.no_grad()
    def score(self, x, batch_size=256):
        z = self.tensor(self.projection.transform(x))
        return torch.cat([nonnegative_variance(self._covariance(q, q).diagonal()) for q in z.split(batch_size)]).cpu().numpy()

    def save(self, path):
        path = Path(path)
        arrays = {**self.projection.arrays(), "support": self.z.cpu().numpy()}
        if self.sparse:
            arrays.update(inducing=self.inducing.cpu().numpy(), lu=self.lu.cpu().numpy(), ls=self.ls.cpu().numpy())
        else:
            arrays["factor"] = self.factor.cpu().numpy()
        atomic_npz(path / "estimator.npz", **arrays)
        atomic_json(path / "estimator_fit.json", {"method": "kv", **self.diagnostics})

    @classmethod
    def load(cls, path, feature_config, gp_config, *, device="cpu"):
        path = Path(path); model = cls(feature_config, gp_config, device=device)
        info = read_json(path / "estimator_fit.json")
        model.dtype = getattr(torch, info["precision"].split(".")[-1])
        with np.load(path / "estimator.npz", allow_pickle=False) as a:
            model.projection = Projection.restore(a, feature_config.compression)
            model.z = model.tensor(a["support"])
            model.sparse = info["inference"] == "sparse"
            for key in (("inducing", "lu", "ls") if model.sparse else ("factor",)):
                setattr(model, key, model.tensor(a[key]))
        model.lengthscale = info["lengthscale"]; model.diagnostics = info
        return model

    def covariance_components(self):
        return [(self.cross_covariance, np.asarray([self.gp.ridge]), 1.)]
