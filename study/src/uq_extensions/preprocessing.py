"""Training-only standardization, random coordinates, PCA and PLS2."""
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.covariance import LedoitWolf
from .cache_lock import cache_lock


class UninformativeBank(ValueError):
    pass


def matrix(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError("Expected a finite [examples, features] matrix")
    return x


@dataclass
class Projection:
    mean: np.ndarray
    scale: np.ndarray
    center: np.ndarray
    weights: np.ndarray
    method: str
    explained_variance: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    def transform(self, x):
        x = matrix(x)
        if x.shape[1] != len(self.mean):
            raise ValueError("Feature width differs from fitted projection")
        return (((x - self.mean) / self.scale - self.center) @ self.weights).astype(np.float32)

    def arrays(self, prefix="projection_"):
        result = {prefix + k: getattr(self, k) for k in ("mean", "scale", "center", "weights", "explained_variance")}
        if self.diagnostics:
            result[prefix+'diagnostics'] = np.asarray(json.dumps(self.diagnostics, sort_keys=True))
        return result

    @classmethod
    def restore(cls, arrays, method, prefix="projection_"):
        diagnostics = json.loads(str(arrays[prefix+'diagnostics'])) if prefix+'diagnostics' in arrays else {}
        return cls(method=method, diagnostics=diagnostics,
                   **{k: arrays[prefix + k] for k in ("mean", "scale", "center", "weights", "explained_variance")})


def shrunk_covariance(x):
    """Training-only Ledoit-Wolf covariance; no validation-score tuning."""
    x = matrix(x)
    estimator = LedoitWolf(store_precision=False).fit(x)
    covariance = estimator.covariance_
    return covariance, dict(rule='ledoit_wolf', shrinkage=float(estimator.shrinkage_),
                            training_rows=len(x), dimensions=x.shape[1], training_only=True)


def whiten_projection(base, x, *, variance_floor=1e-8):
    # Fit in float64 before combining transforms; do not change the selected subspace.
    z = ((matrix(x)-base.mean)/base.scale-base.center) @ base.weights
    covariance, diagnostics = shrunk_covariance(z)
    if base.method == 'random':
        eigenvalues, axes = np.linalg.eigh(covariance)
        eigenvalues, axes = eigenvalues[::-1], axes[:, ::-1]
        diagnostics['basis'] = 'covariance_principal_directions'
    else:
        # PCA/PLS scores are already decorrelated: preserve their component axes.
        eigenvalues, axes = covariance.diagonal().copy(), np.eye(z.shape[1])
        diagnostics['basis'] = 'existing_component_axes'
    floor = max(variance_floor, float(eigenvalues.max())*1e-10)
    scales = np.sqrt(np.maximum(eigenvalues, floor))
    diagnostics.update(variance_floor=floor, floored_directions=int((eigenvalues<floor).sum()),
        minimum_regularized_variance=float(eigenvalues.min()), maximum_regularized_variance=float(eigenvalues.max()),
        scales=scales.tolist(), base_method=base.method)
    return replace(base, method=base.method+'_whiten', weights=base.weights @ (axes/scales),
                   diagnostics=dict(whitening=diagnostics))


def fit_projection(x, *, method="pca", components=512, seed=42, variance_floor=1e-8, targets=None):
    x = matrix(x)
    cache = None
    if os.environ.get('UQ_EXT_PROJECTION_CACHE'):
        from .common import identity, array_hash
        key = identity(dict(x=array_hash(x), targets=array_hash(np.asarray(targets)) if targets is not None and method.removesuffix('_whiten')=='pls' else None,
                            method=method, components=components, seed=seed, floor=variance_floor, version=2))
        cache = Path(os.environ['UQ_EXT_PROJECTION_CACHE']) / (key+'.npz')
        with cache_lock(cache.with_suffix('.lock'),label=f'{method} projection'):
            if cache.exists():
                with np.load(cache,allow_pickle=False) as arrays:return Projection.restore(arrays,method)
            projection=_fit_projection(x,method=method,components=components,seed=seed,
                                       variance_floor=variance_floor,targets=targets)
            from .common import atomic_npz
            atomic_npz(cache,**projection.arrays())
            return projection
    return _fit_projection(x,method=method,components=components,seed=seed,
                           variance_floor=variance_floor,targets=targets)


def _fit_projection(x, *, method, components, seed, variance_floor, targets):
    if method.endswith('_whiten'):
        base = fit_projection(x, method=method.removesuffix('_whiten'), components=components,
                              seed=seed, variance_floor=variance_floor, targets=targets)
        return whiten_projection(base, x, variance_floor=variance_floor)
    if len(x) < 2:
        raise UninformativeBank("At least two support episodes are required")
    mean = x.mean(0)
    std = x.std(0)
    active = np.flatnonzero(std > variance_floor)
    if not len(active):
        raise UninformativeBank("Feature bank is constant on the support episodes")
    scale = np.maximum(std, variance_floor)
    standardized = (x - mean) / scale
    n = min(components, len(x) - 1, len(active))
    if n < 1:
        raise UninformativeBank("Feature bank has no usable rank")
    if method == "random":
        coords = np.sort(np.random.default_rng(seed).choice(active, min(components, len(active)), replace=False))
        weights = np.eye(x.shape[1], len(coords), dtype=float) * 0
        weights[coords, np.arange(len(coords))] = 1
        return Projection(mean, scale, np.zeros(x.shape[1]), weights, method, np.array([]))
    if method == "pca":
        # Repeated instructions must not request hundreds of zero PCs.
        if n > 32:n=min(n,len(np.unique(x,axis=0))-1)
        if n < 1:raise UninformativeBank('PCA has no nonzero support directions')
        pca = PCA(n_components=n, whiten=False, svd_solver="randomized", random_state=seed)
        pca.fit(standardized[:, active])
        valid = pca.explained_variance_ > max(pca.explained_variance_[0] * 1e-10, variance_floor)
        weights = np.zeros((x.shape[1], int(valid.sum())))
        weights[active] = pca.components_[valid].T
        center = np.zeros(x.shape[1]); center[active] = pca.mean_
        if not valid.any():
            raise UninformativeBank("PCA has no nonzero support directions")
        return Projection(mean, scale, center, weights, method, pca.explained_variance_ratio_[valid])
    if method == "pls":
        y = matrix(targets)
        if len(y) != len(x):
            raise ValueError("PLS inputs and targets must be episode-aligned")
        pls = PLSRegression(n_components=n, scale=False, max_iter=1000, tol=1e-7)
        pls.fit(standardized[:, active], y)
        rotation = pls.x_rotations_
        valid = np.isfinite(rotation).all(0) & (np.linalg.norm(rotation, axis=0) > 1e-10)
        weights = np.zeros((x.shape[1], int(valid.sum())))
        weights[active] = rotation[:, valid]
        if not valid.any():
            raise UninformativeBank("PLS found no usable input directions")
        # sklearn centers X even with scale=False. Save that fitted mean explicitly.
        center = np.zeros(x.shape[1]); center[active] = standardized[:, active].mean(0)
        return Projection(mean, scale, center, weights, method, np.array([]))
    raise ValueError(f"Unknown projection method: {method}")


def fit_output_pca(y, components=16, seed=42, variance_floor=1e-8):
    """Center (do not standardize source neurons), PCA, then standardize PCs."""
    y = matrix(y)
    pca = PCA(n_components=min(components, len(y) - 1, y.shape[1]), svd_solver="randomized", random_state=seed)
    scores = pca.fit_transform(y)
    std = scores.std(0)
    valid = std > max(variance_floor, std.max() * 1e-8)
    if not valid.any():
        raise UninformativeBank("Downstream targets have no usable variation")
    weights = pca.components_[valid].T / std[valid]
    projection = Projection(pca.mean_, np.ones(y.shape[1]), np.zeros(y.shape[1]), weights,
                            "output_pca", pca.explained_variance_ratio_[valid])
    return projection, projection.transform(y)


def fit_output_projection(y, *, method='pca', components=16, seed=42, variance_floor=1e-8):
    if method=='pca':
        return fit_output_pca(y, components, seed, variance_floor)
    if method!='random':
        raise ValueError('Unknown target projection')
    projection=fit_projection(y, method='random', components=components, seed=seed, variance_floor=variance_floor)
    projection.method='output_random'
    return projection,projection.transform(y)
