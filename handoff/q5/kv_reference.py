"""Reference implementation of the paper's two kernel-variance heads (numpy, scikit-learn, torch).

KVCtxHead  -- the KV-Ctx recipe. Use it for KV-Ctx and for every context-specific head.
KVDecHead  -- the KV-Dec recipe (identical to KernelVariance(ridge=0.01, projection='whiten',
              seed=0) in lerobot/uncertainty/uncertainty_scoring/kernel_variance.py).

Both take a [rows, features] array of support embeddings and score query embeddings
with the GP conditional variance u(z) = k(z,z) - k(z,Z)(K_ZZ + lambda I)^-1 k(Z,z),
prior variance 1. Check any other implementation against kv_fixture.npz:

    python kv_reference.py kv_fixture.npz
"""
import math
import sys

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


class KVCtxHead:
    """Standardize, PCA (<=512 components), Ledoit-Wolf whitening on the component axes,
    k-nearest-neighbour bandwidth, exact GP with ridge 1e-3, float32 kernel algebra.

    Mirrors uq_followup.estimators.Estimator with the final study's H1 recipe.
    """

    def __init__(self, components=512, ridge=1e-3, jitter=1e-6, seed=42, variance_floor=1e-8, device='cpu'):
        self.components, self.ridge, self.jitter = components, ridge, jitter
        self.seed, self.floor, self.device = seed, variance_floor, torch.device(device)

    def _project(self, x):
        return (((np.asarray(x, np.float64)-self.mean)/self.scale-self.center) @ self.weights).astype(np.float32)

    def fit(self, support):
        x = np.asarray(support, np.float64)
        # 1. Coordinate-wise standardization (population std).
        self.mean = x.mean(0); std = x.std(0)
        active = np.flatnonzero(std > self.floor)
        self.scale = np.maximum(std, self.floor)
        standardized = (x-self.mean)/self.scale
        # 2. PCA. Above 32 components, never request more than (#unique rows - 1).
        n = min(self.components, len(x)-1, len(active))
        if n > 32:
            n = min(n, len(np.unique(x, axis=0))-1)
        pca = PCA(n_components=n, whiten=False, svd_solver='randomized', random_state=self.seed)
        pca.fit(standardized[:, active])
        valid = pca.explained_variance_ > max(pca.explained_variance_[0]*1e-10, self.floor)
        self.weights = np.zeros((x.shape[1], int(valid.sum()))); self.weights[active] = pca.components_[valid].T
        self.center = np.zeros(x.shape[1]); self.center[active] = pca.mean_
        # 3. Ledoit-Wolf shrinkage covariance of the component scores; scale each component axis.
        scores = ((x-self.mean)/self.scale-self.center) @ self.weights
        variances = LedoitWolf(store_precision=False).fit(scores).covariance_.diagonal().copy()
        floor = max(self.floor, float(variances.max())*1e-10)
        self.weights = self.weights/np.sqrt(np.maximum(variances, floor))
        # 4. Deduplicate support rows in the final geometry.
        z = torch.as_tensor(np.unique(self._project(x), axis=0), device=self.device)
        # 5. Bandwidth: kernel = 0.5 at the median distance to the k-th nearest support neighbour, k = ceil(sqrt(N)).
        sample = z
        if len(z) > 4096:
            sample = z[np.sort(np.random.default_rng(0).choice(len(z), 4096, replace=False))]
        k = min(len(sample)-1, math.ceil(math.sqrt(len(sample))))
        distances = torch.cdist(sample, sample); distances.fill_diagonal_(float('inf'))
        median = torch.quantile(distances.kthvalue(k, dim=1).values, 0.5).item()
        self.lengthscale = median/math.sqrt(2*math.log(2))
        # 6. Exact conditioning: Cholesky of K + (ridge + jitter) I, escalating jitter only on failure.
        kernel = self._kernel(z, z)
        eye = torch.eye(len(z), dtype=z.dtype, device=self.device)
        jitter = self.jitter
        for _ in range(7):
            factor, info = torch.linalg.cholesky_ex((kernel+kernel.T)/2+(self.ridge+jitter)*eye)
            if not info.any() and torch.isfinite(factor).all():
                break
            jitter *= 10
        else:
            raise RuntimeError('Cholesky failed')
        self.z, self.factor = z, factor
        return self

    def _kernel(self, a, b):
        return torch.exp(-0.5*torch.cdist(a/self.lengthscale, b/self.lengthscale).square())

    @torch.no_grad()
    def score(self, x, batch_size=256):
        q = torch.as_tensor(self._project(x), device=self.device)
        out = []
        for part in q.split(batch_size):
            v = torch.linalg.solve_triangular(self.factor, self._kernel(self.z, part), upper=False)
            out.append((1-v.square().sum(0)).clamp_min(0))
        return torch.cat(out).cpu().numpy()


class KVDecHead:
    """Standardize, full Ledoit-Wolf whitening (eigendecomposition), bandwidth = median pairwise
    distance of up to 4096 support rows, exact GP with ridge 1e-2, no deduplication, float32."""

    def __init__(self, ridge=1e-2, seed=0, device='cpu'):
        self.ridge, self.seed, self.device = ridge, seed, torch.device(device)

    @staticmethod
    def _ledoit_wolf(rows):
        n, d = rows.shape
        x = rows-rows.mean(dim=0); s = x.T @ x/n
        mu = torch.trace(s)/d
        delta = (s-mu*torch.eye(d, dtype=s.dtype, device=s.device)).square().sum()/d
        beta = (x.square().sum(dim=1).square().sum()-n*s.square().sum())/(n*n*d)
        shrink = torch.clamp(beta/delta, max=1.0) if delta > 0 else torch.tensor(1.0)
        return shrink*mu*torch.eye(d, dtype=s.dtype, device=s.device)+(1-shrink)*s

    def _prepare(self, rows):
        return ((rows-self.mean)/self.scale) @ self.basis

    def fit(self, support):
        x = torch.as_tensor(np.asarray(support), dtype=torch.float32, device=self.device)
        self.mean = x.mean(0); self.scale = x.std(0, unbiased=False).clamp_min(1e-6)
        values, vectors = torch.linalg.eigh(self._ledoit_wolf(((x-self.mean)/self.scale).double()))
        self.basis = ((vectors*values.clamp_min(1e-10).rsqrt()) @ vectors.T).to(x.dtype)
        z = self._prepare(x)
        count = min(4096, len(z))
        index = torch.randperm(len(z), generator=torch.Generator().manual_seed(self.seed))[:count].to(self.device)
        distances = torch.cdist(z[index], z[index])
        mask = ~torch.eye(count, dtype=torch.bool, device=self.device)
        self.lengthscale = float(distances[mask].median().clamp_min(1e-6))
        gram = self._kernel(z, z); gram.diagonal().add_(self.ridge)
        self.factor, self.z = torch.linalg.cholesky(gram), z
        return self

    def _kernel(self, a, b):
        return torch.exp(-torch.cdist(a, b).square()/(2.0*self.lengthscale**2))

    @torch.no_grad()
    def score(self, x):
        q = self._prepare(torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device))
        v = torch.linalg.solve_triangular(self.factor, self._kernel(q, self.z).T, upper=False)
        return (1.0-v.square().sum(0)).clamp(0.0, 1.0).cpu().numpy()


def check(path):
    data = np.load(path)
    worst = 0.0
    for name in ('ctx', 'dense', 'state8'):
        got = KVCtxHead().fit(data[f'{name}_support']).score(data[f'{name}_query'])
        diff = float(np.abs(got-data[f'{name}_expected']).max()); worst = max(worst, diff)
        print(f'KV-Ctx recipe, {name:6s}: max |difference| {diff:.2e}')
    got = KVDecHead().fit(data['dec_support']).score(data['dec_query'])
    diff = float(np.abs(got-data['dec_expected']).max()); worst = max(worst, diff)
    print(f'KV-Dec recipe, dec   : max |difference| {diff:.2e}')
    print('PASS' if worst < 1e-4 else 'FAIL', f'(tolerance 1e-4, worst {worst:.2e})')
    return worst < 1e-4


if __name__ == '__main__':
    sys.exit(0 if check(sys.argv[1] if len(sys.argv) > 1 else 'kv_fixture.npz') else 1)
