"""Matched input geometry for fixed KV and double-precision batched KV-T."""
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch

from uq_extensions.common import atomic_json, read_json
from uq_extensions.config import FeatureConfig, GPConfig
from uq_extensions.estimators import flatten_groups, probe_weights
from uq_extensions.kernel_variance import KernelVariance, geometry_lengthscale, rbf, stable_cholesky
from uq_extensions.preprocessing import Projection, fit_projection, fit_output_projection, whiten_projection
from uq_extensions.target_gp import TargetGP


class FollowupTargetGP(TargetGP):
    def __init__(self, features, gp, *, device='cpu', variant=None, whiten=False, conditioning_columns=0):
        super().__init__(features, gp, device=device)
        self.variant = variant or gp.input_projection
        self.whiten = whiten
        self.conditioning_columns = conditioning_columns

    def _geometry(self, x, y):
        if self.conditioning_columns:
            return self._conditioned_geometry(x, y)
        projection, output, targets = super()._geometry(x, y)
        if self.variant == 'pca_adaptive':
            cumulative = np.cumsum(projection.explained_variance)
            count = min(len(cumulative), int(np.searchsorted(cumulative, .99)) + 1)
            projection = replace(projection, weights=projection.weights[:, :count],
                                 explained_variance=projection.explained_variance[:count])
            self.geometry_diagnostics.update(pca_variance_target=.99,
                pca_variance_retained=float(cumulative[count-1]), pca_criterion_met=bool(cumulative[count-1] >= .99),
                pca_max_components=512, pca_actual_components=count)
        if self.whiten:
            projection = whiten_projection(projection, x, variance_floor=self.features.variance_floor)
            self.geometry_diagnostics.update(projection.diagnostics)
        return projection, output, targets

    def _conditioned_geometry(self, x, y):
        from sklearn.metrics import r2_score
        tail = self.conditioning_columns
        if x.shape[1] <= tail: raise ValueError('Missing embedding before conditioning columns')
        core, conditioning = x[:, :-tail], x[:, -tail:]
        output, targets = fit_output_projection(y, method=self.gp.output_compression,
            components=self.gp.output_components, seed=self.gp.output_seed or 142)
        mean = conditioning.mean(0); scale = np.maximum(conditioning.std(0), self.features.variance_floor)
        c = (conditioning-mean)/scale
        dims = [self.gp.input_components]
        if self.variant == 'pls_adaptive': dims = [32, 128, 512]
        checks = []
        for dim in dims:
            projection = fit_projection(core, method=self.gp.input_projection, components=dim,
                seed=self.features.seed, variance_floor=self.features.variance_floor, targets=targets)
            if self.variant == 'pca_adaptive':
                cumulative = np.cumsum(projection.explained_variance)
                n = min(len(cumulative), int(np.searchsorted(cumulative, .99))+1)
                projection = replace(projection, weights=projection.weights[:, :n], explained_variance=projection.explained_variance[:n])
            if self.gp.input_projection != 'pls': break
            z = np.column_stack([np.ones(len(x)), projection.transform(core), c])
            prediction = z @ np.linalg.lstsq(z, targets, rcond=None)[0]
            scales = 1/np.linalg.norm(output.weights, axis=0)
            score = float(r2_score(targets*scales, prediction*scales, multioutput='variance_weighted'))
            checks.append(dict(requested=dim, actual=projection.weights.shape[1], retained_target_variance_explained=score))
            print(f'PLS {dim} + complete conditioning: retained-target variance explained={score:.6f}', flush=True)
            if self.variant != 'pls_adaptive' or score >= .99: break
        n = projection.weights.shape[1]
        weights = np.zeros((x.shape[1], n+tail))
        weights[:-tail, :n] = projection.weights
        weights[-tail:, n:] = np.eye(tail)
        combined = Projection(np.r_[projection.mean, mean], np.r_[projection.scale, scale],
            np.r_[projection.center, np.zeros(tail)], weights, projection.method, projection.explained_variance)
        if self.whiten:
            # Appended conditioning correlates with embedding components, so
            # diagonal rescaling of PCA/PLS axes is not full covariance whitening.
            whitened = whiten_projection(replace(combined, method='random'), x,
                                          variance_floor=self.features.variance_floor)
            combined = replace(whitened, method=projection.method+'_whiten')
        retained = float(projection.explained_variance.sum()) if len(projection.explained_variance) else None
        self.geometry_diagnostics = dict(embedding_components=n, conditioning_columns=tail,
            actual_input_components=n+tail, conditioning_preserved=True,
            input_dimension_cap_applies_to='embedding components; conditioning appended intact',
            pca_variance_retained=retained, pca_criterion_met=(retained >= .99) if retained is not None else None,
            pls_checks=checks, pls_criterion_met=(checks[-1]['retained_target_variance_explained'] >= .99) if checks else None,
            pls_criterion_units='Retained output variance explained by PLS embedding components plus full conditioning; training only',
            output_pca_variance_retained=float(output.explained_variance.sum()), **combined.diagnostics)
        return combined, output, targets


def condition_fixed_kernel(projection, x, features, gp, device):
    """Condition KV on exactly the representation used by matched KV-T.

    Geometry uses all support rows (including flow probes), rather than fitting
    KV preprocessing on a different 4096-row subset. Kernel-scale estimation
    retains the existing bounded neighborhood calculation.
    """
    model = KernelVariance(features, gp, device=device)
    started = time.monotonic()
    model.projection = projection
    z, counts = np.unique(projection.transform(x), axis=0, return_counts=True)
    model.z = model.tensor(z)
    model.lengthscale, geometry = geometry_lengthscale(model.z)
    model.sparse = gp.inference == 'sparse'
    if model.sparse:
        seed = features.geometry_seed if features.geometry_seed is not None else features.seed
        ix = np.sort(np.random.default_rng(seed).choice(len(z), min(len(z), gp.inducing_points), replace=False))
        model.inducing = model.z[ix]
        model.lu, used = stable_cholesky(rbf(model.inducing, model.inducing, model.lengthscale), gp.jitter)
        model.dtype = model.lu.dtype
        model.z = model.z.to(model.dtype); model.inducing = model.inducing.to(model.dtype)
        sufficient = torch.eye(len(ix), device=device, dtype=model.dtype)
        for part in model.z.split(gp.algebra_batch_size):
            a = torch.linalg.solve_triangular(model.lu, rbf(model.inducing, part, model.lengthscale), upper=False)
            sufficient += a @ a.T / gp.ridge
        model.ls, _ = stable_cholesky(sufficient, gp.jitter)
    else:
        kernel = rbf(model.z, model.z, model.lengthscale)
        model.factor, used = stable_cholesky(kernel + gp.ridge*torch.eye(len(z), device=device, dtype=model.dtype), gp.jitter)
        model.dtype = model.factor.dtype; model.z = model.z.to(model.dtype)
    model.diagnostics = dict(geometry, method='kv', support_rows=len(x), unique_rows=len(z),
        components=z.shape[1], multiplicities=counts.tolist(), lengthscale=model.lengthscale,
        ridge=gp.ridge, jitter_used=used, precision=str(model.dtype), inference=gp.inference,
        geometry_fit_rows=len(x), fit_seconds=time.monotonic()-started, **projection.diagnostics)
    return model


class Estimator:
    def __init__(self, row, device):
        self.row = row; self.device = device
        self.features = FeatureConfig(**row['features']); self.gp = GPConfig(**row['gp'])

    def fit(self, x, y, tasks, directory):
        x, probes = flatten_groups(x)
        if y is not None:
            y, other = flatten_groups(y)
            if probes != other: raise ValueError('Unaligned input/target probes')
        directory = Path(directory)
        target = self.row['method'] == 'kv_t'
        if target:
            self.model = FollowupTargetGP(self.features, self.gp, device=self.device,
                variant=self.row['variant'], whiten=self.row['whiten'], conditioning_columns=self.row.get('conditioning_columns', 0))
            self.model.training_directory = directory / 'optimizer'
            self.model.fit(x, y, tasks=np.repeat(tasks, probes), groups=np.repeat(np.arange(len(tasks)), probes))
        else:
            if self.row['group'] in ('early_to_decoder', 'middle_to_decoder', 'middle_to_final_context', 'decoder_to_velocity'):
                geometry = FollowupTargetGP(self.features, self.gp, variant=self.row['variant'], whiten=self.row['whiten'],
                                            conditioning_columns=self.row.get('conditioning_columns', 0))
                projection, _, _ = geometry._geometry(x, y)
                diagnostics = geometry.geometry_diagnostics
            else:
                projection = fit_projection(x, method=self.features.compression,
                    components=self.features.components, seed=self.features.seed,
                    variance_floor=self.features.variance_floor)
                diagnostics = {}
            self.model = condition_fixed_kernel(projection, x, self.features, self.gp, self.device)
            self.model.diagnostics['geometry'] = diagnostics
            self.model.diagnostics['target_informed_projection'] = self.row['variant'].startswith('pls')
        self.model.diagnostics['support_probes_per_episode'] = probes
        self.model.diagnostics['support_episodes'] = len(tasks)
        self.model.save(directory)
        self.diagnostics = self.model.diagnostics
        return self

    @classmethod
    def load(cls, row, directory, device):
        obj = cls(row, device)
        constructor = TargetGP if row['method'] == 'kv_t' else KernelVariance
        obj.model = constructor.load(directory, obj.features, obj.gp, device=device)
        obj.diagnostics = obj.model.diagnostics
        return obj

    def score(self, x):
        # Bound host-memory projections as well as GPU prediction batches.
        result = []
        for start in range(0, len(x), 16):
            batch, count = flatten_groups(x[start:start+16])
            result.extend(self.model.score(batch).reshape(-1, count) @ probe_weights(count, self.features))
        return np.asarray(result)
