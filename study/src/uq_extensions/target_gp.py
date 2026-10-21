"""Independent Gaussian regressions on downstream PCs, outside the policy.

Exact inference is the default. The optional Gaussian variational fallback uses
the analytically optimal full inducing covariance and the collapsed ELBO.
"""
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import r2_score

from .common import atomic_json, atomic_npz, read_json
from .kernel_variance import geometry_lengthscale, initial_log_parameters, nonnegative_variance, rbf, stable_cholesky
from .preprocessing import Projection, fit_output_projection, fit_projection, matrix


def split_episodes(tasks, seed=42, groups=None):
    if groups is not None:
        groups=np.asarray(groups)
        unique,first=np.unique(groups,return_index=True)
        for g,i in zip(unique,first):
            if len(set(np.asarray(tasks)[groups==g]))!=1:raise ValueError('Episode group spans tasks')
        fit,test=split_episodes(np.asarray(tasks)[first],seed)
        return np.flatnonzero(np.isin(groups,unique[fit])),np.flatnonzero(np.isin(groups,unique[test]))
    tasks = np.asarray(tasks)
    rng = np.random.default_rng(seed)
    train, test = [], []
    for task in sorted(set(tasks)):
        indices = rng.permutation(np.flatnonzero(tasks == task))
        count = min(len(indices) - 1, max(1, round(.2 * len(indices)))) if len(indices) > 1 else 0
        test.extend(indices[:count]); train.extend(indices[count:])
    if len(train) < 3 or len(test) < 2:
        raise ValueError("Too few episodes for an independent 80/20 mapping diagnostic")
    return np.asarray(sorted(train)), np.asarray(sorted(test))


class ScalarGP:
    def __init__(self, config, device="cpu"):
        self.config, self.device = config, torch.device(device)
        self.dtype = getattr(torch, config.precision)

    def tensor(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def _terms(self, length, amplitude, noise, *, kernel_jitter=None):
        kernel_jitter = self.config.jitter if kernel_jitter is None else kernel_jitter
        if self.config.precision == 'float64':
            from .batched_gp import terms
            logs = torch.cat((length.reshape(-1), amplitude.reshape(1), noise.reshape(1))).log()[None, :]
            loss, posterior = terms(self.x, self.y[None, :], self.inducing[None, :, :], logs,
                                    self.config, kernel_jitter=kernel_jitter)
            return loss[0], tuple(t[0] if isinstance(t, torch.Tensor) else t for t in posterior)
        if self.config.inference == "exact":
            gram = rbf(self.x, self.x, length, amplitude)
            factor, jitter = stable_cholesky(gram + noise * torch.eye(len(self.x), device=self.device, dtype=self.dtype), kernel_jitter)
            y = self.y.to(factor.dtype)
            alpha = torch.cholesky_solve(y[:, None], factor)
            loss = .5 * (y @ alpha[:, 0] + 2 * factor.diagonal().log().sum())
            return loss, (factor, alpha, jitter)
        lu, jitter = stable_cholesky(rbf(self.inducing, self.inducing, length, amplitude), kernel_jitter)
        a = torch.linalg.solve_triangular(lu, rbf(self.inducing, self.x, length, amplitude).to(lu.dtype), upper=False)
        b = torch.eye(len(lu), dtype=lu.dtype, device=self.device) + a @ a.T / noise
        lb, _ = stable_cholesky(b, self.config.jitter)
        ay = a @ self.y.to(a.dtype)
        beta = torch.cholesky_solve(ay[:, None], lb) / noise
        quadratic = self.y.square().sum() / noise - ay @ beta[:, 0] / noise
        residual = (len(self.x) * amplitude - a.square().sum()).clamp_min(0)
        loss = .5 * (quadratic + len(self.x) * noise.log() + 2 * lb.diagonal().log().sum() + residual / noise)
        return loss, (lu, lb, beta, jitter)

    def fit(self, x, y, *, seed=42):
        self.x, self.y = self.tensor(x), self.tensor(y)
        initial, self.initialization = initial_log_parameters(self.x, self.config)
        self.inducing = self.x[np.sort(np.random.default_rng(seed).choice(len(x), min(len(x), self.config.inducing_points), replace=False))]
        logs = torch.nn.Parameter(self.tensor(initial))
        optimizer = torch.optim.Adam([logs], lr=self.config.learning_rate)
        history = []
        # BEGIN RUN PROGRESS
        from run_progress import Progress
        progress=Progress(f'GP output seed={seed} {self.config.inference}',self.config.fit_steps,every=max(1,self.config.fit_steps//4))
        # END RUN PROGRESS
        for _ in range(self.config.fit_steps):
            optimizer.zero_grad()
            values = logs.exp()
            loss, _ = self._terms(values[:-2], values[-2], values[-1])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite GP marginal likelihood/variational bound")
            loss.backward()
            if not torch.isfinite(logs.grad).all():
                raise RuntimeError("Nonfinite kernel hyperparameter gradient")
            optimizer.step()
            with torch.no_grad():
                logs.clamp_(-13.8, 9.2)
            history.append(float(loss.detach()))
            # BEGIN RUN PROGRESS
            progress.update(detail=f'objective={history[-1]:.6g}')
            # END RUN PROGRESS
        values = logs.detach().exp()
        self.length, self.amplitude, self.noise = values[:-2], values[-2], values[-1]
        self.history = history
        # BEGIN RUN PROGRESS
        progress.finish(f'final objective={history[-1]:.6g}')
        # END RUN PROGRESS
        self.rebuild()
        return self

    def rebuild(self, *, kernel_jitter=None):
        with torch.no_grad():
            _, self.terms = self._terms(self.length, self.amplitude, self.noise, kernel_jitter=kernel_jitter)

    @torch.no_grad()
    def predict(self, x, y=None):
        x = self.tensor(x)
        y = x if y is None else self.tensor(y)
        prior = rbf(x, y, self.length, self.amplitude)
        if self.config.inference == "exact":
            factor, alpha, _ = self.terms
            kx = rbf(self.x, x, self.length, self.amplitude).to(factor.dtype)
            ky = rbf(self.x, y, self.length, self.amplitude).to(factor.dtype)
            vx = torch.linalg.solve_triangular(factor, kx, upper=False)
            vy = torch.linalg.solve_triangular(factor, ky, upper=False)
            return (kx.T @ alpha)[:, 0], prior - vx.T @ vy
        lu, lb, beta, _ = self.terms
        ax = torch.linalg.solve_triangular(lu, rbf(self.inducing, x, self.length, self.amplitude).to(lu.dtype), upper=False)
        ay = torch.linalg.solve_triangular(lu, rbf(self.inducing, y, self.length, self.amplitude).to(lu.dtype), upper=False)
        return (ax.T @ beta)[:, 0], prior - ax.T @ ay + ax.T @ torch.cholesky_solve(ay, lb)


class TargetGP:
    def __init__(self, feature_config, gp_config, *, device="cpu"):
        self.features, self.gp, self.device = feature_config, gp_config, device
        self.training_directory = None
        self.training_records = {}
        self.prediction_numerics = dict(training_precision=gp_config.precision,
                                       posterior_precision=gp_config.precision, float64_retries=0)
        if gp_config.inference == "sparse" and not gp_config.sparse_justification:
            raise ValueError("Sparse KV-T requires a measured scaling justification")

    def _geometry(self, x, y):
        input_method = self.gp.input_projection.removesuffix('_whiten')
        output, targets = fit_output_projection(y,method=self.gp.output_compression,components=self.gp.output_components,
                                               seed=self.features.seed if self.gp.output_seed is None else self.gp.output_seed,
                                               variance_floor=self.features.variance_floor)
        requested = min(self.gp.input_components, self.gp.pls_max_components) if input_method == 'pls' else self.gp.input_components
        dimensions = [requested]
        if input_method == 'pls' and self.gp.pls_variance_target is not None:
            dimensions += [n for n in (128, 512) if requested < n <= self.gp.pls_max_components]
        checks = []
        target_scales = (1/np.linalg.norm(output.weights,axis=0) if output.method=='output_pca'
                         else np.sqrt(((output.scale[:,None]*output.weights)**2).sum(0)))
        for n in dimensions:
            if checks and min(n,len(x)-1,x.shape[1])<=checks[-1]['actual']:continue
            projection = fit_projection(x, method=self.gp.input_projection, components=n,
                                        seed=self.features.seed, variance_floor=self.features.variance_floor, targets=targets)
            if input_method != 'pls':
                break
            z = projection.transform(x)
            design = np.column_stack([np.ones(len(z)), z])
            fitted = design @ np.linalg.lstsq(design, targets, rcond=None)[0]
            score = float(r2_score(targets*target_scales, fitted*target_scales, multioutput='variance_weighted'))
            checks.append(dict(requested=n, actual=z.shape[1], retained_target_variance_explained=score))
            print(f'PLS input {z.shape[1]}: explained retained-target variance={score:.6f}; limit={self.gp.pls_max_components}', flush=True)
            if self.gp.pls_variance_target is None or score >= self.gp.pls_variance_target:
                break
        self.geometry_diagnostics = dict(pls_checks=checks, pls_variance_target=self.gp.pls_variance_target,
            pls_criterion_met=(checks[-1]['retained_target_variance_explained'] >= (self.gp.pls_variance_target or .99)) if checks else None,
            pls_max_components=self.gp.pls_max_components,
            pls_criterion_units='Variance-weighted R2 of retained targets in their original units; training only',
            output_pca_variance_retained=float(output.explained_variance.sum()) if len(output.explained_variance) else None)
        self.geometry_diagnostics.update(projection.diagnostics)
        return projection, output, targets

    def _fit_outputs(self, z, targets, *, phase='full', validation=None):
        from .batched_gp import fit_batched
        models, records = fit_batched(z, targets, self.gp, device=self.device,
            seed=self.features.seed if self.gp.training_seed is None else self.gp.training_seed,
            directory=Path(self.training_directory)/phase if self.training_directory else None,
            validation=validation, label=f'KV-T {phase}')
        self.training_records[phase] = records
        return models

    def fit(self, x, y, *, tasks, groups=None, mapping_check=True):
        started = time.perf_counter()
        x, y = matrix(x), matrix(y)
        if len(x) != len(y) or len(tasks) != len(x):
            raise ValueError("KV-T inputs, downstream targets and task IDs must align by episode")
        _, inverse, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)
        for group in np.flatnonzero(counts > 1):
            if np.max(np.ptp(y[inverse == group], axis=0)) > 1e-5:
                raise ValueError("Repeated inputs have different downstream targets: missing conditioning information")
        mapping = {"performed": False}
        if mapping_check:
            fit, test = split_episodes(tasks, self.features.seed if self.gp.split_seed is None else self.gp.split_seed,groups)
            proj, out, target = self._geometry(x[fit], y[fit])
            mapping_geometry = self.geometry_diagnostics.copy()
            zfit, ztest = proj.transform(x[fit]), proj.transform(x[test])
            expected = out.transform(y[test])
            linear = np.linalg.lstsq(np.column_stack((np.ones(len(fit)), zfit)), target, rcond=None)[0]
            linear_prediction = np.column_stack((np.ones(len(test)), ztest)) @ linear
            models = self._fit_outputs(zfit, target, phase='mapping_validation', validation=(ztest, expected))
            prediction = np.column_stack([gp.predict(ztest)[0].cpu().numpy() for gp in models])
            gp_r2 = r2_score(expected, prediction, multioutput="raw_values")
            mapping = {"performed": True, "fit_indices": fit.tolist(), "test_indices": test.tolist(),
                       "linear_r2": r2_score(expected, linear_prediction, multioutput="raw_values").tolist(),
                       "linear_variance_weighted_r2": float(r2_score(expected,linear_prediction,multioutput='variance_weighted')),
                       "gp_r2": gp_r2.tolist(), "usable": bool(np.median(gp_r2) > 0),
                       "geometry": mapping_geometry,
                       "transforms_fit_only_on_training_split": True,
                       "grouped_by_episode":groups is not None}
            del models
        self.projection, self.output_projection, targets = self._geometry(x, y)
        self.models = self._fit_outputs(self.projection.transform(x), targets)
        self.diagnostics = dict(method="kv_t", support_rows=len(x), output_count=len(self.models),
                                input_components=self.projection.weights.shape[1], mapping_check=mapping,
                                inference=self.gp.inference, fit_seconds=time.perf_counter() - started,
                                fit_steps=self.gp.fit_steps, learning_rate=self.gp.learning_rate,
                                batched_training=True, output_batch_size=self.gp.output_batch_size,
                                training_records=self.training_records, geometry=self.geometry_diagnostics,
                                log_parameter_bounds=[-13.8, 9.2], latent_variance_excludes_noise=True,
                                prediction_numerics=self.prediction_numerics,
                                output_projection=self.output_projection.method,
                                kernels=[dict(lengthscales=m.length.cpu().tolist(), amplitude=float(m.amplitude),
                                              noise_variance=float(m.noise), jitter_used=m.terms[-1],
                                              loss_history=m.history) for m in self.models])
        return self

    def score(self, x, batch_size=256):
        z = self.projection.transform(x)
        try:
            return self._score_projected(z, batch_size)
        except RuntimeError as error:
            if ('Materially negative or nonfinite latent variance' not in str(error)
                    or all(m.dtype == torch.float64 for m in self.models)):
                raise
            self._rebuild_prediction_float64(str(error))
            # Redo all query batches so a single score array uses one precision.
            return self._score_projected(z, batch_size)

    @torch.no_grad()
    def _rebuild_prediction_float64(self, reason):
        """Recompute kernels and factors, preserving fitted parameters and jitter.

        Casting an already inaccurate factor to double does not fix cancellation.
        No optimizer update, extra likelihood noise, or variance-floor change is
        involved. Subsequent scores and IVR covariances use the rebuilt models.
        """
        print(f'KV-T posterior precision retry: {reason}; rebuilding in float64 from saved '
              'hyperparameters (no refit).', flush=True)
        started = time.perf_counter()
        shared_x = self.models[0].x.double()
        rebuilt = []
        for original in self.models:
            model = ScalarGP(self.gp, original.device)
            model.dtype = torch.float64
            model.x = shared_x
            for name in ('y', 'inducing', 'length', 'amplitude', 'noise'):
                setattr(model, name, getattr(original, name).double())
            if hasattr(original, 'history'):
                model.history = original.history
            # Keep the effective support-kernel regularization of this fit.
            model.rebuild(kernel_jitter=original.terms[-1])
            rebuilt.append(model)
        self.models = rebuilt
        self.prediction_numerics.update(posterior_precision='float64',
            float64_retries=self.prediction_numerics['float64_retries'] + 1,
            retry_reason=reason, rebuild_seconds=time.perf_counter() - started,
            jitter_used=[m.terms[-1] for m in rebuilt])

    def _score_projected(self, z, batch_size):
        from .batched_gp import predict_diagonal
        models=self.models
        logs=torch.stack([torch.cat([m.length,m.amplitude[None],m.noise[None]]).log() for m in models])
        posterior=tuple(torch.stack([m.terms[i] for m in models]) for i in range(len(models[0].terms)-1))+(models[0].terms[-1],)
        inducing=torch.stack([m.inducing for m in models])
        output = []
        for start in range(0, len(z), batch_size):
            q = z[start:start + batch_size]
            _,variance=predict_diagonal(models[0].x,inducing,logs,posterior,models[0].tensor(q),self.gp)
            variance=nonnegative_variance(variance,torch.stack([m.amplitude for m in models])[:,None])
            output.append(variance.mean(0).cpu().numpy())
        return np.concatenate(output)

    def cross_covariance(self, x, y):
        x, y = self.projection.transform(x), self.projection.transform(y)
        return torch.stack([m.predict(x, y)[1] for m in self.models]).mean(0)

    def covariance_components(self):
        return [(lambda x,y,m=m:m.predict(self.projection.transform(x),self.projection.transform(y))[1],
                 np.asarray([float(m.noise)]),1/len(self.models)) for m in self.models]

    def save(self, path):
        path = Path(path)
        arrays = {**self.projection.arrays(), **self.output_projection.arrays("output_")}
        arrays['shared_x'] = self.models[0].x.cpu().numpy()
        for index, model in enumerate(self.models):
            for name in ("y", "inducing", "length", "amplitude", "noise"):
                arrays[f"gp_{index}_{name}"] = getattr(model, name).cpu().numpy()
        atomic_npz(path / "estimator.npz", **arrays)
        atomic_json(path / "estimator_fit.json", self.diagnostics)

    @classmethod
    def load(cls, path, feature_config, gp_config, *, device="cpu"):
        path = Path(path); model = cls(feature_config, gp_config, device=device)
        model.diagnostics = read_json(path / "estimator_fit.json")
        model.diagnostics['prediction_numerics'] = model.prediction_numerics
        with np.load(path / "estimator.npz", allow_pickle=False) as arrays:
            model.projection = Projection.restore(arrays, gp_config.input_projection)
            model.output_projection = Projection.restore(arrays, model.diagnostics.get('output_projection','output_pca'), "output_")
            model.models = []
            shared_x=ScalarGP(gp_config,device).tensor(arrays['shared_x']) if 'shared_x' in arrays else None
            for i in range(model.diagnostics["output_count"]):
                gp = ScalarGP(gp_config, device)
                for name in ("x", "y", "inducing", "length", "amplitude", "noise"):
                    key = 'shared_x' if name == 'x' and 'shared_x' in arrays else f"gp_{i}_{name}"
                    setattr(gp, name, shared_x if name=='x' and shared_x is not None else gp.tensor(arrays[key]))
                gp.rebuild(); model.models.append(gp)
        return model
