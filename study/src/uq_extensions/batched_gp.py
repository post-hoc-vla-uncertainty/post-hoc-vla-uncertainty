"""Independent Gaussian regressions fitted together, with resumable Adam state.

Exact and collapsed sparse objectives match ScalarGP. Batching does not share
hyperparameters across outputs and does not change the statistical model.
"""
from pathlib import Path
import csv
import os
import time

import numpy as np
import torch

from .common import array_hash, atomic_json, identity, plain
from .kernel_variance import geometry_lengthscale, initial_log_parameters


def kernel(x, y, length, amplitude):
    return amplitude[:, None, None] * torch.exp(
        -.5 * torch.cdist(x / length[:, None, :], y / length[:, None, :]).square())


def cholesky(matrix, jitter):
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    current = jitter
    for _ in range(7):
        factor, info = torch.linalg.cholesky_ex((matrix + matrix.transpose(-1, -2)) / 2 + current * eye)
        if not info.any() and torch.isfinite(factor).all():
            return factor, current
        current *= 10
    if matrix.dtype != torch.float64:
        return cholesky(matrix.double(), jitter)
    raise RuntimeError('Batched GP Cholesky failed after jitter escalation')


def terms(x, y, inducing, logs, config, *, kernel_jitter=None):
    # Promote before kernel construction: casting a damaged float32 Gram matrix
    # at Cholesky time cannot recover its lost information.
    if config.precision == 'float64':
        x, y, inducing, logs = (v.double() for v in (x, y, inducing, logs))
    kernel_jitter = config.jitter if kernel_jitter is None else kernel_jitter
    def factorize(matrix, jitter):
        if config.precision != 'float64':return cholesky(matrix, jitter)
        eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
        # Keep the declared numerical regularization fixed; never silently
        # change the fitted model to get a factorization through.
        return torch.linalg.cholesky((matrix + matrix.transpose(-1, -2)) / 2 + jitter * eye), jitter
    values = logs.exp()
    length, amplitude, noise = values[:, :-2], values[:, -2], values[:, -1]
    if config.inference == 'exact':
        gram = kernel(x, x, length, amplitude)
        eye = torch.eye(len(x), device=x.device, dtype=x.dtype)
        factor, jitter = factorize(gram + noise[:, None, None] * eye, kernel_jitter)
        alpha = torch.cholesky_solve(y.to(factor.dtype).unsqueeze(-1), factor)
        loss = .5 * ((y * alpha.squeeze(-1)).sum(-1) + 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1))
        return loss, (factor, alpha, jitter)
    lu, jitter = factorize(kernel(inducing, inducing, length, amplitude), kernel_jitter)
    a = torch.linalg.solve_triangular(lu, kernel(inducing, x, length, amplitude).to(lu.dtype), upper=False)
    b = torch.eye(lu.shape[-1], device=x.device, dtype=lu.dtype) + (a @ a.transpose(-1, -2)) / noise[:, None, None]
    lb, _ = factorize(b, config.jitter)
    ay = a @ y.to(a.dtype).unsqueeze(-1)
    beta = torch.cholesky_solve(ay, lb) / noise[:, None, None]
    if config.precision == 'float64':
        residual_y = y.unsqueeze(-1) - a.transpose(-1, -2) @ beta
        quadratic = residual_y.square().sum((-1, -2)) / noise + (1 + config.jitter) * beta.square().sum((-1, -2))
        residual_diagonal = amplitude[:, None] - a.square().sum(-2)
        if (residual_diagonal < -1e-8 * amplitude[:, None]).any():
            raise RuntimeError('Invalid sparse conditional variance in float64')
        residual = residual_diagonal.clamp_min(0).sum(-1)
    else:
        quadratic = y.square().sum(-1) / noise - (ay * beta).sum((-1, -2)) / noise
        residual = (len(x) * amplitude - a.square().sum((-1, -2))).clamp_min(0)
    loss = .5 * (quadratic + len(x) * noise.log() + 2 * lb.diagonal(dim1=-2, dim2=-1).log().sum(-1) + residual / noise)
    return loss, (lu, lb, beta, jitter)


@torch.no_grad()
def predict_diagonal(x, inducing, logs, posterior, query, config):
    values = logs.exp()
    length, amplitude = values[:, :-2], values[:, -2]
    if config.inference == 'exact':
        factor, alpha, _ = posterior
        k = kernel(x, query, length, amplitude).to(factor.dtype)
        v = torch.linalg.solve_triangular(factor, k, upper=False)
        return (k.transpose(-1, -2) @ alpha).squeeze(-1), amplitude[:, None] - v.square().sum(-2)
    lu, lb, beta, _ = posterior
    a = torch.linalg.solve_triangular(lu, kernel(inducing, query, length, amplitude).to(lu.dtype), upper=False)
    v = torch.linalg.solve_triangular(lb, a, upper=False)
    return (a.transpose(-1, -2) @ beta).squeeze(-1), amplitude[:, None] - a.square().sum(-2) + v.square().sum(-2)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.pt')
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def fit_batched(x, targets, config, *, device='cpu', seed=42, directory=None,
                validation=None, label='GP', step_callback=None):
    """Return ScalarGP-compatible views without fitting the outputs serially."""
    from .target_gp import ScalarGP
    x, targets = np.asarray(x), np.asarray(targets)
    directory = Path(directory) if directory is not None else None
    dtype = getattr(torch, config.precision)
    tx = torch.as_tensor(x, device=device, dtype=dtype)
    initial, initialization = initial_log_parameters(tx, config)
    models, records = [], []
    for start in range(0, targets.shape[1], config.output_batch_size):
        end = min(start + config.output_batch_size, targets.shape[1])
        ty = torch.as_tensor(targets[:, start:end].T.copy(), device=device, dtype=dtype)
        indices = [np.sort(np.random.default_rng(seed + j).choice(len(x), min(len(x), config.inducing_points), replace=False))
                   for j in range(start, end)]
        inducing = tx[torch.as_tensor(np.stack(indices), device=device)]
        logs = torch.nn.Parameter(torch.tensor(np.tile(initial, (end-start,1)),
                                               device=device, dtype=dtype))
        optimizer = torch.optim.Adam([logs], lr=config.learning_rate)
        spec = dict(x=array_hash(x), y=array_hash(targets[:, start:end]), seed=seed,
                    config=plain(config), batch=[start, end],
                    implementation='batched_gaussian_float64_v2' if config.precision == 'float64' else 'batched_gaussian_v1')
        key = identity(spec)
        checkpoint = directory / f'outputs_{start:02d}_{end:02d}.pt' if directory else None
        history, diagnostics, offset, elapsed = [], [], 0, 0.
        if checkpoint and checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            if saved['identity'] != key:
                raise RuntimeError(f'GP optimizer checkpoint identity changed: {checkpoint}')
            with torch.no_grad():
                logs.copy_(saved['logs'])
            optimizer.load_state_dict(saved['optimizer'])
            history, diagnostics, offset, elapsed = saved['history'], saved['diagnostics'], saved['step'], saved['seconds']
        started = time.perf_counter()
        step = offset
        in_update = False

        def save():
            if checkpoint:
                atomic_torch(checkpoint, dict(identity=key, logs=logs.detach(), optimizer=optimizer.state_dict(),
                                              step=step, history=history, diagnostics=diagnostics,
                                              seconds=elapsed + time.perf_counter() - started))
                atomic_json(checkpoint.with_suffix('.json'), dict(step=step, outputs=[start, end],
                    max_steps=config.fit_steps, seconds=elapsed + time.perf_counter()-started, milestones=diagnostics))
                path = checkpoint.with_suffix('.csv')
                temporary = path.with_suffix('.partial.csv')
                with temporary.open('w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', *[f'negative_log_likelihood_output_{j}' for j in range(start, end)]])
                    writer.writerows((i + 1, *row) for i, row in enumerate(history))
                os.replace(temporary, path)

        from run_progress import Progress
        progress = Progress(f'{label}: outputs {start+1}–{end} together', config.fit_steps,
                            initial=offset, every=min(100, config.checkpoint_every))
        try:
            for step in range(offset + 1, config.fit_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                losses, _ = terms(tx, ty, inducing, logs, config)
                if not torch.isfinite(losses).all():
                    raise RuntimeError('Nonfinite batched GP objective')
                losses.sum().backward()
                if not torch.isfinite(logs.grad).all():
                    raise RuntimeError('Nonfinite batched GP gradient')
                in_update = True
                optimizer.step()
                with torch.no_grad():
                    logs.clamp_(-13.8, 9.2)
                history.append(losses.detach().cpu().tolist())
                in_update = False
                if step in config.diagnostic_steps or step == config.fit_steps:
                    with torch.no_grad():
                        objective, posterior = terms(tx, ty, inducing, logs, config)
                        record = dict(step=step, objective=objective.cpu().tolist(),
                            seconds=elapsed+time.perf_counter()-started,
                            peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if str(device).startswith('cuda') else 0)
                        if validation is not None:
                            z, expected = validation
                            means, variances = [], []
                            for part in torch.as_tensor(z, dtype=dtype, device=device).split(256):
                                mean, variance = predict_diagonal(tx, inducing, logs, posterior, part, config)
                                means.append(mean.cpu().numpy()); variances.append(variance.cpu().numpy())
                            prediction = np.concatenate(means, axis=1).T
                            variance = np.maximum(np.concatenate(variances, axis=1).T, 0)
                            expected = np.asarray(expected)[:, start:end]
                            from sklearn.metrics import r2_score
                            total_var = variance + logs.detach().exp()[:, -1].cpu().numpy()[None, :]
                            record.update(validation_r2=r2_score(expected, prediction, multioutput='raw_values').tolist(),
                                validation_nll=(.5*(np.log(2*np.pi*total_var)+(expected-prediction)**2/total_var)).mean(0).tolist(),
                                mean_latent_variance=variance.mean(0).tolist())
                        diagnostics.append(record)
                        if checkpoint:
                            # Small milestone hyperparameters, not duplicated training data/posteriors.
                            atomic_torch(checkpoint.with_name(f'{checkpoint.stem}_step_{step:05d}.pt'),
                                         dict(step=step, identity=key, logs=logs.detach().cpu()))
                progress.update(step, f'objective/output={float(losses.mean()):.6g}')
                if step % config.checkpoint_every == 0 or step == config.fit_steps:
                    save()
                if step_callback:
                    step_callback(step)
        except (KeyboardInterrupt, SystemExit):
            # Never claim an uncompleted optimizer update in the resume record.
            step = len(history)
            if not in_update:save()
            raise
        with torch.no_grad():
            _, posterior = terms(tx, ty, inducing, logs, config)
            values = logs.detach().exp()
        for j in range(end-start):
            model = ScalarGP(config, device)
            model.x, model.y, model.inducing = tx, ty[j], inducing[j]
            model.length, model.amplitude, model.noise = values[j, :-2], values[j, -2], values[j, -1]
            model.history = [row[j] for row in history]
            model.terms = tuple(t[j] if isinstance(t, torch.Tensor) else t for t in posterior)
            models.append(model)
        records.append(dict(outputs=[start, end], steps=len(history), milestones=diagnostics,
                            seconds=elapsed+time.perf_counter()-started, resumed_from_step=offset))
        if initialization:records[-1]['initialization']=initialization
        progress.finish(f'{len(history)} optimizer steps; {end-start} independent outputs trained as a batch')
        del optimizer, logs
    return models, records
