"""Validation fixture for the collaborator's Q5 experiments (CPU only, no policy).

Fits the paper's two kernel heads on synthetic features with the exact code paths
of the final study and stores inputs plus expected scores:

  * KV-Ctx recipe (also used for every context head): uq_followup Estimator with
    the final study's H1 recipe (standardize, PCA<=512, Ledoit-Wolf whitening on
    component axes, k-NN bandwidth, exact GP, ridge 1e-3, float32).
  * KV-Dec recipe: the collaborator's KernelVariance(ridge=0.01, projection='whiten', seed=0),
    loaded from reference_src/vla-ua exactly as the final study loaded it.

    .venv/bin/python scripts/make_q5_fixture.py paper/handoff/q5/kv_fixture.npz
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
from uq_followup.estimators import Estimator  # noqa: E402


def synthetic(rng, n_support, n_query, width, tasks=10, rank=24):
    """Task clusters on a low-rank manifold plus noise; queries mix copies, neighbours, new and far points."""
    basis = rng.normal(size=(rank, width))
    centers = rng.normal(scale=3.0, size=(tasks, rank))
    labels = np.arange(n_support) % tasks
    support = (centers[labels]+rng.normal(size=(n_support, rank))) @ basis + 0.3*rng.normal(size=(n_support, width))
    support = support.astype(np.float32).astype(np.float64)  # stored as float32; fit on exactly these values
    q = n_query//4
    copies = support[rng.choice(n_support, q, replace=False)]
    near = support[rng.choice(n_support, q, replace=False)] + 0.3*rng.normal(size=(q, width))
    new = (centers[rng.integers(tasks, size=q)]+rng.normal(size=(q, rank))) @ basis + 0.3*rng.normal(size=(q, width))
    far = (rng.normal(scale=6.0, size=(n_query-3*q, rank))) @ basis + 0.3*rng.normal(size=(n_query-3*q, width))
    return support, np.concatenate([copies, near, new, far]).astype(np.float32).astype(np.float64)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'paper/handoff/q5/kv_fixture.npz'
    recipe = json.loads((ROOT/'paper/data/final_save_v1/recipes.json').read_text())['h1']
    spec = importlib.util.spec_from_file_location(
        'collaborator_kv', ROOT/'reference_src/vla-ua/src/lerobot/uncertainty/uncertainty_scoring/kernel_variance.py')
    collaborator = importlib.util.module_from_spec(spec); sys.modules['collaborator_kv'] = collaborator
    spec.loader.exec_module(collaborator)
    rng = np.random.default_rng(20261012)
    arrays, meta = {}, {}
    # Widths are reduced to keep the file small; the estimator does not depend on width.
    # 'dense' has more than 513 support rows, so the PCA cap of 512 components is exercised.
    for name, n, width in (('ctx', 379, 1200), ('dense', 700, 600), ('state8', 379, 8)):
        support, query = synthetic(rng, n, 80, width, rank=min(24, width))
        with tempfile.TemporaryDirectory() as directory:
            estimator = Estimator(recipe, 'cpu').fit(support, None, np.arange(len(support)) % 10, directory)
            expected = estimator.score(query)
        arrays.update({f'{name}_support': support, f'{name}_query': query, f'{name}_expected': expected})
        meta[name] = dict(recipe='KV-Ctx (paper H1)', support_rows=n, width=width, lengthscale=estimator.diagnostics['lengthscale'],
                          components=estimator.diagnostics['components'], unique_rows=estimator.diagnostics['unique_rows'])
    support, query = synthetic(rng, 379, 80, 480, rank=24)
    head = collaborator.KernelVariance(ridge=.01, projection='whiten', seed=0).fit(torch.as_tensor(support, dtype=torch.float32))
    arrays.update(dec_support=support, dec_query=query,
                  dec_expected=head.score(torch.as_tensor(query, dtype=torch.float32)).numpy())
    meta['dec'] = dict(recipe='KV-Dec (paper H2, collaborator KernelVariance)', width=480, lengthscale=head.length_scale)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: (v.astype(np.float32) if k.endswith(('_support', '_query')) else v) for k, v in arrays.items()}
    np.savez_compressed(out, **arrays, meta=np.asarray(json.dumps(meta)))
    print(json.dumps(meta, indent=1))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
