"""Paired, task-stratified bootstrap, vectorized over every resample and episode.

All arrays are float64 on CPU. Models, feature caches, and simulation are untouched.
The only Python loops are over tasks, head runs, and bounded-size batches.
"""
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy.stats import rankdata

from uq_extensions.common import identity


METRICS = ['global_nauc', 'within_nauc', 'neg_spearman', 'neg_pearson',
           'benefit_spearman', 'benefit_pearson', 'automation', 'failure_auroc',
           'benefit_time_spearman', 'benefit_time_pearson',
           'benefit_path_spearman', 'benefit_path_pearson', 'global_raw_gain', 'within_raw_gain']
VERSION = 'paired-stratified-fixed-ties-v1'


def row_correlation(x, y, *, ranks=False):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if x.shape[-1] < 3:return np.full(x.shape[:-1], np.nan)
    if ranks:x, y = rankdata(x, axis=-1), rankdata(y, axis=-1)
    x = x - x.mean(-1, keepdims=True);y = y - y.mean(-1, keepdims=True)
    denominator = np.sqrt(np.sum(x*x, axis=-1)*np.sum(y*y, axis=-1))
    return np.divide(np.sum(x*y, axis=-1), denominator,
                     out=np.full(denominator.shape, np.nan), where=denominator > 0)


def fixed_order(scores, ids):
    ties = dict(zip(sorted(ids), np.random.default_rng(42).random(len(ids))))
    order = sorted(range(len(ids)), key=lambda i:(scores[i], ties[ids[i]], ids[i]))
    ranks = np.empty(len(ids), int);ranks[order] = np.arange(len(ids))
    return ranks


def resample_indices(cohort, repetitions=2000, seed=20261001):
    """Draw the observed number of independent initial-state clusters per task.

    Repeated rollouts of one state travel together. Current study: 20 distinct
    states/task, one rollout/state, hence exactly 200 episodes/resample.
    """
    tasks = np.asarray(cohort['tasks']);clusters = np.asarray(cohort.get('clusters', cohort['ids']))
    blocks = []
    for task in sorted(set(tasks)):
        names = sorted(set(clusters[tasks == task]))
        members = [np.flatnonzero((tasks == task) & (clusters == name)) for name in names]
        if len({len(x) for x in members}) != 1:
            raise ValueError('Bootstrap requires equal rollout counts per initial state within a task')
        # Task-specific streams keep identical states paired across checkpoints.
        rng = np.random.default_rng(int(identity(dict(seed=seed, task=task, clusters=names))[:16], 16))
        selected = rng.integers(len(members), size=(repetitions, len(members)))
        blocks.append(np.asarray(members)[selected].reshape(repetitions, -1))
    return np.concatenate(blocks, axis=1)


class Plan:
    def __init__(self, cohort, repetitions=2000, seed=20261001, indices=None):
        self.cohort = cohort
        self.indices = resample_indices(cohort, repetitions, seed) if indices is None else np.asarray(indices)
        self.repetitions, self.size = self.indices.shape
        self.tasks = np.asarray(cohort['tasks'])
        # Plans have a fixed number of episodes per task, in task-contiguous blocks.
        self.names = sorted(set(self.tasks));sample_tasks = self.tasks[self.indices]
        if not np.all(sample_tasks == sample_tasks[0]):
            raise ValueError('Task positions must agree across resamples')
        self.blocks = [np.flatnonzero(sample_tasks[0] == task) for task in self.names]
        self.weights = np.empty(self.size)
        for block in self.blocks:self.weights[block] = 1 / (len(self.blocks)*len(block))
        self.learner = np.asarray(cohort['learner'])[self.indices]
        self.supervisor = np.asarray(cohort['supervisor'])[self.indices]
        self.gain = self.learner - self.supervisor
        self.lbar = self.learner @ self.weights;self.hbar = self.supervisor @ self.weights
        self.task_rates = np.column_stack([self.learner[:, block].mean(1) for block in self.blocks])
        self.quality_gain = {'success': -self.gain}
        for name, (left, right) in cohort['qualities'].items():
            self.quality_gain[name] = (np.asarray(right)-np.asarray(left))[self.indices]
        self.quality_ranks = {key:rankdata(value, axis=1) for key,value in self.quality_gain.items()}
        oracle_scores = np.asarray(cohort['supervisor'])-np.asarray(cohort['learner'])
        self.oracle = self.curves(oracle_scores)
        self.denominators = {key:self.area(value) for key,value in self.oracle.items()}
        self.fingerprint = identity(dict(version=VERSION, seed=seed, repetitions=self.repetitions,
            ids=cohort['ids'], tasks=cohort['tasks'], clusters=cohort.get('clusters'),
            learner=np.asarray(cohort['learner']).tolist(), supervisor=np.asarray(cohort['supervisor']).tolist(),
            qualities={k:[np.asarray(a).tolist(), np.asarray(b).tolist()] for k,(a,b) in cohort['qualities'].items()}))

    @staticmethod
    def area(curve):
        x, actual, random = curve
        return np.trapz(actual-random, x, axis=1)

    def curves(self, scores):
        # A fixed original tie order is retained when an episode is repeated.
        ranks = fixed_order(scores, self.cohort['ids'])[self.indices]
        order = np.argsort(ranks, axis=1, kind='stable')
        weight = self.weights[order]
        x = np.column_stack([np.zeros(self.repetitions), np.cumsum(weight, axis=1)])
        gain = np.take_along_axis(self.gain, order, axis=1)
        y = self.hbar[:, None]+np.column_stack([np.zeros(self.repetitions), np.cumsum(weight*gain, axis=1)])
        random = self.hbar[:, None]+x*(self.lbar-self.hbar)[:, None]
        global_curve = (x, y, random)
        grid = np.array(sorted({j/len(block) for block in self.blocks for j in range(len(block)+1)}))
        within = np.zeros((self.repetitions, len(grid)));expected = within.copy();automation = np.zeros(len(grid))
        for block in self.blocks:
            n = len(block);counts = np.floor(grid*n+1e-9).astype(int);fraction = counts/n
            task_order = np.argsort(ranks[:, block], axis=1, kind='stable')
            task_gain = np.take_along_axis(self.gain[:, block], task_order, axis=1)
            prefix = np.column_stack([np.zeros(self.repetitions), np.cumsum(task_gain, axis=1)])
            h = self.supervisor[:, block].mean(1)
            within += (h[:, None]+prefix[:, counts]/n)/len(self.blocks)
            expected += (h[:, None]+fraction*self.gain[:, block].mean(1)[:, None])/len(self.blocks)
            automation += fraction/len(self.blocks)
        return dict(global_ = global_curve, within=(np.broadcast_to(automation, within.shape), within, expected))

    def evaluate(self, scores, task_means):
        curves = self.curves(scores)
        results = np.full((self.repetitions, len(METRICS)), np.nan)
        def put(name, value):results[:, METRICS.index(name)] = value
        for key, prefix in [('global_', 'global'), ('within', 'within')]:
            gain = self.area(curves[key]);denominator = self.denominators[key]
            put(prefix+'_raw_gain', gain)
            put(prefix+'_nauc', np.divide(gain, denominator, out=np.full_like(gain, np.nan), where=denominator > 1e-12))
        x, y, _ = curves['global_']
        put('automation', np.clip(np.max(np.where(y >= self.hbar[:, None]-.02-1e-12, x, -np.inf), axis=1), 0, 1))
        task_values = np.array([task_means[t] for t in self.names])[None, :]
        put('neg_pearson', -row_correlation(task_values, self.task_rates))
        put('neg_spearman', -row_correlation(task_values, self.task_rates, ranks=True))
        sampled = np.asarray(scores)[self.indices];score_ranks = rankdata(sampled, axis=1)
        for key, value in self.quality_gain.items():
            prefix = 'benefit' if key == 'success' else 'benefit_'+key
            put(prefix+'_pearson', row_correlation(sampled, value))
            put(prefix+'_spearman', row_correlation(score_ranks, self.quality_ranks[key]))
        # Same macro-AUROC convention as score_metrics: average defined tasks.
        aucs = []
        for block in self.blocks:
            failures = 1-self.learner[:, block];n = len(block);positives = failures.sum(1)
            numerator = (rankdata(sampled[:, block], axis=1)*failures).sum(1)-positives*(positives+1)/2
            denominator = positives*(n-positives)
            aucs.append(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0))
        aucs = np.asarray(aucs);count = np.isfinite(aucs).sum(0)
        put('failure_auroc', np.divide(np.nansum(aucs, axis=0), count, out=np.full(self.repetitions, np.nan), where=count > 0))
        return results, curves


def cached_metrics(plan, detail, directory):
    task_means = dict(zip(detail['task_names'], detail['task_means']))
    digest = identity(dict(plan=plan.fingerprint, code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scores=np.asarray(detail['scores']).tolist(), task_means=task_means))
    directory = Path(directory);directory.mkdir(parents=True, exist_ok=True);path = directory/(digest+'.npz')
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                values = saved['metrics']
                if values.shape == (plan.repetitions, len(METRICS)) and np.array_equal(saved['names'], METRICS):return values, True
        except (ValueError, OSError, KeyError):pass
    values, _ = plan.evaluate(detail['scores'], task_means)
    handle, temporary = tempfile.mkstemp(prefix='.bootstrap-', dir=directory)
    try:
        with os.fdopen(handle, 'wb') as stream:np.savez_compressed(stream, metrics=values, names=np.asarray(METRICS))
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()
    return values, False


def seed_draws(seeds, repetitions, seed=20261001):
    """The same seed IDs are resampled together across matched configurations."""
    seeds = sorted(seeds)
    rng = np.random.default_rng(int(identity(dict(seed=seed, feature_seeds=seeds))[:16], 16))
    return rng.integers(len(seeds), size=(len(seeds), repetitions))


def combine(values, seeds, seed=20261001):
    """Input [seeds, resamples, ...]; return evaluation-only and joint replicates."""
    values = np.asarray(values);evaluation = values.mean(axis=0)
    if len(seeds) < 2:return evaluation, None
    order = np.argsort(seeds);values = values[order]
    draws = seed_draws(sorted(seeds), values.shape[1], seed)
    total = values[draws, np.arange(values.shape[1])[None, :]].mean(axis=0)
    return evaluation, total


def interval(values, minimum_valid=.9):
    values = np.asarray(values);valid = np.isfinite(values);count = int(valid.sum())
    if count < max(2, minimum_valid*len(values)):return np.nan, np.nan, count
    lower, upper = np.quantile(values[valid], [.025, .975])
    return float(lower), float(upper), count


def interpolate_rows(x, y, targets):
    """Linear interpolation for B curves without a Python loop over resamples."""
    targets = np.broadcast_to(targets, (len(x), len(targets)))
    lo = np.zeros(targets.shape, int);hi = np.full(targets.shape, x.shape[1]-1, int)
    while np.any(hi-lo > 1):
        mid = (lo+hi)//2;left = np.take_along_axis(x, mid, axis=1) <= targets
        lo = np.where(left, mid, lo);hi = np.where(left, hi, mid)
    xa = np.take_along_axis(x, lo, axis=1);xb = np.take_along_axis(x, hi, axis=1)
    ya = np.take_along_axis(y, lo, axis=1);yb = np.take_along_axis(y, hi, axis=1)
    f = np.clip((targets-xa)/(xb-xa), 0, 1)
    return ya+f*(yb-ya)
