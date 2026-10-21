"""Paired episode-start referral with equal task weighting and signed gains."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

from .protocol import task_weights


PAIR_KEYS = ["task", "init_state_hash", "reset_seed", "repetition", "condition", "split"]


def pair_outcomes(learner, supervisor):
    learner, supervisor = pd.DataFrame(learner), pd.DataFrame(supervisor)
    for data in (learner, supervisor):
        if data.duplicated(PAIR_KEYS).any():
            raise ValueError("Duplicate paired episode identity")
    joined = learner.merge(supervisor, on=PAIR_KEYS, how="outer", suffixes=("_learner", "_supervisor"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Learner and supervisor episode starts do not match exactly")
    for name in ("max_steps", "control_hz", "action_steps", "flow_steps", "observation_spec", "observation_preprocessing", "rendering_protocol"):
        if (name in learner) != (name in supervisor):
            raise ValueError(f"Unmatched learner/supervisor execution setting: {name}")
        left, right = f"{name}_learner", f"{name}_supervisor"
        if left in joined and not joined[left].eq(joined[right]).all():
            raise ValueError(f"Unmatched learner/supervisor execution setting: {name}")
    return joined.drop(columns="_merge")


def clustered_headroom_interval(learner, supervisor, tasks, clusters, seed=42, repetitions=1000):
    rows = pd.DataFrame(dict(task=tasks, cluster=clusters, gain=np.asarray(supervisor)-np.asarray(learner)))
    grouped = rows.groupby(["task", "cluster"]).gain.mean()
    rng = np.random.default_rng(seed)
    samples = np.zeros(repetitions)
    for task in sorted(set(tasks)):
        values = grouped.loc[task].to_numpy()
        samples += values[rng.integers(len(values), size=(repetitions, len(values)))].mean(1) / len(set(tasks))
    return np.quantile(samples, [.025, .975]).tolist()


def referral_curve(scores, learner, supervisor, tasks, ids, *, allocation="global", seed=42, clusters=None):
    scores, learner, supervisor = map(lambda a: np.asarray(a, dtype=float), (scores, learner, supervisor))
    tasks = np.asarray(tasks); ids = np.asarray(ids)
    if not (len(scores) == len(learner) == len(supervisor) == len(tasks) == len(ids)) or len(set(ids)) != len(ids):
        raise ValueError("Referral arrays must have identical unique episode identities")
    if not all(np.isfinite(a).all() for a in (scores, learner, supervisor)):
        raise ValueError("Missing/nonfinite scores or paired outcomes")
    if allocation not in ("global", "within_task"):
        raise ValueError("Unknown referral allocation")
    w = task_weights(tasks)
    tie_rng = np.random.default_rng(seed)
    tie = dict(zip(sorted(ids), tie_rng.random(len(ids))))
    order = sorted(range(len(ids)), key=lambda i: (scores[i], tie[ids[i]], ids[i]))
    task_orders = {task: [i for i in order if tasks[i] == task] for task in sorted(set(tasks))}
    grids = np.arange(len(ids)+1) / len(ids) if allocation == "global" else sorted({j / len(indices) for indices in task_orders.values() for j in range(len(indices)+1)})
    lbar, hbar = float(w @ learner), float(w @ supervisor)
    curve = []
    for fraction in grids:
        mask = np.zeros(len(ids), dtype=bool)
        if allocation == "global":
            mask[order[:round(fraction * len(ids))]] = True
        else:
            for indices in task_orders.values():
                mask[indices[:int(np.floor(fraction * len(indices) + 1e-9))]] = True
        automated = float(w @ mask)
        utility = float(w @ np.where(mask, learner, supervisor))
        random = automated * lbar + (1 - automated) * hbar
        if allocation == "within_task":
            random = sum(float(mask[indices].mean()) * learner[indices].mean() +
                         (1 - float(mask[indices].mean())) * supervisor[indices].mean()
                         for indices in task_orders.values()) / len(task_orders)
        curve.append(dict(automated=automated, referred=1-automated, utility=utility,
                          random_utility=float(random), gain=utility-random))
    x = np.array([r["automated"] for r in curve])
    gain = float(np.trapz([r["gain"] for r in curve], x))
    denominator = float(np.trapz([hbar-r["random_utility"] for r in curve], x))
    interval = clustered_headroom_interval(learner, supervisor, tasks, ids if clusters is None else clusters, seed)
    normalized = gain / denominator if denominator > 1e-12 and interval[0] > 0 else None
    summary = dict(raw_gain=gain, normalized_gain=normalized, headroom_denominator=denominator,
                   headroom_interval=interval, learner_mean=lbar, supervisor_mean=hbar,
                   automation_within_two_points=max(r["automated"] for r in curve if r["utility"] >= hbar-.02-1e-12),
                   allocation=allocation)
    for fraction in (.1, .25, .5):
        row = min(curve, key=lambda r: abs(r["referred"] - fraction))
        summary[f"referral_{fraction:g}"] = row
    return curve, summary


def safe_correlation(x, y, method="spearman"):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = spearmanr(x, y).statistic if method == "spearman" else pearsonr(x, y).statistic
    return float(value) if np.isfinite(value) else None


def discrimination(scores, successes):
    scores, failures = np.asarray(scores, dtype=float), 1 - np.asarray(successes, dtype=int)
    order = np.argsort(scores, kind="stable")
    risk = np.cumsum(failures[order]) / np.arange(1, len(order)+1)
    coverage = np.arange(1, len(order)+1) / len(order)
    two_classes = len(set(failures)) == 2
    return dict(auroc=float(roc_auc_score(failures, scores)) if two_classes else None,
                auprc=float(average_precision_score(failures, scores)) if two_classes else None,
                successes=int((1-failures).sum()), failures=int(failures.sum()),
                risk_coverage_area=float(np.trapz(np.r_[0, risk], np.r_[0, coverage])),
                coverage=coverage.tolist(), risk=risk.tolist())


def score_metrics(scores, successes, tasks, *, initial_pool_task_means, supervisor=None):
    scores, successes, tasks = np.asarray(scores), np.asarray(successes), np.asarray(tasks)
    per_task = {t: discrimination(scores[tasks == t], successes[tasks == t]) for t in sorted(set(tasks))}
    means = [initial_pool_task_means[t] for t in sorted(per_task)]
    rates = [float(successes[tasks == t].mean()) for t in sorted(per_task)]
    result = dict(within_task=per_task, between_task_spearman=safe_correlation(means, rates),
                  between_task_pearson=safe_correlation(means, rates, "pearson"),
                  task_mean_only_scores=[initial_pool_task_means[t] for t in tasks])
    w = task_weights(tasks)
    for label, values in (("pooled", scores), ("pooled_task_mean_only", result['task_mean_only_scores'])):
        failure = 1-successes
        valid = len(set(failure)) == 2
        result[label] = dict(auroc=float(roc_auc_score(failure, values, sample_weight=w)) if valid else None,
                             auprc=float(average_precision_score(failure, values, sample_weight=w)) if valid else None,
                             weighting="equal mass per task")
    for metric in ("auroc", "auprc", "risk_coverage_area"):
        values = [d[metric] for d in per_task.values() if d[metric] is not None]
        result["macro_" + metric] = float(np.mean(values)) if values else None
        result["valid_tasks_" + metric] = len(values)
    if supervisor is not None:
        gains = np.asarray(supervisor) - successes
        result["signed_gain_spearman"] = safe_correlation(scores, gains)
        result["within_task_gain_spearman"] = {t: safe_correlation(scores[tasks==t], gains[tasks==t]) for t in sorted(per_task)}
    return result


def clustered_referral_interval(scores, learner, supervisor, tasks, clusters, *, allocation='global', seed=42, repetitions=1000):
    """Resample complete initial-state clusters within each task, retaining noise repeats."""
    scores,learner,supervisor,tasks,clusters=map(np.asarray,(scores,learner,supervisor,tasks,clusters))
    rng=np.random.default_rng(seed);groups={}
    for task in sorted(set(tasks)):
        groups[task]=[np.flatnonzero((tasks==task)&(clusters==c)) for c in sorted(set(clusters[tasks==task]))]
    samples=[]
    for _ in range(repetitions):
        indices=np.concatenate([np.concatenate([parts[j] for j in rng.integers(len(parts),size=len(parts))]) for parts in groups.values()])
        s,l,h,t=scores[indices],learner[indices],supervisor[indices],tasks[indices]
        weights=task_weights(t);order=np.lexsort((rng.random(len(s)),s));gain=l-h
        if allocation=='global':
            x=np.r_[0,np.cumsum(weights[order])]
            difference=np.r_[0,np.cumsum((weights*gain)[order])]-x*(weights@gain)
            samples.append(float(np.trapz(difference,x)))
        elif allocation=='within_task':
            task_orders={task:[i for i in order if t[i]==task] for task in sorted(set(t))}
            grids=sorted({j/len(ids) for ids in task_orders.values() for j in range(len(ids)+1)})
            xs=[];ys=[]
            for fraction in grids:
                fractions=[];differences=[]
                for ids in task_orders.values():
                    count=int(np.floor(fraction*len(ids)+1e-9));f=count/len(ids);fractions.append(f)
                    differences.append(float(gain[ids[:count]].sum()/len(ids)-f*gain[ids].mean()))
                xs.append(np.mean(fractions));ys.append(np.mean(differences))
            samples.append(float(np.trapz(ys,xs)))
        else:raise ValueError('Unknown referral allocation')
    return np.quantile(samples,[.025,.975]).tolist()
