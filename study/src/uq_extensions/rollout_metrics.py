"""Physical movement and fixed demonstrator-normalized episode outcomes."""
import numpy as np


def episode_metrics(*, success, steps, positions, control_hz, normalizer=None):
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("End-effector positions must be finite world-coordinate XYZ points")
    if steps < 0 or control_hz <= 0:
        raise ValueError("Invalid control-time measurement")
    duration = steps / control_hz
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    result = dict(success=int(bool(success)), steps=int(steps), time_seconds=duration,
                  path_meters=length, normalizer_valid=True)
    for suffix, value, key in (("time", duration, "mean_time"), ("path", length, "mean_path")):
        baseline = None if normalizer is None else normalizer.get(key)
        valid = baseline is not None and np.isfinite(baseline) and baseline > 0
        result["normalizer_valid"] &= valid
        result[f"relative_{suffix}"] = value / baseline if success and valid else None
        result[f"q_{suffix}"] = 0. if not success else min(1., baseline / max(value, 1e-9)) if valid else None
    return result


def learning_summary(records, *, budget_key='new_demonstrations'):
    """One trajectory; success values must already be macro task averages."""
    rows = sorted(records, key=lambda r: r[budget_key])
    if len({r[budget_key] for r in rows}) != len(rows):
        raise ValueError("Duplicate acquisition budgets in one learning curve")
    x = np.array([r[budget_key] for r in rows], dtype=float)
    y = np.array([r["success"] for r in rows], dtype=float)
    area = float(np.trapz(y, x) / (x[-1] - x[0])) if len(x) > 1 and x[-1] > x[0] else None
    prefix='query_slots' if budget_key=='queries' else 'demonstrations'
    return dict(area=area, final_success=float(y[-1]),budget_axis=budget_key,
                **{f'{prefix}_to_50':float(x[np.flatnonzero(y >= .5)[0]]) if np.any(y >= .5) else None,
                   f'{prefix}_to_60':float(x[np.flatnonzero(y >= .6)[0]]) if np.any(y >= .6) else None})
