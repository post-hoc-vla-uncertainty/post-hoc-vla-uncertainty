#!/usr/bin/env python
"""Q5 / E2 final analysis: within-task referral under perturbations (handoff/q5, Sections 7 and 9).

Inputs: rollouts outputs/q5/seed{s}/e2/{primary,supervisor}/*.json (q5_e2.py; 1,300 conditions per seed, three
executions each, shared environment seeds) and the heads' scores of each condition's starting observation from E1
(outputs/q5/final/e1/e1_scores_dense.csv and e1_scores.csv: same seed, task, starting state and variant).

Metrics are the paper's own (vla-uncertainty-paper/scripts/analysis.py, vendored as handoff/q5/analysis.py): within-task and global referral
nAUC, automated fraction, within-task failure AUROC of the primary. Per seed, then mean +- SD over seeds; 95 %
interval from a hierarchical bootstrap (starting states resampled within each task with all their variants, then
seeds; 2,000 draws shared by all scores). Combinations of the eight context heads use percentiles among the task's
clean calibration observations (E1 starting states 25-49): COMB = maximum (handoff), COMB-mean = mean.
Cross-replicate reference: the gain observed in execution 0 ranks the conditions, scored on executions 1-2.

Outputs in outputs/q5/final/e2/: e2_conditions.csv (handoff columns), e2_summary.json.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("paper_analysis", "handoff/q5/analysis.py")
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)

SEEDS, TASKS, STATES = (0, 1, 2), tuple(range(10)), tuple(range(10))
OUT = Path("outputs/q5/final/e2")
CONTEXT = ("pre_ws", "pre_wrist", "pre_lang", "pre_state", "ctx_ws", "ctx_wrist", "ctx_lang", "ctx_state")
HEADS = ("kv_ctx", "kv_dec", *CONTEXT, "vfd")
SCORES = (*HEADS, "comb", "comb_mean", "cross_replicate")
BOOT = 2000
HARMLESS = {"I3:please", "V1:x0.8", "V2:sigma0.01"}


def load_rollouts(seed: int) -> dict:
    out = defaultdict(dict)
    for role in ("primary", "supervisor"):
        for path in Path(f"outputs/q5/seed{seed}/e2/{role}").glob("*.json"):
            d = json.loads(path.read_text())
            out[d["task"], d["state"], d["variant"]][role] = [float(x) for x in d["success"]]
    return out


def load_scores(csv_path: Path) -> dict:
    """(seed, task, state, variant) -> raw head scores; also each head's calibration distribution per seed and task."""
    scores, calib = {}, defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            seed, task, state = int(row["seed"]), int(row["task"].split("/")[1]), int(row["init_state_id"])
            values = {h: float(row[h]) if row[h] != "" else np.nan for h in HEADS}
            if row["split"] == "calibration" and row["variant"] == "clean":
                calib[seed, task].append(values)
            elif state in STATES:
                scores[seed, task, state, row["variant"]] = values
    ref = {k: {h: np.sort([v[h] for v in rows]) for h in CONTEXT} for k, rows in calib.items()}
    return scores, ref


def midrank(ref: np.ndarray, x: float) -> float:
    return 100 * (np.searchsorted(ref, x, "left") + np.searchsorted(ref, x, "right")) / (2 * len(ref))


def seed_table(seed: int, rollouts: dict, scores: dict, ref: dict) -> dict:
    keys = sorted(k for k in rollouts if "primary" in rollouts[k] and "supervisor" in rollouts[k]
                  and (seed, *k) in scores)
    t = {"keys": keys, "tasks": np.array([f"libero_10/{k[0]}" for k in keys]), "state": np.array([k[1] for k in keys]),
         "primary": np.array([np.mean(rollouts[k]["primary"]) for k in keys]),
         "supervisor": np.array([np.mean(rollouts[k]["supervisor"]) for k in keys])}
    for h in HEADS:
        t[h] = np.array([scores[(seed, *k)][h] for k in keys])
    pct = np.array([[midrank(ref[seed, k[0]][h], scores[(seed, *k)][h]) for h in CONTEXT] for k in keys])
    t["comb"], t["comb_mean"] = pct.max(axis=1), pct.mean(axis=1)
    # Cross-replicate reference: rank by the gain in execution 0, evaluate on executions 1 and 2.
    t["cross_replicate"] = np.array([rollouts[k]["supervisor"][0] - rollouts[k]["primary"][0] for k in keys])
    t["primary_eval"] = np.array([np.mean(rollouts[k]["primary"][1:]) for k in keys])
    t["supervisor_eval"] = np.array([np.mean(rollouts[k]["supervisor"][1:]) for k in keys])
    return t


def metrics(t: dict, idx: np.ndarray | None = None) -> dict:
    sel = slice(None) if idx is None else idx
    out = {}
    for name in SCORES:
        p, s = (t["primary_eval"], t["supervisor_eval"]) if name == "cross_replicate" else (t["primary"], t["supervisor"])
        score = t[name][sel]
        if np.isnan(score).any():
            continue
        r, _, _ = paper.referral(score, p[sel], s[sel], t["tasks"][sel])
        out[name] = {"within_nauc": r["within_nauc"], "global_nauc": r["global_nauc"], "automation": r["automation"],
                     "failure_auroc": paper.within_failure_auroc(score, p[sel], t["tasks"][sel])}
    return out


def boot_indices(t: dict, rng) -> np.ndarray:
    """Resample starting states within each task; a state keeps all its variants."""
    idx = []
    for task in paper.TASKS:
        members = {st: np.flatnonzero((t["tasks"] == task) & (t["state"] == st)) for st in STATES}
        draws = rng.integers(len(STATES), size=(BOOT, len(STATES)))
        idx.append(np.stack([np.concatenate([members[STATES[i]] for i in row]) for row in draws]))
    return np.concatenate(idx, axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    seed_pick = rng.integers(len(SEEDS), size=(len(SEEDS), BOOT))
    rollouts = {s: load_rollouts(s) for s in SEEDS}
    result = {"seeds": SEEDS, "supports": {}, "families": {}}
    for support, csv_name in (("dense", "e1_scores_dense.csv"), ("initial", "e1_scores.csv")):
        scores, ref = load_scores(Path("outputs/q5/final/e1") / csv_name)
        tables = {s: seed_table(s, rollouts[s], scores, ref) for s in SEEDS}
        point = {s: metrics(tables[s]) for s in SEEDS}
        draws = {}
        for s in SEEDS:
            idx = boot_indices(tables[s], rng)
            draws[s] = [metrics(tables[s], idx[b]) for b in range(BOOT)]
            print(f"  {support} support, seed {s}: {len(tables[s]['keys'])} conditions, bootstrap done", flush=True)
        summary = {}
        for name in SCORES:
            summary[name] = {}
            for m in ("within_nauc", "global_nauc", "automation", "failure_auroc"):
                vals = [point[s][name][m] for s in SEEDS]
                d = np.array([[draws[s][b][name][m] for b in range(BOOT)] for s in SEEDS])
                total = d[seed_pick, np.arange(BOOT)[None, :]].mean(0)
                summary[name][m] = {"mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals, ddof=1)),
                                    "per_seed": [float(v) for v in vals],
                                    "ci": [float(np.nanquantile(total, 0.025)), float(np.nanquantile(total, 0.975))]}
        result["supports"][support] = summary
        if support == "dense":
            dense_tables = tables
            with open(OUT / "e2_conditions.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["seed", "task", "condition", "init_state_id", "family", "level", "variant",
                            "harmless_expected", "primary", "supervisor", "primary_rep0", "primary_rep1", "primary_rep2",
                            "supervisor_rep0", "supervisor_rep1", "supervisor_rep2", *HEADS, "comb", "comb_mean"])
                for s in SEEDS:
                    t = tables[s]
                    for i, (task, state, variant) in enumerate(t["keys"]):
                        fam, lv = variant.split(":") if ":" in variant else ("clean", "")
                        r = rollouts[s][task, state, variant]
                        w.writerow([s, f"libero_10/{task}", f"libero_10/{task}/start_{state:03d}/{variant}", state, fam, lv,
                                    variant, variant in HARMLESS, f"{t['primary'][i]:.4g}", f"{t['supervisor'][i]:.4g}",
                                    *r["primary"], *r["supervisor"], *[f"{t[h][i]:.6g}" for h in HEADS],
                                    f"{t['comb'][i]:.4g}", f"{t['comb_mean'][i]:.4g}"])
    # Per family: success of primary and supervisor and the mean gain (over seeds, tasks, states).
    fam = defaultdict(lambda: {"primary": [], "supervisor": []})
    for s in SEEDS:
        t = dense_tables[s]
        for i, k in enumerate(t["keys"]):
            fam[k[2]]["primary"].append(t["primary"][i])
            fam[k[2]]["supervisor"].append(t["supervisor"][i])
    result["families"] = {v: {"primary": float(np.mean(d["primary"])), "supervisor": float(np.mean(d["supervisor"])),
                              "gain": float(np.mean(d["supervisor"]) - np.mean(d["primary"])), "harmless_expected": v in HARMLESS}
                          for v, d in fam.items()}
    clean = {}
    for s in SEEDS:
        for role in ("primary", "supervisor"):
            d = json.loads(Path(f"outputs/q5/policies/{role}_s{s}/clean_success.json").read_text())
            clean.setdefault(role, []).append(float(np.mean([np.mean(v) for v in d.values()])))
    result["clean_success"] = clean
    (OUT / "e2_summary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
