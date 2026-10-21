#!/usr/bin/env python
"""Q5 / E1 final analysis over the three seed triples (handoff/q5, Sections 6 and 9).

Inputs per seed s (q5_e1.py, big run): outputs/q5/seed{s}/e1/{support.npz, support_dense.npz, features/, harm/}.
For each seed and support (initial = first frames of the 379 training demonstrations, the paper recipe;
dense = every 10th frame), the ten heads (kv_reference: KVCtxHead / KVDecHead) are fitted on the seed's
primary and every observation is scored; identical inputs are scored once so exact ties stay exact.

Metrics per variant and head:
  AUROC  perturbed vs clean starting observations of the test states 0-24 (VFD: 0-9) within a task, averaged
         over the tasks where the variant exists; then the mean over seeds (and the SD over seeds)
  pct    mean percentile among the task's clean calibration observations (states 25-49), mean over seeds
  CI     hierarchical bootstrap, 2,000 draws: seeds resampled with replacement, and within each drawn seed the
         starting states resampled within each task (a state keeps all its variants); the same draws for all
         heads (paired)
  COMB   maximum percentile over the eight context heads (its own percentile among calibration observations)
Dose-response: mean percentile per level of the graded families. Harm: seed 0, middle level, 3 executions.

Outputs in outputs/q5/final/e1/:
  e1_scores.csv, e1_scores_dense.csv   one row per seed, task, starting state and variant (handoff columns)
  e1_harm.csv                          one row per condition with the three executions
  e1_summary.json                      all metrics; e1_table.html the review table
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["functional_scripts", "handoff/q5"]
import q5_e1_analyze as base  # noqa: E402
from q5_e1 import CALIB, CONTEXT, FAMILIES, HEAD_INPUTS, TEST, VARIANTS, VFD_STATES  # noqa: E402

SEEDS = (0, 1, 2)
TASKS = tuple(range(10))
OUT = Path("outputs/q5/final/e1")
BOOT = 2000
HEADS = base.HEADS
GRADED = {"I1": ["P2", "P3", "P4", "P5"], "V1": ["x0.8", "x0.6", "x0.4"], "V2": ["sigma0.01", "sigma0.03", "sigma0.08"],
          "V3": ["1cm1deg", "2cm3deg", "4cm6deg"], "V4": ["25", "50", "75"], "S1": ["3cm", "6cm", "10cm"],
          "R1": ["0.05rad", "0.1rad", "0.2rad"], "R2": ["1cm", "3cm", "6cm"]}


def seed_rows(seed: int, support_file: str):
    base.ROOT, base.TASKS = Path(f"outputs/q5/seed{seed}/e1"), TASKS
    rows, n_support = base.load(support_file)
    base.add_percentiles(rows)
    for r in rows:
        r["seed"] = seed
    return rows, n_support


def per_seed_stats(rows, rng_counts):
    """Point AUROC per (variant, head) and its bootstrap draws over starting states, for one seed."""
    index = {(r["task"], r["state"], r["variant"]): r for r in rows}
    states = {t: [s for s in TEST if (t, s, "clean") in index] for t in TASKS}
    point, draws, pct, ntasks = {}, {}, {}, {}
    for v in VARIANTS[1:]:
        for h in HEADS:
            p_task, b_task, pc = [], [], []
            for t, st in states.items():
                keep = [i for i, s in enumerate(st) if h != "vfd" or s in VFD_STATES]
                pert = [i for i in keep if (t, st[i], v) in index]
                if not pert:
                    continue
                p = np.array([index[t, st[i], v][h] for i in pert])
                c = np.array([index[t, st[i], "clean"][h] for i in keep])
                if np.isnan(p).all() or np.isnan(c).all():
                    continue
                m = (p[:, None] > c[None, :]) + 0.5 * (p[:, None] == c[None, :])
                p_task.append(m.mean())
                wp, wc = rng_counts[t][:, pert], rng_counts[t][:, keep]
                with np.errstate(invalid="ignore", divide="ignore"):
                    b_task.append(np.einsum("bi,ij,bj->b", wp, m, wc) / (wp.sum(1) * wc.sum(1)))
                pc.append(np.nanmean([index[t, st[i], v][f"pct_{h}"] for i in pert]))
            if p_task:
                point[v, h] = float(np.mean(p_task))
                draws[v, h] = np.nanmean(np.stack(b_task), axis=0)
                pct[v, h] = float(np.mean(pc))
                ntasks[v, h] = len(p_task)
    return point, draws, pct, ntasks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    seed_draws = rng.integers(0, len(SEEDS), size=(BOOT, len(SEEDS)))  # resampled seeds per draw
    summaries, n_support, all_rows = {}, {}, {}
    for support, support_file in base.SUPPORTS.items():
        stats = {}
        for seed in SEEDS:
            rows, n_support[support] = seed_rows(seed, support_file)
            all_rows.setdefault(support, []).extend(rows)
            counts = {t: np.stack([np.bincount(rng.integers(0, 25, 25), minlength=25) for _ in range(BOOT)]).astype(float)
                      for t in TASKS}
            stats[seed] = per_seed_stats(rows, counts)
            print(f"  seed {seed}, {support} support: {len(rows)} observations scored", flush=True)
        summary = {}
        for v in VARIANTS[1:]:
            summary[v] = {}
            for h in HEADS:
                vals = [stats[s][0].get((v, h)) for s in SEEDS]
                if any(x is None for x in vals):
                    summary[v][h] = {"auroc": None, "pct": None, "ci": None, "sd": None, "tasks": 0}
                    continue
                d = np.stack([stats[s][1][v, h] for s in SEEDS])           # seeds x draws
                hier = d[seed_draws, np.arange(BOOT)[:, None]].mean(axis=1)  # resampled seeds, same draw index
                hier = hier[~np.isnan(hier)]
                summary[v][h] = {"auroc": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                                 "per_seed": [float(x) for x in vals],
                                 "pct": float(np.mean([stats[s][2][v, h] for s in SEEDS])),
                                 "ci": [float(np.quantile(hier, 0.025)), float(np.quantile(hier, 0.975))],
                                 "tasks": int(stats[SEEDS[0]][3][v, h])}
        summaries[support] = summary

    # Dose-response (mean percentile per level), both supports.
    dose = {sup: {fam: {h: [summaries[sup].get(f"{fam}:{lv}", {}).get(h, {}).get("pct") for lv in lvls] for h in HEADS}
                  for fam, lvls in GRADED.items()} for sup in summaries}

    # Harm (seed 0).
    harm, harm_rows = {}, []
    for path in sorted(Path("outputs/q5/seed0/e1/harm").glob("*.json")):
        d = json.loads(path.read_text())
        harm.setdefault(d["variant"], []).extend(x for x in d["success"] if x is not None)
        harm_rows.append(d)
    harm_summary = {v: {"success": float(np.mean(x)), "n": len(x)} for v, x in harm.items() if x}

    # Handoff CSVs.
    for support, rows in all_rows.items():
        name = "e1_scores.csv" if support == "initial" else f"e1_scores_{support}.csv"
        with open(OUT / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed", "task", "init_state_id", "split", "family", "level", "variant", "realistic",
                        "harmless_expected", "instruction", *HEAD_INPUTS, "vfd"])
            for r in rows:
                fam, lv = r["variant"].split(":") if ":" in r["variant"] else ("clean", "")
                spec = FAMILIES.get(fam)
                w.writerow([r["seed"], f"libero_10/{r['task']}", r["state"], "test" if r["state"] in TEST else "calibration",
                            fam, lv, r["variant"], spec[3] if spec else "", (lv in spec[4]) if spec else "",
                            r["instruction"], *[f"{r[k]:.6g}" for k in HEAD_INPUTS],
                            "" if np.isnan(r["vfd"]) else f"{r['vfd']:.6g}"])
    with open(OUT / "e1_harm.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "task", "init_state_id", "variant", "primary_rep0", "primary_rep1", "primary_rep2"])
        for d in harm_rows:
            w.writerow([0, f"libero_10/{d['task']}", d["state"], d["variant"], *d["success"]])

    (OUT / "e1_summary.json").write_text(json.dumps(
        {"seeds": SEEDS, "tasks": TASKS, "support_rows": n_support, "variants": summaries, "dose_response": dose,
         "harm_seed0": harm_summary, "bootstrap": {"draws": BOOT, "scheme": "seeds, then starting states within task"}},
        indent=1) + "\n")

    # Review table (the pilot's HTML, with the final protocol in the text and the SD in the tooltips).
    base.ROOT = Path("outputs/q5/seed0/e1")  # thumbnails of seed 0, task 0, starting state 0
    base.TEMPLATE = base.TEMPLATE.replace(
        "Pilot: policy = pretraining lineage seed_1 (VFD with seed_2), LIBERO-10 tasks 0–2, test starting states 0–24\n(VFD 0–9), calibration states 25–49.",
        "Final run: three primary/companion pairs trained on all of LIBERO-10 (paper recipe), all ten tasks, test starting "
        "states 0–24 (VFD 0–9), calibration states 25–49. Values are means over the three seeds; intervals are a hierarchical "
        "bootstrap over seeds and starting states.").replace(
        "first frames of the {n_support} training demonstrations of tasks 0–2", "first frames of the {n_support} training demonstrations").replace(
        "starting states 0–9 × 3 seeds × 3 tasks", "seed 0, starting states 0–9 × 3 executions × 10 tasks")
    base.write_html(summaries, harm_summary, n_support, sum(len(r) for r in all_rows.values()) // 2)
    (base.ROOT / "e1_table.html").rename(OUT / "e1_table.html")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
