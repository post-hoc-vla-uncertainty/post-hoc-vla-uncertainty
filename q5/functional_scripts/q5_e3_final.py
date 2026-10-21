#!/usr/bin/env python
"""Q5 / E3 final analysis over the three seed triples (handoff/q5, Sections 8 and 9).

Inputs per seed s (q5_e3_capture.py, big run): outputs/q5/seed{s}/e3/task{t}_state{k}_{event}.npz, one clean run and
one run per event for every task t and starting state k (0, 1), identical up to the onset at step 100; the inputs of
all ten heads and VFD every 5 environment steps. Heads (kv_reference) are fitted per seed on the dense support
(every 10th frame of the primary's training demonstrations, outputs/q5/seed{s}/e1/support_dense.npz).

Percentiles and alarm thresholds: among all scored steps of separate clean calibration runs of the same task and
seed (starting states 2-4); the clean twins of the test pairs (states 0, 1) are not used for calibration. COMB is the
maximum percentile over the eight context heads (its own percentile among the clean steps).

Metrics per event and head (pair = event run + its clean twin, same seed, task and starting state):
  window AUROC   scored steps 100-149 of the event run vs the same steps of the clean twin (steps both runs reached)
  rise           mean percentile in that window, event minus clean (0-100 scale)
  detection      share of event runs exceeding the head's 99th percentile of clean steps inside the window; delay =
                 median steps from onset to the first exceedance; false alarm = share of clean runs exceeding it
  recovery       transient events (V4, V6: steps 100-159): rise in steps 160-209
  success        event vs clean
Values are means over pairs within a seed, then means over seeds (SD over seeds); 95 % intervals from a
hierarchical bootstrap (seeds, then pairs within seed; 2,000 draws, the same for all heads).

Outputs in outputs/q5/final/e3/: e3_episodes.csv, e3_traces.csv (handoff columns), e3_summary.json, e3_table.html.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["functional_scripts", "handoff/q5"]
from kv_reference import KVCtxHead, KVDecHead  # noqa: E402
from q5_e1 import CONTEXT, HEAD_INPUTS  # noqa: E402

SEEDS, TASKS, STATES, CALIB = (0, 1, 2), tuple(range(10)), (0, 1), (2, 3, 4)
RUNS_DIR = "e3_v2"  # rerun with group-wise sharding (bit-identical twins) and separate calibration runs
ONSET, WINDOW, BOOT = 100, 50, 2000
OUT = Path("outputs/q5/final/e3")
HEADS = ("kv_ctx", "kv_dec", *CONTEXT, "comb", "vfd")
LABEL = {"kv_ctx": "KV-Ctx", "kv_dec": "KV-Dec", "pre_ws": "ws raw", "pre_wrist": "wrist raw", "pre_lang": "lang raw",
         "pre_state": "state raw", "ctx_ws": "ws fused", "ctx_wrist": "wrist fused", "ctx_lang": "lang fused",
         "ctx_state": "state fused", "comb": "COMB", "vfd": "VFD"}
EVENTS = [  # name, group, description, transient
    ("I1", "Language", "Instruction paraphrased (P5)", False), ("M1", "Language", "Other task's instruction", False),
    ("I5", "Language", "Instruction negated", False), ("V1", "Camera images", "Light dimmed ×0.4", False),
    ("V3", "Camera images", "Workspace camera bumped", False), ("V4", "Camera images", "Occluder over target (60 steps)", True),
    ("V6", "Camera images", "Wrist camera frozen (60 steps)", True), ("V7", "Camera images", "Heavy image noise", False),
    ("S1", "Scene", "Target moved 6 cm", False), ("S2", "Scene", "Distractor next to target", False),
    ("S4", "Scene", "Target vanishes", False), ("R3", "Robot", "Arm joints bumped 0.1 rad", False),
    ("D1", "Physics (invisible)", "Objects 10× heavier", False),
]


def midrank(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 100 * (np.searchsorted(ref, x, "left") + np.searchsorted(ref, x, "right")) / (2 * len(ref))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def load_seed(seed: int) -> dict:
    """Runs of one seed: (task, state, event) -> dict with steps, success, raw scores and percentiles per head."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    support = np.load(f"outputs/q5/seed{seed}/e1/support_dense.npz")
    heads = {k: (KVDecHead(device=device) if k == "kv_dec" else KVCtxHead(device=device)).fit(support[k]) for k in HEAD_INPUTS}
    runs, feats = {}, {k: [] for k in HEAD_INPUTS}
    for path in sorted(Path(f"outputs/q5/seed{seed}/{RUNS_DIR}").glob("task*_state*_*.npz")):
        d = np.load(path)
        meta = json.loads(str(d["meta"]))
        if meta["task"] not in TASKS:
            continue
        key = (meta["task"], meta["state"], meta["event"])
        runs[key] = {"steps": d["steps"], "success": bool(d["success"]), "n_steps": int(d["n_steps"]),
                     "vfd": d["vfd"].astype(float), "meta": meta, "path": path}
        for k in HEAD_INPUTS:
            feats[k].append(d[k])
    order = list(runs)
    for k in HEAD_INPUTS:
        stacked = np.concatenate(feats[k])
        unique, inverse = np.unique(stacked, axis=0, return_inverse=True)  # identical inputs, identical scores
        scores = heads[k].score(unique)[inverse.reshape(-1)]
        i = 0
        for key in order:
            n = len(runs[key]["steps"])
            runs[key][k] = scores[i:i + n].astype(float)
            i += n
    # Percentiles and alarm thresholds among the steps of the task's separate calibration runs (clean, starting
    # states 2-4), so the clean twins of the test pairs give out-of-sample false-alarm rates; COMB from the context heads.
    for task in TASKS:
        clean = [runs[task, s, "clean"] for s in CALIB if (task, s, "clean") in runs]
        ref = {h: np.sort(np.concatenate([r[h] for r in clean])) for h in (*HEAD_INPUTS, "vfd")}
        for key, r in runs.items():
            if key[0] != task:
                continue
            for h in ref:
                r[f"pct_{h}"] = midrank(ref[h], r[h])
            r["comb"] = np.max([r[f"pct_{h}"] for h in CONTEXT], axis=0)
        comb_ref = np.sort(np.concatenate([r["comb"] for r in clean]))
        for key, r in runs.items():
            if key[0] == task:
                r["pct_comb"] = midrank(comb_ref, r["comb"])
                r["thr"] = {h: np.quantile(np.concatenate([c[h] for c in clean]), 0.99) for h in HEADS}
    return runs


def pair_metrics(event_run: dict, clean_run: dict, head: str, transient: bool) -> dict:
    steps_e, steps_c = event_run["steps"], clean_run["steps"]
    common = np.intersect1d(steps_e[(steps_e >= ONSET) & (steps_e < ONSET + WINDOW)], steps_c)
    if not len(common):
        return {}
    ie, ic = np.searchsorted(steps_e, common), np.searchsorted(steps_c, common)
    e, c = event_run[head][ie], clean_run[head][ic]
    pe, pc = event_run[f"pct_{head}"][ie], clean_run[f"pct_{head}"][ic]
    thr = event_run["thr"][head]
    window = steps_e[(steps_e >= ONSET) & (steps_e < ONSET + WINDOW)]
    above = window[event_run[head][np.searchsorted(steps_e, window)] > thr]
    out = {"auroc": auroc(e, c), "rise": float(pe.mean() - pc.mean()), "pct_event": float(pe.mean()),
           "pct_clean": float(pc.mean()), "detect": float(len(above) > 0),
           "delay": float(above[0] - ONSET) if len(above) else np.nan}
    if transient:
        after = np.intersect1d(steps_e[(steps_e >= ONSET + 60) & (steps_e < ONSET + 110)], steps_c)
        if len(after):
            out["recovery"] = float(event_run[f"pct_{head}"][np.searchsorted(steps_e, after)].mean()
                                    - clean_run[f"pct_{head}"][np.searchsorted(steps_c, after)].mean())
    return out


def thumb(event: str) -> str:
    from PIL import Image
    path = Path(f"outputs/q5/seed0/e3/task0_state0_{event}.npz")
    if not path.exists():
        return ""
    frames = np.load(path)["external"]
    buffer = io.BytesIO()
    Image.fromarray(frames[min(120, len(frames) - 1)]).resize((84, 84)).save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def main() -> None:
    import argparse
    global TASKS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, nargs="+", default=list(TASKS),
                        help="subset of tasks, e.g. without the kitchen scenes 3, 8, 9 whose resets are not deterministic")
    parser.add_argument("--tag", default="", help="suffix of the output directory")
    args = parser.parse_args()
    TASKS, OUT = tuple(args.tasks), Path(str(OUT) + args.tag)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    per_seed, episodes, false_alarm, identical = {}, [], {}, []
    traces = open(OUT / "e3_traces.csv", "w", newline="")
    tw = csv.writer(traces)
    tw.writerow(["seed", "task", "init_state_id", "event", "run", "env_step", "head", "score", "percentile"])
    for seed in SEEDS:
        runs = load_seed(seed)
        print(f"  seed {seed}: {len(runs)} runs scored", flush=True)
        for (task, state, event), r in runs.items():
            m = r["meta"]
            kind = "calibration" if state in CALIB else ("clean" if event == "clean" else "event")
            episodes.append([seed, f"libero_10/{task}", state, event, m.get("level") or "", kind,
                             m.get("onset") or "", r["n_steps"] - 1, r["success"], r["n_steps"]])
            for h in HEADS:
                for step, score, pct in zip(r["steps"], r[h], r[f"pct_{h}"]):
                    tw.writerow([seed, f"libero_10/{task}", state, event, kind,
                                 int(step), h, f"{score:.6g}", f"{pct:.4g}"])
            if event != "clean" and (task, state, "clean") in runs:
                c = runs[task, state, "clean"]
                pre = (r["steps"] < ONSET)
                identical.append(bool(np.allclose(r["kv_ctx"][pre][:len(c["kv_ctx"][c["steps"] < ONSET])],
                                                  c["kv_ctx"][c["steps"] < ONSET][:pre.sum()], atol=1e-5)))
        false_alarm[seed] = {h: float(np.mean([(runs[t, s, "clean"][h] > runs[t, s, "clean"]["thr"][h]).any()
                                               for t in TASKS for s in STATES if (t, s, "clean") in runs])) for h in HEADS}
        per_seed[seed] = {}
        for event, _, _, transient in EVENTS:
            pairs = [(t, s) for t in TASKS for s in STATES if (t, s, event) in runs and (t, s, "clean") in runs]
            per_seed[seed][event] = {
                "pairs": pairs,
                "success_event": float(np.mean([runs[p + (event,)]["success"] for p in pairs])),
                "success_clean": float(np.mean([runs[p + ("clean",)]["success"] for p in pairs])),
                "values": {h: [pair_metrics(runs[p + (event,)], runs[p + ("clean",)], h, transient) for p in pairs]
                           for h in HEADS}}
        del runs
    traces.close()
    with open(OUT / "e3_episodes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "task", "init_state_id", "event", "level", "run", "onset_step", "end_step", "success", "steps"])
        w.writerows(episodes)

    # Summaries: per seed the mean over pairs, then mean and SD over seeds, hierarchical bootstrap.
    seed_draws = rng.integers(0, len(SEEDS), size=(BOOT, len(SEEDS)))
    summary = {}
    for event, group, desc, transient in EVENTS:
        summary[event] = {"group": group, "description": desc, "transient": transient,
                          "success_clean": float(np.mean([per_seed[s][event]["success_clean"] for s in SEEDS])),
                          "success_event": float(np.mean([per_seed[s][event]["success_event"] for s in SEEDS])),
                          "pairs_per_seed": [len(per_seed[s][event]["pairs"]) for s in SEEDS], "heads": {}}
        pair_draws = {s: rng.integers(0, len(per_seed[s][event]["pairs"]), size=(BOOT, len(per_seed[s][event]["pairs"])))
                      for s in SEEDS}
        for h in HEADS:
            out = {}
            for metric in ("auroc", "rise", "pct_event", "pct_clean", "detect", "delay", "recovery"):
                arrays = {s: np.array([v.get(metric, np.nan) for v in per_seed[s][event]["values"][h]], float) for s in SEEDS}
                if all(np.isnan(a).all() for a in arrays.values()):
                    continue
                means = [float(np.nanmedian(a)) if metric == "delay" else float(np.nanmean(a)) for a in arrays.values()]
                out[metric] = float(np.nanmean(means))
                out[f"{metric}_sd"] = float(np.nanstd(means, ddof=1))
                if metric in ("auroc", "rise", "detect"):
                    boot = np.stack([np.nanmean(arrays[s][pair_draws[s]], axis=1) for s in SEEDS])  # seeds x draws
                    hier = boot[seed_draws, np.arange(BOOT)[:, None]].mean(axis=1)
                    out[f"{metric}_ci"] = [float(np.nanquantile(hier, 0.025)), float(np.nanquantile(hier, 0.975))]
            out["false_alarm"] = float(np.mean([false_alarm[s][h] for s in SEEDS]))
            summary[event]["heads"][h] = out
    result = {"seeds": SEEDS, "onset": ONSET, "window": WINDOW, "pre_onset_identical": float(np.mean(identical)),
              "events": summary}
    (OUT / "e3_summary.json").write_text(json.dumps(result, indent=1) + "\n")
    write_html(result)
    print(f"pre-onset identical pairs: {np.mean(identical):.3f}; wrote {OUT}")


def write_html(result: dict) -> None:
    import e3_table
    rows, group = [], None
    for event, grp, desc, _ in EVENTS:
        r = result["events"][event]
        if grp != group:
            rows.append(f'<tr class="group"><th colspan="{len(HEADS) + 3}">{html.escape(grp)}</th></tr>')
            group = grp
        image = thumb(event)
        img = f'<img src="data:image/jpeg;base64,{image}" alt="">' if image else ""
        cells = ""
        for h in HEADS:
            m = r["heads"][h]
            legacy = {"step_auroc": m.get("auroc", np.nan), "step_auroc_ci": m.get("auroc_ci"), "episode_auroc": np.nan,
                      "pct_delta": m.get("rise", np.nan), "pct_delta_ci": m.get("rise_ci"),
                      "pct_clean": m.get("pct_clean", np.nan), "pct_event": m.get("pct_event", np.nan), "detect_p99": m.get("detect", 0.0),
                      "delay_p99": m.get("delay"), "false_alarm_p99": m["false_alarm"]}
            for metric in ("auroc", "delta", "alarm"):
                cells += e3_table.cell(metric, legacy)
        rows.append(f'<tr><td class="ev">{html.escape(desc)}<span class="set">{event} · {sum(r["pairs_per_seed"])} pairs over 3 seeds'
                    f'</span></td><td class="thumb">{img}</td>{cells}'
                    f'<td class="succ">{r["success_clean"]:.2f} → {r["success_event"]:.2f}</td></tr>')
    fa = "".join(f'<td class="m alarm fa">{100 * result["events"]["I1"]["heads"][h]["false_alarm"]:.0f}%</td>'
                 '<td class="m auroc"></td><td class="m delta"></td>' for h in HEADS)
    rows.append(f'<tr class="farow"><td class="ev">False alarms, clean runs<span class="set">mean over seeds</span></td><td></td>{fa}<td></td></tr>')
    heads = "".join(f'<th class="h">{html.escape(LABEL[h])}</th>' for h in HEADS)
    page = e3_table.TEMPLATE.format(heads=heads, rows="\n".join(rows))
    page = page.replace("LIBERO-10 tasks 0 and 1, starting states 0–9; each event run (disturbance in steps 30–89) is paired with a clean run of the\nsame starting state and seed. Policy: pretraining lineage seed_1; VFD with seed_2. Heads on a dense support (every 10th demonstration frame\nof tasks 0–2); percentiles among all steps of ten clean calibration runs per task.",
                        "Final run: three primary/companion pairs on all ten LIBERO-10 tasks, starting states 0 and 1; each event run (onset at step 100,\npersistent unless marked) is paired with the clean run of the same starting state and seed. Heads on a dense support per seed; percentiles\namong all scored steps of separate clean calibration runs (starting states 2–4); window = steps 100–149. Values are means over three seeds with hierarchical bootstrap intervals.")
    (OUT / "e3_table.html").write_text(page)


if __name__ == "__main__":
    main()
