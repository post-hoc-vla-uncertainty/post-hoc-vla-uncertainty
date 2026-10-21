#!/usr/bin/env python
"""Q5 / E1 analysis: scores, within-task detection AUROC, dose-response, harm, HTML review table.

Inputs (from q5_e1.py): outputs/q5/e1/support.npz, features/task*_state*.npz, harm/*.json.
Depends only on numpy, torch and the handoff's kv_reference, so it can move to the paper
repository unchanged (with paths adjusted).

Outputs in outputs/q5/e1/:
  e1_scores.csv   one row per starting observation and variant, raw scores per head (handoff Section 9)
  e1_harm.csv     one row per condition with the three executions
  e1_summary.json AUROC / percentile / harm per variant and head, with 95 % bootstrap intervals
  e1_table.html   the review table
Metrics:
  AUROC   perturbed vs clean observations of starting states 0-24 within a task (VFD: 0-9),
          averaged over the tasks where the variant exists; raw scores
  pct     mean percentile of the perturbed observations among the task's clean calibration
          observations (starting states 25-49); clean test observations sit near 50
  CI      bootstrap over starting states within each task (a state keeps all its variants),
          1,000 draws, paired across heads
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("outputs/q5/e1")
sys.path.insert(0, "handoff/q5")
sys.path.insert(0, "functional_scripts")
from kv_reference import KVCtxHead, KVDecHead  # noqa: E402
from q5_e1 import CALIB, CONTEXT, FAMILIES, HEAD_INPUTS, TASKS, TEST, VARIANTS, VFD_STATES  # noqa: E402

HEADS = ("kv_ctx", "kv_dec", *CONTEXT, "comb", "vfd")
LABEL = {"kv_ctx": "KV-Ctx", "kv_dec": "KV-Dec", "comb": "COMB", "vfd": "VFD"}
BOOT = 1000


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


SUPPORTS = {"initial": "support.npz", "dense": "support_dense.npz"}


def load(support_file: str):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    support = np.load(ROOT / support_file)
    heads = {k: (KVDecHead(device=device) if k == "kv_dec" else KVCtxHead(device=device)).fit(support[k])
             for k in HEAD_INPUTS}
    rows, features = [], {k: [] for k in HEAD_INPUTS}  # rows: task, state, variant, instruction
    for task in TASKS:
        for state in [*TEST, *CALIB]:
            path = ROOT / "features" / f"task{task}_state{state:02d}.npz"
            if not path.exists():
                continue
            d = np.load(path)
            for k in HEAD_INPUTS:
                features[k].append(d[k])
            for i, variant in enumerate(d["variants"]):
                rows.append({"task": task, "state": state, "variant": str(variant),
                             "instruction": str(d["instructions"][i]), "vfd": float(d["vfd"][i])})
    for k in HEAD_INPUTS:
        # Score each distinct input once: float32 kernel algebra on the GPU gives identical inputs
        # slightly different scores in different batches, which would break the exact ties of heads
        # that are constant on clean inputs (pre_state, pre_lang) and, through the maximum, COMB.
        unique, inverse = np.unique(np.concatenate(features[k]), axis=0, return_inverse=True)
        scores = heads[k].score(unique)[inverse.reshape(-1)]
        for r, value in zip(rows, scores, strict=True):
            r[k] = float(value)
    return rows, len(support["episode"])


def midrank(sorted_ref: np.ndarray, x) -> float:
    """Percentile with ties counted half, so a head that is constant on clean inputs sits at 50."""
    return float(100 * (np.searchsorted(sorted_ref, x, "left") + np.searchsorted(sorted_ref, x, "right"))
                 / (2 * len(sorted_ref)))


def add_percentiles(rows) -> None:
    for task in TASKS:
        calib = [r for r in rows if r["task"] == task and r["state"] in CALIB and r["variant"] == "clean"]
        ref = {k: np.sort([r[k] for r in calib]) for k in (*HEAD_INPUTS, "vfd")}
        for r in (r for r in rows if r["task"] == task):
            for k, values in ref.items():
                r[f"pct_{k}"] = midrank(values, r[k])
            r["comb"] = max(r[f"pct_{k}"] for k in CONTEXT)
        comb = np.sort([r["comb"] for r in calib])
        for r in (r for r in rows if r["task"] == task):
            r["pct_comb"] = midrank(comb, r["comb"])


def metrics(index, variant, head, states_by_task) -> dict:
    aucs, pcts = [], []
    for task, states in states_by_task.items():
        if head == "vfd":
            states = [s for s in states if s in VFD_STATES]
        pert = [index[task, s, variant][head] for s in states if (task, s, variant) in index]
        if not pert:
            continue
        clean = [index[task, s, "clean"][head] for s in states]
        aucs.append(auroc(pert, clean))
        pcts.append(np.nanmean([index[task, s, variant][f"pct_{head}"] for s in states if (task, s, variant) in index]))
    return {"auroc": float(np.nanmean(aucs)) if aucs else None, "pct": float(np.nanmean(pcts)) if pcts else None,
            "tasks": len(aucs)}


def analyze(rows) -> dict:
    index = {(r["task"], r["state"], r["variant"]): r for r in rows}
    full = {t: [s for s in TEST if (t, s, "clean") in index] for t in TASKS}
    present = [v for v in VARIANTS if any((t, s, v) in index for t in TASKS for s in TEST)]
    out = {v: {h: metrics(index, v, h, full) for h in HEADS} for v in present if v != "clean"}
    # Bootstrap over starting states within each task, the same draws for every variant and head.
    # The AUROC of a resample is w_p' M w_c / (sum w_p * sum w_c), with M the fixed comparison matrix
    # (perturbed vs clean starting state) and w the number of times each state was drawn.
    rng = np.random.default_rng(0)
    counts = {t: np.stack([np.bincount(rng.integers(0, len(s), len(s)), minlength=len(s)) for _ in range(BOOT)])
              for t, s in full.items() if s}
    for v in out:
        for h in HEADS:
            per_task = []
            for t, states in full.items():
                keep = [i for i, s in enumerate(states) if h != "vfd" or s in VFD_STATES]
                pert = [i for i in keep if (t, states[i], v) in index]
                if not pert:
                    continue
                p = np.array([index[t, states[i], v][h] for i in pert])
                c = np.array([index[t, states[i], "clean"][h] for i in keep])
                m = (p[:, None] > c[None, :]) + 0.5 * (p[:, None] == c[None, :])
                wp, wc = counts[t][:, pert].astype(float), counts[t][:, keep].astype(float)
                denominator = wp.sum(1) * wc.sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    per_task.append(np.einsum("bi,ij,bj->b", wp, m, wc) / denominator)
            if per_task:
                values = np.nanmean(np.stack(per_task), axis=0)
                values = values[~np.isnan(values)]
                out[v][h]["ci"] = [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            else:
                out[v][h]["ci"] = None
    return out


def harm() -> dict:
    result = defaultdict(list)
    for path in sorted((ROOT / "harm").glob("*.json")):
        d = json.loads(path.read_text())
        result[d["variant"]] += [x for x in d["success"] if x is not None]
    return {v: {"success": float(np.mean(x)), "n": len(x)} for v, x in result.items() if x}


def write_csv(rows, suffix: str) -> None:
    with open(ROOT / f"e1_scores{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "task", "init_state_id", "split", "family", "level", "variant", "realistic",
                    "harmless_expected", "instruction", *HEAD_INPUTS, "vfd"])
        for r in rows:
            fam, lv = r["variant"].split(":") if ":" in r["variant"] else ("clean", "")
            spec = FAMILIES.get(fam)
            w.writerow([1, f"libero_10/{r['task']}", r["state"], "test" if r["state"] in TEST else "calibration",
                        fam, lv, r["variant"], spec[3] if spec else "", (lv in spec[4]) if spec else "",
                        r["instruction"], *[f"{r[k]:.6g}" for k in HEAD_INPUTS],
                        "" if np.isnan(r["vfd"]) else f"{r['vfd']:.6g}"])
    with open(ROOT / "e1_harm.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "task", "init_state_id", "variant", "primary_rep0", "primary_rep1", "primary_rep2"])
        for path in sorted((ROOT / "harm").glob("*.json")):
            d = json.loads(path.read_text())
            w.writerow([1, f"libero_10/{d['task']}", d["state"], d["variant"], *d["success"]])


def thumbnails() -> dict:
    from PIL import Image
    out = {}
    path = ROOT / "features" / "task0_state00.npz"
    if not path.exists():
        return out
    d = np.load(path)
    for name, thumb in zip(d["variants"], d["thumbs"]):
        buffer = io.BytesIO()
        Image.fromarray(thumb).save(buffer, format="JPEG", quality=80)
        out[str(name)] = base64.b64encode(buffer.getvalue()).decode()
    return out


def colour(value: float | None, metric: str) -> str:
    if value is None or np.isnan(value):
        return ""
    x = (value - 0.5) * 2 if metric == "auroc" else (value - 50) / 50
    x = float(np.clip(x, -1, 1))
    return f"--x:{x:.3f}"


def write_html(summaries, harm_result, n_support, n_rows) -> None:
    summary = summaries["initial"]
    thumbs = thumbnails()
    head_cells = "".join(f'<th class="h">{html.escape(LABEL.get(h, h))}</th>' for h in HEADS)
    body = []
    clean_harm = harm_result.get("clean")
    for family, spec in FAMILIES.items():
        levels = [f"{family}:{lv}" for lv in spec[2] if f"{family}:{lv}" in summary]
        for i, variant in enumerate(levels):
            lv = variant.split(":")[1]
            cells = []
            for h in HEADS:
                for support, sm in summaries.items():
                    m = sm[variant][h]
                    ci = m.get("ci")
                    for metric in ("auroc", "pct"):
                        v = m[metric]
                        text = "–" if v is None or np.isnan(v) else (f"{v:.2f}" if metric == "auroc" else f"{v:.0f}")
                        tip = (f"{LABEL.get(h, h)} {variant} ({support} support): AUROC {m['auroc']:.3f}"
                               + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
                               + (f", SD over seeds {m['sd']:.3f}" if m.get("sd") is not None else "") + f", mean pct {m['pct']:.0f}, "
                               f"{m['tasks']} task(s)") if m["auroc"] is not None else ""
                        sig = " sig" if metric == "auroc" and ci and ci[0] > 0.5 else ""
                        cells.append(f'<td class="m {metric} {support}{sig}" style="{colour(v, metric)}" title="{html.escape(tip)}">{text}</td>')
            hm = harm_result.get(variant)
            harm_cell = f"{hm['success']:.2f}" if hm else ""
            img = f'<img src="data:image/jpeg;base64,{thumbs[variant]}" alt="">' if variant in thumbs else ""
            fam_cell = (f'<td class="fam" rowspan="{len(levels)}"><b>{family}</b><br>{html.escape(spec[0])}'
                        f'<br><span class="tag">{"realistic" if spec[3] else "blatant"}</span></td>') if i == 0 else ""
            flag = ' <span class="tag">harmless?</span>' if lv in spec[4] else ""
            body.append(f'<tr class="{"first" if i == 0 else ""}">{fam_cell}<td class="lv">{html.escape(lv)}{flag}</td>'
                        f'<td class="thumb">{img}</td>{"".join(cells)}<td class="harm">{harm_cell}</td></tr>')
    clean_text = f"{clean_harm['success']:.2f} (n={clean_harm['n']})" if clean_harm else "pending"
    clean_img = f'<img src="data:image/jpeg;base64,{thumbs["clean"]}" alt="">' if "clean" in thumbs else ""
    page = TEMPLATE.format(head_cells=head_cells, rows="\n".join(body), n_support=n_support["initial"],
                           n_dense=n_support.get("dense", "–"), n_rows=n_rows,
                           clean_harm=clean_text, clean_img=clean_img, ncols=2 * len(HEADS))
    (ROOT / "e1_table.html").write_text(page)


TEMPLATE = """<title>Q5 E1 Head Selectivity</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
:root {{ --bg:#f6f7f9; --fg:#1b2230; --muted:#5d6676; --line:#dde1e8; --pos:31,119,180; --neg:214,96,77; --card:#ffffff; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme:dark; --bg:#13161c; --fg:#e8ebf0; --muted:#9aa3b2; --line:#2d333d; --card:#1a1e26; --pos:86,160,220; --neg:230,120,100; }} }}
:root[data-theme="dark"] {{ color-scheme:dark; --bg:#13161c; --fg:#e8ebf0; --muted:#9aa3b2; --line:#2d333d; --card:#1a1e26; --pos:86,160,220; --neg:230,120,100; }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 "IBM Plex Sans", system-ui, sans-serif; }}
#page {{ padding-block:16px; padding-inline:16px; }}
td.m {{ font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:12px; }}
h1 {{ text-wrap:balance; }}
button:focus-visible {{ outline:2px solid rgb(var(--pos)); outline-offset:2px; }}
h1 {{ font-size:20px; margin:0 0 4px; }} p {{ margin:4px 0; max-width:1100px; }} .muted {{ color:var(--muted); }}
.controls {{ margin:12px 0; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
button {{ font:inherit; padding:4px 10px; border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:6px; cursor:pointer; }}
button.on {{ border-color:var(--fg); font-weight:600; }}
.wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--card); }}
table {{ border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th, td {{ padding:3px 6px; border-bottom:1px solid var(--line); text-align:center; white-space:nowrap; }}
th {{ position:sticky; top:0; background:var(--card); z-index:1; font-weight:600; }}
th.h {{ writing-mode:vertical-rl; transform:rotate(180deg); height:70px; font-size:12px; }}
tr.first td {{ border-top:2px solid var(--line); }}
td.fam {{ text-align:left; vertical-align:top; font-size:12px; min-width:110px; }}
td.lv {{ text-align:left; font-size:12px; }}
td.thumb img {{ height:44px; display:block; }}
td.m {{ background:rgba(var(--pos), calc(max(var(--x, 0), 0) * 0.75)); }}
td.m[style*="--x:-"] {{ background:rgba(var(--neg), calc(max(-1 * var(--x, 0), 0) * 0.75)); }}
td.sig {{ font-weight:700; }}
.tag {{ font-size:10px; color:var(--muted); border:1px solid var(--line); border-radius:4px; padding:0 3px; }}
#page.show-auroc td.pct, #page.show-pct td.auroc, #page.sup-initial td.dense, #page.sup-dense td.initial {{ display:none; }}
td.harm {{ font-weight:600; }}
</style>
<div id="page" class="show-auroc sup-initial">
<h1>Q5 · E1: context heads on perturbed starting observations</h1>
<p class="muted">Pilot: policy = pretraining lineage seed_1 (VFD with seed_2), LIBERO-10 tasks 0–2, test starting states 0–24
(VFD 0–9), calibration states 25–49. Heads fitted on the first frames of the {n_support} training demonstrations of tasks 0–2
(handoff kv_reference: KVCtxHead / KVDecHead) — <b>initial</b> support, the paper's recipe — or on every 10th frame of the same
demonstrations ({n_dense} rows; <b>dense</b> support, as in E3). {n_rows} scored observations. VFD does not depend on the support.</p>
<p class="muted"><b>AUROC</b>: perturbed vs clean observations of the same starting states, within each task, averaged over tasks;
bold = 95 % bootstrap interval above 0.5 (hover for the interval). <b>pct</b>: mean percentile among the clean calibration observations
(clean ≈ 50). <b>COMB</b>: maximum percentile over the eight context heads. <b>success</b>: policy executed with the middle level for the whole
episode, starting states 0–9 × 3 seeds × 3 tasks; clean: {clean_harm}.</p>
<p class="muted">Approximations: V5 tint = material colour multiplied (texture hue not rotated); V6 frozen frame left to E3; S2 moves an existing
object (in task 2 next to the fixed stove); M1 same scene only for tasks 0 ↔ 1; M3 takes the state of another demonstration's first frame.</p>
<p class="muted">Reading notes: in LIBERO the robot starts every episode in the same pose, so the state vector and the wrist view of the first
observation are identical across starting states — M2 and M3 (same task) therefore change nothing at t = 0 (all heads 0.50 by construction).
With the initial-frame support the 8-d state head saturates (every clean observation scores 1.0), so it cannot respond; the dense support
fixes this. COMB is capped near 0.9: with 25 calibration observations per task, one of eight heads is already at the 100th percentile on
many clean observations.</p>
<div class="controls"><span>Show:</span>
<button class="on" data-show="auroc">AUROC</button><button data-show="pct">mean percentile</button>
<span>Support:</span><button class="on" data-sup="initial">initial frames (paper)</button><button data-sup="dense">dense (every 10th frame)</button>
<span class="muted">Clean reference {clean_img}</span></div>
<div class="wrap"><table>
<thead><tr><th>family</th><th>level</th><th>task 0, state 0</th>{head_cells}<th>success</th></tr></thead>
<tbody>
{rows}
</tbody></table></div>
</div>
<script>
const state = {{show: 'auroc', sup: 'initial'}};
['show', 'sup'].forEach(key => document.querySelectorAll(`button[data-${{key}}]`).forEach(b => b.addEventListener('click', () => {{
  state[key] = b.dataset[key];
  document.getElementById('page').className = `show-${{state.show}} sup-${{state.sup}}`;
  document.querySelectorAll(`button[data-${{key}}]`).forEach(x => x.classList.toggle('on', x === b));
}})));
</script>

"""


def main() -> None:
    summaries, n_support = {}, {}
    for name, support_file in SUPPORTS.items():
        if not (ROOT / support_file).exists():
            continue
        rows, n_support[name] = load(support_file)
        add_percentiles(rows)
        summaries[name] = analyze(rows)
        write_csv(rows, "" if name == "initial" else f"_{name}")
    harm_result = harm()
    (ROOT / "e1_summary.json").write_text(json.dumps({"variants": summaries, "harm": harm_result,
                                                      "support_rows": n_support}, indent=1) + "\n")
    write_html(summaries, harm_result, n_support, len(rows))
    print(f"wrote {ROOT / 'e1_table.html'} ({len(rows)} observations, supports {list(summaries)})")


if __name__ == "__main__":
    main()
