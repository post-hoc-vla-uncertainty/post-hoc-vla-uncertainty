#!/usr/bin/env python
"""Q5 / E3 review table: per event and head, step AUROC, percentile rise and alarm rate.

Reads outputs/live_eval/e3/e3_heads.json (e3_heads_eval.py) and one frame of each event run, and
writes outputs/live_eval/e3/e3_table.html (a page body for the artifact viewer).
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np

E3 = Path("outputs/live_eval/e3")
CAPTURES = {"kvctx": Path("outputs/live_eval/kvctx/captures"), "full": Path("outputs/live_eval/full/captures")}
HEADS = ["VFD", "KV-Ctx", "KV-Dec", "pre_ws", "pre_wrist", "pre_lang", "pre_state", "ctx_ws", "ctx_wrist",
         "ctx_lang", "state_tok", "vl_fused", "COMB"]
LABEL = {"pre_ws": "ws raw", "pre_wrist": "wrist raw", "pre_lang": "lang raw", "pre_state": "state raw",
         "ctx_ws": "ws fused", "ctx_wrist": "wrist fused", "ctx_lang": "lang fused", "state_tok": "state token",
         "vl_fused": "all fused"}
# Event -> (group, readable name, capture set). The target-removal event exists in both sets.
EVENTS = [
    ("kvctx", "instruction_negated", "Language", "Instruction negated"),
    ("kvctx", "noise_light", "Camera images", "Image noise, light (σ 0.05)"),
    ("kvctx", "noise_heavy", "Camera images", "Image noise, heavy (σ 0.35)"),
    ("kvctx", "vanish_distractor", "Scene", "One distractor removed"),
    ("kvctx", "vanish_target", "Scene", "Target removed"),
    ("kvctx", "vanish_all", "Scene", "All objects removed"),
    ("kvctx", "scramble", "Scene", "Objects scrambled"),
    ("kvctx", "state_offset", "Robot", "State reads 3 cm off"),
    ("kvctx", "arm_push", "Robot", "Arm pushed with 20 N"),
    ("kvctx", "action_delay", "Action channel", "Actions 3 steps late"),
    ("kvctx", "action_noise", "Action channel", "Action noise σ 0.3"),
    ("kvctx", "action_scale", "Action channel", "Actions scaled ×0.3"),
    ("kvctx", "mass_heavy", "Physics (invisible)", "Objects 10× heavier"),
    ("kvctx", "friction_low", "Physics (invisible)", "Friction ×0.1"),
]


def thumb(set_name: str, event: str) -> str:
    from PIL import Image
    path = CAPTURES[set_name] / f"task0_ep0_{event}.npz"
    if not path.exists():
        return ""
    frame = np.load(path)["external"][60]
    buffer = io.BytesIO()
    Image.fromarray(frame).resize((84, 84)).save(buffer, format="JPEG", quality=80)
    return base64.b64encode(buffer.getvalue()).decode()


def cell(metric: str, m: dict | None) -> str:
    if m is None:
        return f'<td class="m {metric} na">–</td>'
    if metric == "auroc":
        v, ci = m["step_auroc"], m.get("step_auroc_ci")
        x, text, sig = (v - 0.5) * 2, f"{v:.2f}", ci and ci[0] > 0.5
        tip = f"step AUROC {v:.3f} [{ci[0]:.2f}, {ci[1]:.2f}], episode AUROC {m['episode_auroc']:.2f}"
    elif metric == "delta":
        v, ci = m["pct_delta"], m.get("pct_delta_ci")
        x, text, sig = v / 60, f"{v:+.0f}", ci and ci[0] > 0
        tip = (f"window percentile {m['pct_clean']:.0f} (clean twin) → {m['pct_event']:.0f} (event), "
               f"rise {v:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]")
    else:
        v = m["detect_p99"]
        x, text, sig = v * 2 - 1 if v > 0 else 0, f"{100 * v:.0f}%", False
        delay = m.get("delay_p99")
        tip = (f"alarm in {100 * v:.0f}% of event runs" + (f", median {delay:.0f} steps after onset" if delay == delay
               and delay is not None else "") + f"; false alarms on the clean twins: {100 * m['false_alarm_p99']:.0f}%")
    x = float(np.clip(x, -1, 1))
    return (f'<td class="m {metric}{" sig" if sig else ""}" style="--x:{x:.3f}" title="{html.escape(tip)}">'
            f"{text}</td>")


def main() -> None:
    data = json.loads((E3 / "e3_heads.json").read_text())["sets"]
    rows, group = [], None
    for set_name, event, grp, name in EVENTS:
        r = data[set_name]["events"].get(event)
        if r is None:
            continue
        if grp != group:
            rows.append(f'<tr class="group"><th colspan="{len(HEADS) + 3}">{html.escape(grp)}</th></tr>')
            group = grp
        image = thumb(set_name, event)
        img = f'<img src="data:image/jpeg;base64,{image}" alt="">' if image else ""
        cells = "".join(cell(metric, r["heads"].get(h)) for h in HEADS for metric in ("auroc", "delta", "alarm"))
        rows.append(f'<tr><td class="ev">{html.escape(name)}<span class="set">{set_name} set · {r["n_pairs"]} pairs'
                    f'</span></td><td class="thumb">{img}</td>{cells}'
                    f'<td class="succ">{r["success_clean"]:.2f} → {r["success_event"]:.2f}</td></tr>')
    # False alarms of each head on the clean twins, per capture set (alarm view only).
    for set_name in sorted({e[0] for e in EVENTS}):
        first = next(iter(data[set_name]["events"].values()))["heads"]
        cells = "".join(
            (f'<td class="m alarm fa">{100 * first[h]["false_alarm_p99"]:.0f}%</td>' if h in first
             else '<td class="m alarm fa na">–</td>') + '<td class="m auroc"></td><td class="m delta"></td>'
            for h in HEADS)
        rows.append(f'<tr class="farow"><td class="ev">False alarms, clean twins<span class="set">{set_name} set'
                    f'</span></td><td></td>{cells}<td></td></tr>')
    heads = "".join(f'<th class="h">{html.escape(LABEL.get(h, h))}</th>' for h in HEADS)
    (E3 / "e3_table.html").write_text(TEMPLATE.format(heads=heads, rows="\n".join(rows)))
    print(f"wrote {E3 / 'e3_table.html'}")


TEMPLATE = """<title>Q5 E3 Live Events</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
:root {{ --bg:#f6f7f9; --fg:#1b2230; --muted:#5d6676; --line:#dde1e8; --pos:31,119,180; --neg:214,96,77; --card:#ffffff; --band:#eef1f5; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme:dark; --bg:#13161c; --fg:#e8ebf0; --muted:#9aa3b2; --line:#2d333d; --card:#1a1e26; --band:#20252f; --pos:86,160,220; --neg:230,120,100; }} }}
:root[data-theme="dark"] {{ color-scheme:dark; --bg:#13161c; --fg:#e8ebf0; --muted:#9aa3b2; --line:#2d333d; --card:#1a1e26; --band:#20252f; --pos:86,160,220; --neg:230,120,100; }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 "IBM Plex Sans", system-ui, sans-serif; }}
#page {{ padding-block:16px; padding-inline:16px; }}
h1 {{ font-size:20px; margin:0 0 4px; text-wrap:balance; }} p {{ margin:4px 0; max-width:1100px; }} .muted {{ color:var(--muted); }}
.controls {{ margin:12px 0; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
button {{ font:inherit; padding:4px 10px; border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:6px; cursor:pointer; }}
button.on {{ border-color:var(--fg); font-weight:600; }}
button:focus-visible {{ outline:2px solid rgb(var(--pos)); outline-offset:2px; }}
.wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--card); }}
table {{ border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th, td {{ padding:3px 6px; border-bottom:1px solid var(--line); text-align:center; white-space:nowrap; }}
thead th {{ position:sticky; top:0; background:var(--card); z-index:1; font-weight:600; }}
th.h {{ writing-mode:vertical-rl; transform:rotate(180deg); height:84px; font-size:12px; }}
tr.group th {{ text-align:left; background:var(--band); font-size:12px; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); }}
td.ev {{ text-align:left; font-size:13px; }}
.set {{ display:block; font-size:11px; color:var(--muted); }}
td.thumb img {{ height:42px; display:block; }}
td.m {{ font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:12px; background:rgba(var(--pos), calc(max(var(--x, 0), 0) * 0.75)); }}
td.m[style*="--x:-"] {{ background:rgba(var(--neg), calc(max(-1 * var(--x, 0), 0) * 0.75)); }}
td.sig {{ font-weight:700; }}
td.na {{ color:var(--muted); }}
td.fa {{ color:rgb(var(--neg)); }}
td.succ {{ font-size:12px; }}
#page.show-auroc td.delta, #page.show-auroc td.alarm, #page.show-delta td.auroc, #page.show-delta td.alarm,
#page.show-alarm td.auroc, #page.show-alarm td.delta {{ display:none; }}
#page:not(.show-alarm) tr.farow {{ display:none; }}
</style>
<div id="page" class="show-delta">
<h1>Q5 · E3: heads during perturbed rollouts</h1>
<p class="muted">LIBERO-10 tasks 0 and 1, starting states 0–9; each event run (disturbance in steps 30–89) is paired with a clean run of the
same starting state and seed. Policy: pretraining lineage seed_1; VFD with seed_2. Heads on a dense support (every 10th demonstration frame
of tasks 0–2); percentiles among all steps of ten clean calibration runs per task. “raw” heads read an input before the transformer,
“fused” heads the same token group after it; COMB is the maximum percentile over the raw and fused context heads.</p>
<p class="muted"><b>Rise</b>: mean percentile in the window, event run minus clean twin (0–100 scale; bold = 95 % interval above 0).
<b>AUROC</b>: window steps of the event run vs the same steps of the clean twin, per pair, averaged (bold = interval above 0.5).
<b>Alarm</b>: share of event runs in which the head exceeds the 99th percentile of the clean calibration steps inside the window; the last
rows give the false-alarm rate on the clean twins (whole run). Hover a cell for intervals and details.</p>
<div class="controls"><span>Show:</span>
<button data-show="delta" class="on">Rise in percentile</button><button data-show="auroc">Step AUROC</button>
<button data-show="alarm">Alarm rate</button></div>
<div class="wrap"><table>
<thead><tr><th>event</th><th>step 60</th>{heads}<th>success<br>clean → event</th></tr></thead>
<tbody>
{rows}
</tbody></table></div>
</div>
<script>
document.querySelectorAll('button[data-show]').forEach(b => b.addEventListener('click', () => {{
  document.getElementById('page').className = 'show-' + b.dataset.show;
  document.querySelectorAll('button[data-show]').forEach(x => x.classList.toggle('on', x === b));
}}));
</script>
"""


if __name__ == "__main__":
    main()
