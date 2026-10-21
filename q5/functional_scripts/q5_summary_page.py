#!/usr/bin/env python
"""One-page summary of the final Q5 results (E1 and E3) for review: outputs/q5/final/summary.html.

Reads outputs/q5/final/e1/e1_summary.json and outputs/q5/final/e3/e3_summary.json; every number on the page comes
from them. Heatmaps use a diverging scale around AUROC 0.5 (no signal = neutral grey).
"""

import json
from pathlib import Path

E1 = json.loads(Path("outputs/q5/final/e1/e1_summary.json").read_text())
E3 = json.loads(Path("outputs/q5/final/e3/e3_summary.json").read_text())
OUT = Path("outputs/q5/final/summary.html")

COLUMNS = [("pre_ws", "workspace"), ("pre_wrist", "wrist"), ("pre_lang", "instruction"), ("pre_state", "state"),
           ("ctx_ws", "workspace"), ("ctx_wrist", "wrist"), ("ctx_lang", "instruction"), ("ctx_state", "state"),
           ("kv_ctx", "KV-Ctx"), ("kv_dec", "KV-Dec"), ("vfd", "VFD")]
E1_ROWS = [("I1:P5", "Paraphrase (P5)", "instruction"), ("I2:all", "Object synonyms", "instruction"),
           ("I3:typo", "One typo", "instruction"), ("I4:pronouns", "Pronouns only", "instruction"),
           ("I5:german", "German", "instruction"), ("V1:x0.4", "Light ×0.4", "both cameras"),
           ("V2:sigma0.08", "Image noise σ 0.08", "both cameras"), ("V3:4cm6deg", "Camera moved 4 cm / 6°", "workspace"),
           ("V4:75", "Target 75 % occluded", "workspace"), ("V5:table", "Table texture", "scene"),
           ("V6:blackout", "Wrist camera black", "wrist"), ("V7:grey", "Grey images", "both cameras"),
           ("S1:10cm", "Target moved 10 cm", "scene"), ("S2:near", "Distractor next to target", "scene"),
           ("S3:90deg", "Target rotated 90°", "scene"), ("R1:0.1rad", "Joints offset 0.1 rad", "robot"),
           ("R2:3cm", "State reads 3 cm off", "state"), ("M1:same_scene", "Other task's instruction, same scene", "mismatch"),
           ("M1:other_scene", "Instruction naming absent objects", "mismatch"), ("M3:other_task", "State of another task", "mismatch")]
HARM_OF = {"I1:P5": "I1:P3", "V3:4cm6deg": "V3:2cm3deg", "V4:75": "V4:50", "S1:10cm": "S1:6cm"}


def cell(m):
    if not m or m.get("auroc") is None:
        return None
    ci = m.get("ci") or m.get("auroc_ci")
    return {"v": round(m["auroc"], 3), "lo": round(ci[0], 3) if ci else None, "hi": round(ci[1], 3) if ci else None,
            "sd": round(m["sd"], 3) if m.get("sd") is not None else (round(m["auroc_sd"], 3) if m.get("auroc_sd") is not None else None)}


e1 = E1["variants"]["dense"]
harm = E1["harm_seed0"]
e1_rows = []
for key, label, inp in E1_ROWS:
    h = harm.get(HARM_OF.get(key, key))
    e1_rows.append({"key": key, "label": label, "input": inp, "cells": [cell(e1[key][c]) for c, _ in COLUMNS],
                    "success": round(h["success"], 2) if h else None,
                    "harm_level": HARM_OF.get(key, key) if h else None})
e3_rows = []
for key, r in E3["events"].items():
    e3_rows.append({"key": key, "label": r["description"], "input": r["group"], "cells": [cell(r["heads"][c]) for c, _ in COLUMNS],
                    "success": [round(r["success_clean"], 2), round(r["success_event"], 2)]})
scatter = []
clean = harm["clean"]["success"]
for variant, h in harm.items():
    if variant == "clean" or variant not in e1:
        continue
    scatter.append({"variant": variant, "drop": round(clean - h["success"], 3), "kv_ctx": round(e1[variant]["kv_ctx"]["auroc"], 3),
                    "vfd": round(e1[variant]["vfd"]["auroc"], 3)})
compare = []
for key, label in [("I5", "Instruction negated"), ("V1", "Light ×0.4"), ("S2", "Distractor next to target"), ("S4", "Target vanishes"),
                   ("R3", "Arm joints bumped"), ("V3", "Camera bumped 2 cm / 3°"), ("D1", "Objects 10× heavier (invisible)")]:
    r = E3["events"][key]["heads"]
    compare.append({"exp": "E3", "label": label, **{h: cell(r[h]) for h in ("kv_ctx", "kv_dec", "vfd")}})
for key, label in [("S2:near", "Distractor next to target"), ("S1:10cm", "Target moved 10 cm"), ("V5:table", "Table texture"),
                   ("V2:sigma0.08", "Image noise σ 0.08")]:
    compare.append({"exp": "E1", "label": label, **{h: cell(e1[key][h]) for h in ("kv_ctx", "kv_dec", "vfd")}})

E2_PATH = Path("outputs/q5/final/e2/e2_summary.json")
e2 = None
if E2_PATH.exists():
    E2 = json.loads(E2_PATH.read_text())
    names = [("kv_dec", "KV-Dec", "paper"), ("kv_ctx", "KV-Ctx", "paper"), ("vfd", "VFD (2 policies)", "baseline"),
             ("ctx_state", "state, fused", "context"), ("ctx_lang", "instruction, fused", "context"),
             ("pre_lang", "instruction, raw", "context"), ("ctx_wrist", "wrist, fused", "context"),
             ("pre_wrist", "wrist, raw", "context"), ("ctx_ws", "workspace, fused", "context"),
             ("pre_state", "state, raw", "context"), ("pre_ws", "workspace, raw", "context"),
             ("comb", "COMB (max percentile)", "combination"), ("comb_mean", "COMB-mean (mean percentile)", "combination"),
             ("cross_replicate", "Cross-replicate reference", "reference")]
    rows = []
    for key, label, kind in names:
        m = E2["supports"]["dense"][key]
        rows.append({"key": key, "label": label, "kind": kind,
                     **{k: {"mean": round(m[k]["mean"], 3), "sd": round(m[k]["sd"], 3), "lo": round(m[k]["ci"][0], 3),
                            "hi": round(m[k]["ci"][1], 3)} for k in ("within_nauc", "global_nauc", "automation", "failure_auroc")}})
    order = ["I1:P3", "I1:P5", "I2:all", "I3:please", "V1:x0.8", "V1:x0.4", "V2:sigma0.01", "V2:sigma0.08", "V3:2cm3deg",
             "V4:50", "V6:blackout", "R2:3cm", "clean"]
    fams = [{"variant": v, **{k: round(E2["families"][v][k], 3) for k in ("primary", "supervisor", "gain")},
             "harmless": E2["families"][v]["harmless_expected"]} for v in order]
    init = {k: round(E2["supports"]["initial"][k]["within_nauc"]["mean"], 3) for k in ("kv_ctx", "kv_dec")} \
        if "initial" in E2["supports"] else None
    e2 = {"rows": rows, "families": fams, "clean": {r: [round(x, 3) for x in v] for r, v in E2["clean_success"].items()},
          "initial": init}

data = {"columns": [{"key": k, "label": l} for k, l in COLUMNS], "e1": e1_rows, "e3": e3_rows, "scatter": scatter, "e2": e2,
        "clean_success": round(clean, 2), "compare": compare,
        "e3_identical": round(E3["pre_onset_identical"], 2)}

PAGE = (Path(__file__).with_name("q5_summary_template.html")).read_text().replace("__DATA__", json.dumps(data))
OUT.write_text(PAGE)
print(f"wrote {OUT}")
