# Post-hoc Epistemic Uncertainty for Vision–Language–Action Policies

Code that produced the results of the paper.

| Folder | Content |
|---|---|
| `scripts/`, `data/` | Paper numbers, tables and figures from the frozen result exports in `data/` |
| `study/` | Experiments of the main study (Q1–Q4): policy training, SAVE acquisition, scoring, export |
| `q5/` | Experiments of the context-head study (Q5, E1–E3) |
| `handoff/q5/` | Reference implementation of the KV-Ctx and KV-Dec heads, perturbation library |

Recompute the paper's numbers, tables and figures (numpy, pandas, scipy, matplotlib; no GPU):

```bash
python scripts/analysis.py && python scripts/figures.py      # Q1–Q4
python scripts/q5_analysis.py && python scripts/q5_results.py  # Q5
```

Both analyses check every recomputed value against the exported results and stop on a mismatch.
The experiments in `study/` and `q5/` need GPUs, LIBERO and SmolVLA. Both build on
[UQ-VLA](https://github.com/learnsyslab/uq_vla) (commit `daffea8`): `study/` applies its `patches/` to it, `q5/` runs with
`q5/src` copied over it (`q5/functional_scripts` and `q5/configs` alongside, `handoff/q5` at the root).

Licensed under the Apache License 2.0 (`LICENSE`), as are UQ-VLA and LeRobot, from which parts of `q5/src` derive.
