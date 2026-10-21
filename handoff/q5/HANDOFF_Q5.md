# Handoff: context-specific uncertainty heads (Q5)

For the collaborator who runs the Q5 experiments. Written 2026-10-12 against the paper revision `37a184c`. Everything needed is in this folder (`paper/handoff/q5/`) and the paper repository; no policy, dataset or code from the research machine is required.

| File | Purpose |
| --- | --- |
| `HANDOFF_Q5.md` | This plan |
| `kv_reference.py` | Reference implementation of the paper's two kernel heads (`KVCtxHead`, `KVDecHead`) |
| `kv_features_reference.py` | Reference extraction of all ten head inputs from a SmolVLA policy |
| `kv_fixture.npz` | Synthetic support/query features with the scores the paper's own code produces |
| `perturbations.json` | Perturbation library (23 families, levels, realism, which experiments, supervisor augmentation) |

## 1. What the paper needs

The paper shows that a single-policy kernel-variance head matches ensemble scores (VFD, GU) in task-level calibration, SAVE acquisition and supervisor referral, at a fraction of the cost (Sections 4.1-4.3). Section 4.4 then shows that on standard LIBERO these evaluations mostly test coverage *between* tasks: starting conditions within a task differ little, and much of the remaining outcome variation is execution noise, so every score ranks conditions within a task only weakly.

Q5 (introduction): *Do heads on different context embeddings respond selectively to perturbations of the inputs they read, and does this enable referral within a task?* Your experiments fill Sections 4.5 and 4.6 and support three claims:

1. **Attribution.** A head on one input's embedding responds to perturbations of that input and much less to others; a mismatch between familiar inputs shows up in the fused heads (E1, E3).
2. **Within-task ranking when variation matters.** With deliberately perturbed conditions inside a task, the heads rank the primary's failures and the supervisor's gain within a task, and combining context heads helps or not (E2).
3. **Online use.** The heads detect and attribute changes during execution, compared with VFD (E3).

Do not claim that a response to a perturbation predicts failure; E2 measures that separately. Keep null and mixed results; they are informative here.

## 2. What is fixed and what is free

| Fixed (needed for comparability with the paper) | Free (state your choice) |
| --- | --- |
| KV-Ctx and KV-Dec recipes, validated with the fixture (Section 4) | Implementation level of a perturbation (simulator vs image fallback), as long as it is reported |
| The eight context heads and their estimator (Section 4) | Exact severity values within the listed ranges |
| Policies: primary on clean LIBERO-10, supervisor with label-preserving augmentations (Section 3) | Supervisor training length and augmentation probabilities |
| Three seeds (policy triples) | Number of tasks and starting states, down to the minimums in Section 11 |
| E2 uses only label-preserving families (Section 7) | Scoring frequency in E3; ACE as an extra baseline |
| Metrics, bootstrap and output format (Sections 6-9) | Additional heads (e.g. intermediate layers), reported separately |

## 3. Policies

Train per seed `s` in {0, 1, 2} (nine policies), all fine-tuned from the SmolVLA base checkpoint with the recipe of Römer et al.'s pre-training (batch 32, their optimizer and cosine schedule, warm-up = steps / 30):

| Policy | Data | Updates | Used in |
| --- | --- | --- | --- |
| Primary `P_s` | All 379 LIBERO-10 demonstrations of `HuggingFaceVLA/libero` (revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e`, 101,469 frames), no augmentation | about 14,800 (4.68 passes, as in the paper) | E1, E2, E3: heads are fitted on it and it is executed |
| Companion `C_s` | Same as `P_s`, different seed | same | VFD baseline (symmetric VFD between `P_s` and `C_s`, 5 action samples, released implementation) |
| Supervisor `S_s` | Same demonstrations; each training sample is augmented with probability 0.5 by one label-preserving family (`supervisor_augmentation` in `perturbations.json`) | about 14,800, more allowed | E2 only |

Keep LeRobot's image transforms disabled for primaries and companions, as in the paper, so that every perturbation is new to them. Use distinct training seeds (e.g. primaries 0-2, companions 10-12, supervisors 20-22). Report for every policy its clean success per task on LIBERO's 50 standard starting states.

Requirements for the supervisor: its clean success must be within about 5 points of the primary's, and it must clearly beat the primary on the E2 perturbations it was trained on. Check this before running E2 at full scale. The supervisor may see the same paraphrase prompts that E2 tests (it stands in for a human who understands them); random perturbations (noise draws, camera offsets, patch positions) use different draws at test time.

## 4. Heads

All heads are post hoc on the frozen primary `P_s`, fitted after training and never fed outcomes.

| Head | Input (from `kv_features_reference.py`) | Width | Estimator |
| --- | --- | --- | --- |
| **KV-Ctx** (paper) | Final prefix states: mean over workspace-camera tokens, mean over wrist-camera tokens, mean over instruction tokens, plus the 8 normalized state values | 2,888 | `KVCtxHead` |
| **KV-Dec** (paper) | Action-expert final hidden state at Euler evaluation 9 of 10, initial chunk from seed 0, mean over 50 action tokens | 480 | `KVDecHead` |
| `pre_ws` | Workspace-camera tokens after vision encoder and connector, before the transformer, mean | 960 | `KVCtxHead` |
| `pre_wrist` | Same for the wrist camera | 960 | `KVCtxHead` |
| `pre_lang` | Instruction token embeddings before the transformer, mean | 960 | `KVCtxHead` |
| `pre_state` | The 8 normalized state values (the same 8 that KV-Ctx appends) | 8 | `KVCtxHead` |
| `ctx_ws` | Final prefix states, mean over workspace-camera tokens (part of KV-Ctx) | 960 | `KVCtxHead` |
| `ctx_wrist` | Same for the wrist camera (part of KV-Ctx) | 960 | `KVCtxHead` |
| `ctx_lang` | Final prefix states, mean over instruction tokens (part of KV-Ctx) | 960 | `KVCtxHead` |
| `ctx_state` | Final prefix state at the state token (last prefix position) | 960 | `KVCtxHead` |

The `pre_*` heads read each input before the transformer mixes them; the `ctx_*` heads read the same token groups after it. KV-Ctx is `ctx_ws`, `ctx_wrist`, `ctx_lang` and `pre_state` concatenated, so the context heads decompose it. `pre_lang` is constant within a task on clean inputs.

Baselines: **VFD** (symmetric, between `P_s` and `C_s`, as in the paper). ACE is optional as an entropy-based reference.

**Estimators.**
- `KVCtxHead`: standardize, PCA to at most 512 components (above 32 never more than unique rows - 1), Ledoit-Wolf shrinkage variance per component axis (whitening), deduplicate support rows, bandwidth such that the kernel is 0.5 at the median distance to the k-th nearest support neighbour with k = ceil(sqrt(N)), exact GP with ridge 1e-3, float32.
- `KVDecHead`: standardize, full Ledoit-Wolf whitening, bandwidth = median pairwise distance, exact GP with ridge 1e-2, no deduplication. This is your `KernelVariance(ridge=0.01, projection='whiten', seed=0)`.

**Support sets.**
- E1 and E2: the initial observation of each of the primary's 379 training demonstrations (as in the paper).
- E3: dense support, every 10th frame of each training demonstration (about 10,000 rows). Without it, ordinary arm motion looks unfamiliar. This is a deliberate difference from the paper's heads; state it.

**Extraction rules.** One observation per call (KV-Dec draws its fixed noise per call; your `extract_locations` draws one noise batch for the whole batch, which changes the noise of all but the first observation). Use the policy's own preprocessor. Your `extract_locations` features `vision_external` and `vision_wrist` equal `pre_ws` and `pre_wrist`, and `joint_late` equals the KV-Dec input when called one observation at a time; its `state` (state projection, 960) differs from `pre_state`, which uses the 8 values inside KV-Ctx.

**Validation (do this first).**
1. `python kv_reference.py kv_fixture.npz` must print `PASS`. If you use your own head code, run it on the fixture arrays and require max |difference| < 1e-4 for all four cases (`ctx`, `dense` with more than 512 support rows, `state8`, `dec`).
2. `kv_features_reference.py` was checked against the paper's extractor on a study checkpoint: bit-identical for KV-Ctx (which contains `ctx_ws`, `ctx_wrist`, `ctx_lang` and `pre_state`), `ctx_state` and KV-Dec. The `pre_ws`, `pre_wrist` and `pre_lang` inputs follow SmolVLA's `embed_prefix` and equal your `extract_locations` features; they have no counterpart in the paper's code. Fitting `KVCtxHead` on features extracted with it reproduced the paper's KV-Ctx scores of that checkpoint with Spearman 0.999 (max difference 0.02, from batched versus single-observation extraction). If your LeRobot fork differs, compare its output with this reference on a few observations.
3. Sanity: a support observation scores close to the floor (KV-Ctx about 1e-3, KV-Dec about 1e-2); clean held-out starting states score well below 1; `pre_lang` is identical for all clean starting states of a task.

**Calibration and combination.** For each head and seed, convert scores to percentiles of the head's scores on clean *calibration* observations: LIBERO starting states 25-49 of every task, never used as test conditions (E1 and E2), or all steps of the clean runs of the same task (E3). Percentiles make heads comparable in figures and define the combined head **COMB = maximum percentile over the eight context heads**. Rank metrics (AUROC, nAUC) are computed on raw scores; they are unchanged by the percentile transform. Optional: COMB+ including KV-Ctx and KV-Dec.

## 5. Perturbation library

`perturbations.json` holds all 23 families with levels, implementation, realism, the expected-harmless levels, whether they are label-preserving, the supervisor augmentation and the experiments they belong to. Summary (R = realistic, X = obviously out of distribution; harmless levels marked h):

| ID | Input | Perturbation | Levels | Kind | Label-preserving | E1 | E2 | E3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I1 | instruction | graded paraphrase (released prompts, P1 = original) | P2-P5 | R | yes | x | P3, P5 | x |
| I2 | instruction | object synonyms | one / all object words | R | yes | x | all | |
| I3 | instruction | "please", "could you ...?", one typo | h | R | yes | x | "please" | |
| I4 | instruction | underspecified ("put it in the basket") | 1 | R | no | x | | |
| I5 | instruction | negated, shuffled, German, empty | 4 | X | no | x | | x |
| V1 | both cameras | lighting intensity, tint, direction | x0.8 (h) ... x2.0 | R | yes | x | x0.8, x0.4 | x |
| V2 | both cameras | Gaussian noise, blur, JPEG | sigma 0.01 (h) ... 0.08 | R | yes | x | 0.01, 0.08 | |
| V3 | workspace camera | camera bumped (pose shift) | 1 cm/1 deg ... 4 cm/6 deg | R | yes | x | 2 cm/3 deg | x |
| V4 | workspace camera | target partly occluded | 25 / 50 / 75 % | R | yes | x | 50 % | x |
| V5 | scene appearance | table texture, target colour | 3 | R | approx. | x | | x |
| V6 | wrist camera | blackout, frozen, blur | 3 | R | yes | x | blackout | x |
| V7 | cameras | heavy noise, replaced, grey, swapped, mirrored | 5 | X | no | x | | x |
| S1 | scene | target displaced outside training region | 3 / 6 / 10 cm | R | no | x | | x |
| S2 | scene | extra distractor far (h) / near target | 2 | R | no | x | | x |
| S3 | scene | target rotated | 30 / 90 / 180 deg | R | no | x | | |
| S4 | scene | objects vanish, scrambled (existing events) | 3 | X | no | | | x |
| R1 | robot | initial joint offset | 0.05 / 0.1 / 0.2 rad | R | no | x | | |
| R2 | robot | proprioception bias | 1 / 3 / 6 cm | R | yes | x | 3 cm | |
| R3 | robot | push during execution | 0.05 / 0.1 rad | R | no | | | x |
| M1 | instruction x scene | other task's instruction (same scene: tasks 0 and 1) | 2 | R | no | x | | x |
| M2 | camera x camera | views from different starting states | 1 | X | no | x | | |
| M3 | state x vision | state from a different starting state | 2 | X | no | x | | |
| D1 | dynamics | invisible mass or friction change | 3 | R | no | | | x (control) |

Implement them as one library module with a uniform interface, e.g. `apply(family, level, env_or_observation, rng)` with deterministic seeding per (seed, task, starting state, family, level, execution). Existing pieces in your repository: the scene and camera events of `capture_libero_uq_rollout.py`, `ImagePatchOODConfig` and `BddlSwapOODConfig` in `lerobot/envs/configs.py`, the corruptions of `compare_epistemic_corruptions.py`, and `scripts/calibration/language_variation_prompts.json` for I1. The synonym table in the JSON is a default; correct it where a replacement changes the meaning.

## 6. E1: perturbed starting observations

**Goal.** Selectivity and dose-response of every head, without executing the policy.

**Conditions.** For each seed, task and LIBERO starting state 0-24: the clean starting observation and every E1 family at every level, applied to the starting observation (simulator families: reset with the change, render, read the observation; no action is executed). Calibration: clean starting states 25-49.

**Scores.** KV-Ctx and the eight context heads on every observation (cheap: one prefix pass). KV-Dec on all levels for starting states 0-24 if affordable, otherwise on the middle level of each family. VFD on clean and the middle level of each family for starting states 0-9 (VFD costs about 5 s per observation).

**Metrics.**
- Detection AUROC per head, family and level: perturbed versus clean observations of the same starting states, computed within each task and averaged over tasks (this removes the task-coverage effect of Section 4.4).
- Dose-response: mean percentile per head along the levels of graded families (I1, V1, V2, V3, V4, S1, R1, R2).
- Optional harm: primary success (3 executions) for the realistic non-E2 families I4, S1, S2 (near), S3, R1 and M1 at their middle level on starting states 0-9; E2 provides harm for the E2 families. This allows a response-versus-harm plot.

**Expected pattern (hypothesis, not a requirement).** `pre_*` heads respond to their own input (block-diagonal); `ctx_*` heads show cross-talk; KV-Ctx and KV-Dec respond broadly; mismatch families (M1-M3) appear mainly in the `ctx_*` heads, KV-Ctx and KV-Dec while the `pre_*` heads stay low; harmless levels raise some heads without harm.

## 7. E2: within-task referral under perturbations

**Goal.** Can heads rank conditions within a task when the variation matters, and does combining context heads help?

**Only label-preserving families** (the supervisor can be trained on them, so referring them can help). Default variants per starting state: `clean`, `I1:P3`, `I1:P5`, `I2:all`, `I3:please` (h), `V1:x0.8` (h), `V1:x0.4`, `V2:sigma0.01` (h), `V2:sigma0.08`, `V3:2cm3deg`, `V4:50`, `V6:blackout`, `R2:3cm`: thirteen per starting state, three of them expected harmless.

**Conditions.** For each seed: 10 tasks x starting states 0-9 x 13 variants = 1,300 conditions; each executed 3 times by the primary and 3 times by the supervisor with shared environment seeds per execution (520-step horizon, replanning every 25 steps, as in the paper).

**Scores.** All ten heads, COMB and VFD on the condition's starting observation.

**Metrics** (definitions in the paper, Section 3.4; code in `paper/scripts/analysis.py`):
- Within-task failure AUROC (`within_failure_auroc`).
- Within-task referral nAUC, global referral nAUC and automated fraction (`referral`); signed supervisor gain, negative gains kept.
- Cross-replicate reference for within-task referral (as in the paper).
- Per family: primary success (clean versus perturbed), supervisor success, mean gain, and detection AUROC per head.

`referral` expects task names `libero_10/0` ... `libero_10/9` and the same number of conditions per task.

## 8. E3: events during rollouts

**Goal.** Online detection and attribution of changes during execution, compared with VFD; material for one paper figure and the web page videos.

**Events.** Realistic: `I1:P5` and `M1` as a user changing the instruction mid-task, `V1` light change, `V3` camera bumped, `V4` occluder appears (60 steps), `V6` wrist camera freezes (60 steps), `S1` target moved by someone (6 cm), `S2` distractor appears near the target, `R3` push, `D1` mass x10 (invisible control). Blatant: `S4` vanish target, `V7` heavy noise, `I5` negated instruction.

**Runs.** For each seed, task and two starting states (default 0 and 1): one run per event and one clean run with the same environment seed and policy noise, which serves as the matched control for all events of that starting state. Event and clean runs must be identical up to the onset, so scoring must not consume the policy's random stream (give every score its own generator). Onset at environment step 100 unless the event needs contact; persistent unless a duration is given.

**Scores.** All ten heads (dense support) every 5 environment steps; VFD at every policy query (every 25 steps) or more often if affordable.

**Metrics.**
- Window AUROC per head and event: scores in the 50 steps after onset versus the same steps of the matched clean runs.
- Detection rate and delay at a threshold equal to the head's 99th percentile over all clean-run steps of that task and seed; report the clean runs' false-alarm rate per episode.
- Whether the score returns to clean level after transient events, and event versus clean success.

## 9. Statistics, outputs and provenance

**Statistics.** Report mean +- sample SD over the three seeds and 95 % hierarchical bootstrap intervals (2,000 draws). In E1 and E2, a cluster is a starting state with all its variants and executions: resample starting states within each task, then resample seeds, using the same draws for all heads (paired). `boot_indices` in `analysis.py` resamples single rows and must be adapted to these clusters. E3: resample (task, starting state) pairs.

**Output files** (commit to `paper/data/q5/`; one row per entity, CSV):

| File | Rows | Columns |
| --- | --- | --- |
| `e1_scores.csv` | starting observation | `seed, task, init_state_id, split (test/calibration), family, level, variant, realistic, harmless_expected`, then one column per head: `kv_ctx, kv_dec, pre_ws, pre_wrist, pre_lang, pre_state, ctx_ws, ctx_wrist, ctx_lang, ctx_state, vfd` (`ace` optional) |
| `e1_harm.csv` (optional) | condition | `seed, task, init_state_id, variant, primary_rep0..2` |
| `e2_conditions.csv` | condition | `seed, task, condition, init_state_id, family, level, variant, realistic, harmless_expected, primary, supervisor, primary_rep0..2, supervisor_rep0..2`, head columns as above |
| `e3_episodes.csv` | run | `seed, task, init_state_id, event, level, run (event/clean), onset_step, end_step, success, steps` |
| `e3_traces.csv` | run x step x head | `seed, task, init_state_id, event, run, env_step, head, score` |
| `policies.json` | policy | role, seed, data ids or hash, updates, schedule, checkpoint SHA-256, clean success per task |
| `manifest.json` | | your repository revision and LeRobot version, dataset revision, fixture check output, SHA-256 of every file above |

Use task names `libero_10/k`, condition ids `libero_10/k/start_XXX/<variant>` and variant ids `family:level` (e.g. `V2:sigma0.08`). `primary` and `supervisor` are means over the three executions, exactly as in `data/final_save_v1/conditions.csv`.

## 10. Figures, tables and paper text

Generate figures and tables with scripts in `paper/scripts/` (e.g. `q5_analysis.py`, `q5_figures.py`) from `data/q5/` only, writing to `tables/generated/` and `figures/generated/`; reuse the colour conventions of `scripts/figures.py` (KV-Ctx blue, VFD orange, KV-Dec aqua) and keep the main text within about 1.25 pages:

- **Figure (Section 4.5).** (a) E1 heatmap of within-task detection AUROC, families (mid level) x heads, grouped by input; (b) dose-response for 3-4 graded families and 3-4 heads.
- **Figure (Section 4.5).** (a) one exemplary E3 trace: frame strip at 4 times, percentiles of 4-5 heads and VFD, shaded event window; (b) E3 heatmap of window AUROC or detection rate, events x heads.
- **Table (Section 4.6).** E2 per head (ten heads, COMB, VFD): within-task failure AUROC, within-task nAUC, automated fraction, mean +- SD and 95 % interval; footnote with the cross-replicate reference and per-family supervisor gain.
- **Appendix.** Perturbation library table, protocol details, per-family E2 results, supervisor augmentation, and the policies' clean success.

Paper places to update (the yellow `\revise{}` text marks statements that depend on your results; remove the mark once a statement is supported or rewrite it):
- `sections/methods/kernel_heads.tex`, paragraph "Context-specific heads" and its draft note; `tables/selected_heads.tex` (row "Context heads"); `figures/architecture.tex` (context-head node).
- `sections/results.tex`, Sections 4.5 (`sec:results-rollout`) and 4.6 (`sec:results-perturbed`), replacing the placeholder figure and draft notes.
- `sections/discussion.tex`, paragraph "Context-specific uncertainty".
- Introduction (Q5, third contribution) and related work: check every `\revise{}` statement.
- A new appendix file for the protocol and the library table, included from `main.tex`.

Keep the provenance separate from the matched SAVE study: these are different policies. Write one sentence per line and run `make check` and `make` before pushing (see `CONTRIBUTING.md`).

## 11. Scale and compute

Rollouts take about 10 s each on one RTX 3090 in the paper's evaluator; training one policy on LIBERO-10 about 2.3 h (4,000 updates take 37 min).

| Part | Default (3 seeds) | Minimum |
| --- | --- | --- |
| Training | 9 policies, about 21 GPU-h | same |
| E1 scoring | 10 tasks x (25 states x about 68 variants + 25 calibration states), KV-Dec and VFD on subsets (Section 6), about 15 GPU-h | KV-Dec on middle levels only |
| E1 harm (optional) | 6 families x 10 tasks x 10 states x 3 executions, about 15 GPU-h | skip |
| E2 | 3 x 1,300 conditions x 3 executions x 2 policies = 23,400 rollouts, about 65 GPU-h | 5 tasks, or 9 variants (keep two harmless) |
| E3 | 3 x 10 tasks x 2 states x (13 events + 1 clean run) = 840 runs with per-step scoring, about 25 GPU-h (VFD dominates) | VFD for one seed only |

Check after the first seed of E2 that the supervisor gains on the perturbed variants and that within-task oracle headroom is positive in most tasks; otherwise strengthen the supervisor's augmentation before running the other seeds.

## 12. Checklist

- [ ] Fixture `PASS`; feature reference matches your fork on a few observations.
- [ ] Nine policies trained; clean success per task reported; supervisor meets the requirements of Section 3.
- [ ] Perturbation library implemented with deterministic seeding; a sample image or instruction per family and level saved for the appendix.
- [ ] E1, E2, E3 data in `paper/data/q5/` with `manifest.json`.
- [ ] Analysis and figure scripts in `paper/scripts/`, generated tables and figures, text in the places of Section 10.
- [ ] Videos for the web page (not in the paper), linked from the paper.
