#!/usr/bin/env python
"""Q5 / E3: events during rollouts (handoff/q5, Section 8), one primary policy and its companion.

For every task and starting state (default 0 and 1) one clean run and one run per event, all with the
same environment seed and policy noise, all in one process, so event and clean run are identical up to the onset (every
score draws from its own generator and never touches the policy's random stream). Onset at environment
step 100; events are persistent unless a duration is given.

Events (handoff names):
  realistic  I1 paraphrase P5 · M1 other task's instruction · V1 light x0.4 · V3 camera bumped 2 cm / 3 deg ·
             V4 occluder over the target for 60 steps · V6 wrist camera frozen for 60 steps · S1 target moved
             6 cm · S2 distractor placed next to the target · R3 arm joints bumped by 0.1 rad once ·
             D1 movable objects 10x heavier (invisible control)
  blatant    S4 target vanishes · V7 heavy image noise (sigma 0.35) · I5 negated instruction
Scores every --stride environment steps: the inputs of all ten heads (kv_features_reference.head_inputs,
one observation per call) and VFD between primary and companion (paper_vfd, 5 action samples).

    python functional_scripts/q5_e3_capture.py --primary P --companion C --out DIR --shard 0/4

Writes DIR/task{t}_state{s}_{event}.npz: steps (scored env steps), one array per head input, vfd,
event_active per env step, success, n_steps, meta (JSON); with --frames also both camera images per step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["handoff/q5", "functional_scripts"]
import q5_e1  # noqa: E402
from q5_e1 import STATE, WRIST, WS, Policy, Scene, instruction_variant, observation_variant, rng_for  # noqa: E402

ONSET, HORIZON = 100, 520
EVENTS = {  # name -> (family, level, kind, duration in steps or None = persistent)
    "I1": ("I1", "P5", "instruction", None),
    "M1": ("M1", None, "instruction", None),
    "I5": ("I5", "negated", "instruction", None),
    "V1": ("V1", "x0.4", "simulator", None),
    "V3": ("V3", "2cm3deg", "simulator", None),
    "V4": ("V4", "75", "observation", 60),
    "V6": ("V6", "frozen", "observation", 60),
    "S1": ("S1", "6cm", "simulator", None),
    "S2": ("S2", "near", "simulator", None),
    "R3": ("R1", "0.1rad", "simulator", None),
    "D1": ("D1", "mass10", "simulator", None),
    "S4": ("S4", "vanish_target", "simulator", None),
    "V7": ("V7", "noise0.35", "observation", None),
}


def m1_text(task: int) -> str:
    """Same-scene instruction where one exists (tasks 0 and 1), otherwise one naming absent objects."""
    level = "same_scene" if task in q5_e1.SAME_SCENE else "other_scene"
    return instruction_variant(task, "M1", level, None)


def apply_simulator(scene: Scene, family: str, level: str, rng) -> bool:
    m, d = scene.sim.model, scene.sim.data
    if family == "D1":
        for name in scene.objects():
            body = int(m.jnt_bodyid[m.joint_name2id(f"{name}_joint0")])
            m.body_mass[body] *= 10
            m.body_inertia[body] *= 10
        scene.sim.forward()
        return True
    if family == "S4":
        target = scene.target()
        if target is None:
            return False
        body = int(m.jnt_bodyid[m.joint_name2id(f"{target}_joint0")])
        for g in range(m.ngeom):  # parked objects must not collide with anything on the way down
            if m.geom_bodyid[g] == body or m.body_parentid[m.geom_bodyid[g]] == body:
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        address = scene.objects()[target]
        d.qpos[address:address + 3] = (0.0, 0.0, -5.0)
        joint = m.joint_name2id(f"{target}_joint0")
        d.qvel[int(m.jnt_dofadr[joint]):int(m.jnt_dofadr[joint]) + 6] = 0
        scene.sim.forward()
        return True
    return scene.apply(family, level, rng)


@torch.no_grad()
def run(scene: Scene, policy: Policy, task: int, state: int, event: str, stride: int, frames: bool) -> dict:
    from kv_features_reference import head_inputs
    from lerobot.uncertainty.uncertainty_scoring.paper_vfd import paper_vfd

    seed = q5_e1.SEED0 + 100 * task + state
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    observation = scene.reset(state, seed)
    policy.policy.reset()
    instruction = scene.instruction
    family, level, kind, duration = EVENTS[event] if event != "clean" else (None, None, None, None)
    rng = rng_for(task, state, "E3", event)
    if family in ("V4", "S1", "S4") and scene.target() is None:
        scene.restore()
        return {"skipped": True}  # these events need a named movable target
    stop = ONSET + duration if duration else HORIZON + 1
    scored = {"steps": [], "vfd": []}
    head_rows: dict[str, list] = {}
    active, ws_frames, wrist_frames = [], [], []
    frozen, applied, success, step = None, False, False, 0
    try:
        for step in range(HORIZON):
            is_active = family is not None and ONSET <= step < stop
            if is_active and not applied:
                applied = True
                if kind == "instruction":
                    instruction = m1_text(task) if family == "M1" else instruction_variant(task, family, level, rng)
                elif kind == "simulator":
                    if not apply_simulator(scene, family, level, rng):
                        return {"skipped": True}
                    observation = scene.observe()
            batch = policy.env_batch(observation, instruction)
            if is_active and kind == "observation":
                if family == "V6":
                    if frozen is None:
                        frozen = batch[WRIST].clone()
                    batch[WRIST] = frozen.clone()
                else:
                    ctx = {"target_box": scene.target_box(256, 256)} if family == "V4" else {}
                    batch = observation_variant(batch, family, level, rng, ctx)
            active.append(is_active)
            if frames:
                ws_frames.append((batch[WS][0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
                wrist_frames.append((batch[WRIST][0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
            if step % stride == 0:
                scored["steps"].append(step)
                processed = policy.pre(dict(batch))  # the scores get their own copy of the input
                for k, v in head_inputs(policy.policy, processed).items():
                    head_rows.setdefault(k, []).append(v.reshape(-1))
                generator = torch.Generator(device=policy.device).manual_seed(seed * 1000 + step)
                scored["vfd"].append(float(paper_vfd(policy.adapter, policy.partner, processed,
                                                     num_action_samples=5, generator=generator)[0]))
            action = policy.policy.select_action(policy.pre(dict(batch)))
            action = policy.env_post({"action": policy.post(action)})["action"]
            action = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
            observation, _, terminated, truncated, info = scene.env.step(np.asarray(action).reshape(-1))
            if info.get("is_success"):
                success = True
                break
            if terminated or truncated:
                break
    finally:
        scene.restore()
    out = {"steps": np.asarray(scored["steps"]), "vfd": np.asarray(scored["vfd"], np.float32),
           "event_active": np.asarray(active), "success": success, "n_steps": step + 1,
           **{k: np.asarray(v, np.float32) for k, v in head_rows.items()}}
    if frames:
        out["external"], out["wrist"] = np.asarray(ws_frames), np.asarray(wrist_frames)
    out["meta"] = json.dumps({"task": task, "state": state, "event": event, "family": family, "level": level,
                              "onset": ONSET if family else None, "duration": duration, "seed": seed,
                              "instruction": instruction, "original_instruction": scene.instruction,
                              "success": success, "stride": stride,
                              "members": [str(m) for m in q5_e1.MEMBERS]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--states", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--events", nargs="+", default=["clean", *EVENTS])
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--frames", action="store_true", help="also store both camera images per step")
    parser.add_argument("--calib-states", type=int, nargs="+", default=[2, 3, 4],
                        help="starting states recorded clean only, for percentiles and alarm thresholds")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    q5_e1.MEMBERS = (args.primary, args.companion)
    policy = Policy(args.device, partner=True)
    args.out.mkdir(parents=True, exist_ok=True)
    # A shard takes whole groups (task, starting state): the clean run and all its events run one after another
    # in this process on this GPU, so event and clean run stay bit-identical up to the onset. Runs of the same
    # group on different GPUs (A100 vs RTX 3090) drifted apart before the onset in about 40 % of the pairs.
    groups = [(t, s, list(args.events)) for t in args.tasks for s in args.states]
    groups += [(t, s, ["clean"]) for t in args.tasks for s in args.calib_states]
    i, n = map(int, args.shard.split("/"))
    jobs = [(t, s, e) for t, s, events in groups[i::n] for e in events]
    scenes: dict[int, Scene] = {}
    for task, state, event in jobs:
        path = args.out / f"task{task}_state{state}_{event}.npz"
        if path.exists():
            continue
        started = time.time()
        scene = scenes.setdefault(task, Scene(task))
        result = run(scene, policy, task, state, event, args.stride, args.frames)
        if result.get("skipped"):
            print(f"task {task} state {state} {event}: not applicable in this scene", flush=True)
            continue
        np.savez_compressed(path, **result)
        print(f"task {task} state {state} {event}: success={result['success']} steps={result['n_steps']} "
              f"({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
