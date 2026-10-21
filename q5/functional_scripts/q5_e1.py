#!/usr/bin/env python
"""Q5 / E1: every context head on perturbed starting observations (handoff/q5, Section 6).

Pilot scope (one policy, not the handoff's three primaries trained on all of LIBERO-10):
  policy    pretraining lineage seed_1 (LIBERO-10 tasks 0-2 in its data), VFD with seed_2 as in
            the paper; the same policy as the E3 captures
  tasks     LIBERO-10 tasks 0-2 (the only LIBERO-10 tasks the policy was trained on)
  support   the initial observation of every training demonstration of tasks 0-2 (paper recipe)
  test      starting states 0-24: the clean observation and every E1 family at every level
  calib     starting states 25-49, clean only (percentiles, COMB)
  VFD       all variants of starting states 0-9
  harm      the policy executed (3 seeds) with the middle level of each family applied for the
            whole episode, starting states 0-9

Features come from the handoff's kv_features_reference.head_inputs (one observation per call), the
heads from kv_reference (KVCtxHead for KV-Ctx and the eight context heads, KVDecHead for KV-Dec).

Stages:
  support [--stride n]          head inputs of the training demonstrations: first frames (paper
                                recipe) or every n-th frame (dense, as in E3)
  features --shard i/n          head inputs (+ VFD) of every variant, one file per task and state
  sheet                         contact sheet of every variant (starting state 0), for checking
  harm --shard i/n              rollouts of the middle levels
  analyze                       e1_scores.csv, e1_harm.csv, e1_summary.json and e1_table.html

Perturbation application (perturbations.json conventions):
  instruction  the task text the policy reads is replaced
  observation  applied to the env-processed batch: images float in [0, 1], (1, C, H, W), upright;
               observation.state = eef position (3), axis-angle (3), gripper (2), before normalization
  simulator    the MuJoCo scene/model is changed after the reset, the observation re-rendered;
               model edits are undone after each variant
Approximations, stated in the HTML: V5 target "hue" is a material tint (textures are not re-uploaded);
V6 "frozen frame" is not defined for a starting observation and is left to E3; S2 moves an existing
non-target object; M3 takes the state of another task's / starting state's demonstration first frame.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("outputs/q5/e1")
HANDOFF = Path("handoff/q5")
sys.path[:0] = [str(HANDOFF), "functional_scripts"]

MEMBERS = (Path("outputs/pretrain/libero_pretrain_leak3/seed_1/checkpoints/030000/pretrained_model"),
           Path("outputs/pretrain/libero_pretrain_leak3/seed_2/checkpoints/030000/pretrained_model"))
DATA_ROOT = Path("storage/huggingface/lerobot/HuggingFaceVLA/libero")
TASKS = (0, 1, 2)
TEST, CALIB = range(0, 25), range(25, 50)
VFD_STATES = range(0, 10)
HARM_STATES, HARM_REPS = range(0, 10), 3
SEED0 = 9000
WS, WRIST, STATE = "observation.images.image", "observation.images.image2", "observation.state"
HEAD_INPUTS = ("kv_ctx", "kv_dec", "pre_ws", "pre_wrist", "pre_lang", "pre_state",
               "ctx_ws", "ctx_wrist", "ctx_lang", "ctx_state")
CONTEXT = ("pre_ws", "pre_wrist", "pre_lang", "pre_state", "ctx_ws", "ctx_wrist", "ctx_lang", "ctx_state")
PROMPTS = Path("scripts/calibration/language_variation_prompts.json")

# Hand-written variants for the LIBERO-10 tasks (I4 pronouns, I5 German), as the handoff leaves them free.
PRONOUNS = {0: "put both of them in the basket", 1: "put both of them in the basket",
            2: "turn it on and put it on it", 3: "put it in the bottom drawer of it and close it",
            4: "put it on the left one and put the other one on the right one",
            5: "pick it up and place it in the back compartment of it",
            6: "put it on it and put the other one to the right of it", 7: "put both of them in the basket",
            8: "put both of them on it", 9: "put it in there and close it"}
GERMAN = {0: "lege die Buchstabensuppe und die Tomatensauce in den Korb",
          1: "lege die Frischkäsepackung und die Butter in den Korb",
          2: "schalte den Herd ein und stelle die Mokkakanne darauf",
          3: "stelle die schwarze Schüssel in die untere Schublade des Schranks und schließe sie",
          4: "stelle die weiße Tasse auf den linken Teller und die gelb-weiße Tasse auf den rechten Teller",
          5: "nimm das Buch und stelle es in das hintere Fach des Organizers",
          6: "stelle die weiße Tasse auf den Teller und den Schokoladenpudding rechts neben den Teller",
          7: "lege die Buchstabensuppe und die Frischkäsepackung in den Korb",
          8: "stelle beide Mokkakannen auf den Herd",
          9: "stelle die gelb-weiße Tasse in die Mikrowelle und schließe sie"}
# M1: only tasks 0 and 1 share a scene (LIVING_ROOM_SCENE2); "other scene" takes an instruction whose
# objects are absent from the task's scene.
SAME_SCENE = {0: 1, 1: 0}
OTHER_SCENE = {0: 3, 1: 9, 2: 5, 3: 6, 4: 8, 5: 0, 6: 3, 7: 9, 8: 4, 9: 5}

# Family -> (input, kind, levels, realistic, harmless levels, harm level).
# kind: instruction | observation | simulator | mismatch.
FAMILIES = {
    "I1": ("instruction", "instruction", ["P2", "P3", "P4", "P5"], True, [], "P3"),
    "I2": ("instruction", "instruction", ["one", "all"], True, [], "all"),
    "I3": ("instruction", "instruction", ["please", "could_you", "typo"], True, ["please", "could_you", "typo"], "typo"),
    "I4": ("instruction", "instruction", ["pronouns"], True, [], "pronouns"),
    "I5": ("instruction", "instruction", ["negated", "shuffled", "german", "empty"], False, [], "german"),
    "V1": ("vision", "simulator", ["x0.8", "x0.6", "x0.4", "x1.4", "x2.0", "warm", "cool", "dir45"], True, ["x0.8"], "x0.4"),
    "V2": ("vision", "observation", ["sigma0.01", "sigma0.03", "sigma0.08", "blur3", "blur7", "blur11",
                                     "jpeg60", "jpeg25", "jpeg10"], True, ["sigma0.01", "jpeg60"], "sigma0.08"),
    "V3": ("workspace camera", "simulator", ["1cm1deg", "2cm3deg", "4cm6deg"], True, [], "2cm3deg"),
    "V4": ("workspace camera", "observation", ["25", "50", "75"], True, [], "50"),
    "V5": ("scene appearance", "simulator", ["table", "tint30", "tint90"], True, [], "table"),
    "V6": ("wrist camera", "observation", ["blackout", "blur15"], True, [], "blackout"),
    "V7": ("vision", "observation", ["noise0.35", "replaced", "grey", "swapped", "mirrored"], False, [], "grey"),
    "S1": ("scene", "simulator", ["3cm", "6cm", "10cm"], True, [], "6cm"),
    "S2": ("scene", "simulator", ["far", "near"], True, ["far"], "near"),
    "S3": ("scene", "simulator", ["30deg", "90deg", "180deg"], True, [], "90deg"),
    "R1": ("robot state", "simulator", ["0.05rad", "0.1rad", "0.2rad"], True, [], "0.1rad"),
    "R2": ("robot state", "observation", ["1cm", "3cm", "6cm"], True, [], "3cm"),
    "M1": ("instruction x scene", "instruction", ["same_scene", "other_scene"], True, [], "same_scene"),
    "M2": ("camera x camera", "mismatch", ["same_task"], False, [], None),
    "M3": ("state x vision", "mismatch", ["same_task", "other_task"], False, [], None),
}
VARIANTS = ["clean"] + [f"{f}:{lv}" for f, spec in FAMILIES.items() for lv in spec[2]]


def rng_for(*key) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(repr(key).encode()).hexdigest()[:12], 16))


def task_instruction(task: int) -> str:
    from lerobot.envs.libero import get_task_instruction
    return get_task_instruction("libero_10", task)


# ---------------------------------------------------------------------------------------------
# Instruction families
def instruction_variant(task: int, family: str, level: str, rng) -> str | None:
    text = task_instruction(task)
    if family == "I1":
        return json.loads(PROMPTS.read_text())[str(task)][int(level[1]) - 1]
    if family == "I2":
        synonyms = json.loads((HANDOFF / "perturbations.json").read_text())["families"][1]["synonyms"][str(task)]
        # Longest phrases first, so "yellow and white mug" wins over "white mug".
        out = text
        for word in sorted(synonyms, key=len, reverse=True):
            if word in out:
                out = out.replace(word, synonyms[word])
                if level == "one":
                    break
        return out
    if family == "I3":
        if level == "please":
            return "please " + text
        if level == "could_you":
            return "Could you " + text + "?"
        words = text.split()
        nouns = {w for phrase in json.loads((HANDOFF / "perturbations.json").read_text())["families"][1]
                 ["synonyms"][str(task)] for w in phrase.split()}
        i = next(i for i, w in enumerate(words) if len(w) > 3 and w not in nouns)
        w = words[i]
        words[i] = w[:1] + w[2] + w[1] + w[3:]  # swap the second and third letter
        return " ".join(words)
    if family == "I4":
        return PRONOUNS[task]
    if family == "I5":
        if level == "negated":
            return "do not " + text
        if level == "shuffled":
            words = text.split()
            rng.shuffle(words)
            return " ".join(words)
        if level == "german":
            return GERMAN[task]
        return ""
    if family == "M1":
        if level == "same_scene":
            return task_instruction(SAME_SCENE[task]) if task in SAME_SCENE else None
        return task_instruction(OTHER_SCENE[task])
    raise KeyError(family)


# ---------------------------------------------------------------------------------------------
# Observation families (on the env-processed batch)
def _blur(image: torch.Tensor, size: int, motion: bool) -> torch.Tensor:
    c = image.shape[-3]
    if motion:
        kernel = torch.zeros(size, size)
        kernel[size // 2] = 1.0 / size
    else:
        x = torch.arange(size) - size // 2
        g = torch.exp(-x.float() ** 2 / (2 * (size / 4) ** 2))
        kernel = torch.outer(g, g)
        kernel /= kernel.sum()
    kernel = kernel.to(image)[None, None].repeat(c, 1, 1, 1)
    return torch.nn.functional.conv2d(torch.nn.functional.pad(image, [size // 2] * 4, mode="replicate"),
                                      kernel, groups=c)


def _jpeg(image: torch.Tensor, quality: int) -> torch.Tensor:
    from PIL import Image
    out = []
    for frame in image:
        array = (frame.permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
        decoded = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"), np.float32) / 255
        out.append(torch.from_numpy(decoded).permute(2, 0, 1))
    return torch.stack(out).to(image)


def observation_variant(batch: dict, family: str, level: str, rng, ctx: dict) -> dict:
    """Return a perturbed copy of an env-processed single-observation batch."""
    b = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    both = (WS, WRIST)
    if family == "V2":
        for key in both:
            if level.startswith("sigma"):
                noise = torch.from_numpy(rng.normal(0, float(level[5:]), b[key].shape)).to(b[key])
                b[key] = (b[key] + noise).clamp(0, 1)
            elif level.startswith("blur"):
                b[key] = _blur(b[key], int(level[4:]), motion=True)
            else:
                b[key] = _jpeg(b[key], int(level[4:]))
    elif family == "V4":
        if ctx.get("target_box") is None:
            return None  # no named movable target in this scene
        r0, r1, c0, c1 = ctx["target_box"]
        width = int(round((c1 - c0) * int(level) / 100))
        if width > 0 and r1 > r0:
            image = b[WS]
            ring = torch.cat([image[..., max(r0 - 4, 0):r0, c0:c1].flatten(-2),
                              image[..., r1:r1 + 4, c0:c1].flatten(-2)], -1)
            colour = ring.median(dim=-1).values if ring.shape[-1] else image.mean((-2, -1))
            image[..., r0:r1, c0:c0 + width] = colour[..., None, None]
    elif family == "V6":
        b[WRIST] = torch.zeros_like(b[WRIST]) if level == "blackout" else _blur(b[WRIST], 15, motion=False)
    elif family == "V7":
        if level == "noise0.35":
            for key in both:
                b[key] = (b[key] + torch.from_numpy(rng.normal(0, 0.35, b[key].shape)).to(b[key])).clamp(0, 1)
        elif level == "replaced":
            for key in both:
                b[key] = torch.from_numpy(rng.random(b[key].shape)).to(b[key])
        elif level == "grey":
            for key in both:
                b[key] = torch.full_like(b[key], 0.5)
        elif level == "swapped":
            b[WS], b[WRIST] = batch[WRIST].clone(), batch[WS].clone()
        else:
            for key in both:
                b[key] = b[key].flip(-1)
    elif family == "R2":
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        b[STATE][..., :3] += torch.from_numpy(direction * float(level[:-2]) / 100).to(b[STATE])
    elif family == "M2":
        b[WRIST] = ctx["other_state_batch"][WRIST].clone()
    elif family == "M3":
        b[STATE] = ctx["other_task_state" if level == "other_task" else "other_state_state"].clone().to(b[STATE])
    else:
        raise KeyError(family)
    return b


# ---------------------------------------------------------------------------------------------
# Simulator families
def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _axis_quat(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[math.cos(angle / 2), math.sin(angle / 2) * axis]


class Scene:
    """Handle on one LIBERO env: reset to a starting state, apply and undo simulator edits."""

    MODEL_FIELDS = ("light_diffuse", "light_specular", "light_dir", "light_pos", "cam_pos", "cam_quat",
                    "geom_matid", "mat_rgba", "body_mass", "body_inertia", "geom_contype", "geom_conaffinity",
                    "geom_friction")

    def __init__(self, task: int):
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.libero import LiberoEnv, _get_suite
        cfg = LiberoEnvConfig(task="libero_10", task_ids=[task])
        self.cfg, self.task = cfg, task
        self.env = LiberoEnv(task_suite=_get_suite("libero_10"), task_id=task, task_suite_name="libero_10",
                             obs_type=cfg.obs_type, camera_name=cfg.camera_name, init_states=cfg.init_states,
                             observation_width=256, observation_height=256, episode_index=0,
                             control_mode=cfg.control_mode)
        self.instruction = self.env.task_description

    @property
    def sim(self):
        return self.env._env.env.sim

    def reset(self, state: int, seed: int):
        self.env._init_state_id = state
        self.env.task_description = self.instruction
        observation, _ = self.env.reset(seed=seed)
        m = self.sim.model
        self._model = {f: getattr(m, f).copy() for f in self.MODEL_FIELDS}
        self._headlight = m.vis.headlight.diffuse.copy()
        self._qpos, self._qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
        return observation

    def restore(self):
        m = self.sim.model
        for f, value in self._model.items():
            getattr(m, f)[:] = value
        m.vis.headlight.diffuse[:] = self._headlight
        self.sim.data.qpos[:] = self._qpos
        self.sim.data.qvel[:] = self._qvel
        self.sim.forward()
        self.env.task_description = self.instruction

    def observe(self):
        inner = self.env._env.env
        return self.env._format_raw_obs(inner._get_observations(force_update=True))

    # -- scene queries ---------------------------------------------------------------------
    def objects(self) -> dict[str, int]:
        from capture_libero_uq_rollout import free_joint_objects
        return free_joint_objects(self.sim)

    # How LIBERO-10's instructions name objects whose asset names differ.
    ALIASES = {"porcelain_mug": "white mug", "white_yellow_mug": "yellow and white mug",
               "red_coffee_mug": "red mug", "akita_black_bowl": "black bowl", "black_book": "book",
               "chefmate_8_frypan": "frying pan"}

    def named(self) -> list[str]:
        """Movable objects the instruction names, in the order it names them. An object is named by its alias,
        its full name, or a trailing part of at least two words ("akita_black_bowl" -> "black bowl"); a match
        lying inside another object's longer match does not count ("white mug" in "yellow and white mug").
        Objects sharing a last word ("plate_1", "plate_2") are named by that word unless one of them was
        already named more specifically (the red mug is not named by "white mug"). Plurals count."""
        text = " " + self.instruction.lower().replace(",", " ") + " "
        stems = {name: name.rsplit("_", 1)[0] for name in self.objects()}

        def spans(phrases):
            out = []
            for ph in phrases:
                for end in ("", "s"):
                    start = text.find(f" {ph}{end} ")
                    if start >= 0:
                        out.append((start + 1, start + 1 + len(ph) + len(end)))
            return out

        specific = {}
        for name, stem in stems.items():
            words = stem.split("_")
            phrases = ([self.ALIASES[stem]] if stem in self.ALIASES else []) + \
                      [" ".join(words[k:]) for k in range(len(words) - 1)]
            if spans(phrases):
                specific[name] = min(spans(phrases), key=lambda sp: (sp[0], -(sp[1] - sp[0])))
        # Drop matches nested inside a longer match of another object.
        named = {n: sp for n, sp in specific.items()
                 if not any(o != n and o_sp[0] <= sp[0] and sp[1] <= o_sp[1] and o_sp != sp
                            for o, o_sp in specific.items())}
        for name, stem in stems.items():
            if name in named:
                continue
            word = stem.split("_")[-1]
            same = [n for n, st in stems.items() if st.split("_")[-1] == word]
            if not any(n in specific for n in same) and spans([word]):
                named[name] = min(spans([word]))
        return [name for name, _ in sorted(named.items(), key=lambda kv: kv[1])]

    def target(self) -> str | None:
        named = self.named()
        movable = [t for t in named if "basket" not in t]
        return (movable or named or [None])[0]

    def distractors(self) -> list[str]:
        named = set(self.named())
        return [o for o in self.on_table() if o not in named]

    def position(self, name: str) -> np.ndarray:
        return self.sim.data.qpos[self.objects()[name]:self.objects()[name] + 3].copy()

    def on_table(self) -> list[str]:
        """Objects resting on the work surface (some scenes park objects out of view, higher up)."""
        heights = {o: self.position(o)[2] for o in self.objects()}
        surface = min(heights.values())
        return [o for o, z in heights.items() if z < surface + 0.12]

    def _place(self, name: str, xy) -> None:
        address = self.objects()[name]
        self.sim.data.qpos[address:address + 2] = xy
        joint = self.sim.model.joint_name2id(f"{name}_joint0")
        vel = int(self.sim.model.jnt_dofadr[joint])
        self.sim.data.qvel[vel:vel + 6] = 0

    def _free_spot(self, name: str, centre, radius: tuple[float, float], rng, clearance=0.07):
        others = [self.position(o)[:2] for o in self.on_table() if o != name]
        for _ in range(400):
            angle, r = rng.uniform(0, 2 * np.pi), rng.uniform(*radius)
            xy = np.asarray(centre[:2]) + r * np.array([np.cos(angle), np.sin(angle)])
            if not (-0.30 <= xy[0] <= 0.22 and -0.32 <= xy[1] <= 0.32):
                continue
            if all(np.linalg.norm(xy - o) > clearance for o in others):
                return xy
        return None

    def target_box(self, height: int, width: int) -> tuple[int, int, int, int]:
        """Pixel box (rows, cols) of the target in the upright workspace image."""
        from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
        name = self.target()
        if name is None:
            return None
        m, d = self.sim.model, self.sim.data
        body = m.body_name2id(f"{name}_main") if f"{name}_main" in [m.body_id2name(i) for i in range(m.nbody)] \
            else int(m.jnt_bodyid[m.joint_name2id(f"{name}_joint0")])
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body or m.body_parentid[m.geom_bodyid[g]] == body]
        corners = []
        for g in geoms:
            r = float(m.geom_rbound[g]) or 0.02
            for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3):
                corners.append(d.geom_xpos[g] + 0.6 * r * s)
        matrix = get_camera_transform_matrix(self.sim, "agentview", height, width)
        pixels = project_points_from_world_to_camera(np.array(corners), matrix, height, width)
        rows, cols = pixels[:, 0], pixels[:, 1]
        # robosuite's convention is the vertically flipped render; the upright image also mirrors columns.
        r0, r1 = int(np.clip(rows.min(), 0, height)), int(np.clip(rows.max() + 1, 0, height))
        c0, c1 = int(np.clip(width - 1 - cols.max(), 0, width)), int(np.clip(width - cols.min(), 0, width))
        return r0, r1, c0, c1

    # -- edits -------------------------------------------------------------------------------
    def apply(self, family: str, level: str, rng) -> bool:
        """Edit the scene; returns False if the variant does not exist in this scene."""
        m, d = self.sim.model, self.sim.data
        if family == "V1":
            if level.startswith("x"):
                f = float(level[1:])
                m.light_diffuse[:] *= f
                m.light_specular[:] *= f
                m.vis.headlight.diffuse[:] *= f
            elif level in ("warm", "cool"):
                tint = np.array((1.0, 0.85, 0.7) if level == "warm" else (0.75, 0.85, 1.0))
                m.light_diffuse[:] *= tint
                m.vis.headlight.diffuse[:] *= tint
            else:
                q = _axis_quat([0, 0, 1], math.radians(45))
                rot = np.array([[1 - 2 * (q[2] ** 2 + q[3] ** 2), -2 * q[0] * q[3], 0],
                                [2 * q[0] * q[3], 1 - 2 * (q[1] ** 2 + q[3] ** 2), 0], [0, 0, 1]])
                m.light_dir[:] = m.light_dir @ rot.T
                m.light_pos[:] = m.light_pos @ rot.T
        elif family == "V3":
            cm, deg = {"1cm1deg": (1, 1), "2cm3deg": (2, 3), "4cm6deg": (4, 6)}[level]
            camera = m.camera_name2id("agentview")
            shift, axis = rng.normal(size=3), rng.normal(size=3)
            m.cam_pos[camera] += shift / np.linalg.norm(shift) * cm / 100
            m.cam_quat[camera] = _quat_mul(_axis_quat(axis, math.radians(deg)), m.cam_quat[camera])
        elif family == "V5":
            import mujoco
            raw = m._model if hasattr(m, "_model") else m
            names = [mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_MATERIAL, i) or "" for i in range(m.nmat)]
            if level == "table":
                tables = [i for i, n in enumerate(names) if "table" in n and "leg" not in n]
                replacement = names.index("walls_mat") if "walls_mat" in names else names.index("floorplane")
                for g in range(m.ngeom):
                    if m.geom_matid[g] in tables:
                        m.geom_matid[g] = replacement
            else:
                target = self.target()
                hue = {"tint30": 30, "tint90": 90}[level]
                tint = colorsys.hsv_to_rgb(hue / 360, 0.6, 1.0)
                body = int(m.jnt_bodyid[m.joint_name2id(f"{target}_joint0")])
                mats = {int(m.geom_matid[g]) for g in range(m.ngeom)
                        if m.geom_matid[g] >= 0 and (m.geom_bodyid[g] == body or m.body_parentid[m.geom_bodyid[g]] == body)}
                for i in mats:
                    m.mat_rgba[i, :3] *= tint
        elif family in ("S1", "S3"):
            target = self.target()
            if target is None:
                return False
            if family == "S1":
                start = self.position(target)
                xy = self._free_spot(target, start, (float(level[:-2]) / 100,) * 2, rng)
                if xy is None:
                    return False
                self._place(target, xy)
            else:
                address = self.objects()[target]
                q = d.qpos[address + 3:address + 7].copy()
                d.qpos[address + 3:address + 7] = _quat_mul(_axis_quat([0, 0, 1], math.radians(float(level[:-3]))), q)
        elif family == "S2":
            target = self.target()
            distractors = self.distractors()
            if target is None or not distractors:
                return False
            name = distractors[int(rng.integers(len(distractors)))]
            xy = self._free_spot(name, self.position(target),
                                 (0.22, 0.32) if level == "far" else (0.07, 0.09), rng, clearance=0.06)
            if xy is None:
                return False
            self._place(name, xy)
        elif family == "R1":
            robot = self.env._env.env.robots[0]
            signs = rng.choice([-1.0, 1.0], size=len(robot._ref_joint_pos_indexes))
            d.qpos[robot._ref_joint_pos_indexes] += signs * float(level[:-3])
        else:
            raise KeyError(family)
        self.sim.forward()
        return True


# ---------------------------------------------------------------------------------------------
class Policy:
    def __init__(self, device: str, partner: bool):
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies.factory import make_flow_matching_adapter_from_policy, make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.device = torch.device(device)
        self.policy = SmolVLAPolicy.from_pretrained(MEMBERS[0]).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(MEMBERS[0]),
            preprocessor_overrides={"device_processor": {"device": device}})
        self.env_pre, self.env_post = make_env_pre_post_processors(
            env_cfg=LiberoEnvConfig(task="libero_10", task_ids=[0]), policy_cfg=self.policy.config)
        self.adapter = make_flow_matching_adapter_from_policy(self.policy)
        self.partner = None
        if partner:
            other = SmolVLAPolicy.from_pretrained(MEMBERS[1]).to(self.device).eval()
            self.partner = make_flow_matching_adapter_from_policy(other)

    def env_batch(self, observation: dict, instruction: str) -> dict:
        from lerobot.envs.utils import preprocess_observation
        batch = preprocess_observation(observation)
        batch["task"] = [instruction]
        return self.env_pre(batch)

    @torch.no_grad()
    def features(self, env_batch: dict, vfd_seed: int | None):
        from kv_features_reference import head_inputs
        from lerobot.uncertainty.uncertainty_scoring.paper_vfd import paper_vfd
        batch = self.pre(dict(env_batch))
        out = {k: v.reshape(-1) for k, v in head_inputs(self.policy, batch).items()}
        vfd = float("nan")
        if vfd_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(vfd_seed)
            vfd = float(paper_vfd(self.adapter, self.partner, batch, num_action_samples=5, generator=generator)[0])
        return out, vfd


def shard_items(items: list, shard: str) -> list:
    i, n = map(int, shard.split("/"))
    return items[i::n]


def build_variant(scene: Scene, policy: Policy, variant: str, task: int, state: int, clean_obs, ctx):
    """Env-processed batch for one variant (None if the variant does not exist here), and the
    instruction used."""
    if variant == "clean":
        return policy.env_batch(clean_obs, scene.instruction), scene.instruction
    family, level = variant.split(":")
    rng = rng_for(task, state, family, level)
    kind = FAMILIES[family][1]
    if kind == "instruction":
        text = instruction_variant(task, family, level, rng)
        return (None, None) if text is None else (policy.env_batch(clean_obs, text), text)
    if kind == "simulator":
        try:
            ok = scene.apply(family, level, rng)
            batch = policy.env_batch(scene.observe(), scene.instruction) if ok else None
        finally:
            scene.restore()
        return batch, scene.instruction
    batch = observation_variant(policy.env_batch(clean_obs, scene.instruction), family, level, rng, ctx)
    return batch, scene.instruction


def first_frames() -> dict:
    """Env-processed-like first frames of the training demonstrations: states and head inputs."""
    from types import SimpleNamespace
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import filter_libero_episodes
    meta = LeRobotDatasetMetadata("HuggingFaceVLA/libero", root=DATA_ROOT)
    demos = sorted(int(e) for e in filter_libero_episodes(SimpleNamespace(meta=meta),
                                                         tasks_to_use={"libero_10": list(TASKS)}))
    dataset = LeRobotDataset("HuggingFaceVLA/libero", root=DATA_ROOT, episodes=demos)
    first = {}
    for index, episode in enumerate(dataset.hf_dataset["episode_index"]):
        first.setdefault(int(episode), index)
    return dataset, demos, first


def run_support(device: str, stride: int) -> None:
    """stride 0: the first frame of every demonstration (paper recipe, support.npz); stride n: every
    n-th frame (dense support as in E3, support_dense.npz)."""
    from score_fttgp_initial_pool import make_batch
    from kv_features_reference import head_inputs
    if (ROOT / ("support_dense.npz" if stride else "support.npz")).exists():
        print(f"support ({'dense' if stride else 'initial'}) already in {ROOT}, skipped", flush=True)
        return
    policy = Policy(device, partner=False)
    dataset, demos, first = first_frames()
    frames = {}
    for index, episode in enumerate(dataset.hf_dataset["episode_index"]):
        frames.setdefault(int(episode), []).append(index)
    rows = {k: [] for k in HEAD_INPUTS}
    meta = {"episode": [], "task": [], "state": []}
    instructions = {task_instruction(t): t for t in TASKS}
    for episode in demos:
        for row in (frames[episode][::stride] if stride else [first[episode]]):
            sample = dataset[row]
            batch = make_batch(policy.pre, dataset, [row], policy.device)
            for k, v in head_inputs(policy.policy, batch).items():
                rows[k].append(v.reshape(-1))
            meta["episode"].append(episode)
            meta["task"].append(instructions.get(sample["task"], -1))
            meta["state"].append(sample[STATE].numpy())
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / ("support_dense.npz" if stride else "support.npz"),
                        **{k: np.asarray(v, np.float32) for k, v in rows.items()},
                        episode=np.asarray(meta["episode"]), task=np.asarray(meta["task"]),
                        raw_state=np.asarray(meta["state"], np.float32))
    print(f"support: {len(meta['episode'])} frames of {len(demos)} demonstrations, tasks {sorted(set(meta['task']))}",
          flush=True)


def run_features(device: str, shard: str) -> None:
    support = np.load(ROOT / "support.npz")
    policy = Policy(device, partner=True)
    out_dir = ROOT / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    items = shard_items([(t, s) for t in TASKS for s in [*TEST, *CALIB]], shard)
    scenes: dict[int, Scene] = {}
    for task, state in items:
        path = out_dir / f"task{task}_state{state:02d}.npz"
        if path.exists():
            continue
        scene = scenes.setdefault(task, Scene(task))
        seed = SEED0 + 100 * task + state
        variants = VARIANTS if state in TEST else ["clean"]
        ctx = {}
        if state in TEST:
            other = scene.reset(25 + state, seed)  # mismatch partner: a calibration starting state
            ctx["other_state_batch"] = policy.env_batch(other, scene.instruction)
            ctx["other_state_state"] = ctx["other_state_batch"][STATE]
            pick = rng_for(task, state, "M3")
            other_task = TASKS[(TASKS.index(task) + 1) % len(TASKS)]
            others = np.flatnonzero(support["task"] == other_task)
            ctx["other_task_state"] = torch.from_numpy(support["raw_state"][pick.choice(others)])[None]
        clean_obs = scene.reset(state, seed)
        if state in TEST:
            ctx["target_box"] = scene.target_box(256, 256)
        feats, vfd, names, texts, thumbs = {k: [] for k in HEAD_INPUTS}, [], [], [], []
        for variant in variants:
            batch, text = build_variant(scene, policy, variant, task, state, clean_obs, ctx)
            if batch is None:
                continue
            f, v = policy.features(batch, seed if state in VFD_STATES else None)
            for k in HEAD_INPUTS:
                feats[k].append(f[k])
            vfd.append(v)
            names.append(variant)
            texts.append(text)
            if state in (0, 1):
                thumbs.append(np.concatenate([(batch[key][0].permute(1, 2, 0).cpu().numpy()[::2, ::2] * 255)
                                              .round().clip(0, 255).astype(np.uint8) for key in (WS, WRIST)], 1))
        np.savez_compressed(path, variants=np.asarray(names), instructions=np.asarray(texts),
                            vfd=np.asarray(vfd), **{k: np.asarray(v, np.float32) for k, v in feats.items()},
                            **({"thumbs": np.asarray(thumbs)} if thumbs else {}))
        print(f"task {task} state {state}: {len(names)} variants", flush=True)


def run_harm(device: str, shard: str) -> None:
    from iterative_fine_tuning.evaluation import _extract_batch_successes  # noqa: F401  (same success rule)
    policy = Policy(device, partner=False)
    out_dir = ROOT / "harm"
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = ["clean"] + [f"{f}:{spec[5]}" for f, spec in FAMILIES.items() if spec[5]]
    items = shard_items([(t, s, v) for t in TASKS for s in HARM_STATES for v in variants], shard)
    scenes: dict[int, Scene] = {}
    for task, state, variant in items:
        path = out_dir / f"task{task}_state{state:02d}_{variant.replace(':', '-')}.json"
        if path.exists():
            continue
        scene = scenes.setdefault(task, Scene(task))
        results = []
        for rep in range(HARM_REPS):
            seed = SEED0 + 100 * task + state + 10_000 * (rep + 1)
            results.append(rollout(scene, policy, task, state, variant, seed))
        path.write_text(json.dumps({"task": task, "state": state, "variant": variant, "success": results}) + "\n")
        print(f"harm task {task} state {state} {variant}: {results}", flush=True)


@torch.no_grad()
def rollout(scene: Scene, policy: Policy, task: int, state: int, variant: str, seed: int) -> bool | None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    observation = scene.reset(state, seed)
    instruction = scene.instruction
    family, level = variant.split(":") if variant != "clean" else (None, None)
    rng = rng_for(task, state, family, level)
    kind = FAMILIES[family][1] if family else None
    ctx = {}
    try:
        if kind == "instruction":
            instruction = instruction_variant(task, family, level, rng)
            if instruction is None:
                return None
        elif kind == "simulator":
            if not scene.apply(family, level, rng):
                return None
            observation = scene.observe()
        elif kind == "observation" and family == "V4":
            ctx["target_box"] = scene.target_box(256, 256)
            if ctx["target_box"] is None:
                return None  # no named movable target to occlude
        policy.policy.reset()
        for _ in range(520):
            batch = policy.env_batch(observation, instruction)
            if kind == "observation":
                if family == "V4":  # the occluder follows the target
                    ctx["target_box"] = scene.target_box(256, 256)
                batch = observation_variant(batch, family, level, rng_for(task, state, family, level), ctx)
            action = policy.policy.select_action(policy.pre(batch))
            action = policy.env_post({"action": policy.post(action)})["action"]
            action = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else action
            observation, _, terminated, truncated, info = scene.env.step(np.asarray(action).reshape(-1))
            if info.get("is_success"):
                return True
            if terminated or truncated:
                return False
        return False
    finally:
        scene.restore()


def run_sheet() -> None:
    from PIL import Image, ImageDraw
    for task in TASKS:
        d = np.load(ROOT / "features" / f"task{task}_state00.npz")
        thumbs, names = d["thumbs"], list(d["variants"])
        h, w = thumbs.shape[1:3]
        cols = 6
        rows = math.ceil(len(names) / cols)
        sheet = Image.new("RGB", (cols * w, rows * (h + 14)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (thumb, name) in enumerate(zip(thumbs, names)):
            x, y = (i % cols) * w, (i // cols) * (h + 14)
            sheet.paste(Image.fromarray(thumb), (x, y + 14))
            draw.text((x + 2, y + 1), name, fill="black")
        sheet.save(ROOT / f"sheet_task{task}.png")
        print(f"wrote {ROOT / f'sheet_task{task}.png'}")


def main() -> None:
    global MEMBERS, ROOT, TASKS, HARM_REPS
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("support", "features", "sheet", "harm", "analyze"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--stride", type=int, default=0, help="support: 0 = first frames, n = every n-th frame")
    parser.add_argument("--primary", type=Path, default=MEMBERS[0], help="acting policy (heads are fitted on it)")
    parser.add_argument("--companion", type=Path, default=MEMBERS[1], help="second VFD member")
    parser.add_argument("--root", type=Path, default=ROOT, help="output directory")
    parser.add_argument("--tasks", type=int, nargs="+", default=list(TASKS))
    parser.add_argument("--harm-reps", type=int, default=HARM_REPS)
    args = parser.parse_args()
    MEMBERS, ROOT, TASKS, HARM_REPS = (args.primary, args.companion), args.root, tuple(args.tasks), args.harm_reps
    if args.stage == "support":
        run_support(args.device, args.stride)
    elif args.stage == "features":
        run_features(args.device, args.shard)
    elif args.stage == "sheet":
        run_sheet()
    elif args.stage == "harm":
        run_harm(args.device, args.shard)
    else:
        import q5_e1_analyze
        q5_e1_analyze.ROOT, q5_e1_analyze.TASKS = ROOT, TASKS
        q5_e1_analyze.main()


if __name__ == "__main__":
    main()
