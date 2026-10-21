#!/usr/bin/env python
"""Roll out the paper's leak3 LIBERO ensemble and record frames plus per-step UQ.

The ensemble is the paper's M = 2 SmolVLA ensemble: two independent pretraining
lineages (`seed_1` and `seed_2` of `libero_pretrain_leak3`), both leaked the same
three LIBERO-10 tasks (0, 1, 2) and neither shown tasks 3-9. Member 0 acts; at
every policy step the paper's two-member VFD is evaluated between the two
lineages on the observation the policy is about to act on, so the score series is
aligned with the recorded frames. FT-TGP scores the acting member alone, through
the shared flow head fitted on the same leak3 pool.

A task in {0, 1, 2} is therefore "known" to the ensemble and any other LIBERO-10
task is "unknown". The capture is written to a compressed npz so the rendering
pass can be re-run without repeating the rollout.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import LiberoEnv, _get_suite, get_task_instruction
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.common.flow_matching.ode_solver import make_sampling_time_grid
from lerobot.policies.factory import make_flow_matching_adapter_from_policy, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.uncertainty.uncertainty_scoring.fttgp import PostHocFTTGP, append_flow_time
from kernel_variance import KernelVariance, extract_locations
from lerobot.uncertainty.uncertainty_scoring.kernel_variance import flow_level_features
from lerobot.uncertainty.uncertainty_scoring.paper_vfd import paper_vfd

TASK_GROUP = "libero_10"

# The paper's two-member ensemble: two independent pretraining lineages, not two
# fine-tunes of one lineage. Member 0 is the acting policy.
MEMBERS = (
    Path("outputs/pretrain/libero_pretrain_leak3/seed_1/checkpoints/030000/pretrained_model"),
    Path("outputs/pretrain/libero_pretrain_leak3/seed_2/checkpoints/030000/pretrained_model"),
)

# The shared flow head fitted on the leak3 pool, with 2048 random features - the
# same head `analyze_in_episode_uq.py` uses as `flow_shared`.
FTTGP_SHARED = Path("outputs/fttgp/libero_leak3_balanced_rff2048/head.pt")

# The three LIBERO-10 tasks leaked into pretraining. Everything else is unseen.
KNOWN_TASKS = (0, 1, 2)


def role_of(task_id: int) -> str:
    return "known" if task_id in KNOWN_TASKS else "unknown"


# Where a vanished object is parked: far below the floor, out of every camera.
VANISH_POSITION = (0.0, 0.0, -5.0)

# Camera corruptions. Sigma is in [0, 1] image units, matching the scale the
# repository's own eval corruptions use after `preprocess_observation`; here it is
# applied to the uint8 frames so the recorded video shows exactly what the policy
# and both scorers were given. "noise-replace" is the full-replacement corruption
# `IMAGE_CORRUPTION_MODE=noise` performs in eval_uncertainty_estimation.py.
IMAGE_NOISE_SIGMA = {
    "noise-light": 0.05,
    "noise-moderate": 0.15,
    "noise-heavy": 0.35,
}
# "noise-ramp": Gaussian noise whose sigma rises linearly from 0 at --event-start to
# NOISE_RAMP_MAX at --event-stop and then stays there until the episode ends.
NOISE_RAMP_MAX = 1.0
# Two more ramps over the same window, then held:
#   "camera-orbit"  the workspace camera orbits the table, about the vertical axis through the
#                   point it looks at on the table (ORBIT_PIVOT), from 0 to ORBIT_MAX_DEG degrees.
#                   A real camera move in the simulation, so perspective and occlusion stay right.
#   "scene-blend"   both camera images are cross-faded towards the time-matched frames of a
#                   capture from another scene (--blend-source), from 0 to 100 %.
ORBIT_MAX_DEG = 45.0
ORBIT_PIVOT = np.array([-0.21, 0.0, 0.45])  # agentview's line of sight meets the table here
RAMP_KINDS = ("noise-ramp", "camera-orbit", "scene-blend")
# "gripper-block": from --event-start on, for the rest of the episode, the Panda gripper is gone and a
# rigid grey block of about its size sits in its place. The fingers turn invisible and stop
# colliding; the hand housing's two geoms become a box (one drawn, one colliding) reaching from the
# hand to where the fingertips were. The finger joints still move on command, so proprioception
# reads as if the gripper worked. Half-extents in the hand frame (x, y, z) and its centre.
BLOCK_HALF = np.array([0.102, 0.032, 0.066])  # the gripper's full envelope, measured from its meshes
BLOCK_CENTRE = np.array([0.002, 0.0, 0.04])
# Windowed disturbances of the robot's state, body, action channel and physics (steps start..stop-1):
#   "state-offset"   the reported end-effector position is shifted by STATE_OFFSET (policy input only)
#   "arm-push"       a constant world-frame force PUSH_FORCE on the hand ("someone leans on the arm")
#   "action-delay"   each action is executed ACTION_DELAY steps late
#   "action-noise"   Gaussian noise of std ACTION_NOISE on the six arm commands
#   "action-scale"   the six arm commands scaled by ACTION_SCALE (weaker, slower motors)
#   "mass-heavy"     every movable object's mass and inertia x MASS_FACTOR (invisible before contact)
#   "friction-low"   every movable object's sliding friction x FRICTION_FACTOR (invisible before contact)
STATE_OFFSET = np.array([0.0, 0.03, 0.0])
PUSH_FORCE = np.array([0.0, 20.0, 0.0])
ACTION_DELAY = 3
ACTION_NOISE = 0.3
ACTION_SCALE = 0.3
MASS_FACTOR = 10.0
FRICTION_FACTOR = 0.1
STATE_KINDS = ("state-offset",)
ACTION_KINDS = ("action-delay", "action-noise", "action-scale")
PHYSICS_KINDS = ("arm-push", "mass-heavy", "friction-low")
# The block follows --event-start/--event-stop like any windowed event (a huge stop = for good).
PERSISTENT_KINDS = RAMP_KINDS
IMAGE_KINDS = tuple(IMAGE_NOISE_SIGMA) + ("noise-replace", "noise-ramp", "scene-blend")


def free_joint_objects(sim) -> dict[str, int]:
    """Movable objects in the scene, mapped to the qpos address of their free joint."""
    objects = {}
    for joint in range(sim.model.njnt):
        if sim.model.jnt_type[joint] != 0:  # 0 == mjJNT_FREE
            continue
        name = sim.model.joint_id2name(joint)
        objects[name.replace("_joint0", "")] = int(sim.model.jnt_qposadr[joint])
    return objects


def classify_objects(objects: dict[str, int], instruction: str) -> tuple[list[str], list[str]]:
    """Split scene objects into those the instruction names and the rest.

    Object names are MuJoCo-style (`alphabet_soup_1`); the instruction is English
    ("put both the alphabet soup and the tomato sauce in the basket"). Matching the
    underscore-free stem against the instruction keeps this working for any task
    rather than hard-coding a table of task ids.
    """
    targets, distractors = [], []
    lowered = instruction.lower()
    for name in objects:
        stem = name.rsplit("_", 1)[0].replace("_", " ")
        (targets if stem in lowered else distractors).append(name)
    return sorted(targets), sorted(distractors)


class SceneEvent:
    """Disturbs the simulated scene over a window of steps.

    Objects are teleported far below the floor (or, for `scramble`, flung to fresh
    random poses every step) and then restored to the pose and velocity they had at
    onset. This is a genuine change to the scene, not an edit of the rendered image:
    physics, contacts and every camera agree.

    The kinds form a severity ladder, from one irrelevant object going missing to
    the entire tabletop being emptied or thrown into chaos.
    """

    # Where the scramble mode may fling objects: across the reachable tabletop and
    # up into the air above it.
    SCRAMBLE_BOX = ((-0.26, 0.20), (-0.30, 0.30), (0.45, 0.80))

    def __init__(self, kind: str, start: int, stop: int, override: str | None = None,
                 seed: int = 0, blend_source: Path | None = None, orbit_max_deg: float = ORBIT_MAX_DEG):
        self.kind = kind
        self.orbit_max_deg = orbit_max_deg
        self._camera_home: tuple[int, np.ndarray, np.ndarray] | None = None
        self._blend: dict[str, np.ndarray] | None = None
        if kind == "scene-blend":
            if blend_source is None:
                raise SystemExit("--event scene-blend needs --blend-source")
            other = np.load(blend_source)
            # Captures store frames turned upright for display; the policy sees them as rendered.
            self._blend = {"image": other["external"][:, ::-1, ::-1], "image2": other["wrist"][:, ::-1, ::-1]}
        self.start = start
        self.stop = stop
        self.override = override
        self.objects: list[str] = []
        self.label: str | None = None
        self._saved: dict[str, tuple[int, int, np.ndarray, np.ndarray]] = {}
        # Collision flags of the parked objects' geoms, restored when they return.
        self._collisions: dict[int, tuple[int, int]] = {}
        self._rng = np.random.default_rng(seed + 991)

    @property
    def enabled(self) -> bool:
        return self.kind != "none"

    @property
    def scrambles(self) -> bool:
        return self.kind == "scramble"

    @property
    def corrupts_images(self) -> bool:
        """Image-space corruptions leave the simulation entirely alone."""
        return self.kind in IMAGE_KINDS

    def active(self, step: int) -> bool:
        if self.kind in PERSISTENT_KINDS:  # ramps hold their maximum; the block never comes off
            return self.start <= step
        return self.enabled and self.start <= step < self.stop

    def level(self, step: int) -> float:
        """How far a ramp has progressed at this step, 0..1 (0 for every other kind)."""
        if self.kind not in RAMP_KINDS or not self.active(step):
            return 0.0
        return min(1.0, (step - self.start) / max(self.stop - self.start, 1))

    def sigma(self, step: int) -> float:
        """Noise level (fraction of the pixel range) applied at this step; 0 when none."""
        if not self.active(step):
            return 0.0
        if self.kind == "noise-ramp":
            return NOISE_RAMP_MAX * min(1.0, (step - self.start) / max(self.stop - self.start, 1))
        return IMAGE_NOISE_SIGMA.get(self.kind, 0.0)

    def _choose(self, sim, instruction: str, eef_pos: np.ndarray) -> list[str]:
        objects = free_joint_objects(sim)
        if self.override:
            missing = [n for n in self.override.split(",") if n not in objects]
            if missing:
                raise SystemExit(f"object(s) {missing} not in scene; have: {sorted(objects)}")
            return self.override.split(",")

        targets, distractors = classify_objects(objects, instruction)
        if self.kind in ("vanish-all", "scramble"):
            return sorted(objects)
        if self.kind == "vanish-all-distractors":
            if not distractors:
                raise SystemExit("no distractor objects in this scene")
            return distractors

        pool = targets if self.kind == "vanish-target" else distractors
        if not pool:
            raise SystemExit(f"no {'target' if 'target' in self.kind else 'distractor'} objects")

        # Whichever candidate is nearest the gripper at onset, so a single-object
        # event lands on what the robot is actually dealing with.
        def distance(name: str) -> float:
            address = objects[name]
            return float(np.linalg.norm(sim.data.qpos[address : address + 3] - eef_pos))

        return [min(pool, key=distance)]

    def _chip_label(self) -> str:
        if self.kind == "noise-replace":
            return "camera feed replaced by noise"
        if self.kind == "noise-ramp":
            return f"camera noise rising to σ={NOISE_RAMP_MAX:.2f}"
        if self.kind == "camera-orbit":
            return f"camera orbiting to {self.orbit_max_deg:.0f}°"
        if self.kind == "gripper-block":
            return "gripper replaced by a block"
        labels = {"state-offset": f"reported hand position off by {100 * np.linalg.norm(STATE_OFFSET):.0f} cm",
                  "arm-push": f"arm pushed with {np.linalg.norm(PUSH_FORCE):.0f} N",
                  "action-delay": f"actions {ACTION_DELAY} steps late",
                  "action-noise": f"action noise std {ACTION_NOISE}",
                  "action-scale": f"arm commands x{ACTION_SCALE}",
                  "mass-heavy": f"objects x{MASS_FACTOR:.0f} heavier",
                  "friction-low": f"object friction x{FRICTION_FACTOR}"}
        if self.kind in labels:
            return labels[self.kind]
        if self.kind == "scene-blend":
            return "another scene fading in"
        if self.kind in IMAGE_NOISE_SIGMA:
            return f"camera noise σ={IMAGE_NOISE_SIGMA[self.kind]:.2f}"
        if self.scrambles:
            return f"{len(self.objects)} objects scrambled"
        if len(self.objects) == 1:
            return f"{self.objects[0]} gone"
        return f"all {len(self.objects)} objects gone" if self.kind == "vanish-all" \
            else f"{len(self.objects)} distractors gone"

    def _begin(self, sim, instruction: str, eef_pos: np.ndarray, step: int) -> None:
        self.objects = self._choose(sim, instruction, eef_pos)
        addresses = free_joint_objects(sim)
        for name in self.objects:
            address = addresses[name]
            joint = sim.model.joint_name2id(f"{name}_joint0")
            vel_address = int(sim.model.jnt_dofadr[joint])
            self._saved[name] = (
                address,
                vel_address,
                sim.data.qpos[address : address + 7].copy(),
                sim.data.qvel[vel_address : vel_address + 6].copy(),
            )
        if not self.scrambles:
            # A parked object must not collide with anything: the floor's half-space and
            # the kitchen scene's cabinets otherwise fling it back into the scene, where
            # it hits the other objects and the robot.
            bodies = {int(sim.model.jnt_bodyid[sim.model.joint_name2id(f"{n}_joint0")])
                      for n in self.objects}
            for body in range(sim.model.nbody):
                parent = body
                while parent not in bodies and parent != 0:
                    parent = int(sim.model.body_parentid[parent])
                if parent not in bodies:
                    continue
                for geom in np.flatnonzero(sim.model.geom_bodyid == body):
                    self._collisions[int(geom)] = (int(sim.model.geom_contype[geom]),
                                                   int(sim.model.geom_conaffinity[geom]))
                    sim.model.geom_contype[geom] = 0
                    sim.model.geom_conaffinity[geom] = 0
        self.label = self._chip_label()
        print(f"  [event] {self.kind} at step {step}: {self.label} "
              f"({', '.join(self.objects)})", flush=True)

    def _random_pose(self) -> tuple[np.ndarray, np.ndarray]:
        (x0, x1), (y0, y1), (z0, z1) = self.SCRAMBLE_BOX
        position = np.array([
            self._rng.uniform(x0, x1),
            self._rng.uniform(y0, y1),
            self._rng.uniform(z0, z1),
        ])
        quaternion = self._rng.normal(size=4)
        return position, quaternion / np.linalg.norm(quaternion)

    def transform_action(self, action: np.ndarray, step: int) -> np.ndarray:
        """The action actually executed this step (action-channel disturbances)."""
        if self.kind not in ACTION_KINDS:
            return action
        history = self.__dict__.setdefault("_actions", [])
        history.append(np.array(action, copy=True))
        if not self.active(step):
            return action
        out = np.array(action, copy=True)
        if self.kind == "action-delay":
            out = history[max(0, len(history) - 1 - ACTION_DELAY)].copy()
        elif self.kind == "action-noise":
            out[:6] = out[:6] + self._rng.normal(0.0, ACTION_NOISE, 6)
        elif self.kind == "action-scale":
            out[:6] = out[:6] * ACTION_SCALE
        return out

    def corrupt(self, observation: dict, step: int) -> dict:
        """Apply the camera corruption to this step's frames, in place."""
        if self.kind == "state-offset" and self.active(step):
            eef = observation["robot_state"]["eef"]
            eef["pos"] = np.asarray(eef["pos"], dtype=np.float64) + STATE_OFFSET
            return observation
        if not self.corrupts_images or not self.active(step):
            return observation
        pixels = observation["pixels"]
        if self.kind == "scene-blend":
            alpha = self.level(step)
            for key, image in list(pixels.items()):
                frames = self._blend[key]
                other = frames[min(step, len(frames) - 1)].astype(np.float32)
                pixels[key] = ((1 - alpha) * image.astype(np.float32) + alpha * other).clip(0, 255).astype(np.uint8)
            return observation
        for key, image in list(pixels.items()):
            if self.kind == "noise-replace":
                pixels[key] = self._rng.integers(0, 256, image.shape, dtype=np.uint8)
                continue
            sigma = self.sigma(step) * 255.0
            noisy = image.astype(np.float32) + self._rng.normal(0.0, sigma, image.shape)
            pixels[key] = np.clip(noisy, 0.0, 255.0).astype(np.uint8)
        return observation

    GEOM_FIELDS = ("geom_type", "geom_dataid", "geom_size", "geom_pos", "geom_quat", "geom_rbound",
                   "geom_rgba", "geom_contype", "geom_conaffinity")

    def _swap_gripper(self, sim) -> None:
        m = sim.model
        names = ("gripper0_finger1_visual", "gripper0_finger2_visual", "gripper0_finger1_collision",
                 "gripper0_finger2_collision", "gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision",
                 "gripper0_hand_visual", "gripper0_hand_collision")
        fields = self.GEOM_FIELDS + (("geom_aabb",) if hasattr(m, "geom_aabb") else ())
        # Everything the swap touches, so the real gripper can be put back at --event-stop.
        self._gripper_saved = {(n, f): np.array(getattr(m, f)[m.geom_name2id(n)]).copy() for n in names for f in fields}
        for name in ("gripper0_finger1_visual", "gripper0_finger2_visual", "gripper0_finger1_collision",
                     "gripper0_finger2_collision", "gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"):
            g = m.geom_name2id(name)
            m.geom_rgba[g, 3] = 0.0
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
        for name, drawn in (("gripper0_hand_visual", True), ("gripper0_hand_collision", False)):
            g = m.geom_name2id(name)
            m.geom_type[g] = 6  # mjGEOM_BOX
            m.geom_dataid[g] = -1
            m.geom_size[g] = BLOCK_HALF
            m.geom_pos[g] = BLOCK_CENTRE
            m.geom_quat[g] = [1.0, 0.0, 0.0, 0.0]
            m.geom_rbound[g] = float(np.linalg.norm(BLOCK_HALF))
            if hasattr(m, "geom_aabb"):
                m.geom_aabb[g] = np.concatenate([np.zeros(3), BLOCK_HALF])
            if drawn:
                m.geom_rgba[g] = [0.55, 0.56, 0.6, 1.0]
            else:
                m.geom_contype[g] = 1
                m.geom_conaffinity[g] = 1
        sim.forward()

    def _restore_gripper(self, sim) -> None:
        m = sim.model
        for (name, field), value in self._gripper_saved.items():
            getattr(m, field)[m.geom_name2id(name)] = value
        self._gripper_saved = None
        sim.forward()

    def update(self, sim, step: int, instruction: str, eef_pos: np.ndarray) -> bool:
        """Apply, hold or lift the event for this step. Returns whether it is active."""
        if self.kind in STATE_KINDS + ACTION_KINDS:
            if self.active(step) and self.label is None:
                self.label = self._chip_label()
                print(f"  [event] {self.kind} at step {step}: {self.label}", flush=True)
            return self.active(step)
        if self.kind in PHYSICS_KINDS:
            m = sim.model
            if self.active(step) and self.label is None:
                self.label = self._chip_label()
                print(f"  [event] {self.kind} at step {step}: {self.label}", flush=True)
                if self.kind in ("mass-heavy", "friction-low"):
                    bodies = [int(m.jnt_bodyid[m.joint_name2id(f"{n}_joint0")]) for n in free_joint_objects(sim)]
                    self._physics_saved = {"mass": {b: (float(m.body_mass[b]), m.body_inertia[b].copy()) for b in bodies},
                                           "friction": {}}
                    for b in bodies:
                        if self.kind == "mass-heavy":
                            m.body_mass[b] *= MASS_FACTOR
                            m.body_inertia[b] *= MASS_FACTOR
                        else:
                            for g in np.flatnonzero(m.geom_bodyid == b):
                                self._physics_saved["friction"][int(g)] = m.geom_friction[g].copy()
                                m.geom_friction[g, 0] *= FRICTION_FACTOR
                    sim.forward()
            hand = m.body_name2id("gripper0_right_gripper")
            if self.kind == "arm-push":
                sim.data.xfrc_applied[hand, :3] = PUSH_FORCE if self.active(step) else 0.0
            if not self.active(step) and step >= self.stop and getattr(self, "_physics_saved", None):
                for b, (mass, inertia) in self._physics_saved["mass"].items():
                    m.body_mass[b] = mass
                    m.body_inertia[b] = inertia
                for g, friction in self._physics_saved["friction"].items():
                    m.geom_friction[g] = friction
                self._physics_saved = None
                sim.forward()
                print(f"  [event] {self.kind} lifted at step {step}", flush=True)
            return self.active(step)
        if self.kind == "gripper-block":
            if self.active(step) and self.label is None:
                self._swap_gripper(sim)
                self.label = self._chip_label()
                print(f"  [event] gripper-block from step {step}: {self.label}", flush=True)
            elif not self.active(step) and getattr(self, "_gripper_saved", None) and step >= self.stop:
                self._restore_gripper(sim)
                print(f"  [event] gripper restored at step {step}", flush=True)
            return self.active(step)
        if self.kind == "camera-orbit":
            camera = sim.model.camera_name2id("agentview")
            if self._camera_home is None:
                self._camera_home = (camera, sim.model.cam_pos[camera].copy(), sim.model.cam_quat[camera].copy())
                self.label = self._chip_label()
                print(f"  [event] camera-orbit from step {step}: {self.label}", flush=True)
            _, home_pos, home_quat = self._camera_home
            angle = np.radians(self.orbit_max_deg * self.level(step))
            c, s_ = np.cos(angle), np.sin(angle)
            offset = home_pos - ORBIT_PIVOT
            sim.model.cam_pos[camera] = ORBIT_PIVOT + np.array([c * offset[0] - s_ * offset[1],
                                                                s_ * offset[0] + c * offset[1], offset[2]])
            # World-frame rotation about z, composed onto the camera's orientation (w, x, y, z).
            qz = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])
            w1, x1, y1, z1 = qz
            w2, x2, y2, z2 = home_quat
            sim.model.cam_quat[camera] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                                          w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                                          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
            sim.forward()
            return self.active(step)
        if self.corrupts_images:
            if self.active(step) and self.label is None:
                self.label = self._chip_label()
                print(f"  [event] {self.kind} at step {step}: {self.label}", flush=True)
            return self.active(step)

        if not self.enabled or step < self.start or step >= self.stop:
            if self._saved and step >= self.stop:
                for address, vel_address, qpos, qvel in self._saved.values():
                    sim.data.qpos[address : address + 7] = qpos
                    sim.data.qvel[vel_address : vel_address + 6] = qvel
                self._saved = {}
                for geom, (contype, conaffinity) in self._collisions.items():
                    sim.model.geom_contype[geom] = contype
                    sim.model.geom_conaffinity[geom] = conaffinity
                self._collisions = {}
                sim.forward()
            return False

        if not self._saved:
            self._begin(sim, instruction, eef_pos, step)

        # Re-assert every step: gravity would otherwise carry the objects somewhere
        # else, and the restore pose would no longer be the pose they left from.
        for index, (name, (address, vel_address, _, _)) in enumerate(self._saved.items()):
            if self.scrambles:
                position, quaternion = self._random_pose()
                sim.data.qpos[address : address + 3] = position
                sim.data.qpos[address + 3 : address + 7] = quaternion
            else:
                # Spread the parked objects out so they do not all interpenetrate
                # at one point, which upsets the constraint solver.
                sim.data.qpos[address : address + 3] = (
                    VANISH_POSITION[0] + 0.5 * index,
                    VANISH_POSITION[1],
                    VANISH_POSITION[2],
                )
            sim.data.qvel[vel_address : vel_address + 6] = 0.0
        sim.forward()
        return True


@torch.no_grad()
def score_shared_flow(adapter, heads: dict, batch, num_samples: int, generator,
                      collect: list | None = None, keep_tokens: int = 0) -> dict[str, float]:
    """FT-TGP variance along the flow path, pooled the paper's late-level way.

    Every head is scored on one shared integration of the policy's flow, so two
    heads differ only in the head, never in the path they were asked about. Only
    shared flow heads are needed for the score the videos show, so the per-level
    and VLM heads (which live in separate, separately fitted directories) are not
    loaded at all.
    """
    batch_size = next(value.shape[0] for value in batch.values() if isinstance(value, torch.Tensor))
    conditioning = adapter.prepare_conditioning(batch, num_action_samples=num_samples)
    velocity_and_features = adapter.make_velocity_and_features_fn(conditioning)
    state = adapter.sample_prior(batch_size * num_samples, generator=generator)
    grid = make_sampling_time_grid(
        step_size=adapter.ode_solver_config["step_size"], device=adapter.device, dtype=adapter.dtype
    )
    levels = {name: [] for name in heads}
    for start, stop in zip(grid[:-1], grid[1:], strict=True):
        velocity, hidden = velocity_and_features(x_t=state, t=start)
        gp_input = append_flow_time(hidden, start)
        if collect is not None and keep_tokens:
            # The GP inputs do not depend on any head's hyperparameters, so caching
            # them lets a parameter search be replayed offline without re-running
            # the policy. Tokens are subsampled evenly to keep the file small.
            index = torch.linspace(0, gp_input.shape[1] - 1, keep_tokens,
                                   device=gp_input.device).round().long()
            # Keep every action sample: the score averages over them, so caching
            # only one would replay a noisier estimator than the live one.
            collect.append(gp_input[:, index].to(torch.float16).cpu())
        for name, entry in heads.items():
            head = entry["head"]
            if entry.get("mode") == "predictive":
                # How far the policy's own velocity sits from the velocity the GP
                # expects, in units of its predictive standard deviation.
                mean, variance = head.predict(gp_input)
                residual = (velocity[..., : mean.shape[-1]] - mean).square().sum(dim=-1)
                token = residual / (variance + head.observation_noise)
            else:
                token = head.predictive_variance(gp_input)
            levels[name].append(token.mean(dim=-1).reshape(batch_size, num_samples).mean(dim=1))
        state = state + (stop - start) * velocity
    times = grid[:-1]
    weights = times / (1 - times).clamp_min(1e-3)
    weights = weights / weights.sum().clamp_min(1e-12)
    scores = {}
    for name, stacked in levels.items():
        stacked = torch.stack(stacked)
        entry = heads[name]
        scores[f"{name}_mean"] = float(stacked.mean(dim=0)[0])
        if entry.get("level_mean") is not None:
            # Per-level standardization against ID statistics, so no Euler level
            # dominates the weighted sum.
            standardized = (stacked - entry["level_mean"][:, None]) / entry["level_std"][:, None]
            scores[f"{name}_paper_weighted"] = float((standardized * weights[:, None]).sum(dim=0)[0])
        else:
            scores[f"{name}_paper_weighted"] = float((stacked * weights[:, None]).sum(dim=0)[0])
    return scores


def policy_batch(env, env_preprocessor, preprocessor, observation: dict) -> dict:
    """One observation, converted to what the policy and the scorers consume."""
    batch = preprocess_observation(observation)
    batch = add_envs_task(env, batch)
    batch = env_preprocessor(batch)
    return preprocessor(batch)


@torch.no_grad()
def capture(
    task_id: int,
    episode_index: int,
    seed: int,
    device: torch.device,
    num_action_samples: int,
    max_steps: int | None,
    with_fttgp: bool,
    event: SceneEvent | None = None,
    extra_heads: dict[str, Path] | None = None,
    cache_stride: int = 0,
    cache_tokens: int = 10,
    kv_heads: dict[str, Path] | None = None,
    save_locations: bool = False,
    instruction_override: str | None = None,
    instruction_window: tuple[int, int] | None = None,
    instruction_label: str = "instruction negated",
    save_flow_levels: bool = False,
    save_ctx: bool = False,
    render_size: int | None = None,
) -> dict:
    event = event or SceneEvent("none", 0, 0)
    instruction = get_task_instruction(TASK_GROUP, task_id)
    actor = SmolVLAPolicy.from_pretrained(MEMBERS[0]).to(device).eval()
    partner = SmolVLAPolicy.from_pretrained(MEMBERS[1]).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        actor.config,
        pretrained_path=str(MEMBERS[0]),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    actor_adapter = make_flow_matching_adapter_from_policy(actor)
    partner_adapter = make_flow_matching_adapter_from_policy(partner)

    heads: dict = {}
    if with_fttgp:
        heads["flow_shared"] = {"head": PostHocFTTGP.load(FTTGP_SHARED, device=device),
                                "mode": "variance"}
        print(f"  loaded FT-TGP shared head from {FTTGP_SHARED}", flush=True)
        for name, path in (extra_heads or {}).items():
            entry = {"head": PostHocFTTGP.load(path, device=device), "mode": "variance"}
            # A sidecar written by compare_fttgp_variants.py carries the scoring rule.
            sidecar = Path(str(path).removesuffix(".pt") + ".json")
            if sidecar.exists():
                meta = json.loads(sidecar.read_text())
                entry["mode"] = meta.get("mode", "variance")
                if meta.get("level_mean") is not None:
                    entry["level_mean"] = torch.tensor(meta["level_mean"], device=device)
                    entry["level_std"] = torch.tensor(meta["level_std"], device=device)
            heads[name] = entry
            print(f"  loaded {name} head from {path} (mode={entry['mode']}"
                  f"{', level-standardized' if 'level_mean' in entry else ''})", flush=True)

    # KV heads are named <location>_<bank>; the location picks the feature.
    kv: dict[str, tuple[str, KernelVariance]] = {}
    for name, path in (kv_heads or {}).items():
        kv[name] = (name.rsplit("_", 1)[0], KernelVariance.load(path, device=device))
        print(f"  loaded KV head {name} from {path}", flush=True)
    kv_traces: dict[str, list[float]] = {name: [] for name in kv}
    # Raw per-step features at every KV location, so new heads can be replayed offline.
    location_rows: dict[str, list[np.ndarray]] = {}
    # Pooled action-expert state at every Euler level, for the flow-level GP (iVFD) offline.
    flow_rows: list[np.ndarray] = []
    # KV-Ctx input per step (paper's head: final prefix states per camera and instruction + state).
    ctx_rows: list[np.ndarray] = []

    # Build the environment from the repo's own LIBERO config so the capture runs
    # under the same observation size and control mode the evaluations use.
    env_cfg = LiberoEnvConfig(task=TASK_GROUP, task_ids=[task_id])
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=actor.config)
    env = LiberoEnv(
        task_suite=_get_suite(TASK_GROUP),
        task_id=task_id,
        task_suite_name=TASK_GROUP,
        obs_type=env_cfg.obs_type,
        camera_name=env_cfg.camera_name,
        init_states=env_cfg.init_states,
        # --render-size 256 matches the evaluator's vector envs and the training images.
        observation_width=render_size or env_cfg.observation_width,
        observation_height=render_size or env_cfg.observation_height,
        episode_index=episode_index,
        control_mode=env_cfg.control_mode,
    )
    step_budget = max_steps or env._max_episode_steps
    original_instruction = env.task_description
    if instruction_override and instruction_window is None:
        # The policy reads its instruction from the env; the scene and the success
        # check stay those of the original task.
        env.task_description = instruction_override
    instruction_trace: list[bool] = []

    external_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    frame_scores: list[float] = []
    gripper_trace: list[float] = []
    eef_height: list[float] = []
    event_trace: list[bool] = []
    sigma_trace: list[float] = []
    level_trace: list[float] = []
    cached_features: list = []
    cached_steps: list[int] = []
    fttgp_traces: dict[str, list[float]] = {
        f"{name}{suffix}": []
        for name in heads
        for suffix in ("_paper_weighted", "_mean")
    }
    started = time.time()
    success = False
    sim_steps = 0

    try:
        # SmolVLA draws its action samples from the global torch RNG, so the
        # trajectory only repeats if that is seeded too.
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        observation, _ = env.reset(seed=seed)
        actor.reset()

        while sim_steps < step_budget:
            # Apply the scene event before anything looks at the scene, so the
            # policy and both scorers see the same perturbed world this step.
            if event.enabled:
                inner = env._env.env
                was_active = event.update(
                    inner.sim,
                    sim_steps,
                    instruction,
                    np.asarray(observation["robot_state"]["eef"]["pos"]),
                )
                observation = event.corrupt(
                    # A moved camera must be re-rendered now; the cached frames are a step old.
                    env._format_raw_obs(inner._get_observations(
                        force_update=event.kind in ("camera-orbit", "gripper-block"))),
                    sim_steps
                )
                event_trace.append(was_active)
                sigma_trace.append(event.sigma(sim_steps))
                level_trace.append(event.level(sim_steps))

            if instruction_window is not None:
                # An instruction event: the override is read only inside the window.
                swapped = instruction_window[0] <= sim_steps < instruction_window[1]
                env.task_description = instruction_override if swapped else original_instruction
                instruction_trace.append(swapped)

            # Score the observation the policy is about to act on. Separate
            # preprocessor calls keep the policy's own input untouched.
            scored = policy_batch(env, env_preprocessor, preprocessor, observation)
            generator = torch.Generator(device=device).manual_seed(seed * 1000 + sim_steps)
            score = float(
                paper_vfd(
                    actor_adapter,
                    partner_adapter,
                    scored,
                    num_action_samples=num_action_samples,
                    generator=generator,
                )[0]
            )
            fttgp_now = {key: float("nan") for key in fttgp_traces}
            if with_fttgp:
                scored_gp = policy_batch(env, env_preprocessor, preprocessor, observation)
                gp_generator = torch.Generator(device=device).manual_seed(
                    seed * 1000 + sim_steps + 500_000
                )
                collect = None
                if cache_stride and sim_steps % cache_stride == 0:
                    collect = []
                fttgp_now = score_shared_flow(
                    actor_adapter, heads, scored_gp, num_action_samples, gp_generator,
                    collect=collect, keep_tokens=cache_tokens if collect is not None else 0,
                )
                if collect:
                    cached_features.append(torch.stack(collect))
                    cached_steps.append(sim_steps)

            if kv or save_locations:
                locations = extract_locations(
                    actor_adapter, policy_batch(env, env_preprocessor, preprocessor, observation)
                )
                for name, (location, head) in kv.items():
                    kv_traces[name].append(float(head.score(locations[location].float())[0]))
                if save_locations:
                    for location, value in locations.items():
                        location_rows.setdefault(location, []).append(
                            value[0].float().cpu().numpy().astype(np.float32))

            if save_ctx:
                from lerobot.uncertainty.uncertainty_scoring.context_kv import final_context_features
                ctx = final_context_features(
                    actor_adapter, policy_batch(env, env_preprocessor, preprocessor, observation))
                ctx_rows.append((ctx.float().cpu().numpy() if torch.is_tensor(ctx) else np.asarray(ctx))
                                .reshape(-1).astype(np.float32))

            if save_flow_levels:
                flow_rows.append(flow_level_features(
                    actor_adapter, policy_batch(env, env_preprocessor, preprocessor, observation)
                )[0].cpu().numpy().astype(np.float32))

            # Record the frame the score belongs to before stepping it on. LIBERO
            # renders upside down and mirrored, so flip both axes as `render()` does.
            pixels = observation["pixels"]
            external_frames.append(np.ascontiguousarray(pixels["image"][::-1, ::-1]))
            wrist_frames.append(np.ascontiguousarray(pixels["image2"][::-1, ::-1]))
            frame_scores.append(score)
            state = observation["robot_state"]
            gripper_trace.append(float(state["gripper"]["qpos"][0]))
            eef_height.append(float(state["eef"]["pos"][2]))
            for key, value in fttgp_now.items():
                fttgp_traces[key].append(value)

            processed = policy_batch(env, env_preprocessor, preprocessor, observation)
            action = postprocessor(actor.select_action(processed))[0].cpu().numpy()
            if event.enabled:
                action = event.transform_action(action, sim_steps)

            observation, _, terminated, truncated, info = env.step(action)
            sim_steps += 1
            if info.get("is_success"):
                success = True
            if terminated or truncated:
                break
            if sim_steps % 20 == 0:
                print(
                    f"  step {sim_steps:3d}/{step_budget}  vfd={score:8.2f}  "
                    f"({time.time() - started:.0f}s elapsed)",
                    flush=True,
                )
    finally:
        env.close()

    result_event = {
        "kind": event.kind, "objects": event.objects, "label": event.label,
        "start": event.start, "stop": event.stop,
        **({"max": event.orbit_max_deg} if event.kind == "camera-orbit" else {}),
    } if event.enabled else None
    variant = event.kind if event.enabled else "clean"
    event_active = np.asarray(event_trace, dtype=bool)
    if instruction_window is not None:
        variant = "instruction-swap"
        result_event = {"kind": "instruction-swap", "objects": [],
                        "label": instruction_label, "start": instruction_window[0],
                        "stop": instruction_window[1], "instruction": instruction_override}
        event_active = np.asarray(instruction_trace, dtype=bool)
    return {
        "arm": role_of(task_id),
        "variant": variant,
        "task_group": TASK_GROUP,
        "task_id": task_id,
        "instruction": (instruction_override if instruction_window is None else None)
        or get_task_instruction(TASK_GROUP, task_id),
        "original_instruction": get_task_instruction(TASK_GROUP, task_id),
        "episode_index": episode_index,
        "seed": seed,
        "success": success,
        "sim_steps": sim_steps,
        "num_action_samples": num_action_samples,
        "members": [str(path) for path in MEMBERS],
        "external": np.stack(external_frames),
        "wrist": np.stack(wrist_frames),
        "vfd": np.asarray(frame_scores, dtype=np.float32),
        "gripper": np.asarray(gripper_trace, dtype=np.float32),
        "eef_z": np.asarray(eef_height, dtype=np.float32),
        "event_active": event_active,
        **({"gp_features": torch.stack(cached_features).numpy(),
            "gp_feature_steps": np.asarray(cached_steps, dtype=np.int32)}
           if cached_features else {}),
        "event": result_event,
        "with_fttgp": with_fttgp,
        "fttgp_heads": {name: str(path) for name, path in (extra_heads or {}).items()},
        **{f"fttgp_{key}": np.asarray(values, dtype=np.float32)
           for key, values in fttgp_traces.items()},
        "kv_heads": {name: str(path) for name, path in (kv_heads or {}).items()},
        **{f"kv_{name}": np.asarray(values, dtype=np.float32)
           for name, values in kv_traces.items()},
        **{f"loc_{name}": np.stack(rows) for name, rows in location_rows.items()},
        **({"flow_levels": np.stack(flow_rows)} if flow_rows else {}),
        **({"loc_kv_ctx": np.stack(ctx_rows)} if ctx_rows else {}),
        **({"noise_sigma": np.asarray(sigma_trace, dtype=np.float32)} if event.kind == "noise-ramp" else {}),
        **({"ramp_level": np.asarray(level_trace, dtype=np.float32)} if event.kind in RAMP_KINDS else {}),
        "seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", type=int, default=[0, 3],
                        help="LIBERO-10 task ids. 0-2 are known to the ensemble, 3-9 unknown.")
    parser.add_argument("--episodes", nargs="+", type=int, default=[0],
                        help="Episode indices; each picks a fixed initial state of its task.")
    parser.add_argument("--num-action-samples", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Defaults to the suite's own limit (520 for libero_10).")
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-fttgp", action="store_true",
                        help="Also score the leak3 shared flow head at every policy step.")
    parser.add_argument("--event", default="none",
                        choices=("none", "vanish-distractor", "vanish-target",
                                 "vanish-all-distractors", "vanish-all", "scramble", "camera-orbit",
                                 "gripper-block", "state-offset", "arm-push", "action-delay", "action-noise",
                                 "action-scale", "mass-heavy", "friction-low")
                                + IMAGE_KINDS,
                        help="Anomaly to inject. Scene events, in rough order of "
                             "severity: one irrelevant object, the object being reached "
                             "for, every irrelevant object, the entire tabletop, or "
                             "every object flung to a fresh random pose each step. "
                             "Camera events corrupt both views instead: additive "
                             "Gaussian noise at three strengths, or full replacement.")
    parser.add_argument("--event-start", type=int, default=30,
                        help="First simulation step of the anomaly window.")
    parser.add_argument("--event-stop", type=int, default=90,
                        help="First step after the anomaly window; the object returns here.")
    parser.add_argument("--cache-stride", type=int, default=0,
                        help="Cache the GP input features every N steps (0 disables). "
                             "Lets a head parameter search be replayed offline.")
    parser.add_argument("--cache-tokens", type=int, default=10,
                        help="How many action tokens per level to keep when caching.")
    parser.add_argument("--extra-head", action="append", metavar="NAME:PATH", default=None,
                        help="Score an additional shared flow head alongside the shipped "
                             "one. Repeatable; stored as fttgp_<NAME>_paper_weighted.")
    parser.add_argument("--kv-head", action="append", metavar="NAME:PATH", default=None,
                        help="Score a KV (kernel variance) head written by fit_kv_heads.py. "
                             "NAME is <location>_<bank>; stored as kv_<NAME>. Repeatable.")
    parser.add_argument("--save-locations", action="store_true",
                        help="Store the raw features at every KV location per step "
                             "as loc_<location> (float32: camera features overflow float16), for replaying heads offline.")
    parser.add_argument("--orbit-max-deg", type=float, default=ORBIT_MAX_DEG,
                        help="For --event camera-orbit: the angle reached at the end of the ramp (360 = a full circle).")
    parser.add_argument("--blend-source", type=Path, default=None,
                        help="For --event scene-blend: a capture (npz) of another scene whose frames fade in.")
    parser.add_argument("--save-ctx", action="store_true",
                        help="Store the KV-Ctx input per step as loc_kv_ctx (2,888 values).")
    parser.add_argument("--render-size", type=int, default=None,
                        help="Camera resolution (256 = the evaluator's and training images'; default: env config, 360).")
    parser.add_argument("--save-flow-levels", action="store_true",
                        help="Store the pooled action-expert state at every Euler level per step "
                             "as flow_levels (steps x levels x dim), for the flow-level GP (iVFD).")
    parser.add_argument("--instruction", default=None,
                        help="Give the policy this instruction instead of the task's own. The "
                             "scene and the success check stay the original task's.")
    parser.add_argument("--instruction-window", nargs=2, type=int, default=None, metavar=("START", "STOP"),
                        help="Read --instruction only for steps START..STOP-1, the original "
                             "instruction before and after (an instruction event).")
    parser.add_argument("--instruction-label", default="instruction negated",
                        help="How the instruction event is labelled in the capture and videos.")
    parser.add_argument("--name-suffix", default="",
                        help="Appended to the capture filename, e.g. _negated.")
    parser.add_argument("--event-object", default=None,
                        help="Override the automatic choice with a comma-separated list "
                             "of scene object names.")
    args = parser.parse_args()

    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extra_heads = {}
    for spec in args.extra_head or []:
        name, _, path = spec.partition(":")
        if not name or not path:
            raise SystemExit(f"--extra-head expects NAME:PATH, got {spec!r}")
        extra_heads[name] = Path(path)
    kv_heads = {}
    for spec in args.kv_head or []:
        name, _, path = spec.partition(":")
        if not name or not path:
            raise SystemExit(f"--kv-head expects NAME:PATH, got {spec!r}")
        kv_heads[name] = Path(path)

    for episode_index in args.episodes:
        for task_id in args.tasks:
            suffix = "" if args.event == "none" else f"_{args.event.replace('-', '_')}"
            target = args.output_dir / f"task{task_id}_ep{episode_index}{suffix}{args.name_suffix}.npz"
            if target.exists():
                print(f"skip {target} (exists)")
                continue
            seed = args.seed_base + episode_index
            print(
                f"capturing task={task_id} ({role_of(task_id)}) episode={episode_index} seed={seed}",
                flush=True,
            )
            result = capture(
                task_id,
                episode_index,
                seed,
                device,
                args.num_action_samples,
                args.max_steps,
                args.with_fttgp,
                SceneEvent(args.event, args.event_start, args.event_stop,
                           args.event_object, seed, args.blend_source, args.orbit_max_deg),
                extra_heads,
                args.cache_stride,
                args.cache_tokens,
                kv_heads,
                args.save_locations,
                args.instruction,
                tuple(args.instruction_window) if args.instruction_window else None,
                args.instruction_label,
                args.save_flow_levels,
                args.save_ctx,
                args.render_size,
            )
            meta = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
            arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
            np.savez_compressed(target, meta=json.dumps(meta), **arrays)
            print(
                f"  -> {target.name}: success={result['success']} steps={result['sim_steps']} "
                f"vfd mean={result['vfd'].mean():.2f} max={result['vfd'].max():.2f} "
                f"({result['seconds']:.0f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
