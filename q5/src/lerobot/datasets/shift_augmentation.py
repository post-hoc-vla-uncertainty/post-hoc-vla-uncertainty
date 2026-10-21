"""Mild distribution shifts for LIBERO-10 tasks 0-2, applied identically in training and evaluation.

Each shift is a pure function of the policy's input, so the teacher can be trained on shifted
copies of the stored demonstrations and the rollouts can be shifted the same way:

  tilt       the workspace camera ("observation.images.image") rotated about its optical centre by
             an angle in degrees (pitch). A pure rotation is exactly a homography, H = K R K⁻¹, so
             warping the stored frame reproduces what the rotated camera would see; the part of the
             view it never saw is black. The wrist camera is untouched.
  dim        both cameras' pixel values multiplied by a factor (a dimmer exposure).
  paraphrase the instruction replaced by a rewording of increasing distance from the original
             (1 = word order, 2 = synonyms, 3 = casual).

Images are float tensors in [0, 1], shape (..., C, H, W), in the orientation of the dataset (the
env processor has already rotated LIBERO's renders when the evaluation applies a shift).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

WORKSPACE_CAMERA = "observation.images.image"
# LIBERO's agentview camera: 45° vertical field of view.
FOVY_DEGREES = 45.0

PARAPHRASES = {
    "put both the alphabet soup and the tomato sauce in the basket": (
        "put the tomato sauce and the alphabet soup in the basket",
        "place both the alphabet soup and the tomato sauce into the basket",
        "can you put the soup and the tomato sauce in the basket",
    ),
    "put both the cream cheese box and the butter in the basket": (
        "put the butter and the cream cheese box in the basket",
        "place both the cream cheese box and the butter into the basket",
        "can you put the cheese box and the butter in the basket",
    ),
    "turn on the stove and put the moka pot on it": (
        "turn the stove on and put the moka pot on it",
        "switch on the stove and place the moka pot on it",
        "can you turn on the stove and then put the coffee pot on top",
    ),
}


@dataclass(frozen=True)
class Shift:
    kind: str  # "clean", "tilt", "dim" or "paraphrase"
    value: float = 0.0  # degrees, brightness factor, or paraphrase level 1-3

    @property
    def name(self) -> str:
        if self.kind == "clean":
            return "clean"
        value = int(self.value) if float(self.value).is_integer() else self.value
        return f"{self.kind}_{value}"

    @classmethod
    def parse(cls, name: str) -> Shift:
        if name == "clean":
            return cls("clean")
        kind, value = name.rsplit("_", 1)
        return cls(kind, float(value))


def tilt(image: torch.Tensor, degrees: float) -> torch.Tensor:
    """Warp (..., C, H, W) as if the camera had pitched by `degrees` about its optical centre."""
    if degrees == 0:
        return image
    height, width = image.shape[-2:]
    focal = (height / 2) / math.tan(math.radians(FOVY_DEGREES) / 2)
    angle = math.radians(degrees)
    k = torch.tensor([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=torch.float64)
    r = torch.tensor([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]],
                     dtype=torch.float64)
    # Output pixel -> source pixel in the original frame.
    inverse = k @ r.T @ torch.linalg.inv(k)
    ys, xs = torch.meshgrid(torch.arange(height, dtype=torch.float64), torch.arange(width, dtype=torch.float64),
                            indexing="ij")
    points = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1) @ inverse.T
    source = points[..., :2] / points[..., 2:3]
    grid = torch.stack([source[..., 0] / (width - 1) * 2 - 1, source[..., 1] / (height - 1) * 2 - 1], dim=-1)
    flat = image.reshape(-1, *image.shape[-3:])
    grid = grid.to(flat.dtype).to(flat.device).unsqueeze(0).expand(flat.shape[0], -1, -1, -1)
    warped = F.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return warped.reshape(image.shape)


def paraphrase(instruction: str, level: int) -> str:
    if instruction not in PARAPHRASES:
        raise KeyError(f"no paraphrases for {instruction!r}")
    return PARAPHRASES[instruction][level - 1]


def apply_shift(item: dict, shift: Shift) -> dict:
    """Shift one sample or batch: image tensors under their keys, "task" a string or list of strings."""
    if shift.kind == "clean":
        return item
    item = dict(item)
    if shift.kind == "tilt":
        item[WORKSPACE_CAMERA] = tilt(item[WORKSPACE_CAMERA], shift.value)
    elif shift.kind == "dim":
        # Only the images themselves, not their boolean padding masks ("..._is_pad").
        for key in [k for k, v in item.items() if k.startswith("observation.images.")
                    and isinstance(v, torch.Tensor) and v.is_floating_point()]:
            item[key] = (item[key] * shift.value).clamp(0.0, 1.0)
    elif shift.kind == "paraphrase":
        task = item["task"]
        item["task"] = (paraphrase(task, int(shift.value)) if isinstance(task, str)
                        else [paraphrase(t, int(shift.value)) for t in task])
    else:
        raise ValueError(f"unknown shift kind {shift.kind!r}")
    return item


class ShiftAugmentation:
    """Training-time sampler: each item stays clean with probability p_clean, otherwise it gets one
    shift drawn uniformly from `shifts`, applied consistently to its cameras and instruction."""

    def __init__(self, shifts: list[str], p_clean: float):
        self.shifts = [Shift.parse(name) for name in shifts]
        self.p_clean = p_clean

    def __call__(self, item: dict) -> dict:
        if random.random() < self.p_clean:
            return item
        return apply_shift(item, random.choice(self.shifts))
