"""Training-time augmentation of the Q5 supervisor (vla-uncertainty-paper/handoff/q5, Section 3).

Each training sample stays clean with probability `p_clean` (0.5); otherwise it is changed by exactly one
label-preserving family, drawn uniformly, with the ranges of `supervisor_augmentation` in perturbations.json:

  I1  instruction replaced by one of the five released prompts P1-P5 (P1 = original)
  I2  a random subset of the task's object words replaced by their synonyms
  I3  "please " prefix, "Could you ...?" with a capital, or one typo in a non-object word
  V1  colour jitter on both cameras: brightness 0.4-2.0, contrast 0.7-1.3, per-channel gain 0.75-1.1
  V2  sensor degradation on both cameras: Gaussian noise sigma 0-0.1, motion blur 3-11 px, or JPEG quality 10-90
  V3  random affine on the workspace camera: shift up to 12 px, rotation up to 8 deg, scale 0.95-1.05
  V4  one or two cutouts of 5-30 % of the image side on the workspace camera, table colour or grey
  V6  wrist camera blanked
  R2  end-effector position in the state offset by N(0, 3 cm) per axis
V5 (object colour) is not augmentable and is left out, as the handoff says. Test-time perturbations (E2) are
separate deterministic draws, so the supervisor never sees the exact noise or offsets it is evaluated on.

Items are single samples from LeRobotDataset.__getitem__: images float in [0, 1] with shape (..., C, H, W) in the
orientation the policy reads, "observation.state" unnormalized (eef position first), "task" a string.
"""

from __future__ import annotations

import io
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

WS, WRIST, STATE = "observation.images.image", "observation.images.image2", "observation.state"
FAMILIES = ("I1", "I2", "I3", "V1", "V2", "V3", "V4", "V6", "R2")
_ROOT = Path(__file__).resolve().parents[3]
PROMPTS = _ROOT / "scripts/calibration/language_variation_prompts.json"
PERTURBATIONS = _ROOT / "vla-uncertainty-paper/handoff/q5/perturbations.json"


def _images(item: dict) -> list[str]:
    return [k for k, v in item.items() if k.startswith("observation.images.") and isinstance(v, torch.Tensor)
            and v.is_floating_point()]


class SupervisorAugmentation:
    def __init__(self, p_clean: float = 0.5, families: tuple[str, ...] = FAMILIES):
        self.p_clean, self.families = p_clean, families
        prompts = json.loads(PROMPTS.read_text())
        # Every prompt of a task maps to the task's five prompts (the dataset uses P1 = original).
        self.prompts = {p[0]: p for p in prompts.values()}
        self.task_of = {p[0]: int(t) for t, p in prompts.items()}
        families_json = {f["id"]: f for f in json.loads(PERTURBATIONS.read_text())["families"]}
        self.synonyms = {int(t): s for t, s in families_json["I2"]["synonyms"].items()}

    # -- instruction families ---------------------------------------------------------------------
    def _instruction(self, family: str, text: str) -> str:
        if text not in self.task_of:
            return text  # not a LIBERO-10 instruction: leave it
        if family == "I1":
            return random.choice(self.prompts[text])
        synonyms = self.synonyms.get(self.task_of[text], {})
        if family == "I2":
            out = text
            for word in sorted(synonyms, key=len, reverse=True):
                if word in out and random.random() < 0.5:
                    out = out.replace(word, synonyms[word])
            return out
        kind = random.choice(("please", "could_you", "typo"))
        if kind == "please":
            return "please " + text
        if kind == "could_you":
            return "Could you " + text + "?"
        nouns = {w for phrase in synonyms for w in phrase.split()}
        words = text.split()
        candidates = [i for i, w in enumerate(words) if len(w) > 3 and w not in nouns]
        if not candidates:
            return text
        i = random.choice(candidates)
        w = words[i]
        j = random.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        return " ".join(words)

    # -- image families ---------------------------------------------------------------------------
    @staticmethod
    def _colour(image: torch.Tensor, brightness: float, contrast: float, gain: list[float]) -> torch.Tensor:
        gain = torch.tensor(gain, dtype=image.dtype)
        mean = image.mean(dim=(-3, -2, -1), keepdim=True)
        out = (image - mean) * contrast + mean
        return (out * brightness * gain.view(3, 1, 1)).clamp(0, 1)

    @staticmethod
    def _degrade(image: torch.Tensor, kind: str, value: float) -> torch.Tensor:
        if kind == "noise":
            return (image + torch.randn_like(image) * value).clamp(0, 1)
        if kind == "blur":
            size = int(value)
            kernel = torch.zeros(size, size, dtype=image.dtype)
            kernel[size // 2] = 1.0 / size
            flat = image.reshape(-1, 1, *image.shape[-2:])
            out = F.conv2d(F.pad(flat, [size // 2] * 4, mode="replicate"), kernel[None, None])
            return out.reshape(image.shape)
        from PIL import Image
        flat = image.reshape(-1, *image.shape[-3:])
        frames = []
        for frame in flat:
            array = (frame.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="JPEG", quality=int(value))
            decoded = torch.from_numpy(__import__("numpy").asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"),
                                                                    dtype="float32") / 255)
            frames.append(decoded.permute(2, 0, 1))
        return torch.stack(frames).reshape(image.shape).to(image.dtype)

    @staticmethod
    def _affine(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        angle = math.radians(random.uniform(-8, 8))
        scale = random.uniform(0.95, 1.05)
        tx, ty = random.uniform(-12, 12) / (width / 2), random.uniform(-12, 12) / (height / 2)
        theta = torch.tensor([[math.cos(angle) / scale, -math.sin(angle) / scale, tx],
                              [math.sin(angle) / scale, math.cos(angle) / scale, ty]], dtype=image.dtype)
        flat = image.reshape(-1, *image.shape[-3:])
        grid = F.affine_grid(theta.unsqueeze(0).expand(flat.shape[0], -1, -1), flat.shape, align_corners=False)
        return F.grid_sample(flat, grid, mode="bilinear", padding_mode="border", align_corners=False).reshape(image.shape)

    @staticmethod
    def _cutout(image: torch.Tensor) -> torch.Tensor:
        out = image.clone()
        height, width = image.shape[-2:]
        for _ in range(random.choice((1, 2))):
            h, w = int(height * random.uniform(0.05, 0.3)), int(width * random.uniform(0.05, 0.3))
            y, x = random.randrange(0, height - h + 1), random.randrange(0, width - w + 1)
            colour = image.median(dim=-1).values.median(dim=-1).values if random.random() < 0.5 \
                else torch.full(image.shape[:-2], 0.5, dtype=image.dtype)
            out[..., y:y + h, x:x + w] = colour[..., None, None]
        return out

    def __call__(self, item: dict) -> dict:
        self.last_family = None
        if random.random() < self.p_clean:
            return item
        family = random.choice(self.families)
        self.last_family = family
        item = dict(item)
        if family in ("I1", "I2", "I3"):
            item["task"] = self._instruction(family, item["task"])
        elif family == "V1":  # one lighting change, so the same jitter on both cameras
            params = (random.uniform(0.4, 2.0), random.uniform(0.7, 1.3), [random.uniform(0.75, 1.1) for _ in range(3)])
            for key in _images(item):
                item[key] = self._colour(item[key], *params)
        elif family == "V2":
            kind = random.choice(("noise", "blur", "jpeg"))
            value = {"noise": random.uniform(0, 0.1), "blur": random.choice((3, 5, 7, 9, 11)),
                     "jpeg": random.randint(10, 90)}[kind]
            for key in _images(item):
                item[key] = self._degrade(item[key], kind, value)
        elif family == "V3" and WS in item:
            item[WS] = self._affine(item[WS])
        elif family == "V4" and WS in item:
            item[WS] = self._cutout(item[WS])
        elif family == "V6" and WRIST in item:
            item[WRIST] = torch.zeros_like(item[WRIST])
        elif family == "R2" and STATE in item:
            state = item[STATE].clone()
            state[..., :3] += torch.randn(3, dtype=state.dtype) * 0.03
            item[STATE] = state
        return item
