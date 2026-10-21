"""Single-policy uncertainty sampling with an external post-hoc FTTGP head."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.uncertainty.uncertainty_scoring.fttgp import PostHocFTTGP, append_flow_time

from .configuration_uncertainty_sampler import FTTGPSamplerConfig
from .uncertainty_sampler import UncertaintySampler


class FTTGPSampler(UncertaintySampler):
    """Run the unchanged policy flow while scoring its hidden features with FTTGP."""

    def __init__(self, config: FTTGPSamplerConfig, model: BaseFlowMatchingAdapter):
        super().__init__(model=model, num_action_samples=config.num_action_samples)
        if not config.checkpoint_path:
            raise ValueError("FTTGP requires a non-empty checkpoint_path.")
        if self.ode_solver_config["solver_method"] != "euler":
            raise ValueError("FTTGP hidden-feature collection currently supports Euler sampling only.")
        self.method_name = "fttgp"
        self.config = config
        self.gp = None
        self.level_gps = None
        self.secondary_gp = None
        self.calibration = None
        if config.variant == "flow_levelwise":
            checkpoint_dir = Path(config.checkpoint_path)
            self.level_gps = [
                PostHocFTTGP.load(checkpoint_dir / f"head_level_{level}.pt", device=self.device)
                for level in range(len(self.sampling_time_grid) - 1)
            ]
        else:
            self.gp = PostHocFTTGP.load(config.checkpoint_path, device=self.device)
        if config.variant == "flow_structured_fused":
            self.secondary_gp = PostHocFTTGP.load(config.secondary_checkpoint_path, device=self.device)
            self.calibration = json.loads(Path(config.calibration_path).read_text())
        self.level_uncertainties: Tensor | None = None

    @torch.no_grad()
    def _modality_embeddings(self, observation: Dict[str, Tensor]) -> dict[str, Tensor]:
        policy_model = self.model.model
        images, masks = policy_model.prepare_images(observation)
        camera_embeddings = []
        for image, mask in zip(images, masks, strict=True):
            tokens = policy_model.vlm_with_expert.embed_image(image)
            tokens = tokens * math.sqrt(tokens.shape[-1])
            valid = mask[:, None, None].to(tokens)
            camera_embeddings.append((tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0))
        language = policy_model.vlm_with_expert.embed_language_tokens(
            observation["observation.language.tokens"]
        )
        language = language * math.sqrt(language.shape[-1])
        language_mask = observation["observation.language.attention_mask"].to(language).unsqueeze(-1)
        language = (language * language_mask).sum(dim=1) / language_mask.sum(dim=1).clamp_min(1.0)
        state = policy_model.state_proj(policy_model.prepare_state(observation))
        if state.ndim == 3:
            state = state.mean(dim=1)
        return {
            "external_camera": camera_embeddings[0], "wrist_camera": camera_embeddings[1],
            "language": language, "state": state,
        }

    def _structured_features(self, observation: Dict[str, Tensor]) -> Tensor:
        blocks = self._modality_embeddings(observation)
        return torch.cat(
            [blocks[name] for name in ("external_camera", "wrist_camera", "language", "state")], dim=-1
        )

    def _vlm_features(self, observation: Dict[str, Tensor]) -> Tensor:
        # This exactly matches the earlier masked-mean VLM head: a token-count-weighted
        # mean of the two image-token blocks and language-token block.
        policy_model = self.model.model
        images, image_masks = policy_model.prepare_images(observation)
        token_sums, token_counts = [], []
        for image, image_mask in zip(images, image_masks, strict=True):
            tokens = policy_model.vlm_with_expert.embed_image(image) * math.sqrt(
                policy_model.vlm_with_expert.config.text_config.hidden_size
            )
            valid = image_mask[:, None, None].to(tokens)
            token_sums.append((tokens * valid).sum(dim=1))
            token_counts.append(valid.sum(dim=1))
        language_tokens = policy_model.vlm_with_expert.embed_language_tokens(
            observation["observation.language.tokens"]
        )
        language_tokens = language_tokens * math.sqrt(language_tokens.shape[-1])
        language_mask = observation["observation.language.attention_mask"].to(language_tokens).unsqueeze(-1)
        token_sums.append((language_tokens * language_mask).sum(dim=1))
        token_counts.append(language_mask.sum(dim=1))
        return sum(token_sums) / sum(token_counts).clamp_min(1.0)

    @staticmethod
    def _calibrated_z(values: Tensor, calibration: dict[str, float]) -> Tensor:
        return (values - float(calibration["mean"])) / max(float(calibration["std"]), 1e-12)

    def _aggregate_levels(self, level_scores: Tensor, times: Tensor) -> Tensor:
        # level_scores shape: (levels, batch)
        if self.config.aggregation == "max":
            return level_scores.max(dim=0).values
        if self.config.aggregation == "paper_weighted":
            weights = times / (1.0 - times).clamp_min(1e-3)
            weights = weights / weights.sum().clamp_min(1e-12)
            return (level_scores * weights[:, None]).sum(dim=0)
        return level_scores.mean(dim=0)

    @torch.no_grad()
    def _batched_uncertainty(
        self,
        observation: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, int]:
        batch_size = self._observation_batch_size(observation)
        conditioning = self.model.prepare_conditioning(observation, self.num_action_samples)
        velocity_and_features_fn = self.model.make_velocity_and_features_fn(conditioning)
        x_t = self.model.sample_prior(
            num_samples=batch_size * self.num_action_samples, generator=generator
        )

        static_uncertainty = None
        if self.config.variant == "structured":
            static_uncertainty = self.gp.predictive_variance(self._structured_features(observation))
        elif self.config.variant == "vlm_mean":
            static_uncertainty = self.gp.predictive_variance(self._vlm_features(observation))
        elif self.config.variant == "flow_structured_fused":
            structured = self.secondary_gp.predictive_variance(self._structured_features(observation))

        level_scores = []
        level_times = []
        for start, stop in zip(self.sampling_time_grid[:-1], self.sampling_time_grid[1:], strict=True):
            velocity, hidden = velocity_and_features_fn(x_t=x_t, t=start)
            if self.config.variant in {"flow_shared", "flow_structured_fused"}:
                gp_input = append_flow_time(hidden, start) if self.config.include_flow_time else hidden
                token_variance = self.gp.predictive_variance(gp_input)
                observation_variance = token_variance.mean(dim=-1).reshape(
                    batch_size, self.num_action_samples
                ).mean(dim=1)
                level_scores.append(observation_variance)
            elif self.config.variant == "flow_levelwise":
                token_variance = self.level_gps[len(level_scores)].predictive_variance(hidden)
                observation_variance = token_variance.mean(dim=-1).reshape(
                    batch_size, self.num_action_samples
                ).mean(dim=1)
                level_scores.append(observation_variance)
            level_times.append(start)
            x_t = x_t + (stop - start) * velocity

        if static_uncertainty is not None:
            uncertainty = static_uncertainty
            self.level_uncertainties = None
        else:
            stacked_scores = torch.stack(level_scores)
            times = torch.stack(level_times)
            uncertainty = self._aggregate_levels(stacked_scores, times)
            self.level_uncertainties = stacked_scores.detach().cpu()
            if self.config.variant == "flow_structured_fused":
                uncertainty = torch.maximum(
                    self._calibrated_z(uncertainty, self.calibration["flow_shared"]),
                    self._calibrated_z(structured, self.calibration["structured"]),
                )
        action_candidates = x_t.reshape(
            batch_size, self.num_action_samples, self.horizon, self.action_dim
        )
        return action_candidates, uncertainty, batch_size

    def conditional_sample_with_uncertainty_batch(
        self,
        observation: Dict[str, Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor]:
        candidates, uncertainty, _ = self._batched_uncertainty(observation, generator)
        actions, _ = self.rand_pick_action_batch(candidates)
        self.action_candidates = candidates.detach().cpu()
        self.uncertainty = float(uncertainty.mean().item())
        return actions, uncertainty.detach().cpu()

    def conditional_sample_with_uncertainty(
        self,
        conditioning: Dict[str, Tensor] | None = None,
        generator: Optional[torch.Generator] = None,
        observation: Dict[str, Tensor] | None = None,
    ) -> Tuple[Tensor, float]:
        if conditioning is None:
            conditioning = observation
        if conditioning is None:
            raise ValueError("FTTGP requires an observation/conditioning dictionary.")
        actions, uncertainty = self.conditional_sample_with_uncertainty_batch(conditioning, generator)
        if uncertainty.numel() != 1:
            raise ValueError("Single-observation FTTGP API received a batched observation.")
        return actions, float(uncertainty.item())

    def reset(self):
        self.action_candidates = None
        self.uncertainty = None
        self.level_uncertainties = None
