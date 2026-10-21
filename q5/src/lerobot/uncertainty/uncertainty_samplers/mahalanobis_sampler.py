"""Live uncertainty sampler for a frozen post-hoc Mahalanobis detector."""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from lerobot.policies.common.flow_matching.adapter import BaseFlowMatchingAdapter
from lerobot.uncertainty.uncertainty_scoring.mahalanobis import PostHocMahalanobis
from lerobot.uncertainty.uncertainty_scoring.fttgp import append_flow_time
from .uncertainty_sampler import UncertaintySampler

class MahalanobisSampler(UncertaintySampler):
    def __init__(self, checkpoint_path: str, model: BaseFlowMatchingAdapter, num_action_samples: int = 1):
        super().__init__(model=model, num_action_samples=num_action_samples)
        self.method_name = "mahalanobis"
        self.detector = PostHocMahalanobis.load(checkpoint_path, device=self.device)

    @torch.no_grad()
    def _batched_uncertainty(self, observation: Dict[str, Tensor], generator: Optional[torch.Generator] = None):
        batch_size = self._observation_batch_size(observation)
        conditioning = self.model.prepare_conditioning(observation, self.num_action_samples)
        vf = self.model.make_velocity_and_features_fn(conditioning)
        x_t = self.model.sample_prior(batch_size * self.num_action_samples, generator=generator)
        scores = []
        for start, stop in zip(self.sampling_time_grid[:-1], self.sampling_time_grid[1:], strict=True):
            velocity, hidden = vf(x_t=x_t, t=start)
            detector_input = hidden
            if self.detector.input_dim == hidden.shape[-1] + 1:
                detector_input = append_flow_time(hidden, start)
            scores.append(self.detector.squared_distance(detector_input).mean(dim=(-1, -2)))
            x_t = x_t + (stop - start) * velocity
        uncertainty = torch.stack(scores).mean(dim=0).reshape(batch_size, self.num_action_samples).mean(dim=1)
        candidates = x_t.reshape(batch_size, self.num_action_samples, self.horizon, self.action_dim)
        return candidates, uncertainty, batch_size

    def conditional_sample_with_uncertainty_batch(self, observation, generator=None):
        candidates, uncertainty, _ = self._batched_uncertainty(observation, generator)
        actions, _ = self.rand_pick_action_batch(candidates)
        self.action_candidates = candidates.detach().cpu()
        self.uncertainty = float(uncertainty.mean().item())
        return actions, uncertainty.detach().cpu()

    def conditional_sample_with_uncertainty(self, conditioning=None, generator=None, observation=None):
        if conditioning is None:
            conditioning = observation
        actions, uncertainty = self.conditional_sample_with_uncertainty_batch(conditioning, generator)
        return actions, float(uncertainty.item())

    def reset(self):
        self.action_candidates = None
        self.uncertainty = None
