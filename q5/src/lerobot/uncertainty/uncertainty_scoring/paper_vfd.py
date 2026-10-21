"""Velocity-field disagreement from Römer et al. (2026), Algorithm 2."""

from __future__ import annotations

import torch
from torch import Tensor

from lerobot.policies.common.flow_matching.ode_solver import make_sampling_time_grid


@torch.no_grad()
def paper_vfd(
    first_adapter,
    second_adapter,
    observation: dict[str, Tensor],
    num_action_samples: int = 5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Compute the paper's two-member, two-directed-path VFD per observation.

    Each ensemble member starts from independent Gaussian noise. At every Euler
    state on each member's path, both vector fields are evaluated at that same
    state. Squared disagreement is weighted by κₛ=s/(1-s), averaged over the
    two directed paths, the action samples, and the integration steps.
    """
    if num_action_samples <= 0:
        raise ValueError("num_action_samples must be positive.")
    if first_adapter.horizon != second_adapter.horizon or first_adapter.action_dim != second_adapter.action_dim:
        raise ValueError("VFD ensemble members must have equal horizon and action dimensions.")
    tensor_values = [value for value in observation.values() if isinstance(value, Tensor)]
    if not tensor_values:
        raise ValueError("observation must contain at least one tensor.")
    batch_size = int(tensor_values[0].shape[0])

    cond_1 = first_adapter.prepare_conditioning(observation, num_action_samples)
    cond_2 = second_adapter.prepare_conditioning(observation, num_action_samples)
    vf_1 = first_adapter.make_velocity_fn(cond_1)
    vf_2 = second_adapter.make_velocity_fn(cond_2)
    sample_count = batch_size * num_action_samples
    state_1 = first_adapter.sample_prior(sample_count, generator=generator)
    state_2 = second_adapter.sample_prior(sample_count, generator=generator)
    grid = make_sampling_time_grid(
        step_size=first_adapter.ode_solver_config["step_size"],
        device=first_adapter.device,
        dtype=first_adapter.dtype,
    )
    num_steps = len(grid) - 1
    score = torch.zeros(batch_size, device=first_adapter.device, dtype=first_adapter.dtype)

    for start, stop in zip(grid[:-1], grid[1:], strict=True):
        own_1 = vf_1(t=start, x_t=state_1)
        other_at_1 = vf_2(t=start, x_t=state_1)
        own_2 = vf_2(t=start, x_t=state_2)
        other_at_2 = vf_1(t=start, x_t=state_2)
        directed_1 = (own_1 - other_at_1).square().sum(dim=(1, 2))
        directed_2 = (other_at_2 - own_2).square().sum(dim=(1, 2))
        expected = (directed_1 + directed_2).reshape(batch_size, num_action_samples).mean(dim=1)
        kappa = start / (1.0 - start)
        score += kappa * expected / (2.0 * num_steps)
        state_1 = state_1 + (stop - start) * own_1
        state_2 = state_2 + (stop - start) * own_2
    return score
