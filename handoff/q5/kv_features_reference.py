"""Reference extraction of the embeddings read by the Q5 heads from a SmolVLA policy.

Derived from the code that produced the paper's KV-Ctx and KV-Dec features (LeRobot
SmolVLA as used by Römer et al.). Call with ONE preprocessed observation per batch:
KV-Dec draws its fixed Gaussian action chunk with seed 0 per call, so batching would
change the noise of all but the first observation.

    policy = SmolVLAPolicy.from_pretrained(path).eval().to(device)
    batch = preprocessor(raw_observation)          # the policy's own preprocessor
    features = head_inputs(policy, batch)          # dict name -> np.ndarray [1, width]

Returned keys (widths for SmolVLA with a 960-d backbone, two cameras, 8-d state):
    kv_ctx      2,888  final prefix states: mean over workspace-camera tokens, mean over
                       wrist-camera tokens, mean over instruction tokens, plus the 8 state values
    kv_dec        480  action-expert final hidden state at Euler evaluation 9 of 10 (one-based),
                       fixed initial chunk (seed 0), mean over the 50 action tokens
    pre_ws        960  workspace-camera tokens after vision encoder + connector, before the transformer
    pre_wrist     960  same for the wrist camera
    pre_lang      960  instruction token embeddings, before the transformer
    pre_state       8  the policy's normalized state input (first 8 coordinates)
    ctx_ws        960  final prefix states, mean over workspace-camera tokens   (part of kv_ctx)
    ctx_wrist     960  final prefix states, mean over wrist-camera tokens       (part of kv_ctx)
    ctx_lang      960  final prefix states, mean over instruction tokens        (part of kv_ctx)
    ctx_state     960  final prefix state at the state token (last prefix position)
"""
import math

import numpy as np
import torch

OBS_LANGUAGE_TOKENS = 'observation.language.tokens'
OBS_LANGUAGE_ATTENTION_MASK = 'observation.language.attention_mask'


def masked_mean(tokens, mask):
    mask = mask.to(tokens.dtype)[..., None]
    return (tokens*mask).sum(1)/mask.sum(1).clamp_min(1)


@torch.no_grad()
def prefix_features(policy, batch):
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    model = policy.model
    vlm = model.vlm_with_expert
    images, image_masks = model.prepare_images(batch)
    state = model.prepare_state(batch)
    language_tokens = batch[OBS_LANGUAGE_TOKENS]
    language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
    # Before the transformer: the same token embeddings embed_prefix places in the prefix.
    cameras = [vlm.embed_image(image) for image in images]
    width = cameras[0].shape[-1]
    pre = [masked_mean(tokens*math.sqrt(width), torch.ones(tokens.shape[:2], device=tokens.device)) for tokens in cameras]
    language = vlm.embed_language_tokens(language_tokens)*math.sqrt(width)
    # After the transformer: final (normalized) prefix hidden states.
    prefix, pad, attention = model.embed_prefix(images, image_masks, language_tokens, language_masks, state=state)
    output, _ = vlm.forward(attention_mask=make_att_2d_masks(pad, attention), position_ids=torch.cumsum(pad, 1)-1,
                            past_key_values=None, inputs_embeds=[prefix, None], use_cache=False, fill_kv_cache=True)
    hidden = output[0]
    # Token positions: per camera [start token] image tokens [end token], then instruction, then state.
    groups, offset = [], 0
    special = model.add_image_special_tokens
    for tokens in cameras:
        offset += model.global_image_start_token.numel() if special else 0
        groups.append((offset, offset+tokens.shape[1]))
        offset += tokens.shape[1]+(model.image_end_token.numel() if special else 0)
    language_group = (offset, offset+language_tokens.shape[1])
    ctx = [masked_mean(hidden[:, a:b], pad[:, a:b]) for a, b in (*groups, language_group)]
    values = dict(
        pre_ws=pre[0], pre_wrist=pre[1], pre_lang=masked_mean(language, language_masks), pre_state=state[:, :8],
        ctx_ws=ctx[0], ctx_wrist=ctx[1], ctx_lang=ctx[2], ctx_state=hidden[:, -1],
        kv_ctx=torch.cat([*ctx, state[:, :8]], 1))
    return {k: v.float().cpu().numpy() for k, v in values.items()}


@torch.no_grad()
def pooled_decoder(policy, batch):
    """KV-Dec input: ninth Euler evaluation, mean over action tokens, initial chunk from seed 0."""
    from lerobot.policies.smolvla.smolvla_adapter import SmolVLAAdapter
    from lerobot.policies.common.flow_matching.ode_solver import make_sampling_time_grid
    adapter = SmolVLAAdapter(policy.config, policy.model)
    conditioning = adapter.prepare_conditioning(batch, num_action_samples=1)
    velocity = adapter.make_velocity_fn(conditioning)
    generator = torch.Generator(device=adapter.device).manual_seed(0)
    x = adapter.sample_prior(1, generator=generator)
    grid = make_sampling_time_grid(step_size=adapter.ode_solver_config['step_size'], device=adapter.device, dtype=adapter.dtype)
    for level, (start, stop) in enumerate(zip(grid[:-1], grid[1:]), start=1):
        hidden = []
        hook = policy.model.action_out_proj.register_forward_pre_hook(lambda module, args: hidden.append(args[0].detach())) if level == 9 else None
        try:
            v = velocity(t=start, x_t=x)
        finally:
            if hook is not None:
                hook.remove()
        if level == 9:
            return hidden[0].mean(dim=1).float().cpu().numpy()
        x = x+(stop-start)*v
    raise RuntimeError('Flow solver has fewer than nine evaluations')


def head_inputs(policy, batch):
    if len(next(v for v in batch.values() if isinstance(v, torch.Tensor))) != 1:
        raise ValueError('Pass one observation per call (fixed KV-Dec noise per observation)')
    return {**prefix_features(policy, batch), 'kv_dec': pooled_decoder(policy, batch)}
