"""Deterministic observation-only feature banks from the pinned SmolVLA."""
import math
from dataclasses import replace
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from .common import atomic_json, atomic_npz, identity, isolated_rng, model_identity, read_json, plain
from .config import OBSERVATION_BANKS, FLOW_BANKS, feature_uses_flow
from .cache_lock import cache_lock

CONTEXT_BANKS = (*OBSERVATION_BANKS, 'contextual_language', 'contextual_vision',
                 'contextual_state_intermediate', 'contextual_state_late')
EXPERT_ALIASES = ('action_expert', 'action_expert_1', 'action_expert_3')


def spatial_pool(tokens, grid):
    side = math.isqrt(tokens.shape[1])
    if side * side != tokens.shape[1]:
        raise ValueError(f"Vision token layout is not a square grid: {tokens.shape}")
    images = tokens.transpose(1, 2).reshape(len(tokens), tokens.shape[2], side, side)
    return F.adaptive_avg_pool2d(images.float(), (grid, grid)).flatten(1)


def masked_mean(tokens, mask):
    mask = mask.to(tokens.dtype)[..., None]
    return (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)


def ordered_language_pool(tokens, mask, bins=8):
    values, empty = [], []
    for row, valid in zip(tokens, mask):
        selected = row[valid.bool()]
        positions = torch.tensor_split(selected, bins)
        values.append(torch.cat([part.float().mean(0) if len(part) else torch.zeros(row.shape[-1], device=row.device) for part in positions]))
        empty.append(torch.tensor([not len(part) for part in positions], device=row.device))
    lengths = mask.sum(1, keepdim=True).float()
    return torch.cat((torch.stack(values), lengths), 1), torch.stack(empty)


def load_policy(path, device="cpu"):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    cfg = PreTrainedConfig.from_pretrained(str(path))
    cfg.device = device
    with isolated_rng(0):
        policy = SmolVLAPolicy.from_pretrained(str(path), config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(path),
                                          preprocessor_overrides={"device_processor": {"device": device}},
                                          postprocessor_overrides={"device_processor": {"device": "cpu"}})
    return policy, pre, post


def preprocess_inputs(observations, preprocessor):
    batch = {"observation.state": torch.as_tensor(np.stack([o["state"] for o in observations]), dtype=torch.float32),
             "task": [str(o["instruction"]) for o in observations]}
    for key in ("image", "image2"):
        batch[f"observation.images.{key}"] = torch.as_tensor(np.stack([o[key] for o in observations])).permute(0, 3, 1, 2).float() / 255
    return preprocessor(batch)


class FeatureExtractor:
    def __init__(self, policy, config):
        self.policy, self.config = policy, config
        self.spec = {}

    @torch.no_grad()
    def extract(self, batch, banks=OBSERVATION_BANKS, *, probe_count=None):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        original_training = self.policy.training
        self.policy.eval()
        model = self.policy.model
        vlm = model.vlm_with_expert
        vision = vlm.get_vlm_model().vision_model
        collected = {"early": [], "late": [], "connected": []}
        hooks = []
        def capture(key):
            def hook(module, inputs, output):
                value = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0] if isinstance(output, tuple) else output
                collected[key].append(value.detach().clone())
            return hook
        try:
            hooks.append(vision.encoder.layers[0].register_forward_hook(capture("early")))
            hooks.append(vision.register_forward_hook(capture("late")))
            hooks.append(vlm.get_vlm_model().connector.register_forward_hook(capture("connected")))
            images, image_masks = model.prepare_images(batch)
            state = model.prepare_state(batch)
            language_tokens = batch[OBS_LANGUAGE_TOKENS]
            language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
            prefix, pad, attention = model.embed_prefix(images, image_masks, language_tokens, language_masks, state=state)
            groups, offset = [], 0
            for value in collected["connected"]:
                offset += model.global_image_start_token.numel() if model.add_image_special_tokens else 0
                groups.append((offset, offset + value.shape[1]))
                offset += value.shape[1] + (model.image_end_token.numel() if model.add_image_special_tokens else 0)
            language_group = (offset, offset + language_tokens.shape[1])
            midpoint = max(0, vlm.num_vlm_layers // 2 - 1)
            middle = []
            def observer(index, hidden):
                if index == midpoint:
                    middle.append(hidden.detach().clone())
            use_flow=feature_uses_flow(self.config)
            output, past = vlm.forward(attention_mask=make_att_2d_masks(pad, attention),
                                     position_ids=torch.cumsum(pad, 1) - 1, past_key_values=None,
                                     inputs_embeds=[prefix, None], use_cache=use_flow, fill_kv_cache=True,
                                     hidden_observer=observer)
            def joint(hidden):
                pooled = [masked_mean(hidden[:, start:end], pad[:, start:end]) for start, end in [*groups, language_group]]
                return torch.cat([*pooled, state[:, :8]], 1).float()
            language, empty = ordered_language_pool(vlm.embed_language_tokens(language_tokens), language_masks, self.config.language_bins)
            values = {
                "visual_local": torch.cat([spatial_pool(v, self.config.local_grid) * mask[:, None] for v, mask in zip(collected["early"], image_masks)], 1),
                "visual_scene": torch.cat([spatial_pool(v, self.config.scene_grid) * mask[:, None] for v, mask in zip(collected["late"], image_masks)], 1),
                "language": language, "state": state[:, :8].float(),
                "joint_intermediate": joint(middle[0]), "joint_late": joint(output[0]),
                "contextual_language": ordered_language_pool(output[0][:, language_group[0]:language_group[1]],language_masks,self.config.language_bins)[0],
                "contextual_vision": torch.cat([masked_mean(output[0][:,a:b],pad[:,a:b]) for a,b in groups],1).float(),
                # embed_prefix appends state last. Unlike image/language tokens,
                # this attention segment can attend to images, language AND state.
                "contextual_state_intermediate": middle[0][:, -1].float(),
                "contextual_state_late": output[0][:, -1].float(),
            }
            if 'visual_patches' in banks:
                if not all(bool(mask.all()) for mask in image_masks):
                    raise ValueError('Patch bank requires valid images for each declared camera')
                values['visual_patches']=torch.stack(collected['early'],1).float()
            if use_flow:
                values.update(self._flow(model,vlm,past,pad,values,banks,probe_count or self.config.flow_samples))
            self.spec = {
                "vision_early": "model.vlm_with_expert.vlm.model.vision_model.encoder.layers.0",
                "vision_late": "model.vlm_with_expert.get_vlm_model().vision_model.last_hidden_state",
                "joint_mid_layer": midpoint, "joint_late": "normalized final prefix hidden state",
                "camera_token_groups": groups, "language_token_group": language_group,
                "dimensions": {k: list(v.shape[1:]) for k, v in values.items()},
                "state_coordinates": 8, "flow_noise_used": use_flow,
                "camera_order": list(model.config.image_features),
                "flow_indices": list(self.config.flow_indices) if use_flow else [],
                "probe_count": (probe_count or self.config.flow_samples) if use_flow else 0,
                "expert_distance": self.config.expert_distance if use_flow else None,
            }
            for key in banks:
                if not torch.isfinite(values[key]).all():
                    raise RuntimeError(f"Nonfinite feature bank: {key}")
            masks = {"language_empty_bins": empty.cpu().numpy(), "language_mask": language_masks.cpu().numpy(),
                     "camera_masks": torch.stack(image_masks, 1).cpu().numpy()}
            return {k: values[k].cpu().float().numpy() for k in banks}, masks
        finally:
            for hook in hooks:
                hook.remove()
            self.policy.train(original_training)

    def _flow(self,model,vlm,past,pad,values,banks,probe_count):
        """One current policy, common Gaussian starts, no simulator/expert actions."""
        count=model.config.num_steps
        indices=tuple(sorted(self.config.flow_indices))
        if max(indices)>=count:
            raise ValueError(f'Flow index exceeds the policy\'s {count} solver steps')
        if tuple(self.config.flow_indices)!=indices:
            raise ValueError('Flow indices must be increasing so weights retain their declared order')
        expert_layers=vlm.get_model_layers([vlm.get_vlm_model().text_model,vlm.lm_expert])[1]
        active=[i for i,layer in enumerate(expert_layers) if layer is not None]
        if self.config.expert_distance>=len(active):
            raise ValueError('Requested expert layer lies before the first expert block')
        selected=active[-1-self.config.expert_distance]
        distances = ({key: (self.config.expert_distance if key == 'action_expert' else int(key.rsplit('_',1)[1]))
                      for key in banks if key in EXPERT_ALIASES})
        result={k:[] for k in banks}
        size=pad.shape[0]; device=pad.device
        for sample in range(probe_count):
            generator=torch.Generator(device=device).manual_seed(self.config.flow_seed+sample)
            initial=torch.randn((1,model.config.chunk_size,model.config.max_action_dim),device=device,generator=generator)
            x=initial.expand(size,-1,-1).clone()
            dt=torch.tensor(-1./count,dtype=torch.float32,device=device)
            flow_time=torch.tensor(1.,dtype=torch.float32,device=device)
            captured_steps=[]
            for step in range(count):
                t=flow_time.expand(size)
                hidden={}
                def observe(index,value):
                    for key,distance in distances.items():
                        if distance and index==active[-1-distance]:hidden[key]=value.detach().clone()
                hook=None
                if step in indices and any(distance == 0 for distance in distances.values()):
                    def final_hook(module,args):
                        for key,distance in distances.items():
                            if distance==0:hidden[key]=args[0].detach().clone()
                    hook=model.action_out_proj.register_forward_pre_hook(final_hook)
                try:
                    velocity=model.denoise_step(pad,past,x,t,
                        expert_hidden_observer=observe if step in indices and any(distances.values()) else None)
                finally:
                    if hook is not None:hook.remove()
                if step in indices:
                    if set(hidden)!=set(distances):raise RuntimeError('Missing declared action-decoder features')
                    context=values['joint_late']
                    raw=x.float().flatten(1)
                    row={'action_context':torch.cat([context,raw,t[:,None]],1),
                         'velocity':velocity[:,:,:7].float().flatten(1)}
                    for key,value in hidden.items():
                        pooled=F.adaptive_avg_pool1d(value.float().transpose(1,2),self.config.action_bins).transpose(1,2).flatten(1)
                        parts=[pooled,raw,t[:,None]]
                        if self.config.expert_context:parts.append(context)
                        row[key]=torch.cat(parts,1)
                    for key in banks:
                        if key not in (*FLOW_BANKS, *EXPERT_ALIASES):
                            # Complete flow conditioning; source modality itself may intentionally be incomplete.
                            conditioning=initial.expand(size,-1,-1).flatten(1) if self.config.target_bank=='action_chunk' else raw
                            row[key]=torch.cat([values[key],conditioning,t[:,None]],1)
                    captured_steps.append(row)
                x=x+dt*velocity
                flow_time=flow_time+dt
            for row in captured_steps:
                row['action_chunk']=x[:,:,:7].float().flatten(1)
                for key in banks:result[key].append(row[key])
        return {key:torch.stack(rows,1) for key,rows in result.items()}


def cache_features(checkpoint, view, config, path, *, device="cpu", batch_size=16, banks=OBSERVATION_BANKS, role="support", support_ids=()):
    requested = tuple(banks)
    aliases = {k:k for k in requested}
    if os.environ.get('UQ_EXT_BUNDLE_FEATURES') == '1':
        if not feature_uses_flow(config):
            banks = ('visual_patches',) if 'visual_patches' in requested else CONTEXT_BANKS
            config = replace(config,bank=banks[0],target_bank='joint_late',expert_distance=0)
        elif config.target_bank == 'action_chunk':
            banks = ('joint_late','action_chunk')
            config = replace(config,bank='joint_late',expert_distance=0)
        else:
            if config.bank == 'action_expert' and config.expert_distance:
                aliases['action_expert'] = f'action_expert_{config.expert_distance}'
            banks = (*EXPERT_ALIASES, 'action_context', 'velocity', 'language', 'visual_scene',
                     'contextual_language', 'contextual_vision', 'contextual_state_late', 'joint_late')
            if not set(aliases.values()) <= set(banks):
                raise ValueError('Unsupported shared flow feature bundle')
            config = replace(config,bank='action_expert',target_bank='velocity',expert_distance=0)
    path = Path(path)
    probes=config.support_flow_samples if role in ('support','hypothetical_support') else config.reference_flow_samples if role=='integration' else config.flow_samples
    extraction=plain(config)
    for field in ('compression','components','seed','geometry_seed','variance_floor','patch_aggregation','patch_top_fraction','camera_aggregation','geometry_samples'):
        extraction.pop(field,None)
    key = identity(dict(checkpoint=model_identity(checkpoint), ids=view.ids,
                        inputs=[c.input_hash for c in view.candidates], config=extraction, banks=banks,
                        role=role, support_ids=sorted(support_ids), probes=probes, feature_code=Path(__file__).read_text()))
    # BEGIN RUN PROGRESS
    from run_progress import Progress,strip_reporting_blocks
    # Recompute the same extraction key excluding reporting-only source additions.
    key = identity(dict(checkpoint=model_identity(checkpoint), ids=view.ids,
                        inputs=[c.input_hash for c in view.candidates], config=extraction, banks=banks,
                        role=role, support_ids=sorted(support_ids), probes=probes, feature_code=strip_reporting_blocks(Path(__file__).read_text())))
    # END RUN PROGRESS
    if os.environ.get('UQ_EXT_FEATURE_CACHE'):
        shared=Path(os.environ['UQ_EXT_FEATURE_CACHE'])/key
        atomic_json(path/'cache_link.json',dict(identity=key,path=str(shared.resolve())))
        path=shared
    with cache_lock(path/'.features.lock',label=f'{role} features'):
        meta = path / "feature_spec.json"
        if meta.exists():
            if read_json(meta)["identity"] != key:
                raise RuntimeError(f"Incompatible feature cache: {path}")
            if 'array_files' in read_json(meta) and not read_json(meta).get('arrays_retired'):
                return {k:np.load(path/read_json(meta)['array_files'][aliases[k]],mmap_mode='r',allow_pickle=False) for k in requested}
            if 'array_files' not in read_json(meta):
                with np.load(path / "features.npz", allow_pickle=False) as a:return {k:a[aliases[k]] for k in requested}
        # BEGIN RUN PROGRESS
        partial=path/'feature_progress.json'
        completed=0
        if partial.exists():
            saved=read_json(partial)
            if saved['identity']!=key:raise RuntimeError('Partial feature cache identity changed')
            completed=saved['completed']
        progress=Progress(f'features {role}: {",".join(banks)}',len(view.ids),initial=completed,unit='observations',every=max(batch_size,64))
        print(f'Feature policy={checkpoint}; probes/observation={probes if feature_uses_flow(config) else 0}; cache={path}',flush=True)
        # END RUN PROGRESS
        policy, preprocessor, _ = load_policy(checkpoint, device)
        extractor = FeatureExtractor(policy, config)
        path.mkdir(parents=True,exist_ok=True)
        arrays,masks={},{}
        if completed:
            extractor.spec=saved['spec']
            for k in banks:
                file=path/(k+'.partial.npy')
                arrays[k]=np.lib.format.open_memmap(file if file.exists() else path/(k+'.npy'),mode='r+')
        started = time.perf_counter()
        with isolated_rng(0):
            for start in range(completed, len(view.ids), batch_size):
                observations = [view.observation(id) for id in view.ids[start:start + batch_size]]
                values, batch_masks = extractor.extract(preprocess_inputs(observations, preprocessor), banks,probe_count=probes)
                for k,value in values.items():
                    if k not in arrays:
                        arrays[k]=np.lib.format.open_memmap(path/(k+'.partial.npy'),mode='w+',dtype=np.float32,shape=(len(view.ids),*value.shape[1:]))
                    arrays[k][start:start+len(value)]=value
                for array in arrays.values():array.flush()
                atomic_npz(path/'mask_chunks'/f'{start:07d}.npz',**batch_masks)
                atomic_json(partial,dict(identity=key,completed=start+len(observations),spec=extractor.spec))
                # BEGIN RUN PROGRESS
                progress.update(start+len(observations))
                # END RUN PROGRESS
        if not arrays:raise ValueError('Cannot extract an empty feature bank')
        for k,array in arrays.items():
            array.flush()
            if (path/(k+'.partial.npy')).exists():(path/(k+'.partial.npy')).replace(path/(k+'.npy'))
        for file in sorted((path/'mask_chunks').glob('*.npz')):
            with np.load(file,allow_pickle=False) as a:
                for k in a.files:masks.setdefault(k,[]).append(a[k])
        if "language_mask" in masks:
            width = max(a.shape[1] for a in masks["language_mask"])
            masks["language_mask"] = [np.pad(a, ((0, 0), (0, width-a.shape[1]))) for a in masks["language_mask"]]
        atomic_npz(path / "masks.npz",ids=np.asarray(view.ids),**{k:np.concatenate(v) for k,v in masks.items()})
        atomic_json(meta, {"identity": key, "role": role, "support_ids": sorted(support_ids),
                           "ids": view.ids, "feature_config": config, **extractor.spec,
                           "seconds": time.perf_counter() - started,"array_files":{k:k+'.npy' for k in arrays}})
        del policy, extractor
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # BEGIN RUN PROGRESS
        progress.finish(f'cached banks={list(arrays)} at {path}')
        # END RUN PROGRESS
        return {k:np.load(path/(aliases[k]+'.npy'),mmap_mode='r',allow_pickle=False) for k in requested}
