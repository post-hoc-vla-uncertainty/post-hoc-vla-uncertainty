"""Explicit scientific identity and resource settings for extension experiments."""
from dataclasses import dataclass, field
from pathlib import Path
import json

from .common import identity, plain

OBSERVATION_BANKS = ("visual_local", "visual_scene", "language", "state", "joint_intermediate", "joint_late")
BANKS = (*OBSERVATION_BANKS, "visual_patches", "contextual_language", "contextual_vision",
         "contextual_state_intermediate", "contextual_state_late",
         "action_expert", "action_context", "velocity", "action_chunk")
FLOW_BANKS = ("action_expert", "action_context", "velocity", "action_chunk")
ARMS = ("random", "vfd", "gu", "kv", "kv_t", "kv_ivr", "kv_t_ivr")
MODES = ("save_warm_50_50", "pristine_refit")
ELIGIBILITY_BY_MODE = {"save_warm_50_50": "upstream_reference", "pristine_refit": "upstream_reference"}


@dataclass
class FeatureConfig:
    bank: str = "joint_intermediate"
    target_bank: str = "joint_late"
    compression: str = "pca"
    components: int = 512
    seed: int = 42
    geometry_seed: int | None = None
    variance_floor: float = 1e-8
    language_bins: int = 8
    local_grid: int = 4
    scene_grid: int = 2
    expert_distance: int = 0
    expert_context: bool = True
    action_bins: int = 5
    flow_samples: int = 50
    support_flow_samples: int = 4
    reference_flow_samples: int = 4
    flow_indices: tuple[int, ...] = (5,)
    flow_weights: tuple[float, ...] = ()
    flow_seed: int = 1729
    patch_aggregation: str = "top_mean"
    patch_top_fraction: float = .1
    camera_aggregation: str = "max"
    geometry_samples: int = 4096


@dataclass
class GPConfig:
    ridge: float = 1e-3
    jitter: float = 1e-6
    precision: str = "float32"
    output_components: int = 16
    input_components: int = 32
    input_projection: str = "pls"
    learning_rate: float = 0.05
    fit_steps: int = 200
    inference: str = "exact"
    inducing_points: int = 512
    sparse_justification: str | None = None
    output_compression: str = "pca"
    output_dependence: str = "independent"
    coreg_rank: int = 4
    algebra_batch_size: int = 1024
    output_batch_size: int = 16
    checkpoint_every: int = 100
    diagnostic_steps: tuple[int, ...] = (200, 1000, 3000, 10000)
    # Separate input-coordinate repetitions from targets, splits and GP seeds.
    output_seed: int | None = None
    split_seed: int | None = None
    training_seed: int | None = None
    pls_variance_target: float | None = None
    pls_max_components: int = 512
    lengthscale_initialization: str = field(default='geometry', metadata={'omit_default': True})


@dataclass
class ExperimentConfig:
    name: str = "pilot"
    phase: str = "development"
    arm: str = "kv"
    training_mode: str = "save_warm_50_50"
    eligibility_protocol: str | None = None
    seed: int = 1000
    rounds: int = 15
    batch_queries: int = 5
    fit_steps: int = 4000
    refit_steps: int = 4000
    train_batch_size: int = 32
    save_frequency: int = 1000
    num_workers: int = 8
    feature_batch_size: int = 16
    companion: bool = False
    base_models: list[str] = field(default_factory=list)
    data_manifest: str = "storage/extensions/data/data_manifest.json"
    output_root: str = "storage/extensions/runs"
    frozen_recipe: str | None = None
    features: FeatureConfig = field(default_factory=FeatureConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    evaluation_split: str = "development"
    evaluation_max_steps: int = 520
    evaluation_batch_size: int = 15
    evaluation_repetitions: int = 1
    budget_gb: float = 100
    allow_smoke_base: bool = False
    evaluation_per_task: int | None = None

    def __post_init__(self):
        if self.eligibility_protocol is None:
            self.eligibility_protocol = ELIGIBILITY_BY_MODE.get(self.training_mode)

    @property
    def policy_count(self):
        return 2 if self.arm in ("vfd", "gu") or (self.arm == "random" and self.companion) else 1

    @property
    def steps(self):
        return self.fit_steps if self.training_mode == "save_warm_50_50" else self.refit_steps

    def validate(self):
        if self.arm not in ARMS or self.training_mode not in MODES:
            raise ValueError("Unknown acquisition arm or training mode")
        if self.eligibility_protocol not in ("unique_demo", "upstream_reference"):
            raise ValueError("Unknown eligibility protocol")
        if self.features.bank not in BANKS or self.features.target_bank not in BANKS:
            raise ValueError("Unknown feature bank")
        if self.features.compression not in ("pca", "random", "pls", "pca_whiten", "random_whiten", "pls_whiten"):
            raise ValueError("Unknown KV compression")
        if self.gp.input_projection not in ("pls", "pca", "random", "pca_whiten", "random_whiten", "pls_whiten"):
            raise ValueError("Unknown KV-T input projection")
        if self.gp.lengthscale_initialization not in ('geometry', 'variance'):
            raise ValueError('Unknown length-scale initialization')
        if self.gp.lengthscale_initialization == 'variance' and (self.gp.input_projection not in ('pca','pls') or self.gp.output_dependence != 'independent'):
            raise ValueError('Variance initialization requires independent GPs on unwhitened PCA or PLS components')
        if self.features.compression.startswith('pls') and (self.features.bank == 'visual_patches' or feature_uses_flow(self.features)):
            raise ValueError('Target-informed KV is declared only for observation embeddings')
        if self.gp.output_compression not in ('pca','random') or self.gp.output_dependence not in ('independent','coregionalized'):
            raise ValueError('Unknown output compression/dependence')
        if self.gp.output_dependence=='coregionalized' and self.gp.inference!='sparse':
            raise ValueError('Coregionalized comparison uses sparse variational inference; match its independent control')
        if self.features.patch_aggregation not in ('mean','max','top_mean') or self.features.camera_aggregation not in ('mean','max'):
            raise ValueError('Unknown patch/camera aggregation')
        if not 0 < self.features.patch_top_fraction <= 1 or self.features.expert_distance < 0:
            raise ValueError('Invalid patch fraction or expert distance')
        if not self.features.flow_indices or len(set(self.features.flow_indices))!=len(self.features.flow_indices) or min(self.features.flow_indices)<0:
            raise ValueError('Flow indices must be distinct nonnegative indices')
        if self.features.flow_weights and (len(self.features.flow_weights)!=len(self.features.flow_indices) or min(self.features.flow_weights)<0 or sum(self.features.flow_weights)<=0):
            raise ValueError('Invalid flow weights')
        if min(self.features.flow_samples,self.features.support_flow_samples,self.features.reference_flow_samples,
               self.features.action_bins,self.features.geometry_samples,self.gp.coreg_rank,self.gp.algebra_batch_size)<1:
            raise ValueError('Invalid probe or algebra budget')
        if self.features.bank=='visual_patches' and self.arm in ('kv_t','kv_t_ivr','kv_ivr'):
            raise ValueError('Patch heads support KV scoring/SAVE; no undeclared patch targets or max-score IVR')
        if self.gp.inference not in ("exact", "sparse"):
            raise ValueError("GP inference must be exact or sparse")
        if self.gp.inference == "sparse" and not self.gp.sparse_justification:
            raise ValueError("Sparse inference needs a recorded measured scaling justification")
        if self.gp.precision not in ("float32", "float64"):
            raise ValueError("Unsupported kernel precision")
        if min(self.gp.ridge,self.gp.jitter,self.gp.learning_rate,self.features.variance_floor)<=0:
            raise ValueError('GP and normalization scales must be positive')
        if min(self.gp.input_components,self.gp.output_components,self.gp.fit_steps,
               self.gp.inducing_points,self.gp.output_batch_size,self.gp.checkpoint_every,
               self.gp.pls_max_components,self.features.components,self.features.language_bins,
               self.features.local_grid,self.features.scene_grid)<1:
            raise ValueError('Feature and GP dimensions/counts must be positive')
        if self.gp.pls_variance_target is not None and not 0 < self.gp.pls_variance_target <= 1:
            raise ValueError('PLS explained-target-variance criterion must be in (0, 1]')
        if self.gp.pls_max_components > 512:
            raise ValueError('The PLS input dimension is capped at 512')
        if self.companion and self.arm != "random":
            raise ValueError("Companions are only for the common Random score panel")
        if self.phase not in ("smoke", "development", "final"):
            raise ValueError("Unknown experiment phase")
        if self.phase != 'final' and self.evaluation_split != 'development':
            raise ValueError('Development and smoke runs cannot consume final outcomes')
        if self.evaluation_per_task is not None and (self.evaluation_per_task<1 or self.phase=='final'):
            raise ValueError('Per-task evaluation limits are development/preview only')
        if self.evaluation_repetitions != 1:
            raise ValueError('Use the declared gain-panel split for repeated policy-noise evaluations')
        if self.evaluation_batch_size < 1 or self.evaluation_max_steps < 1:
            raise ValueError('Invalid evaluation batch or step limit')
        if self.allow_smoke_base and self.phase != "smoke":
            raise ValueError("Incomplete starting policies are only allowed for explicit smoke checks")
        for value in (self.rounds, self.batch_queries, self.steps, self.train_batch_size,
                      self.save_frequency, self.feature_batch_size, self.budget_gb):
            if value <= 0:
                raise ValueError("Counts and budgets must be positive")
        if self.phase == "final" and (not self.frozen_recipe or self.evaluation_split != "final"):
            raise ValueError("Final runs require a frozen development recipe and final split")
        if self.phase == "final" and (self.rounds != 15 or self.batch_queries != 5 or self.evaluation_max_steps != 520):
            raise ValueError("Main final runs require the declared 15-round/5-demo/520-step recipe")
        if self.phase != "smoke" and self.training_mode == "save_warm_50_50" and self.fit_steps != 4000:
            raise ValueError("SAVE uses 4,000 updates; shorter checks must be labeled smoke")

    def scientific_dict(self):
        data = plain(self)
        for key in ("output_root", "budget_gb", "num_workers", "feature_batch_size", "evaluation_batch_size"):
            data.pop(key)
        return data

    def run_dir(self):
        return Path(self.output_root) / self.name / self.training_mode / self.eligibility_protocol / self.arm / f"seed_{self.seed}"

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(path.read_text())
        else:
            data = json.loads(path.read_text())
        data["features"] = FeatureConfig(**data.get("features", {}))
        data["gp"] = GPConfig(**data.get("gp", {}))
        cfg = cls(**data)
        cfg.validate()
        return cfg


def feature_uses_flow(features):
    return features.bank in FLOW_BANKS or features.target_bank in ('velocity','action_chunk')


def is_target_arm(arm):
    return arm in ('kv_t','kv_t_ivr')
