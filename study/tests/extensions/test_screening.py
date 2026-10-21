from pathlib import Path
import pytest
from uq_extensions.common import atomic_json,read_json,identity,plain
from uq_extensions.config import ExperimentConfig,FeatureConfig,GPConfig
from uq_extensions.runner import validate_frozen
from uq_extensions.screening import freeze


def test_freeze_requires_usable_development_mapping_and_pristine_evidence(tmp_path):
    selection=dict(kv_features=plain(FeatureConfig()),kv_t_features=plain(FeatureConfig(bank='visual_local',target_bank='visual_scene')),
                   gp=plain(GPConfig()),objective='negative_macro_risk_coverage_area',selected_labels=['kv','kv_t'],declaration={'split':'development'})
    atomic_json(tmp_path/'selection.json',selection)
    atomic_json(tmp_path/'screen_results.json',[dict(label='kv',usable=True),dict(label='kv_t',usable=False)])
    with pytest.raises(ValueError):freeze(tmp_path,tmp_path/'frozen.json')
    atomic_json(tmp_path/'screen_results.json',[dict(label='kv',usable=True),dict(label='kv_t',usable=True)])
    freeze(tmp_path,tmp_path/'frozen.json')
    cfg=ExperimentConfig(phase='final',evaluation_split='final',frozen_recipe=str(tmp_path/'frozen.json'))
    validate_frozen(cfg)
    cfg.arm='kv_t'
    with pytest.raises(ValueError,match='settings'):validate_frozen(cfg)
    cfg.features=FeatureConfig(**selection['kv_t_features']);validate_frozen(cfg)
    cfg.training_mode='pristine_refit'
    with pytest.raises(ValueError,match='budget'):validate_frozen(cfg)


def test_condition_verification_cannot_use_final_outcomes(tmp_path):
    from uq_extensions.conditions import validate_condition
    condition=dict(name='language',family='undertrained_language',tasks=['libero_10/3'],instructions={'libero_10/3':'example'})
    evaluation=tmp_path/'evaluation.json';atomic_json(evaluation,dict(spec={'split':'final'}))
    evidence=tmp_path/'verification.json';atomic_json(evidence,dict(condition_hash=identity(condition),meaning_preserved=True,
         physical_feasibility_preserved=True,named_attributes_preserved=True,development_evaluation=str(evaluation)))
    condition['development_verification']=str(evidence)
    with pytest.raises(ValueError,match='development evaluation'):validate_condition(condition,final=True)
