import numpy as np
import pytest
from uq_extensions.rollout_metrics import episode_metrics,learning_summary
from uq_extensions.referral import referral_curve,pair_outcomes,score_metrics


def test_physical_path_and_failure_efficiency():
    points=np.array([[0,0,0],[3,0,0],[3,4,0]])
    m=episode_metrics(success=True,steps=20,positions=points,control_hz=10,normalizer={'mean_time':1,'mean_path':3.5})
    assert m['path_meters']==7 and m['time_seconds']==2 and m['q_path']==.5
    m=episode_metrics(success=False,steps=1,positions=points,control_hz=10)
    assert m['q_time']==m['q_path']==0 and m['relative_time'] is None


def test_referral_signed_gain_and_complementarity():
    # Perfect knowledge of supervisor benefit; learner is better on first half.
    score=[0,0,1,1];l=[1,1,0,0];h=[0,0,1,1]
    curve,result=referral_curve(score,l,h,['a']*4,['0','1','2','3'])
    assert result['raw_gain']>0 and result['normalized_gain'] is None
    _,bad=referral_curve(score[::-1],l,h,['a']*4,['0','1','2','3'])
    assert bad['raw_gain']<0
    assert curve[0]['utility']==.5 and curve[-1]['utility']==.5


def test_referral_equal_task_weighting_and_ties():
    a=referral_curve([1]*6,[0,0,1,1,1,1],[1]*6,['a']*2+['b']*4,list('abcdef'),allocation='within_task')
    assert a[1]['learner_mean']==pytest.approx(.5)
    assert a==referral_curve([1]*6,[0,0,1,1,1,1],[1]*6,['a']*2+['b']*4,list('abcdef'),allocation='within_task')


def test_pair_rejects_different_execution_or_initial_states():
    row=dict(id='1',task='a',init_state_hash='x',reset_seed=1,repetition=0,condition='original',split='final',success=1,max_steps=520)
    assert len(pair_outcomes([row],[row]))==1
    with pytest.raises(ValueError,match='setting'):pair_outcomes([row],[dict(row,max_steps=100)])
    with pytest.raises(ValueError,match='match'):pair_outcomes([row],[dict(row,reset_seed=2)])


def test_undefined_discrimination_and_thresholds():
    m=score_metrics([1,2,3,4],[1,1,0,1],['a','a','b','b'],initial_pool_task_means={'a':1.5,'b':3.5})
    assert m['within_task']['a']['auroc'] is None and m['valid_tasks_auroc']==1
    summary=learning_summary([dict(new_demonstrations=0,success=.2),dict(new_demonstrations=5,success=.5)])
    assert summary['area']==pytest.approx(.35) and summary['demonstrations_to_60'] is None


def test_cluster_bootstrap_retains_signed_referral_gains():
    from uq_extensions.referral import clustered_referral_interval
    low,high=clustered_referral_interval([0,0,1,1],[1,1,0,0],[0,0,1,1],['a']*4,['x','x','y','y'],repetitions=30)
    assert 0<=low<=high


def test_simulator_images_match_upstream_but_dataset_images_are_not_rotated():
    import torch
    from lerobot.processor.env_processor import LiberoProcessorStep
    from uq_extensions.evaluation import initial_observation
    from uq_extensions.features import preprocess_inputs
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    raw = dict(pixels=dict(image=image, image2=image + 100), robot_state=dict(
        eef=dict(pos=np.array([.1, .2, .3]), quat=np.array([0., 0., 0., 1.])),
        gripper=dict(qpos=np.array([.04, -.04]))))
    converted = initial_observation(raw, 'move the mug')
    np.testing.assert_allclose(converted['state'], [.1, .2, .3, 0, 0, 0, .04, -.04])
    assert str(converted['instruction']) == 'move the mug'
    for key in ('image', 'image2'):
        np.testing.assert_array_equal(converted[key], raw['pixels'][key][::-1, ::-1])
        assert converted[key].flags.c_contiguous
    batch = {f'observation.images.{k}': torch.as_tensor(v).permute(2, 0, 1)[None].float()/255
             for k, v in raw['pixels'].items()}
    expected = LiberoProcessorStep().observation(batch)
    processed = preprocess_inputs([converted], lambda b: b)
    for key in batch:
        torch.testing.assert_close(processed[key], expected[key])
    # Training/demonstration inputs use the shared preprocessor directly and must
    # retain their already-canonical orientation.
    dataset = dict(converted, image=image)
    tensor = preprocess_inputs([dataset], lambda b: b)['observation.images.image']
    torch.testing.assert_close(tensor, torch.as_tensor(image).permute(2, 0, 1)[None].float()/255)


def test_referral_rejects_old_and_new_camera_preprocessing():
    from uq_extensions.evaluation import OBSERVATION_PREPROCESSING
    old = dict(id='1', task='a', init_state_hash='x', reset_seed=1, repetition=0,
               condition='original', split='development', success=1)
    new = dict(old, observation_preprocessing=OBSERVATION_PREPROCESSING)
    with pytest.raises(ValueError, match='observation_preprocessing'):
        pair_outcomes([old], [new])
    with pytest.raises(ValueError, match='observation_preprocessing'):
        pair_outcomes([dict(new, observation_preprocessing='v1')], [new])


def test_supervisor_cache_separates_preprocessing_versions(monkeypatch):
    import uq_extensions.panels as panels
    from uq_extensions.common import identity
    from pathlib import Path
    kwargs = dict(split='development', condition=None, model='/tmp/teacher',
                  manifest='m', max_steps=520, per_task=2)
    legacy = identity(dict(kwargs, model=str(Path(kwargs['model']).resolve())))[:20]
    current = panels.supervisor_cache_key(**kwargs)
    assert current != legacy
    monkeypatch.setattr(panels, 'OBSERVATION_PREPROCESSING', 'future-version')
    assert panels.supervisor_cache_key(**kwargs) != current


def test_referral_and_caches_separate_rendering_protocols(monkeypatch):
    import uq_extensions.panels as panels
    row = dict(id='1', task='a', init_state_hash='x', reset_seed=1, repetition=0,
               condition='original', split='development', success=1)
    with pytest.raises(ValueError, match='rendering_protocol'):
        pair_outcomes([row], [dict(row, rendering_protocol=panels.RENDERING_PROTOCOL)])
    kwargs = dict(split='development', condition=None, model='/tmp/teacher',
                  manifest='m', max_steps=520, per_task=2)
    current = panels.supervisor_cache_key(**kwargs)
    monkeypatch.setattr(panels, 'RENDERING_PROTOCOL', 'old-context')
    assert panels.supervisor_cache_key(**kwargs) != current


def test_evaluation_refuses_stale_cache_before_loading_policy(tmp_path, monkeypatch):
    import uq_extensions.evaluation as evaluation
    from uq_extensions.common import atomic_json
    atomic_json(tmp_path/'evaluation.json', {'identity': 'legacy-unrotated', 'episodes': []})
    monkeypatch.setattr(evaluation, 'model_identity', lambda _: 'checkpoint')
    monkeypatch.setattr(evaluation, 'load_policy', lambda *a: pytest.fail('Loaded policy before rejecting stale cache'))
    manifest = dict(manifest_hash='m', evaluation_splits={'development': []})
    with pytest.raises(RuntimeError, match='cache identity mismatch'):
        evaluation.evaluate('checkpoint', manifest, tmp_path)
