"""Render tolerance must not hide a changed simulator condition or stale job."""
from copy import deepcopy
import numpy as np
import pytest

from uq_extensions.common import array_hash, atomic_json, atomic_npz, identity, read_json
from uq_followup.observation_pairing import ObservationPairing, PAIRING_PROTOCOL
from uq_followup.protocol import aggregate_executions


def execution(tmp_path, rep=0, *, tag='learner', change=None):
    arrays = dict(image=np.full((256, 256, 3), 100, np.uint8),
                  image2=np.full((256, 256, 3), 100, np.uint8),
                  state=np.arange(8, dtype=np.float32), instruction=np.asarray('pick up the cup'))
    if change: change(arrays)
    path = tmp_path/f'{tag}_{rep}.npz'; atomic_npz(path, **arrays)
    id_ = f'task/start_0/rep_{rep}'
    return dict(id=id_, task='task', init_state_id=0, init_state_hash='initial-physics', reset_seed=42,
        repetition=rep, condition='original', condition_hash='condition', observation_spec='cameras/state',
        observation_preprocessing='correct', rendering_protocol='per_environment_gl_context_v1',
        max_steps=520, control_hz=20, action_steps=25, flow_steps=10,
        success=rep != 0, q_time=float(rep != 0), q_path=float(rep != 0),
        time_seconds=3., path_meters=1., steps=60,
        initial_input=dict(id=id_, task='task', episode_index=0, observation_file=str(path),
                           input_hash=identity({k: array_hash(v) for k, v in arrays.items()})))


def sparse_roundoff(arrays):
    arrays['image2'].flat[:30] += 2


def test_sparse_render_roundoff_uses_common_unchanged_inputs_and_preserves_outcomes(tmp_path):
    reference = execution(tmp_path, tag='canonical')
    canonical = {('task', 'initial-physics'): reference}
    pairing = ObservationPairing('original_seed0')
    conditions = []
    for policy in ('learner', 'supervisor'):
        records = [execution(tmp_path, rep, tag=policy, change=sparse_roundoff if rep else None) for rep in range(3)]
        original = deepcopy(records)
        result = aggregate_executions(records, canonical=canonical, pairing=pairing, policy_id=policy)
        assert records == original
        row = result[0]; conditions.append(row)
        assert row['success'] == pytest.approx(2/3)
        assert row['execution_successes'] == [False, True, True]
        assert row['initial_input']['observation_file'] == reference['initial_input']['observation_file']
        assert row['initial_input']['id'] == 'task/start_0'
        assert row['execution_initial_inputs'] == [r['initial_input'] for r in records]
        assert row['observation_pairing_protocol'] == PAIRING_PROTOCOL
    assert conditions[0]['initial_input'] == conditions[1]['initial_input']
    summary = pairing.summary()
    assert summary['compared_executions'] == 6
    assert summary['accepted_render_differences'] == 4
    assert summary['observed_max_image_delta'] == 2
    assert summary['observed_max_changed_channel_values'] == 30
    # Reporting's exact learner/supervisor pairing remains intact.
    from uq_followup.reporting import cohort_for
    for name, row in zip(('original_seed0', 'supervisor_seed0'), conditions):
        atomic_json(tmp_path/'evaluations'/name/'conditions.json', [row])
    cohort = cohort_for(tmp_path, 'original_seed0')
    assert cohort['learner'] == cohort['supervisor'] == [pytest.approx(2/3)]


@pytest.mark.parametrize('change,match', [
    (lambda a: a['image2'].flat.__setitem__(0, 103), 'bounds'),
    (lambda a: a['image'].flat.__setitem__(slice(0, 200), 101), 'bounds'),
    (lambda a: a['state'].__setitem__(0, .000001), 'state'),
    (lambda a: a.__setitem__('instruction', np.asarray('pick up another cup')), 'instruction'),
    (lambda a: a.__setitem__('image', a['image'][:128]), 'RGB'),
    (lambda a: a.__setitem__('image', a['image'].astype(np.float32)), 'uint8'),
])
def test_tolerance_rejects_real_changes(tmp_path, change, match):
    reference = execution(tmp_path)
    row = execution(tmp_path, 1, change=change)
    with pytest.raises(ValueError, match=match): ObservationPairing('original_seed0').validate(reference, row)


@pytest.mark.parametrize('field', ['task', 'init_state_hash', 'init_state_id', 'reset_seed',
                                  'condition_hash', 'observation_preprocessing', 'rendering_protocol', 'max_steps'])
def test_identical_images_do_not_bypass_start_and_protocol_checks(tmp_path, field):
    reference = execution(tmp_path)
    row = execution(tmp_path, 1); row[field] = 'changed'
    with pytest.raises(ValueError, match=field): ObservationPairing('original_seed0').validate(reference, row)


def test_actual_observation_checksum_is_verified_even_for_equal_saved_hashes(tmp_path):
    reference = execution(tmp_path)
    row = execution(tmp_path, 1)
    with np.load(row['initial_input']['observation_file']) as data: arrays = {k: data[k] for k in data.files}
    arrays['image2'].flat[0] += 1  # Small enough for tolerance, but not the recorded file!
    atomic_npz(row['initial_input']['observation_file'], **arrays)
    with pytest.raises(ValueError, match='checksum'): ObservationPairing('original_seed0').validate(reference, row)


def test_canonicalization_requires_audit_and_repetitions_remain_strict(tmp_path):
    records = [execution(tmp_path, rep) for rep in range(3)]
    canonical = {('task', 'initial-physics'): records[0]}
    with pytest.raises(ValueError, match='pairing audit'): aggregate_executions(records, canonical=canonical)
    with pytest.raises(ValueError, match='Missing or duplicate'):
        aggregate_executions(records[:2], canonical=canonical, pairing=ObservationPairing('original_seed0'))


def test_completed_jobs_reused_only_for_the_audited_source_transition(tmp_path):
    from uq_followup.launcher import job
    old_declaration = dict(project=str(tmp_path), source_hash='old')
    request = job(tmp_path, old_declaration, 'evaluate', 'evaluate_original_seed0_rep0',
                  policy_id='original_seed0', repetition=0, evaluation_batch_size=15)
    atomic_json(request['path'].with_name('complete.json'), dict(identity=request['request']['identity']))
    atomic_json(tmp_path/'evaluations/original_seed0/rep_0/evaluation.json', dict(episodes=[{}]*500))
    atomic_json(tmp_path/'source_migration.json', dict(protocol=PAIRING_PROTOCOL,
        from_source_hash='old', to_source_hash='new',
        completed_requests={'evaluate_original_seed0_rep0': request['request']['identity']}))
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.json')}
    new_declaration = dict(old_declaration, source_hash='new')
    assert job(tmp_path, new_declaration, 'evaluate', 'evaluate_original_seed0_rep0',
               policy_id='original_seed0', repetition=0, evaluation_batch_size=15) is None
    assert all(p.read_bytes() == data for p, data in before.items())
    for declaration, batch in [(dict(new_declaration, source_hash='future'), 15), (new_declaration, 20)]:
        with pytest.raises(RuntimeError, match='request changed'):
            job(tmp_path, declaration, 'evaluate', 'evaluate_original_seed0_rep0',
                policy_id='original_seed0', repetition=0, evaluation_batch_size=batch)
    (tmp_path/'evaluations/original_seed0/rep_0/evaluation.json').unlink()
    with pytest.raises(RuntimeError, match='Missing artifact'):
        job(tmp_path, new_declaration, 'evaluate', 'evaluate_original_seed0_rep0',
            policy_id='original_seed0', repetition=0, evaluation_batch_size=15)
    assert request['path'].with_name('complete.json').exists()


def test_compatibility_never_authorizes_relabeling_heads(tmp_path):
    from uq_followup.resume_compatibility import reusable_completed_request
    spec = dict(id='head', kind='head', source_hash='old')
    previous = dict(spec=spec, identity=identity(spec))
    new = dict(spec, source_hash='new'); current = dict(spec=new, identity=identity(new))
    atomic_json(tmp_path/'source_migration.json', dict(protocol=PAIRING_PROTOCOL,
        from_source_hash='old', to_source_hash='new', completed_requests={'head': previous['identity']}))
    assert not reusable_completed_request(tmp_path, previous, current)
