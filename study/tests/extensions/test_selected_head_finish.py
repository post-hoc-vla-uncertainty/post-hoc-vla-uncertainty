"""Restricted scheduling preserves recipes and defers unrelated old-study work."""
from copy import deepcopy
import pytest

from finish_selected_heads import (
    FINAL_ACTION, SELECTED, VELOCITY, finish_in_order, select_recipes, selected_readiness,
)


def test_selection_preserves_recipe_identity_and_order():
    rows = [dict(id=FINAL_ACTION, conditional_on='fixed32'), dict(id='unrelated'), dict(id=VELOCITY)]
    before = deepcopy(rows)
    selected = select_recipes(rows)
    assert [row['id'] for row in selected] == list(SELECTED)
    assert selected[0] is rows[2] and selected[1] is rows[0]
    assert rows == before
    with pytest.raises(ValueError, match='absent'):
        select_recipes(rows[:1])


def test_standalone_adaptive_final_action_retains_prerequisite_metadata():
    request = dict(spec=dict(recipe=dict(id=FINAL_ACTION, conditional_on='fixed32')))
    before = deepcopy(request)
    assert selected_readiness(request, lambda _: 'wait') == 'ready'
    assert request == before
    request['spec']['recipe']['id'] = VELOCITY
    assert selected_readiness(request, lambda _: 'wait') == 'wait'


def test_two_phase_queue_counts_later_jobs_and_continues_after_failure():
    jobs = [dict(spec=dict(round=r, recipe=dict(id=name))) for name in SELECTED for r in (0, 10)]
    calls = []; snapshots = []
    def queue(phase, slots, **kwargs):
        calls.append([j['spec']['recipe']['id'] for j in phase])
        kwargs['on_update']([], phase)
        kwargs['on_update']([], [])
        return ['failure'] if len(calls) == 1 else []
    assert finish_in_order(jobs, ['GPU'], queue,
                           on_update=lambda running, pending: snapshots.append(len(pending))) == ['failure']
    assert calls == [[VELOCITY]*2, [FINAL_ACTION]*2]
    assert snapshots == [4, 2, 2, 0]


def test_resume_with_only_final_action_jobs_does_not_recreate_velocity_jobs():
    jobs = [dict(spec=dict(round=10, recipe=dict(id=FINAL_ACTION)))]
    selected = []
    def queue(phase, slots, **kwargs):
        selected.extend(phase)
        return []
    assert finish_in_order(jobs, [], queue, on_update=lambda *args: None) == []
    assert selected == jobs
    with pytest.raises(ValueError, match='Unexpected'):
        finish_in_order([dict(spec=dict(recipe=dict(id='other')))], [], queue)
