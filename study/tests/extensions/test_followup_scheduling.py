"""Head-first ordering and explicit-only aggregate reporting."""
from copy import deepcopy
import sys

import pytest

from uq_extensions.common import atomic_json, read_json
from followup_study import head_first, run


def test_head_family_precedes_policy_and_coordinate_repeats_stay_together():
    recipes = [dict(id='a__kv_random512_s42'), dict(id='a__kv_random512_s42__whiten_lw'),
               dict(id='a__kv_random512_s43'), dict(id='a__kv_random512_s43__whiten_lw'),
               dict(id='b__kv_pca512_s42')]
    policies = [dict(id=f'{condition}_seed{seed}', condition=condition, seed=seed)
                for seed in range(3) for condition in ('original', 'plus35')]
    jobs = [dict(request=dict(spec=dict(policy_id=p['id'], recipe=r))) for p in policies for r in recipes]
    saved = deepcopy(jobs)
    ordered = head_first([None, *jobs], dict(recipes=recipes, policies=policies))
    first = [(j['request']['spec']['policy_id'], j['request']['spec']['recipe']['id']) for j in ordered[:12]]
    assert first == [(f'{condition}_seed{seed}', f'a__kv_random512_s{coordinate}')
                     for condition in ('original', 'plus35') for seed in range(3) for coordinate in (42, 43)]
    assert all('__whiten_lw' in j['request']['spec']['recipe']['id'] for j in ordered[12:24])
    assert all(j['request']['spec']['recipe']['id'] == 'b__kv_pca512_s42' for j in ordered[24:])
    assert jobs == saved and {id(j) for j in ordered} == {id(j) for j in jobs}


@pytest.mark.parametrize('interrupt', [False, True])
def test_run_orders_heads_without_rebuilding_reports_and_restores_hooks(tmp_path, monkeypatch, interrupt):
    from uq_followup import launcher, reporting
    recipes = [dict(id='a__kv_pca512_s42'), dict(id='b__kv_pca512_s42')]
    policies = [dict(id=f'original_seed{seed}', condition='original', seed=seed) for seed in (0, 1)]
    declaration = dict(recipes=recipes, policies=policies)
    atomic_json(tmp_path/'declaration.json', declaration)
    jobs = [dict(request=dict(spec=dict(policy_id=p['id'], recipe=r))) for p in policies for r in recipes]
    drained = []; refreshed = []
    def drain(root, items, slots, stage):
        drained.append((stage, items))
        reporting.refresh_index(root)
        return dict(failures=[], deferred=[])
    def report(_): pytest.fail('An automatic aggregate-report rebuild was attempted')
    def refresh(root): refreshed.append(root)
    monkeypatch.setattr(launcher, 'drain', drain)
    monkeypatch.setattr(reporting, 'aggregate_report', report)
    monkeypatch.setattr(reporting, 'refresh_index', refresh)
    args = ['--gpus', '4,5,6,7', '--output', str(tmp_path)]
    def main():
        assert sys.argv[1:] == args
        for stage in ('policy-evaluation', 'observation-heads', 'action-heads'):
            launcher.drain(tmp_path, jobs, [], stage)
            assert 'not regenerated automatically' in reporting.aggregate_report(tmp_path)
        if interrupt: raise KeyboardInterrupt()
    monkeypatch.setattr(launcher, 'main', main)
    argv = sys.argv
    if interrupt:
        with pytest.raises(KeyboardInterrupt): run(args)
    else: run(args)
    assert drained[0][1] is jobs
    assert drained[1][1] == drained[2][1] == head_first(jobs, declaration)
    assert refreshed == [tmp_path]*3
    assert read_json(tmp_path/'scheduling.json')['aggregate_report'] == 'on_request_only'
    assert launcher.drain is drain and reporting.aggregate_report is report
    assert reporting.refresh_index is refresh and sys.argv is argv


def test_report_only_calls_real_report_without_patching_scheduler(tmp_path, monkeypatch):
    from uq_followup import launcher, reporting
    drain = launcher.drain
    reports = []
    def report(root): reports.append(root)
    def main():
        assert sys.argv[1:] == ['--report-only', '--output', str(tmp_path)]
        assert launcher.drain is drain and reporting.aggregate_report is report
        reporting.aggregate_report(tmp_path)
    monkeypatch.setattr(reporting, 'aggregate_report', report)
    monkeypatch.setattr(launcher, 'main', main)
    argv = sys.argv
    run(['--report-only', '--output', str(tmp_path)])
    assert reports == [tmp_path] and sys.argv is argv
