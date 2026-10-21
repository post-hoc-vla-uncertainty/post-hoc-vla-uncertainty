"""Scientific contracts for the isolated three-seed follow-up study."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch

from uq_extensions.common import atomic_json, read_json
from uq_extensions.config import FeatureConfig, GPConfig
from uq_followup.design import recipes, MAPPINGS
from uq_followup.protocol import sampled_additions, policy_specs, evaluation_manifest, aggregate_executions


def manifest_fixture(tmp_path):
    originals = [f'libero_10/{t}/episode_{i}' for t in range(3) for i in range(6)]
    pool = originals + [f'libero_10/{t}/episode_{i}' for t in range(3, 10) for i in range(12)]
    ids = ['libero_spatial/0/episode_1000', *pool]
    original = ['libero_spatial/0/episode_1000', *originals]
    rows = [dict(id=id, dataset_from_index=i*10, dataset_to_index=i*10+10) for i, id in enumerate(ids)]
    atomic_json(tmp_path/'trajectory_index.json', rows)
    starts = [dict(id=f'libero_10/{t}/start_{i}/rep_0', task=f'libero_10/{t}',
                   init_state_id=i, init_state_hash=f'{t}-{i}', repetition=0, reset_seed=1000+t*50+i)
              for t in range(10) for i in range(50)]
    return dict(pretraining_ids=original, target_ids=pool, episode_map=dict(zip(ids, range(len(ids)))),
        evaluator_directory=str(tmp_path), manifest_hash='source',
        evaluation_splits=dict(development=[s for s in starts if s['init_state_id'] < 20],
                               final=[s for s in starts if s['init_state_id'] >= 20]))


def test_matrix_counts_and_agreed_omissions():
    rows = recipes()
    from collections import Counter
    assert Counter(r['method'] for r in rows)['kv'] == 100
    assert Counter(r['method'] for r in rows)['kv_t'] == 26
    assert len(rows) == 133
    assert not {'visual_patches', 'state', 'expert_1', 'expert_3'} & {r['group'] for r in rows}
    for r in rows:
        if r['method'] == 'kv_t':
            assert r['group'] in MAPPINGS
            assert r['gp']['precision'] == 'float64'
            assert r['gp']['output_batch_size'] == 16
            assert r['gp']['output_dependence'] == 'independent'
        if r['method'] in ('kv', 'kv_t') and 'adaptive' not in r['variant']:
            assert r['features']['components'] == 512
    assert sum(r.get('gp', {}).get('output_compression') == 'random' for r in rows) == 1


def test_new_demonstrations_equal_epochs_and_independent_seed_sampling(tmp_path):
    manifest = manifest_fixture(tmp_path)
    a = sampled_additions(manifest, 0); b = sampled_additions(manifest, 1)
    assert len(a) == len(set(a)) == 35 and a != b
    assert not set(a) & set(manifest['pretraining_ids'])
    assert all(sum(i.startswith(f'libero_10/{t}/') for i in a) == 5 for t in range(3, 10))
    specs = policy_specs(manifest)
    assert len(specs) == 9
    for p in specs:
        assert 0 <= p['effective_epochs']-p['target_epochs'] < 32/p['frames']+1e-9
        if p['condition'] == 'plus35':
            assert set(manifest['pretraining_ids']) <= set(p['ids'])
            assert len(p['ids']) == len(manifest['pretraining_ids'])+35


def test_all_fifty_starts_three_noise_streams_same_reset(tmp_path):
    m = evaluation_manifest(manifest_fixture(tmp_path))
    assert list(m['evaluation_splits']) == [f'study_rep{i}' for i in range(3)]
    splits = list(m['evaluation_splits'].values())
    assert all(len(s) == 500 for s in splits)
    for a, b, c in zip(*splits):
        assert a['reset_seed'] == b['reset_seed'] == c['reset_seed']
        assert a['init_state_hash'] == b['init_state_hash'] == c['init_state_hash']
        assert len({a['id'], b['id'], c['id']}) == 3


def test_average_repetitions_before_hindsight_referral():
    records = []
    for rep, success in enumerate((0, 1, 1)):
        records.append(dict(id=f't/start_0/rep_{rep}', task='t', init_state_hash='s', init_state_id=0,
            repetition=rep, success=success, q_time=float(success), q_path=float(success),
            time_seconds=3., path_meters=1., steps=60,
            initial_input=dict(input_hash='identical', id=f't/start_0/rep_{rep}')))
    result = aggregate_executions(records)
    assert len(result) == 1 and result[0]['success'] == pytest.approx(2/3)
    assert result[0]['id'] == result[0]['initial_input']['id'] == 't/start_0'
    records[1]['initial_input']['input_hash'] = 'different'
    with pytest.raises(ValueError, match='identical'): aggregate_executions(records)


def test_raw_decoder_target_excludes_appended_conditioning():
    from uq_followup.features import FeatureExtractor
    model = SimpleNamespace(config=SimpleNamespace(num_steps=2, chunk_size=2, max_action_dim=2),
        action_out_proj=torch.nn.Linear(3, 2, bias=False))
    def denoise(pad, past, x, t, **kwargs):
        hidden = torch.cat([x, t[:, None, None].expand(-1, 2, 1)], -1)
        return model.action_out_proj(hidden)
    model.denoise_step = denoise
    vlm = SimpleNamespace(get_model_layers=lambda _: ([], [object()]),
                          get_vlm_model=lambda: SimpleNamespace(text_model=None), lm_expert=None)
    f = FeatureConfig(bank='joint_early_flow', target_bank='decoder_hidden', flow_indices=(0, 1), action_bins=2)
    extractor = FeatureExtractor(None, f)
    contexts = dict(joint_early=torch.ones(2, 5), joint_intermediate=torch.ones(2, 5)*2, joint_late=torch.ones(2, 5)*3)
    values = extractor._flow(model, vlm, None, torch.ones(2, 2), contexts,
        ('joint_early_flow', 'joint_intermediate_flow', 'decoder_hidden', 'action_expert'), 2)
    assert values['decoder_hidden'].shape == (2, 4, 6)
    assert values['joint_early_flow'].shape == (2, 4, 10)
    assert values['action_expert'].shape == (2, 4, 16)
    torch.testing.assert_close(values['action_expert'][:, :, :6], values['decoder_hidden'])
    torch.testing.assert_close(values['joint_early_flow'][:, :, 5:], values['joint_intermediate_flow'][:, :, 5:])
    assert not torch.equal(values['decoder_hidden'][:, 0], values['decoder_hidden'][:, 2])


@pytest.mark.parametrize('variant', ['pca512', 'pca_adaptive', 'pls512', 'pls_adaptive', 'random512'])
def test_matched_kv_and_kvt_share_full_support_geometry(variant, tmp_path):
    from uq_followup.estimators import Estimator, FollowupTargetGP
    rng = np.random.default_rng(3); x = rng.normal(size=(24, 7)); y = x[:, :3]**2 + x[:, 3:6]
    row = deepcopy(next(r for r in recipes() if r['group']=='middle_to_final_context' and r['variant']==variant and r['method']=='kv' and not r['whiten']))
    row['gp'].update(input_components=5, output_components=3, pls_max_components=5)
    row['features']['components'] = 5
    fit = Estimator(row, 'cpu').fit(x, y, ['a']*12+['b']*12, tmp_path)
    target = FollowupTargetGP(FeatureConfig(**row['features']), GPConfig(**row['gp']), variant=variant)
    projection, _, _ = target._geometry(x, y)
    np.testing.assert_allclose(fit.model.projection.transform(x), projection.transform(x), atol=1e-6)
    assert np.isfinite(fit.score(x)).all()
    loaded = Estimator.load(row, tmp_path, 'cpu')
    np.testing.assert_allclose(loaded.score(x), fit.score(x), atol=1e-6)


def test_kvt_small_batched_fit_and_resume_artifact(tmp_path):
    from uq_followup.estimators import Estimator
    row = deepcopy(next(r for r in recipes() if r['id']=='middle_to_final_context__kv_t_pca512_s42'))
    row['gp'].update(fit_steps=3, input_components=4, output_components=2, checkpoint_every=1, diagnostic_steps=(1, 3))
    rng = np.random.default_rng(2); x = rng.normal(size=(24, 6)); y = x[:, :2]
    fit = Estimator(row, 'cpu').fit(x, y, ['a']*12+['b']*12, tmp_path)
    assert fit.diagnostics['batched_training']
    assert fit.diagnostics['mapping_check']['grouped_by_episode']
    loaded = Estimator.load(row, tmp_path, 'cpu')
    np.testing.assert_allclose(loaded.score(x), fit.score(x), atol=1e-8)


def test_conditioning_survives_embedding_projection_and_matches_kv(tmp_path):
    from uq_followup.estimators import FollowupTargetGP, Estimator
    rng = np.random.default_rng(4); x = rng.normal(size=(30, 12)); y = x[:, -2:] + x[:, :2]
    f = FeatureConfig(compression='random', components=4)
    g = GPConfig(input_projection='random', input_components=4, output_components=2)
    estimator = FollowupTargetGP(f, g, variant='random512', conditioning_columns=3)
    projection, _, _ = estimator._geometry(x, y)
    z = projection.transform(x)
    assert z.shape == (30, 7)
    np.testing.assert_allclose(z[:, -3:], (x[:, -3:]-x[:, -3:].mean(0))/x[:, -3:].std(0), atol=2e-7)
    changed = x.copy(); changed[:, -1] += 2
    assert not np.array_equal(projection.transform(changed), z)
    row = deepcopy(next(r for r in recipes() if r['id']=='early_to_decoder__kv_random512_s42'))
    row['conditioning_columns'] = 3
    row['gp'].update(input_components=4, output_components=2, inducing_points=8)
    row['features']['components'] = 4
    fitted = Estimator(row, 'cpu').fit(x, y, ['t']*30, tmp_path)
    np.testing.assert_allclose(fitted.model.projection.transform(x), z, atol=2e-7)


def test_feature_cache_interrupt_resumes_completed_batches(tmp_path, monkeypatch):
    import uq_followup.features as f
    monkeypatch.setattr(f, 'model_identity', lambda _: 'model')
    monkeypatch.setattr(f, 'load_policy', lambda *a: (None, None, None))
    monkeypatch.setattr(f, 'preprocess_inputs', lambda x, _: x)
    calls = []; interrupted = [True]
    class Fake:
        def __init__(self, *a): self.spec = {}
        def extract(self, batch, banks, **kwargs):
            calls.append(batch)
            if batch == [2, 3] and interrupted[0]: raise KeyboardInterrupt()
            return {b: np.asarray(batch, dtype='float32')[:, None] for b in banks}, {}
    monkeypatch.setattr(f, 'FeatureExtractor', Fake)
    view = SimpleNamespace(ids=list(range(5)), candidates=[SimpleNamespace(id=i, input_hash=str(i)) for i in range(5)], observation=lambda i: i)
    cfg = FeatureConfig()
    with pytest.raises(KeyboardInterrupt): f.cache_bank(tmp_path, 'm', view, cfg, ('joint_intermediate',), role='support', device='cpu', batch_size=2)
    interrupted[0] = False
    arrays = f.cache_bank(tmp_path, 'm', view, cfg, ('joint_intermediate',), role='support', device='cpu', batch_size=2)
    assert calls == [[0, 1], [2, 3], [2, 3], [4]]
    np.testing.assert_array_equal(arrays['joint_intermediate'][:, 0], range(5))


def test_routing_all_gpus_for_policy_work_a100_only_for_kvt():
    from uq_followup.launcher import accepts
    a100 = dict(name='NVIDIA A100-PCIE-40GB'); rtx = dict(name='NVIDIA GeForce RTX 3090')
    for kind in ('train', 'evaluate'):
        assert accepts(dict(kind=kind), a100) and accepts(dict(kind=kind), rtx)
    assert accepts(dict(kind='head', recipe=dict(method='kv_t')), a100)
    assert not accepts(dict(kind='head', recipe=dict(method='kv_t')), rtx)
    assert accepts(dict(kind='head', recipe=dict(method='kv')), rtx)


def test_scheduler_continues_after_failure_and_skips_valid_completed_job(tmp_path, monkeypatch):
    import uq_followup.launcher as launch
    declaration = dict(project=str(tmp_path), source_hash='code')
    requests = [launch.job(tmp_path, declaration, 'evaluate', f'eval_{i}', policy_id=f'p{i}', repetition=0) for i in range(2)]
    calls = []
    class Process:
        def __init__(self, arguments, **kwargs):
            path = Path(arguments[-1]); request = read_json(path); self.pid = 12345
            calls.append(request['spec']['id']); self.code = 1 if len(calls)==1 else 0
            if self.code == 0:
                atomic_json(path.with_name('complete.json'), dict(identity=request['identity']))
                atomic_json(tmp_path/'evaluations'/'p1'/'rep_0'/'evaluation.json', dict(episodes=[{}]*500))
        def poll(self): return self.code
        def wait(self): return self.code
    monkeypatch.setattr(launch.subprocess, 'Popen', Process)
    outcome = launch.drain(tmp_path, requests, [dict(label='6', name='RTX 3090', environment={})], 'test')
    assert calls == ['eval_0', 'eval_1'] and len(outcome['failures']) == 1
    assert not outcome['deferred']
    assert launch.job(tmp_path, declaration, 'evaluate', 'eval_1', policy_id='p1', repetition=0) is None
    assert launch.job(tmp_path, declaration, 'evaluate', 'eval_0', policy_id='p0', repetition=0) is not None


@pytest.mark.parametrize('first_type', ['RTX 3090', 'A100'])
def test_gpu_subset_defers_incompatible_heads_and_resume_reuses_results(tmp_path, monkeypatch, first_type):
    import uq_followup.launcher as launch
    import uq_followup.reporting as reporting
    declaration = dict(project=str(tmp_path), source_hash='code')
    methods = ['kv', 'kv_t', 'vfd']
    def requests():
        return [launch.job(tmp_path, declaration, 'head', method,
                           directory=str(tmp_path/'heads'/method), recipe=dict(method=method))
                for method in methods]
    calls = []
    class Process:
        def __init__(self, arguments, **kwargs):
            path = Path(arguments[-1]); request = read_json(path); self.pid = 12345
            spec = request['spec']; calls.append(spec['id'])
            directory = Path(spec['directory']); directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory/'result.json', dict(status='complete', recipe=spec['recipe']))
            (directory/'report.html').write_text('saved report')
            atomic_json(path.with_name('complete.json'), dict(identity=request['identity']))
        def poll(self): return 0
        def wait(self): return 0
    monkeypatch.setattr(launch.subprocess, 'Popen', Process)
    monkeypatch.setattr(launch.time, 'sleep', lambda _: None)
    monkeypatch.setattr(reporting, 'refresh_index', lambda _: None)
    slots = [dict(label='6', name='RTX 3090', environment={}), dict(label='4', name='A100', environment={})]
    selected = [s for s in slots if s['name'] == first_type]
    first_requests = requests()
    expected = ['kv', 'vfd'] if first_type == 'RTX 3090' else ['kv_t']
    missing = [m for m in methods if m not in expected]
    outcome = launch.drain(tmp_path, first_requests, selected, 'first-pass')
    assert calls == expected and not outcome['failures'] and outcome['deferred'] == missing
    queue = read_json(tmp_path/'queue.json')
    assert queue['completed_this_stage'] == len(expected) and queue['total_this_stage'] == 3
    assert queue['deferred'] == missing and not queue['pending'] and not queue['running']
    for method in missing:
        assert not (tmp_path/'jobs'/method/'complete.json').exists()
        assert not (tmp_path/'jobs'/method/'failure.json').exists()
    # An invocation with only deferred work must terminate without launching anything.
    repeated = launch.drain(tmp_path, requests(), selected, 'same-gpus')
    assert repeated == outcome and calls == expected
    # Changing GPU availability does not alter scientific requests or redo saved heads.
    result = launch.drain(tmp_path, requests(), slots, 'all-gpus')
    assert not result['deferred'] and not result['failures']
    assert sorted(calls) == sorted(methods)
    assert all(r is None for r in requests())
    for original in first_requests:
        assert read_json(original['path']) == original['request']


def test_fractional_outcome_reports_and_policy_seed_aggregation(tmp_path):
    from uq_followup.reporting import head_report, aggregate_report
    atomic_json(tmp_path/'declaration.json', dict(seeds=[0, 1, 2], recipes=[
        dict(id=f'example__kv_random512_s{seed}') for seed in (42, 43)]))
    for seed in range(3):
        learner = []; teacher = []
        for task in range(3):
            for i in range(4):
                id_ = f't{task}/s{i}'
                common = dict(id=id_, task=f't{task}', init_state_id=i, init_state_hash=id_, reset_seed=i,
                    initial_input=dict(input_hash=id_))
                l = ((i+task+seed)%4)/3
                h = 1 if i else 0
                learner.append(dict(common, success=l, q_time=l*.5, q_path=l*.8))
                teacher.append(dict(common, success=h, q_time=h*.6, q_path=h*.7))
        atomic_json(tmp_path/'evaluations'/f'original_seed{seed}'/'conditions.json', learner)
        atomic_json(tmp_path/'evaluations'/f'supervisor_seed{seed}'/'conditions.json', teacher)
        for coordinate_seed in (42, 43):
            name = f'example__kv_random512_s{coordinate_seed}'
            directory = tmp_path/'heads'/f'original_seed{seed}'/name
            atomic_json(directory/'scores.json', dict(ids=[r['id'] for r in learner],
                scores=[h['success']-l['success'] for h, l in zip(teacher, learner)], pool_task_means={f't{i}':i for i in range(3)}))
            metrics = head_report(tmp_path, f'original_seed{seed}', directory)
            assert metrics['global_nauc'] == pytest.approx(1)
            assert metrics['within_nauc'] == pytest.approx(1)
            assert 'failure_auroc' not in metrics
            atomic_json(directory/'result.json', dict(status='complete', recipe=dict(id=name), policy_id=f'original_seed{seed}', metrics=metrics))
    aggregate_report(tmp_path)
    summary = read_json(tmp_path/'report/summary.json')
    assert len(summary) == 1 and summary[0]['policy_seeds'] == 3
    assert summary[0]['coordinate_repetitions'] == '2,2,2'
    assert summary[0]['global_nauc_mean'] == pytest.approx(1)
    assert summary[0]['global_nauc_std'] == pytest.approx(0, abs=1e-12)
    assert summary[0]['global_nauc_total95'] == pytest.approx([1, 1])
    assert summary[0]['coverage'] == 'complete' and summary[0]['complete_policy_seeds'] == 3
    assert summary[0]['planned_coordinate_repetitions'] == 2
    # Three policy IDs alone do not imply full coverage if one coordinate repeat is missing.
    (tmp_path/'heads/original_seed2/example__kv_random512_s43/result.json').unlink()
    aggregate_report(tmp_path)
    partial = read_json(tmp_path/'report/summary.json')[0]
    assert partial['policy_seeds'] == 3 and partial['coverage'] == 'partial'
    assert partial['complete_policy_seeds'] == 2 and partial['coordinate_repetitions'] == '2,2,1'
    assert 'Partial: 2/3 fully scored policies' in (tmp_path/'report/report.html').read_text()
