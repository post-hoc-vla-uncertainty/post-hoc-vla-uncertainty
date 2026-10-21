"""Scientific matching, seed reuse, referral ties and restart scheduling."""
from pathlib import Path
from itertools import permutations
import json

import numpy as np
import pytest
import torch

from uq_extensions.common import atomic_json, read_json
from uq_finalstudy.design import ARMS, SCORES, jobs, recipes
from uq_finalstudy.launcher import accepts, completed, ready_jobs, validate_graph
from uq_finalstudy.reporting import ReferralPlan, expected_ties, METRICS


def cohort():
    learner = np.array([0., .3, .7, 1., .1, .5, .8, 0.])
    teacher = np.array([1., 1., 0., .7, 1., .2, 1., 1.])
    return dict(ids=[str(i) for i in range(8)], tasks=['a']*4+['b']*4,
        clusters=[str(i) for i in range(8)], learner=learner, supervisor=teacher,
        qualities=dict(time=(learner, teacher), path=(learner, teacher)))


def test_exact_tie_expectation_matches_exhaustive_permutations():
    values = np.array([-.4, .1, .8, .5])
    actual = expected_ties(np.array([[1.,1.,1.,2.]]), values[None,:])[0]
    expectation = np.mean([np.r_[values[list(p)], values[3]] for p in permutations(range(3))], axis=0)
    np.testing.assert_allclose(actual, expectation)


def test_task_constant_score_has_zero_within_task_referral_even_in_bootstrap():
    plan = ReferralPlan(cohort(), repetitions=200)
    values, _ = plan.evaluate(np.array([0.]*4+[1.]*4), {'a':0.,'b':1.})
    assert np.nanmax(np.abs(values[:,METRICS.index('within_nauc')])) < 1e-12


def test_oracle_is_one_and_harm_is_not_dropped():
    c = cohort(); plan = ReferralPlan(c, indices=np.arange(8)[None,:])
    scores = c['supervisor']-c['learner']
    values, curves = plan.evaluate(scores, {'a':.1,'b':.2})
    assert scores.min() < 0
    assert values[0,METRICS.index('global_nauc')] == pytest.approx(1.)
    assert values[0,METRICS.index('within_nauc')] == pytest.approx(1.)
    assert curves['global_'][1][0,0] == pytest.approx(np.mean(c['supervisor']))
    assert curves['global_'][1][0,-1] == pytest.approx(np.mean(c['learner']))


def test_queue_has_three_seeds_all_rounds_no_hidden_extra_training(tmp_path):
    queue = jobs(tmp_path, {'groups':[{'seed':s} for s in (1000,1001,1002)]})
    validate_graph(queue)
    assert len(ARMS)==8 and len(SCORES)==11
    assert sum(j['kind']=='train' for j in queue)==(3*2+5)*3*15
    assert sum(j['kind']=='score' for j in queue)==3*16*11
    for j in queue:
        if j['kind']=='score':
            assert j['arm']=='random'
            assert f'collect_supervisor_{j["seed"]}' in j['dependencies']


def test_sparse_schedule_keeps_training_jobs_identical_and_only_reduces_evaluations(tmp_path):
    declaration = dict(version=2, groups=[{'seed':s} for s in (1000,1001,1002)])
    queue = jobs(tmp_path, declaration); validate_graph(queue)
    original = {j['id']:j for j in jobs(tmp_path, declaration | {'version':1})}
    assert all(j == original[j['id']] for j in queue)
    assert sum(j['kind']=='train' for j in queue) == 495
    assert sum(j['kind']=='acquire' for j in queue) == 360
    assert sum(j['kind']=='reference' for j in queue) == 75
    assert sum(j['kind']=='evaluate' for j in queue) == 27  # Six primaries + three supervisors, three repeats.
    assert sum(j['kind']=='score' for j in queue) == 66
    assert {j['round'] for j in queue if j['kind']=='reference'} == {0,5,10,15}
    assert {j['round'] for j in queue if j['kind'] in ('score','quality')} == {0,10}
    assert {j['id'] for j in queue if j['kind'] in ('train','acquire')} == {
        j['id'] for j in original.values() if j['kind'] in ('train','acquire')}


def test_sparse_report_ignores_old_extra_rounds_and_labels_coarse_acquisition(tmp_path):
    from uq_finalstudy.reporting import generate
    from uq_finalstudy.design import round_dir
    seeds = (1000,1001); c = cohort()
    manifest = tmp_path/'manifest.json'; atomic_json(manifest, dict(pretraining_ids=[]))
    atomic_json(tmp_path/'declaration.json',dict(version=2, groups=[{'seed':s} for s in seeds], manifest=str(manifest)))
    for seed in seeds:
        for r in (0,10):
            pid = f'random_{seed}_r{r:02d}'
            for label, values in [(pid,c['learner']), (f'supervisor_{seed}',c['supervisor'])]:
                rows = [dict(id=id_,task=task,init_state_hash=id_,reset_seed=i,initial_input=dict(input_hash=id_),
                             success=float(value),q_time=float(value),q_path=float(value))
                        for i,(id_,task,value) in enumerate(zip(c['ids'],c['tasks'],values))]
                atomic_json(tmp_path/f'evaluations/{label}/conditions.json',rows)
            for head in ('h1','vfd'):
                directory = tmp_path/f'scores/{pid}/{head}'
                atomic_json(directory/'scores.json',dict(ids=c['ids'],scores=[.9,.7,.2,.1,.8,.4,.3,.6],pool_task_means={'a':.4,'b':.6}))
                atomic_json(directory/'result.json',dict(policy_id=pid,seed=seed,head=head,round=r,status='complete'))
        # Old off-schedule output must not be used, even if its paired outcomes are absent.
        atomic_json(tmp_path/f'scores/random_{seed}_r05/h1/result.json',
                    dict(policy_id=f'random_{seed}_r05',seed=seed,head='h1',round=5,status='complete'))
        for r in range(16):
            directory = round_dir(tmp_path,'random',seed,r)
            source = directory/'outcomes.json'
            atomic_json(source,dict(per_task=[dict(per_rollout_results=[dict(success=r/15)])]))
            atomic_json(directory/'reference.json',dict(source=str(source),provenance='original_reproduction_outcomes'))
            if r: atomic_json(directory/'selection.json',dict(selected_ids=[f'{r}-{i}' for i in range(5)]))
    generate(tmp_path)
    result = read_json(tmp_path/'report/summary.json')
    assert {r['round'] for r in result['per_round']} == {0,10}
    assert all(r['rounds']==[0,10] and r['seeds']==list(seeds) for r in result['round_averaged'])
    assert result['paired_differences'][0]['rounds'] == [0,10]
    acquisition = next(r for r in result['acquisition_summary'] if r['arm']=='random')
    assert acquisition['rounds'] == [0,5,10,15]
    assert acquisition['metrics']['normalized_learning_auc']['mean'] == pytest.approx(.5)
    assert acquisition['individual_seeds'][0]['slots_to_50'] == 50
    assert {r['round'] for r in result['acquisition']} == {0,5,10,15}
    text = (tmp_path/'report/report.html').read_text()
    assert 'all 16 rounds' not in text and 'crossings between evaluations are unknown' in text


def test_schedule_migration_preserves_checkpoints_and_can_resume(tmp_path, monkeypatch):
    import migrate_final_study_schedule as migration
    from uq_extensions.common import identity
    from uq_finalstudy.launcher import request
    root = tmp_path/'study'; root.mkdir(); snapshot = tmp_path/'snapshot'; snapshot.mkdir()
    original = dict(version=1, groups=[{'seed':1000}], source_hash='old', project=str(tmp_path))
    old_jobs = jobs(root, original)
    for path in (root,snapshot):
        atomic_json(path/'declaration.json',original); atomic_json(path/'job_graph.json',old_jobs)
    training = next(j for j in old_jobs if j['kind']=='train')
    weights = root/'weights/model.safetensors'; weights.parent.mkdir(); weights.write_bytes(b'finished weights')
    partial = root/'partial/optimizer.pt'; partial.parent.mkdir(); partial.write_bytes(b'resume state')
    atomic_json(Path(training['output']),dict(model=str(weights.parent)))
    req = request(root,original,training)
    from uq_extensions.common import file_hash
    atomic_json(req.with_name('complete.json'),dict(identity=read_json(req)['identity'],output_hash=file_hash(training['output']),seconds=123))
    monkeypatch.setattr(migration,'verify_source',lambda *args:None)
    monkeypatch.setattr(migration,'source_hash',lambda *args:'new')
    plan = migration.build_plan(root,snapshot); atomic_json(snapshot/'migration_plan.json',plan)
    # Simulate interruption after the first metadata write.
    atomic_json(root/plan['writes'][0]['path'],plan['writes'][0]['after'])
    audit = migration.migrate(root,snapshot)
    assert training['id'] in audit['preserved_completed_jobs']
    assert read_json(req)['spec']['declaration'] == identity(read_json(root/'declaration.json'))
    assert read_json(req.with_name('complete.json'))['identity'] == read_json(req)['identity']
    assert weights.read_bytes()==b'finished weights' and partial.read_bytes()==b'resume state'
    assert completed(training,root)
    migration.migrate(root,snapshot)  # Idempotent before restarting the queue.


def test_gpu_routing_restricts_only_double_precision_fit():
    a100 = dict(name='NVIDIA A100-PCIE-40GB')
    rtx = dict(name='NVIDIA GeForce RTX 3090')
    assert accepts(dict(kind='train'),rtx) and accepts(dict(kind='train'),a100)
    assert accepts(dict(a100=True),a100) and not accepts(dict(a100=True),rtx)
    assert accepts(dict(kind='score',head='h1'),rtx)


def test_failed_arm_does_not_block_independent_jobs():
    queue=[dict(id='broken',dependencies=[]),dict(id='child',dependencies=['broken']),
           dict(id='other',dependencies=[]),dict(id='after_done',dependencies=['done'])]
    assert {j['id'] for j in ready_jobs(queue, {'done'}, {'broken'})} == {'other','after_done'}


def test_missing_checkpoint_never_skips_as_complete(tmp_path):
    path=tmp_path/'model.json'; atomic_json(path,dict(model=str(tmp_path/'missing')))
    with pytest.raises(RuntimeError,match='Missing retained policy'):
        completed(dict(kind='train',output=str(path)))


def test_output_without_worker_commit_is_retried(tmp_path):
    path=tmp_path/'selection.json'
    job=dict(kind='acquire',id='acquire_example',output=str(path))
    atomic_json(path,dict(selected_ids=list('abcde'),provenance='new_acquisition'))
    assert not completed(job,tmp_path)
    from uq_extensions.common import file_hash
    atomic_json(tmp_path/'jobs/acquire_example/complete.json',dict(output_hash=file_hash(path)))
    assert completed(job,tmp_path)


def test_frozen_selected_recipes():
    r=recipes()
    assert r['h1']['features']['compression']=='pca_whiten'
    assert r['h3']['features']['components']==512 and r['h3']['whiten']
    assert r['h4']['gp']['precision']=='float64' and r['h4']['gp']['fit_steps']==10000
    assert r['h4']['gp']['output_components']==16
    assert r['h2']['flow_level']==9 and r['h2']['flow_level_indexing']=='one_based'


def test_collaborator_kernel_is_exact_rbf_conditional_variance():
    from uq_finalstudy.heads import collaborator_module
    project=Path(__file__).resolve().parents[2]
    path=project/'reference_src/vla-ua/src/lerobot/uncertainty/uncertainty_scoring/kernel_variance.py'
    if not path.exists(): pytest.skip('Collaborator checkout not installed')
    module=collaborator_module(project)
    generator=torch.Generator().manual_seed(11)
    x=torch.randn(24,5,generator=generator); q=torch.randn(8,5,generator=generator)
    model=module.KernelVariance(ridge=.01,projection='whiten').fit(x)
    z=model._prepare(x); query=model._prepare(q)
    k=torch.exp(-torch.cdist(z,z).square()/(2*model.length_scale**2))
    cross=torch.exp(-torch.cdist(query,z).square()/(2*model.length_scale**2))
    expected=1-(cross*torch.linalg.solve(k+.01*torch.eye(len(x)),cross.T).T).sum(1)
    torch.testing.assert_close(model.score(q),expected,atol=2e-5,rtol=2e-5)


def test_collaborator_tap_is_ninth_euler_evaluation_before_projection(monkeypatch):
    from types import SimpleNamespace
    import lerobot.policies.smolvla.smolvla_adapter as adapters
    from uq_finalstudy.heads import pooled_decoder
    model=SimpleNamespace(action_out_proj=torch.nn.Linear(3,2,bias=False))
    class Adapter:
        def __init__(self, config, model):
            self.model=model; self.device=torch.device('cpu'); self.dtype=torch.float32
            self.ode_solver_config=dict(step_size=.1)
        def prepare_conditioning(self,batch,num_action_samples): return {}
        def sample_prior(self,n,generator): return torch.randn(n,2,2,generator=generator)
        def make_velocity_fn(self,conditioning):
            def velocity(t,x_t):
                features=torch.full((1,2,3),float(t))
                return self.model.action_out_proj(features)
            return velocity
    monkeypatch.setattr(adapters,'SmolVLAAdapter',Adapter)
    out=pooled_decoder(SimpleNamespace(config=None,model=model),{})
    torch.testing.assert_close(out,torch.full((1,3),.8))
    assert not model.action_out_proj._forward_pre_hooks


def test_random_selection_reproduces_existing_acquisition_sets():
    from uq_finalstudy.worker import upstream_select
    from uq_extensions.data import candidate_view,load_manifest
    project=Path(__file__).resolve().parents[2]
    path=project/'storage/extensions/data/data_manifest.json'
    state=project/'storage/outputs/reproduction/random/seed_1000/run_state.json'
    if not path.exists() or not state.exists(): pytest.skip('Local reproduction evidence unavailable')
    manifest=load_manifest(path); pool=candidate_view(manifest,manifest['target_ids'])
    inverse={v:k for k,v in manifest['episode_map'].items()}
    for r in read_json(state)['rounds'][:12]:
        result=upstream_select(pool,dict.fromkeys(pool.ids,0.),'random',42,r['round_index']+1)
        assert set(result)=={inverse[i] for i in r['selected_episode_ids']}


def test_report_can_be_generated_before_any_heads(tmp_path):
    from uq_finalstudy.reporting import generate
    atomic_json(tmp_path/'declaration.json',dict(groups=[{'seed':s} for s in (1000,1001,1002)]))
    path=generate(tmp_path)
    text=path.read_text()
    assert 'data:image/png;base64,' in text and 'fetch(' not in text
    assert (tmp_path/'report/all_rounds.csv').exists()


def test_report_seed_mean_and_bootstrap_use_completed_results(tmp_path):
    from uq_finalstudy.reporting import generate
    seeds=(1000,1001)
    atomic_json(tmp_path/'declaration.json',dict(groups=[{'seed':s} for s in seeds]))
    c=cohort()
    for seed in seeds:
        pid=f'random_{seed}_r00'
        for label,values in [(pid,c['learner']), (f'supervisor_{seed}',c['supervisor'])]:
            rows=[dict(id=id_,task=task,init_state_hash=id_,reset_seed=i,initial_input=dict(input_hash=id_),
                       success=float(value),q_time=float(value),q_path=float(value))
                  for i,(id_,task,value) in enumerate(zip(c['ids'],c['tasks'],values))]
            atomic_json(tmp_path/f'evaluations/{label}/conditions.json',rows)
        directory=tmp_path/f'scores/{pid}/h1'
        scores=np.array([.9,.7,.2,.1,.8,.4,.3,.6])+(seed-1000)*.01
        atomic_json(directory/'scores.json',dict(ids=c['ids'],scores=scores.tolist(),pool_task_means={'a':.4,'b':.6}))
        atomic_json(directory/'result.json',dict(policy_id=pid,seed=seed,head='h1',round=0,status='complete'))
    generate(tmp_path)
    result=read_json(tmp_path/'report/summary.json')
    assert not result['round_averaged']  # Never turn round 0 into an all-round average.
    row=result['per_round'][0]
    assert row['seeds']==list(seeds)
    assert row['metrics']['global_nauc']['seed_std']==pytest.approx(0.)
    assert row['metrics']['global_nauc']['total_ci95'][0] is not None
    assert 'fetch(' not in (tmp_path/'report/report.html').read_text()


def test_reference_job_does_not_require_companion_to_finish(tmp_path,monkeypatch):
    from uq_finalstudy.worker import reference
    from uq_finalstudy.design import model_path
    import uq_extensions.reference as original
    atomic_json(model_path(tmp_path,'vfd',1000,1),dict(model='primary'))
    seen=[]
    def evaluate(config,model,directory,round_index,*,device):
        seen.append((model,round_index));atomic_json(directory/'reference_evaluation.json',{})
    monkeypatch.setattr(original,'evaluate_reference',evaluate)
    reference(tmp_path,{},dict(arm='vfd',seed=1000,round=1,output=str(tmp_path/'done.json')),'cpu')
    assert seen==[('primary',1)]
