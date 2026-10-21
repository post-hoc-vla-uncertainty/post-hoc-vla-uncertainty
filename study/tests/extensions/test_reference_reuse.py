from dataclasses import replace
from pathlib import Path
import pytest

from uq_extensions.common import atomic_json,read_json,file_hash,identity
from uq_extensions.config import ExperimentConfig,MODES
from uq_extensions.protocol import AcquisitionLedger
from uq_extensions.reference import import_reproduction,validate_source,available_panel_rounds,reference_curve,EVALUATION_PROTOCOL
from uq_extensions.rollout_metrics import learning_summary


def evaluation(index,model):
    return dict(round_index=index,member_index=0,model_path=str(model),policy_type='smolvla',env_type='libero',
        eval_batch_size=15,n_rollouts_per_task=30,seed_start=100,seed_step=1,max_steps=520,use_async_envs=False,
        num_tasks=10,total_rollouts=300,timings={'total':{'duration_s':3000}},
        per_task=[dict(task_group='libero_10',task_id=t,n_rollouts=30,
            per_rollout_results=[dict(success=(j%2==0),steps=250,seed=100+j) for j in range(30)]) for t in range(10)])


def fixture(tmp_path,monkeypatch):
    import reproduction_storage
    root=Path(__file__).resolve().parents[2];monkeypatch.setenv('VLA_PROJECT_ROOT',str(root))
    source=tmp_path/'reproduction/random/seed_1000'
    bases=[str(tmp_path/f'base_{i}/pretrained_model') for i in (0,1)]
    cfg=ExperimentConfig(arm='random',companion=True,base_models=bases,output_root=str(tmp_path/'extensions'),data_manifest=str(tmp_path/'manifest.json'))
    native=read_json(root/'configs/reproduction/random.json')
    atomic_json(source/'iterative_fine_tuning_config.json',native)
    manifest=dict(manifest_hash='manifest',source_revision=native['dataset']['revision'],dataset_root=native['dataset']['root'],
        episode_map={f'd{i}':i for i in range(6)},pretraining_ids=['d0'],target_ids=[f'd{i}' for i in range(6)],
        observations=[dict(id=f'd{i}',task=f'libero_10/{i}') for i in range(6)],evaluator_directory=str(tmp_path/'private'))
    atomic_json(cfg.data_manifest,manifest)
    atomic_json(tmp_path/'private/trajectory_index.json',[dict(id=f'd{i}',dataset_from_index=0,dataset_to_index=100) for i in range(6)])
    monkeypatch.setattr(reproduction_storage,'validate_checkpoint',lambda p:30000 if p.name.startswith('base_') else 4000)
    atomic_json(source/'initial_evaluation/member_00/member_evaluation.json',evaluation(-1,bases[0]))
    rows=[]
    for i in range(15):
        d=source/f'round_{i:03d}'
        models=[str(d/f'member_{m}/checkpoints/004000/pretrained_model') for m in (0,1)]
        ids=[0,1,2,3,4] if i%2==0 else [1,2,3,4,5]
        selection=dict(selected_episodes=[dict(episode_id=id) for id in ids])
        train=dict(history_episode_ids=[0],final_model_paths=models)
        atomic_json(d/'selection_manifest.json',selection);atomic_json(d/'training_manifest.json',train)
        atomic_json(d/'training/member_00/member_evaluation.json',evaluation(i,models[0]))
        evidence={name:file_hash(d/name) for name in ('selection_manifest.json','training_manifest.json','training/member_00/member_evaluation.json')}
        atomic_json(d/'reproduction_commit.json',dict(round_index=i,evidence=evidence))
        if i not in (4,9,14):
            for m in models:atomic_json(Path(m).parent/'retention.json',dict(weights_retired=True))
        rows.append(dict(round_index=i,start_model_paths=bases if i==0 else rows[-1]['final_model_paths'],
            final_model_paths=models,selected_episode_ids=ids,
            timings={'selection':{'duration_s':180},'training':{'duration_s':1920}}))
    atomic_json(source/'run_state.json',dict(rounds=rows))
    return cfg,source,manifest


def test_both_modes_keep_original_eligibility_and_audit_repeats(tmp_path):
    for mode in MODES:
        c=ExperimentConfig(training_mode=mode);assert c.eligibility_protocol=='upstream_reference'
        assert reference_curve(replace(c,phase='final'))
        assert not reference_curve(c)
    ledger=AcquisitionLedger(tmp_path/'ledger.json',manifest_hash='m',protocol='upstream_reference',
        pretraining_ids=['a'],candidates=['a','b','c'])
    ledger.commit(0,['a','b'],audit={},expected_count=2)
    second=ledger.commit(1,['a','b'],audit={},expected_count=2)
    assert second['prior_pretraining']==['a'] and second['prior_acquisition']==['a','b']
    assert ledger.eligible()==['a','b','c'] and ledger.support==['a','b']
    with pytest.raises(ValueError,match='distinct'):ledger.commit(2,['c','c'],audit={},expected_count=2)


def test_query_slot_area_allows_zero_new_data_rounds():
    rows=[dict(queries=0,new_demonstrations=0,success=.2),dict(queries=5,new_demonstrations=0,success=.4),
          dict(queries=10,new_demonstrations=2,success=.6)]
    result=learning_summary(rows,budget_key='queries')
    assert result['area']==pytest.approx(.4) and result['query_slots_to_50']==10
    assert result['budget_axis']=='queries' and 'demonstrations_to_50' not in result


def test_import_reuses_all_outcomes_and_only_retained_weights(tmp_path,monkeypatch):
    cfg,source,manifest=fixture(tmp_path,monkeypatch)
    before={str(p):file_hash(p) for p in source.rglob('*.json')}
    out=import_reproduction(cfg,source,manifest)
    assert available_panel_rounds(out,[0,1,5,10,15])==[0,5,10,15]
    assert read_json(out/'round_001/policy_fit.json')['models']==[]
    assert read_json(out/'round_005/policy_fit.json')['models']
    assert len(read_json(out/'round_015/complete.json')['acquired'])==75
    for r in range(16):
        record=read_json(out/f'round_{r:03d}/evaluation/evaluation.json')
        assert len(record['episodes'])==300 and record['evaluation_protocol']==EVALUATION_PROTOCOL
        assert record['episodes'][0]['q_path'] is None
    assert not list(out.rglob('*.safetensors'))
    assert import_reproduction(cfg,source,manifest)==out
    assert before=={str(p):file_hash(p) for p in source.rglob('*.json')}
    from uq_extensions.reporting import report_run
    report=report_run(out)
    assert report['budget_axis']=='queries' and report['curve'][-1]['queries']==75
    assert report['curve'][-1]['new_demonstrations']==5
    assert report['curve'][-1]['prior_acquisition_queries']>0
    assert report['curve'][1]['policy_updates']==8000
    assert report['evaluation_protocol']==EVALUATION_PROTOCOL


def test_import_refuses_incomplete_or_changed_reference(tmp_path,monkeypatch):
    cfg,source,manifest=fixture(tmp_path,monkeypatch)
    state=read_json(source/'run_state.json');atomic_json(source/'run_state.json',dict(rounds=state['rounds'][:-1]))
    with pytest.raises(ValueError,match='not complete'):import_reproduction(cfg,source,manifest)
    assert not cfg.run_dir().exists()
    atomic_json(source/'run_state.json',state)
    atomic_json(source/'round_002/selection_manifest.json',{'corrupt':True})
    with pytest.raises(RuntimeError,match='evidence changed'):validate_source(cfg,source,manifest)


def test_import_refuses_protocol_or_training_changes(tmp_path,monkeypatch):
    cfg,source,manifest=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Only original'):validate_source(replace(cfg,eligibility_protocol='unique_demo'),source,manifest)
    p=source/'iterative_fine_tuning_config.json';native=read_json(p);native['replay']['history_fraction']=.2;atomic_json(p,native)
    with pytest.raises(ValueError,match='replay'):validate_source(cfg,source,manifest)


def test_reference_evaluation_dispatch_preserves_pinned_settings(tmp_path,monkeypatch):
    import uq_extensions.reference as ref
    import lerobot.envs.libero as libero_env
    monkeypatch.setattr(ref,'model_identity',lambda p:'modelhash')
    monkeypatch.setattr(libero_env,'get_task_instruction',lambda suite,i:f'task {i}')
    captured=[]
    class Child:
        returncode=0
        def __init__(self,cmd,**kwargs):
            assert cmd[1:3]==['-m','iterative_fine_tuning.eval_worker']
            request=read_json(cmd[-1]);captured.append(request)
            atomic_json(request['result_path'],evaluation(request['round_index'],request['model_path']))
        def poll(self):return 0
    monkeypatch.setattr(ref.subprocess,'Popen',Child)
    cfg=ExperimentConfig(phase='final')
    ref.evaluate_reference(cfg,'model',tmp_path,5,device='cpu')
    assert captured[0]['round_index']==4
    assert (captured[0]['seed_start'],captured[0]['n_rollouts'],captured[0]['eval_batch_size'],captured[0]['max_steps'])==(100,30,15,520)
    assert captured[0]['use_async_envs'] is False and len(captured[0]['task_specs'])==10
    assert captured[0]['task_specs'][9]['task_instruction']=='task 9'
    ref.evaluate_reference(cfg,'model',tmp_path,5,device='cpu')
    assert len(captured)==1
    monkeypatch.setattr(ref,'model_identity',lambda p:'changed')
    with pytest.raises(RuntimeError,match='identity changed'):ref.evaluate_reference(cfg,'model',tmp_path,5,device='cpu')


def test_training_seed_matches_upstream_zero_based_round(tmp_path):
    from uq_extensions.training import make_training_config
    from iterative_fine_tuning.training import _member_seed
    cfg=ExperimentConfig(seed=1001,base_models=['base'])
    path=make_training_config(cfg,dict(dataset_root='storage/datasets/HuggingFaceVLA/libero'),['base'],tmp_path,7)
    resolved=read_json(path)
    assert _member_seed(resolved['seed'],7,0)==_member_seed(1001,6,0)
    assert resolved['replay']['history_fraction']==.5 and resolved['replay']['new_fraction']==.5


def test_common_history_stage_imports_without_training(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import uq_extensions.experiments as ex
    import uq_extensions.reference as ref
    import uq_extensions.refit_check as refit
    import uq_extensions.data as data
    import uq_extensions.runner as runner
    evidence={'preview':'reviewed'}
    atomic_json(tmp_path/'01_preview/result.json',evidence)
    atomic_json(tmp_path/'decisions/01_outputs.json',dict(evidence_hash=identity(evidence),outputs_ready=True))
    imported=[]
    monkeypatch.setattr(runner,'run',lambda *a,**kw:pytest.fail('Duplicate SAVE policy training'))
    monkeypatch.setattr(data,'load_manifest',lambda *a:{})
    monkeypatch.setattr(ref,'import_reproduction',lambda cfg,source,manifest:imported.append((cfg,source)))
    monkeypatch.setattr(refit,'check_refit',lambda *a,**kw:dict(selected_refit_steps=4000))
    ex.stage_histories(SimpleNamespace(root=str(tmp_path)),read_json('configs/extensions/experiments/workflow.json'),['cpu'],'cpu')
    assert len(imported)==1 and imported[0][0].eligibility_protocol=='upstream_reference'
    assert str(imported[0][1]).endswith('random/seed_1000')
    assert read_json(tmp_path/'decisions/02_refit_budget.json')['refit_steps'] is None
