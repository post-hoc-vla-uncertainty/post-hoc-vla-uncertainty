from dataclasses import replace
from types import SimpleNamespace
import sys
import pytest
from uq_extensions.common import atomic_json,read_json,plain
from uq_extensions.config import FeatureConfig,GPConfig,ARMS,MODES
from uq_extensions.experiments import decision,require_decision,selected_config
from uq_extensions.matrix import broad_recipes,focused_recipes


def settings():return read_json('configs/extensions/experiments/workflow.json')


def test_decision_boundary_is_not_implicitly_accepted_and_binds_evidence(tmp_path):
    evidence={'results':['a']};path=decision(tmp_path,'review',evidence,{'ready':None})
    with pytest.raises(ValueError,match='still required'):require_decision(tmp_path,'review',evidence,['ready'])
    data=read_json(path);data['ready']=True;atomic_json(path,data)
    assert require_decision(tmp_path,'review',evidence,['ready'])['ready']
    with pytest.raises(ValueError,match='evidence changed'):require_decision(tmp_path,'review',{'results':['b']},['ready'])


def test_plan_never_initializes_cuda_or_creates_workspace(tmp_path,monkeypatch):
    import uq_extensions.cli as cli
    from uq_extensions.experiments import main
    monkeypatch.setattr(cli,'device_setup',lambda *a:pytest.fail('CUDA initialized by --plan'))
    root=tmp_path/'not_created';monkeypatch.setattr(sys,'argv',['experiments','05_active_learning','--root',str(root),'--plan'])
    main();assert not root.exists()


def test_all_42_configs_use_target_recipe_for_kvt_ivr(tmp_path):
    recipe={'kv_features':plain(FeatureConfig()),'kv_t_features':plain(FeatureConfig()),
            'kv_gp':plain(GPConfig()),'kv_t_gp':plain(GPConfig(output_compression='random')),'refit_steps':8000}
    configs=[selected_config(settings(),tmp_path,'frozen.json',recipe,arm=a,mode=m,seed=s)
             for a in ARMS for m in MODES for s in (1000,1001,1002)]
    assert len(configs)==42
    for c in configs:
        c.validate();assert c.gp.output_compression==('random' if c.arm in ('kv_t','kv_t_ivr') else 'pca')
        assert c.eligibility_protocol=='upstream_reference'
    assert sum(c.policy_count for c in configs)==60


def test_declared_matrix_covers_families_span_patch_compression_and_dependence():
    broad=broad_recipes(flow=settings()['features']);byid={r['id']:r for r in broad}
    assert {'S04_E','S04_V','S09_distance_1','S09_distance_3','S10_final_action','S11_patch_max','S13_vfd'}<=set(byid)
    assert len(byid)==len(broad)
    focused=focused_recipes([byid['S04_E'],byid['S04_V']])
    assert len({r['id'] for r in focused})==len(focused)
    assert {r['gp']['output_dependence'] for r in focused}=={'independent','coregionalized'}
    assert {r['features']['seed'] for r in focused}>={42,43,44}
    assert sum(r['group']=='S12' for r in focused)==4


def test_preview_is_one_bounded_run_and_stops_with_pending_decision(tmp_path,monkeypatch):
    import uq_extensions.experiments as ex
    import uq_extensions.runner as runner
    import uq_extensions.reporting as reporting
    calls=[]
    monkeypatch.setattr(ex,'supervisor',lambda *a:'teacher')
    monkeypatch.setattr(runner,'run',lambda c,**kw:calls.append(c))
    monkeypatch.setattr(ex,'run_matrix',lambda c,run,rounds,recipes,out,**kw:[])
    monkeypatch.setattr(reporting,'experiment_report',lambda root,**kw:root/'report.html')
    ex.stage_preview(SimpleNamespace(root=str(tmp_path)),settings(),['cpu'],'cpu')
    assert len(calls)==1 and calls[0].rounds==1 and calls[0].steps==100 and calls[0].arm=='kv'
    assert calls[0].evaluation_per_task==2
    assert read_json(tmp_path/'decisions/01_outputs.json')['outputs_ready'] is None
    assert not (tmp_path/'02_histories').exists()
