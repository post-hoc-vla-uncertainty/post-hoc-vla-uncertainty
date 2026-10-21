from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import pytest
from uq_extensions.common import atomic_json,identity,read_json
from uq_extensions.config import ExperimentConfig
from uq_extensions.runner import run


def test_resume_uses_historical_acquired_data_without_reselection(tmp_path,monkeypatch):
    import uq_extensions.runner as runner
    import uq_extensions.evaluation as evaluation
    import uq_extensions.reporting as reporting
    from uq_extensions.protocol import Candidate,CandidateView
    manifest=dict(manifest_hash='m',pretraining_ids=['0'],target_ids=['1','2','3','4'],integration_ids=['1','2','3','4'],episode_map={str(i):i for i in range(5)})
    v=CandidateView([Candidate(str(i),i,'task','not-read','input') for i in range(5)])
    monkeypatch.setattr(runner,'verify_bases',lambda c:['base'])
    monkeypatch.setattr(runner,'load_manifest',lambda p:manifest)
    monkeypatch.setattr(runner,'model_identity',lambda p:p)
    monkeypatch.setattr(runner,'source_identity',lambda :'code')
    monkeypatch.setattr(runner,'candidate_view',lambda *a:v)
    monkeypatch.setattr(runner,'check_budget',lambda *a:None)
    monkeypatch.setattr(runner,'retire_optimizers',lambda *a:None)
    monkeypatch.setattr(evaluation,'evaluate',lambda *a,**kw:[])
    monkeypatch.setattr(reporting,'report_run',lambda *a:None)
    selections=[];fits=[]
    def acquire(c,models,support,candidates,integration,directory,r,device):
        selections.append((r,support.ids,candidates.ids))
        return candidates.ids[:1],{}
    def fit(c,m,starts,acquired,**kw):
        fits.append((kw['round_index'],list(starts),list(acquired)))
        return [f"model_{kw['round_index']}"]
    monkeypatch.setattr(runner,'acquire',acquire);monkeypatch.setattr(runner,'fit_policies',fit)
    cfg=ExperimentConfig(phase='smoke',rounds=2,batch_queries=1,output_root=str(tmp_path),arm='random',eligibility_protocol='unique_demo')
    run(cfg,gpu_ids=['cpu'],device='cpu')
    assert fits==[(1,['base'],['1']),(2,['model_1'],['1','2'])]
    assert selections[1][1]==['0','1']
    fits.clear();run(cfg,gpu_ids=['cpu'],device='cpu')
    assert len(selections)==2
    assert fits==[(1,['base'],['1']),(2,['model_1'],['1','2'])]
    cfg=replace(cfg,name='pristine',training_mode='pristine_refit')
    fits.clear();run(cfg,gpu_ids=['cpu'],device='cpu')
    assert all(start==['base'] for _,start,_ in fits)
