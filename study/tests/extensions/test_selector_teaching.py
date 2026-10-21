import numpy as np
from uq_extensions.common import atomic_json,atomic_npz,array_hash,identity
from uq_extensions.config import ExperimentConfig
from uq_extensions.protocol import Candidate,CandidateView


def test_three_selectors_share_condition_inputs_and_demo_rules(tmp_path,monkeypatch):
    import uq_extensions.teaching as m
    import uq_extensions.acquisition as acquisition
    import uq_extensions.reporting as reporting
    source=tmp_path/'source';atomic_json(source/'experiment.json',{'base_models':['base']})
    condition={'name':'language','family':'undertrained_language','tasks':['gap'],'instructions':{'gap':'alternate'}}
    path=tmp_path/'condition.json';atomic_json(path,condition);records=[]
    for i in range(12):
        obs=dict(image=np.zeros((4,4,3),np.uint8),image2=np.zeros((4,4,3),np.uint8),state=np.zeros(8),instruction=np.asarray('original'))
        target=tmp_path/f'{i}.npz';atomic_npz(target,**obs)
        records.append(Candidate(str(i),i,'gap' if i<8 else 'other',str(target),identity({k:array_hash(v) for k,v in obs.items()})))
    view=CandidateView(records);support=view.subset(['0']);manifest={'target_ids':view.ids,'integration_ids':view.ids,'episode_map':{id:int(id) for id in view.ids}}
    monkeypatch.setattr(m,'round_context',lambda *a:(['current'],manifest,support))
    monkeypatch.setattr(m,'candidate_view',lambda manifest,ids:view.subset(ids))
    seen=[]
    def acquire(cfg,models,support,pool,*a):
        instructions={id:str(pool.observation(id)['instruction']) for id in pool.ids};seen.append(instructions)
        assert instructions['1']=='alternate' and instructions['9']=='original'
        return ['1','2','3','9','10'],{'scores':dict.fromkeys(pool.ids,0)}
    monkeypatch.setattr(acquisition,'acquire',acquire)
    monkeypatch.setattr(m,'evaluate',lambda *a,**kw:[dict(id='e',task='gap',success=1),dict(id='f',task='other',success=0)])
    monkeypatch.setattr(m,'evaluation_view',lambda *a:support)
    fits=[]
    def fit(cfg,manifest,starts,ids,**kw):fits.append((starts,ids,kw.get('overlay')));return ['trained']
    monkeypatch.setattr(m,'fit_policies',fit);monkeypatch.setattr(m,'retire_optimizers',lambda *a:None)
    monkeypatch.setattr(m,'variance_scores',lambda *a,**kw:({'e':.2,'f':.1},None,None))
    monkeypatch.setattr(reporting,'teaching_report',lambda *a:None)
    cfg=ExperimentConfig(phase='smoke',training_mode='pristine_refit',fit_steps=2,refit_steps=2)
    result=m.teach_selectors(cfg,source,0,path,tmp_path/'teaching',gpu_ids=['cpu'],device='cpu')
    assert len(fits)==3 and seen[0]==seen[1]==seen[2]
    for starts,ids,overlay in fits:
        assert starts==['base'] and len(ids)==5 and set(overlay)=={'1','2','3'}
    assert all(v['condition_demo_count']==3 for v in result['branches'].values())
