import numpy as np
from uq_extensions.common import atomic_json,atomic_npz,array_hash,identity
from uq_extensions.config import ExperimentConfig
from uq_extensions.protocol import Candidate,CandidateView


def test_teaching_matches_budgets_and_only_transforms_relevant_branch(tmp_path,monkeypatch):
    import uq_extensions.teaching as module
    source=tmp_path/'source';atomic_json(source/'experiment.json',{'base_models':['base']})
    condition={'name':'language','family':'undertrained_language','tasks':['gap'],'instructions':{'gap':'clear alternate instruction'}}
    cpath=tmp_path/'condition.json';atomic_json(cpath,condition)
    candidates=[]
    for i in range(9):
        obs=dict(image=np.zeros((4,4,3),np.uint8),image2=np.zeros((4,4,3),np.uint8),state=np.zeros(8),instruction=np.asarray('original'))
        path=tmp_path/f'{i}.npz';atomic_npz(path,**obs)
        candidates.append(Candidate(str(i),i,'gap',str(path),identity({k:array_hash(v) for k,v in obs.items()})))
    view=CandidateView(candidates);support=view.subset(['0'])
    manifest={'target_ids':view.ids,'episode_map':{id:int(id) for id in view.ids}}
    monkeypatch.setattr(module,'round_context',lambda *a:(['current'],manifest,support))
    monkeypatch.setattr(module,'candidate_view',lambda m,ids:view.subset(ids))
    monkeypatch.setattr(module,'evaluate',lambda model,m,path,**kw:[dict(id='evaluation',task='gap',success=int('relevant' in str(path)))])
    monkeypatch.setattr(module,'evaluation_view',lambda *a:support)
    fits=[]
    def fit(c,m,starts,ids,**kw):
        fits.append(dict(starts=starts,ids=ids,overlay=kw.get('overlay'),steps=c.steps))
        return ['fitted']
    monkeypatch.setattr(module,'fit_policies',fit)
    monkeypatch.setattr(module,'retire_optimizers',lambda *a:None)
    def variance(c,model,support,query,*a,**kw):
        instructions=[str(support.observation(id)['instruction']) for id in support.ids]
        if '/relevant/' in str(a[0]):assert instructions.count('clear alternate instruction')==5
        return {'evaluation':.5},None,None
    monkeypatch.setattr(module,'variance_scores',variance)
    config=ExperimentConfig(phase='smoke',training_mode='pristine_refit',fit_steps=2,refit_steps=2)
    result=module.teach(config,source,0,cpath,tmp_path/'teaching',gpu_ids=['cpu'],device='cpu')
    assert [f['starts'] for f in fits]==[['base'],['base']]
    assert all(len(f['ids'])==5 and f['steps']==2 for f in fits)
    assert len(fits[0]['overlay'])==5 and fits[1]['overlay'] is None
    assert result['relevant_minus_random']==1
