"""Shared report structure, nested policy statistics, portable plots, and safe publication."""
from html.parser import HTMLParser
from pathlib import Path
import base64
import json

import numpy as np
from PIL import Image
import pytest

from uq_extensions.common import atomic_json, atomic_npz, file_hash
from head_report.bootstrap import Plan, METRICS
from head_report import followup, followup_data
from head_report.html import clean


@pytest.fixture
def report_study(tmp_path):
    recipes=[]
    for method,variant,seed,white in [('kv','random512',42,False),('kv','random512',43,False),
                                    ('kv','pca512',42,False),('kv','pca512',42,True),('kv_t','pca512',42,False)]:
        recipes.append(dict(id=f'middle_to_final_context__{method}_{variant}_s{seed}'+('__whiten_lw' if white else ''),
            group='middle_to_final_context',method=method,variant=variant,whiten=white,scope='observation',
            features=dict(bank='joint_intermediate',target_bank='joint_late',seed=seed),
            gp=dict(precision='float64' if method=='kv_t' else 'float32',output_compression='pca',output_components=16)))
    policies=[dict(id=f'{c}_seed{s}',condition=c,seed=s) for c in ('original','plus35') for s in range(3)]
    atomic_json(tmp_path/'declaration.json',dict(seeds=[0,1,2],policies=policies,recipes=recipes))
    for p in policies:
        pid=p['id'];seed=p['seed'];left=[];right=[]
        for t in range(3):
            for i in range(4):
                id_=f't{t}/s{i}';common=dict(id=id_,task=f't{t}',init_state_hash=id_,reset_seed=i,initial_input=dict(input_hash=id_))
                l=((i+t+seed+(p['condition']=='plus35'))%4)/3;h=1 if i else 0
                left.append(dict(common,success=l,q_time=l*.5,q_path=l*.8))
                right.append(dict(common,success=h,q_time=h*.6,q_path=h*.7))
        atomic_json(tmp_path/'evaluations'/pid/'conditions.json',left)
        atomic_json(tmp_path/'evaluations'/f'supervisor_seed{seed}'/'conditions.json',right)
        from uq_followup.reporting import cohort_for
        cohort=cohort_for(tmp_path,pid);point=Plan(cohort,indices=np.arange(12)[None,:]);boot=Plan(cohort,repetitions=2000)
        for recipe in recipes:
            feature_seed=recipe['features']['seed'];scores=np.random.default_rng(seed+feature_seed).normal(size=12)
            means={f't{i}':float(scores[i*4:(i+1)*4].mean()) for i in range(3)}
            values,_=point.evaluate(scores,means);samples,_=boot.evaluate(scores,means)
            metrics=clean({m:values[0,j] for j,m in enumerate(METRICS) if m!='failure_auroc'})
            d=tmp_path/'heads'/pid/recipe['id']
            atomic_json(d/'scores.json',dict(ids=cohort['ids'],scores=scores.tolist(),pool_task_means=means))
            atomic_npz(d/'bootstrap.npz',metrics=samples,names=np.array(METRICS))
            atomic_json(d/'result.json',dict(status='complete',policy_id=pid,recipe=recipe,metrics=metrics,seconds=5))
    return tmp_path


def test_policy_seed_statistics_and_matched_intervals(report_study):
    snapshot=followup_data.capture(report_study);review=followup_data.aggregate(snapshot)
    family='middle_to_final_context__kv_random512'
    bundle=review['bundles'][('original',family)]
    row=next(r for r in review['summaries'] if r['condition']=='original' and r['head']==family)
    expected=np.array([np.mean([r['observed'] for r in bundle['runs'][seed]],axis=0) for seed in (0,1,2)])
    assert row['global_nauc_mean']==pytest.approx(expected[:,0].mean())
    assert row['global_nauc_std']==pytest.approx(expected[:,0].std(ddof=1))
    assert row['coordinate_repetitions']=='2,2,2' and row['coverage']=='complete'
    selected,_=followup_data.select(review)
    pairs,ablations,comparisons=followup_data.paired_comparisons(review,selected)
    assert set(pairs.family)=={'KV → KV-T (matched inputs)','Covariance whitening','Random512 → PCA512'}
    matching=comparisons[(comparisons.family=='KV → KV-T (matched inputs)')&(comparisons.metric=='global_nauc')]
    assert len(matching)==2
    assert np.allclose(matching[['mean_difference','total_low','total_high']],0)
    # Policy ID availability is insufficient when a required coordinate subset is absent.
    (report_study/'heads/original_seed2/middle_to_final_context__kv_random512_s43/result.json').unlink()
    updated=followup_data.aggregate(followup_data.capture(report_study))
    row=next(r for r in updated['summaries'] if r['condition']=='original' and r['head']==family)
    assert row['coverage']=='partial' and row['complete_policy_seeds']==2


def test_standalone_structure_plots_provenance_and_failed_publication(report_study,monkeypatch):
    def figures(snapshot,review,selected,heads,destination,cache):
        destination.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(10,10),'blue').save(destination/'leaders_original.png')
        (destination/'leaders_original.pdf').write_bytes(b'%PDF-1.4\n')
        return [dict(name='leaders_original',title='A standalone plot',caption='Saved test figure')]
    monkeypatch.setattr(followup.followup_plots,'generate',figures)
    inputs={str(p):file_hash(p) for p in report_study.rglob('*') if p.is_file()}
    output=followup.build_report(report_study);text=output.read_text()
    class Document(HTMLParser):
        def __init__(self):super().__init__();self.ids=[];self.images=[];self.links=[]
        def handle_starttag(self,tag,attrs):
            d=dict(attrs)
            if 'id' in d:self.ids.append(d['id'])
            if tag=='img':self.images.append(d['src'])
            if tag=='a':self.links.append(d.get('href',''))
    parsed=Document();parsed.feed(text)
    assert all(anchor in parsed.ids for anchor,_ in followup.NAV)
    assert len(parsed.images)==1 and parsed.images[0].startswith('data:image/png;base64,')
    assert base64.b64decode(parsed.images[0].split(',',1)[1]).startswith(b'\x89PNG')
    for href in parsed.links:
        if href and not href.startswith(('#','data:','https:','http:')):assert (output.parent/href).exists(),href
    assert 'head-choice' in parsed.ids and 'report-data' in parsed.ids and 'all-heads' in parsed.ids
    assert 'inspector-status' in parsed.ids
    assert 'Open saved per-head report' not in text
    assert 'three independently trained policies' in text and 'feature-selection seeds' not in text
    assert 'Mean success improvement (−1 to +1)' in text
    assert all(file_hash(path)==digest for path,digest in inputs.items())
    latest=json.loads((output.parent/'latest.json').read_text())
    snapshot=output.parent/'snapshots'/latest['snapshot']
    before=output.read_bytes();old_summary=(output.parent/'summary.json').read_bytes();immutable=(snapshot/'report.html').read_bytes()
    def fail(*args,**kwargs):raise RuntimeError('render failed')
    monkeypatch.setattr(followup,'render',fail)
    with pytest.raises(RuntimeError,match='render failed'):followup.build_report(report_study,make_plots=False)
    assert output.read_bytes()==before and (output.parent/'summary.json').read_bytes()==old_summary
    assert (snapshot/'report.html').read_bytes()==immutable
    assert not list(output.parent.glob('.building-*'))
