"""Reporting must preserve evidence and compare the same statistical quantities."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from head_report import analysis
from head_report.data import capture, curve, curve_statistics
from generate_head_report import build_report
from uq_extensions.common import atomic_json, identity
from uq_extensions.referral import referral_curve
from uq_extensions.reporting import referral_report_statistics


@pytest.mark.parametrize('allocation',['global','within_task'])
def test_fast_curve_matches_existing_ties_weights_signed_gain_and_continuous_outcomes(allocation):
    scores=[.2,.2,.8,.8,.3,.1];learner=np.array([1,0,1,0,0,.4]);teacher=np.array([0,1,.5,1,.6,.5])
    tasks=['a']*2+['b']*4;ids=list('fedcba')
    actual=curve(scores,learner,teacher,tasks,ids,allocation)
    expected,summary=referral_curve(scores,learner,teacher,tasks,ids,allocation=allocation)
    np.testing.assert_allclose(actual,[[r['automated'],r['utility'],r['random_utility']] for r in expected])
    oracle=curve(teacher-learner,learner,teacher,tasks,ids,allocation)
    reported,_,_=referral_report_statistics(scores,learner,teacher,tasks,ids,allocation=allocation)
    stat=curve_statistics(actual,oracle)
    assert stat['nauc']==pytest.approx(reported['oracle_normalized_gain'])
    assert stat['automation']==pytest.approx(summary['automation_within_two_points'])
    assert curve_statistics(oracle,oracle)['nauc']==pytest.approx(1)


def test_undefined_oracle_headroom_and_nonfinite_data():
    actual=curve([1,2,3],[0,0,0],[1,1,1],['a']*3,list('abc'))
    assert curve_statistics(actual,actual)['nauc'] is None
    with pytest.raises(ValueError,match='Nonfinite'):
        curve([1,np.nan],[0,0],[1,1],['a','a'],list('ab'))


def recipe(name='kv',method='kv',precision='float32'):
    return dict(id=name,method=method,scope='observation',group='joint_late',
        features=dict(bank='joint_late',compression='pca',components=32,seed=42,expert_distance=0),
        gp=dict(input_projection='pca',input_components=32,output_compression='pca',output_components=16,
                output_seed=142,precision=precision,inference='exact',output_dependence='independent',
                jitter=1e-6,pls_variance_target=None,lengthscale_initialization='geometry'))


@pytest.fixture
def study(tmp_path):
    root=tmp_path/'study';cache=tmp_path/'supervisor'
    recipes=[recipe('kv'),recipe('kvt','kv_t'),recipe('later')]
    atomic_json(root/'declaration.json',dict(rounds=[0,10],recipes=recipes,supervisor_hash='teacher',split='development'))
    spec=dict(checkpoint='learner',split='development',observation_preprocessing='correct',max_steps=500)
    def episodes(success):
        return [dict(id=i,task=t,success=s,init_state_hash=i,reset_seed=0,repetition=0,
            condition='original',split='development',q_time=float(s)*.8,q_path=float(s)*.9)
            for i,t,s in zip('abcdef',['a','a','b','b','c','c'],success)]
    for r in (0,10):
        atomic_json(root/f'round_{r:03d}/learner/evaluation.json',dict(spec=spec,identity=f'eval{r}',episodes=episodes([1,0,0,0,1,0])))
    atomic_json(cache/'correct/evaluation.json',dict(spec=dict(spec,checkpoint='teacher'),episodes=episodes([0,1,1,1,1,1])))
    atomic_json(cache/'wrong/evaluation.json',dict(spec=dict(spec,checkpoint='teacher',observation_preprocessing='old'),episodes=episodes([0]*6)))

    def complete(name,r,*,wrong_identity=False):
        item=next(copy.deepcopy(x) for x in recipes if x['id']==name)
        directory=root/f'round_{r:03d}'/name
        request=dict(recipe=item,round=r)
        atomic_json(directory/'request.json',dict(identity=identity(request),spec=request))
        atomic_json(directory/'result.json',dict(recipe=item,round=r,status='complete',request_identity='wrong' if wrong_identity else identity(request),
                    seconds=1,metrics=dict(gp_diagnostics=dict(components=32,input_components=32,inference='exact'))))
        atomic_json(directory/item['method']/'scores.json',dict(scores=dict(zip('abcdef',[0,2,3,3,0,1])),pool_task_means=dict(a=1,b=3,c=.5)))
    for r in (0,10):complete('kv',r)
    complete('kvt',0)
    # Old float32 marker must not be mistaken for a current request's failure.
    req=dict(recipe=recipes[1],round=10)
    atomic_json(root/'round_010/kvt/request.json',dict(spec=req,identity=identity(req)))
    atomic_json(root/'round_010/kvt/failure.json',dict(request_identity='old-float32',status='failed',reason='old Cholesky error'))
    req=dict(recipe=recipes[2],round=0)
    atomic_json(root/'round_000/later/request.json',dict(spec=req,identity=identity(req)))
    atomic_json(root/'round_000/later/failure.json',dict(request_identity=identity(req),status='failed',reason='current error'))
    return root,cache,complete


def test_capture_preserves_files_separates_failures_and_excludes_mismatched_evidence(study):
    root,cache,complete=study
    before={p:p.read_bytes() for parent in (root,cache) for p in parent.rglob('*.json')}
    snapshot=capture(root,cache);data=analysis.frame(snapshot)
    assert snapshot['meta']['successful']==3
    assert snapshot['meta']['historical_failure_markers']==1
    assert snapshot['meta']['current_failures']==1
    assert not snapshot['issues']
    assert set(analysis.balanced_summary(data,[0,10])['head'])=={'kv'}
    assert all(p.read_bytes()==content for p,content in before.items())
    assert all('wrong/evaluation' not in p for p in snapshot['files'])
    complete('kvt',10,wrong_identity=True)
    invalid=capture(root,cache)
    row=next(r for r in invalid['rows'] if r['round']==10 and r['head']=='kvt')
    assert row['status']=='data_error' and not row['eligible']
    assert invalid['issues'][0]['kind']=='data_error'


def test_matched_pairs_flag_precision_and_do_not_mix_unrelated_inputs(study):
    root,cache,complete=study;snapshot=capture(root,cache);data=analysis.frame(snapshot)
    pairs=analysis.matched_pairs(snapshot,data)
    assert len(pairs)==1 and pairs.iloc[0]['family']=='KV → KV-T (PCA16 targets)'
    assert pairs.iloc[0].comparable_numerics
    data.loc[data.method=='kv_t','precision']='float64'
    pairs=analysis.matched_pairs(snapshot,data)
    assert not pairs.iloc[0].comparable_numerics
    assert 'different precision' in pairs.iloc[0].caveats
    snapshot['recipes'][(0,'kvt')]['features']['bank']='different_context'
    assert analysis.matched_pairs(snapshot,data).empty


def test_ablation_summary_keeps_mixed_precision_correlations_and_checkpoint_reversals(study):
    root,cache,complete=study;complete('kvt',10)
    snapshot=capture(root,cache);data=analysis.frame(snapshot)
    data.loc[data.method=='kv_t','precision']='float64'
    for r,delta in [(0,.2),(10,-.1)]:
        for metric in analysis.ABLATION_METRICS:
            data.loc[(data['round']==r)&(data.method=='kv'),metric]=0.
            data.loc[(data['round']==r)&(data.method=='kv_t'),metric]=delta
    pairs=analysis.matched_pairs(snapshot,data)
    assert len(pairs)==2 and not pairs.comparable_numerics.any()
    summaries=analysis.ablation_summary(pairs).set_index('subset')
    main=summaries.loc[analysis.ALL_PAIRS]
    for metric in analysis.ABLATION_METRICS:
        assert main['median_delta_'+metric]==pytest.approx(.05)
        assert main['defined_pairs_'+metric]==2
    assert 'mixed' in main.interpretation and 'reverses between checkpoints' in main.interpretation
    assert summaries.loc[analysis.MATCHED_PAIRS,'pairs']==0
    assert 'no conclusion' in summaries.loc[analysis.MATCHED_PAIRS,'interpretation']
    pairs['delta_global_nauc']=[.1,-.1]
    text=analysis.ablation_interpretation(pairs)
    assert 'below 0.001' in text and 'unchanged in all' not in text


def test_decoder_ablation_names_positions_and_holds_other_extraction_settings_fixed(study):
    root,cache,_=study;snapshot=capture(root,cache);data=analysis.frame(snapshot)
    data=data[(data['round']==0)&data.eligible].copy()
    for head,distance in [('kv',0),('kvt',1)]:
        rec=snapshot['recipes'][(0,head)]
        rec.update(method='kv',scope='action',group=f'expert_{distance}')
        rec['features'].update(bank='action_expert',expert_distance=distance,flow_samples=50)
        data.loc[data['head']==head,['method','scope','group']]=['kv','action',f'expert_{distance}']
    pairs=analysis.matched_pairs(snapshot,data)
    assert len(pairs)==1
    assert pairs.iloc[0]['family']==analysis.DECODER_ABLATION
    assert pairs.iloc[0]['contrast']=='Decoder: before output projection → 1 block earlier'
    snapshot['recipes'][(0,'kvt')]['features']['flow_samples']=4
    assert analysis.matched_pairs(snapshot,data).empty


def test_current_report_keeps_patch_raw_state_and_contextual_state_positions(study):
    root,cache,complete=study
    complete('later',0);complete('later',10)
    for out in root.glob('round_*/*'):
        if not (out/'result.json').exists():continue
        group={'kv':'state','kvt':'visual_patches','later':'contextual_state_late'}[out.name]
        req=json.loads((out/'request.json').read_text());result=json.loads((out/'result.json').read_text())
        req['spec']['recipe']['group']=group;req['identity']=identity(req['spec'])
        result['recipe']['group']=group;result['request_identity']=req['identity']
        atomic_json(out/'request.json',req);atomic_json(out/'result.json',result)
    before={p:p.read_bytes() for p in root.rglob('*.json')}
    snapshot=capture(root,cache);data=analysis.frame(snapshot)
    assert set(data[data.eligible]['group'])=={'state','visual_patches','contextual_state_late'}
    retained=data[data['group'].isin(['state','visual_patches'])]
    assert len(retained)==3 and retained.eligible.all() and retained.global_nauc.notna().all()
    assert all(p.read_bytes()==value for p,value in before.items())


def test_pairing_checks_initial_states_for_each_episode_id(study):
    root,cache,_=study
    path=cache/'correct/evaluation.json';value=json.loads(path.read_text())
    value['episodes'][0]['id'],value['episodes'][1]['id']=value['episodes'][1]['id'],value['episodes'][0]['id']
    atomic_json(path,value)
    snapshot=capture(root,cache)
    assert snapshot['meta']['eligible']==0
    assert any('different learner/supervisor initial states' in issue['reason'] for issue in snapshot['issues'])


def test_publish_twice_updates_latest_preserves_snapshot_and_sources(study,tmp_path):
    root,cache,complete=study;output=tmp_path/'report'
    first=build_report(root,output,cache,make_plots=False)
    old=(first['snapshot']/'report.html').read_bytes()
    assert 'Inspect any completed head' in old.decode()
    assert (first['snapshot']/'input_hashes.json').exists()
    complete('kvt',10)
    before={p:p.read_bytes() for parent in (root,cache) for p in parent.rglob('*.json')}
    second=build_report(root,output,cache,make_plots=False)
    assert second['meta']['successful']==4 and first['meta']['successful']==3
    assert first['snapshot']!=second['snapshot']
    assert (first['snapshot']/'report.html').read_bytes()==old
    assert all(p.read_bytes()==content for p,content in before.items())
    assert len(pd.read_csv(second['snapshot']/'balanced_heads.csv'))==2
    latest=json.loads((output/'latest.json').read_text())
    assert latest['snapshot']==second['snapshot'].name
    text=(output/'report.html').read_text()
    assert 'snapshots/'+second['snapshot'].name+'/all_heads.csv' in text
    assert 'Median Δ −Pearson (task)' in text and 'Median Δ signed-benefit Pearson' in text
    assert 'Short interpretation' in text and 'KV does not need to be retrained in float64' in text
    assert 'one transformer block earlier' in text
    assert 'Sensitivity check: matching numerical settings and seed IDs' in text
    assert not list(output.glob('.building-*'))


def test_failed_generation_does_not_replace_previous_report(study,tmp_path,monkeypatch):
    import generate_head_report as module
    root,cache,_=study;out=tmp_path/'out'
    build_report(root,out,cache,make_plots=False)
    before=(out/'report.html').read_bytes()
    def fail(*a,**kw):raise RuntimeError('render error')
    monkeypatch.setattr(module,'render',fail)
    with pytest.raises(RuntimeError,match='render error'):build_report(root,out,cache,make_plots=False)
    assert (out/'report.html').read_bytes()==before
    assert not list(out.glob('.building-*'))
    with pytest.raises(ValueError,match='separate'):build_report(root,root/'report',cache)
