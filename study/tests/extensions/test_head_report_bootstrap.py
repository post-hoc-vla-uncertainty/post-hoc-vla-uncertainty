"""Check the resampling unit, metric equivalence, seed grouping, and joint pairing."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr, pearsonr

from head_report.bootstrap import METRICS, Plan, cached_metrics, combine, fixed_order, interval, resample_indices, seed_draws, interpolate_rows
from head_report.data import curve, curve_statistics, extract_scores
from head_report.analysis import frame
from head_report.seed_summary import prepare
from uq_extensions.referral import score_metrics


def cohort():
    ids=list('abcdefgh');tasks=['a','a','b','b','c','c','c','c']
    learner=np.array([1,0,0,0,1,0,1,0.]);teacher=np.array([0,1,1,1,0,1,1,1.])
    rates={t:float(learner[np.array(tasks)==t].mean()) for t in set(tasks)}
    return dict(ids=ids,tasks=tasks,clusters=ids,learner=learner,supervisor=teacher,rates=rates,
        qualities={'time':(learner*.8,teacher*.6),'path':(learner*.6,teacher*.9)},
        oracles={a:curve(teacher-learner,learner,teacher,tasks,ids,a) for a in ('global','within_task')})


def test_vectorized_original_sample_matches_existing_report_all_metrics():
    c=cohort();scores=np.array([.2,.2,.8,.3,.7,.3,.1,.7]);means={'a':.2,'b':.8,'c':.5}
    plan=Plan(c,indices=np.arange(8)[None,:]);values,curves=plan.evaluate(scores,means)
    expected,_=extract_scores(dict(scores=dict(zip(c['ids'],scores)),pool_task_means=means),c,{})
    for metric in METRICS:
        if metric=='failure_auroc':continue
        assert values[0,METRICS.index(metric)]==pytest.approx(expected[metric],abs=1e-12)
    discrimination=score_metrics(scores,c['learner'],c['tasks'],initial_pool_task_means=means)
    assert values[0,METRICS.index('failure_auroc')]==pytest.approx(discrimination['macro_auroc'])
    for old,new in [('global','global_'),('within_task','within')]:
        np.testing.assert_allclose(np.column_stack([a[0] for a in curves[new]]),curve(scores,c['learner'],c['supervisor'],c['tasks'],c['ids'],old),atol=1e-14)


def test_vectorized_resamples_agree_with_explicit_threshold_masks_and_correlations():
    c=cohort();scores=np.array([.2,.2,.8,.3,.7,.3,.1,.7]);means={'a':.2,'b':.8,'c':.5}
    plan=Plan(c,repetitions=17,seed=95);values,curves=plan.evaluate(scores,means)
    ranks=fixed_order(scores,c['ids'])
    for b,indices in enumerate(plan.indices):
        tasks=np.asarray(c['tasks'])[indices];lv=c['learner'][indices];tv=c['supervisor'][indices]
        w=np.array([1/(3*np.sum(tasks==t)) for t in tasks]);order=np.argsort(ranks[indices],kind='stable')
        expected=[]
        for k in range(len(indices)+1):
            mask=np.zeros(len(indices),bool);mask[order[:k]]=True
            x=w@mask;y=w@np.where(mask,lv,tv);random=x*(w@lv)+(1-x)*(w@tv)
            expected.append([x,y,random])
        np.testing.assert_allclose(np.column_stack([a[b] for a in curves['global_']]),expected,atol=1e-14)
        sx=scores[indices];gain=tv-lv
        for name,fun in [('benefit_spearman',spearmanr),('benefit_pearson',pearsonr)]:
            expected_value=fun(sx,gain).statistic if np.ptp(sx) and np.ptp(gain) else np.nan
            np.testing.assert_allclose(values[b,METRICS.index(name)],expected_value,atol=1e-14,equal_nan=True)


def test_clusters_preserve_repeated_rollouts_and_task_sizes_and_match_checkpoints():
    c=cohort();c['clusters']=['a','a','b','b','c','c','d','d']
    indices=resample_indices(c,40,123)
    assert indices.shape==(40,8)
    for sample in indices:
        assert np.array_equal(np.asarray(c['tasks'])[sample],c['tasks'])
        for state in set(c['clusters']):
            counts=np.bincount(sample,minlength=8)[np.asarray(c['clusters'])==state]
            assert len(set(counts))==1
    other=deepcopy(c);other['learner']=1-c['learner']
    np.testing.assert_array_equal(indices,resample_indices(other,40,123))
    c['clusters'][-1]='other'
    with pytest.raises(ValueError,match='equal rollout counts'):resample_indices(c)


def test_joint_bootstrap_resamples_seeds_without_breaking_shared_evaluation_draws():
    samples=np.array([[[1.],[2.],[3.],[4.]],[[10.],[20.],[30.],[40.]]])
    evaluation,total=combine(samples,[42,43],seed=4)
    draws=seed_draws([42,43],4,seed=4)
    np.testing.assert_allclose(evaluation[:,0],[5.5,11,16.5,22])
    expected=np.array([np.mean([samples[s,b,0] for s in draws[:,b]]) for b in range(4)])
    np.testing.assert_allclose(total[:,0],expected)
    reversed_eval,reversed_total=combine(samples[::-1],[43,42],seed=4)
    np.testing.assert_array_equal(reversed_eval,evaluation);np.testing.assert_array_equal(reversed_total,total)
    assert combine(samples[:1],[42])[1] is None
    # Identical heads must have exactly zero paired bootstrap difference.
    _,same=combine(samples,[42,43],seed=4)
    np.testing.assert_array_equal(total-same,0)


def test_undefined_replicates_are_counted_not_imputed_and_cache_reused(tmp_path):
    assert np.isnan(interval([1,np.nan,np.nan])[0])
    assert interval([1,np.nan,np.nan])[2]==1
    c=cohort();plan=Plan(c,30,seed=8)
    detail=dict(scores=np.arange(8.),task_names=['a','b','c'],task_means=[1,2,3])
    a,hit=cached_metrics(plan,detail,tmp_path);assert not hit
    b,hit=cached_metrics(plan,detail,tmp_path);assert hit
    np.testing.assert_array_equal(a,b)
    detail['scores']=-detail['scores'];_,hit=cached_metrics(plan,detail,tmp_path);assert not hit
    x=np.array([[0,.1,.8,1],[0,.6,.9,1]]);y=np.array([[1,2,4,5],[1,1.5,5,6]])
    targets=np.linspace(0,1,30)
    np.testing.assert_allclose(interpolate_rows(x,y,targets),[np.interp(targets,a,b) for a,b in zip(x,y)])


def seed_study():
    c=cohort();rows=[];recipes={};payload={}
    for r in (0,10):
        for seed in (42,43):
            head=f'joint_late__kv_random32_s{seed}'
            recipe=dict(id=head,method='kv',scope='observation',group='joint_late',
                features=dict(bank='joint_late',compression='random',components=32,seed=seed,expert_distance=0),
                gp=dict(input_projection='pca',input_components=32,output_compression='pca',output_components=16,
                    output_seed=142,precision='float32',inference='exact',output_dependence='independent',
                    jitter=1e-6,pls_variance_target=None,lengthscale_initialization='geometry'))
            scores=np.array([.1,.3,.2,.9,.7,.5,.8,.4])*(1 if seed==42 else -1)
            saved=dict(scores=dict(zip(c['ids'],scores)),pool_task_means=dict(a=.2,b=.8,c=.5))
            metrics,detail=extract_scores(saved,c,{})
            row=dict(round=r,head=head,method='kv',scope='observation',group='joint_late',label=head,status='complete',
                eligible=True,precision='float32',posterior_precision='float32',inference='exact',jitter_escalated=False,
                quality='no recorded escalation',projection_seed=seed,actual_input_dim=32,actual_output_dim=None,
                **metrics)
            rows.append(row);recipes[r,head]=recipe;payload[r,head]=detail
    return dict(meta=dict(rounds=[0,10]),cohorts={0:c,10:c},rows=rows,recipes=recipes,payload=payload)


def test_seed_means_and_sd_use_configurations_and_common_seeds_across_rounds(tmp_path):
    s=seed_study();result=prepare(s,frame(s),tmp_path,repetitions=40)
    assert len(result['data'])==2 and len(result['balanced'])==1
    for r,group in result['data'].groupby('round'):
        values=[x['global_nauc'] for x in s['rows'] if x['round']==r]
        assert group.iloc[0].global_nauc==pytest.approx(np.mean(values))
        assert group.iloc[0].global_nauc_sd==pytest.approx(np.std(values,ddof=1))
        assert group.iloc[0].seed_count==2
        assert np.isfinite(group.iloc[0].global_nauc_total_low)
    # If round 10 has only seed 42, the balanced result uses seed 42 at BOTH rounds.
    s['rows'][-1]['eligible']=False;s['rows'][-1]['status']='pending'
    partial=prepare(s,frame(s),tmp_path,repetitions=40)
    row=partial['balanced'].iloc[0]
    assert row.seed_count==1
    assert row.mean_global_nauc==pytest.approx(s['rows'][0]['global_nauc'])
    assert np.isnan(row.mean_global_nauc_sd) and np.isnan(row.mean_global_nauc_total_low)


def test_numerical_versions_do_not_become_seed_sd(tmp_path):
    s=seed_study()
    for row in s['rows']:
        if row['projection_seed']==43:
            row.update(precision='float64',posterior_precision='float64')
            s['recipes'][row['round'],row['head']]['gp']['precision']='float64'
    result=prepare(s,frame(s),tmp_path,repetitions=20)
    assert len(result['data'])==4
    assert set(result['data'].seed_count)=={1}
    assert result['data'].global_nauc_sd.isna().all()
    assert result['data'].global_nauc_total_low.isna().all()
    assert set(result['data'].seed_coverage)=={'1/2'}


def test_degenerate_oracle_and_task_correlations_remain_undefined():
    c=cohort();c['learner']=np.zeros(8);c['supervisor']=np.ones(8);c['qualities']={}
    values,_=Plan(c,100).evaluate(np.arange(8.),{'a':1,'b':2,'c':3})
    for metric in ('global_nauc','within_nauc','neg_spearman','neg_pearson','benefit_spearman'):
        assert np.isnan(values[:,METRICS.index(metric)]).all()
        assert interval(values[:,METRICS.index(metric)])[2]==0
