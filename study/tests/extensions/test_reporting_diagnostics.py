"""Check metric meaning and regeneration from saved evidence, without GPU work."""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uq_extensions.common import atomic_json
from uq_extensions.reporting import automation_capacity,referral_report_statistics,referral_diagnostic_report


def test_oracle_normalization_handles_complementarity_and_bad_ranking():
    # Same aggregate success does not eliminate the benefit of correct referral.
    learner=[1,1,0,0];supervisor=[0,0,1,1];tasks=['a']*4;ids=list('abcd')
    good,_,_=referral_report_statistics([0,0,1,1],learner,supervisor,tasks,ids)
    bad,_,_=referral_report_statistics([1,1,0,0],learner,supervisor,tasks,ids)
    assert good['oracle_normalized_gain']==pytest.approx(1)
    assert good['random_auc_success']==pytest.approx(.5)
    assert good['oracle_auc_success']==pytest.approx(.75)
    assert bad['oracle_normalized_gain']==pytest.approx(-1)
    # Expected random ranking has zero normalized gain when averaged over orders.
    gains=[referral_report_statistics(order,learner,supervisor,tasks,ids)[0]['oracle_normalized_gain']
           for order in itertools.permutations(range(4))]
    assert np.mean(gains)==pytest.approx(0,abs=1e-12)


@pytest.mark.parametrize('allocation',['global','within_task'])
def test_oracle_normalization_uses_equal_task_mass_and_matching_allocation(allocation):
    learner=np.array([1,0,1,0,0,0]);supervisor=1-learner
    statistics,ranked,oracle=referral_report_statistics(supervisor-learner,learner,supervisor,
        ['a']*2+['b']*4,list('abcdef'),allocation=allocation)
    assert statistics['oracle_normalized_gain']==pytest.approx(1)
    assert ranked.iloc[-1].utility==pytest.approx((.5+.25)/2)
    np.testing.assert_allclose(ranked.utility,oracle.utility)


def test_undefined_oracle_gap_and_spearman_are_not_reported_as_zero():
    statistics,_,_=referral_report_statistics([1,2,3],[0,0,0],[1,1,1],['a']*3,list('abc'))
    assert statistics['oracle_normalized_gain'] is None
    assert statistics['oracle_normalization_applicable'] is False
    assert statistics['signed_gain_spearman'] is None
    with pytest.raises(ValueError,match='finite'):
        referral_report_statistics([np.nan,2],[0,0],[1,1],['a']*2,list('ab'))


def test_automation_uses_last_feasible_point_on_nonmonotonic_curve():
    curve=pd.DataFrame(dict(automated=[0,.25,.5,.75,1],utility=[.8,.75,.9,.79,.7]))
    assert automation_capacity(curve,.02)==.75
    assert automation_capacity(curve,0)==.5
    # No interpolation to manufacture more automation between tested counts.
    assert automation_capacity(curve,.005)==.5
    with pytest.raises(ValueError):automation_capacity(curve,-.1)


def test_report_regeneration_matches_renderer_and_preserves_source_files(tmp_path,monkeypatch):
    import uq_extensions.common as common
    monkeypatch.setattr(common,'model_identity',lambda _: 'teacher')
    scoring=tmp_path/'scoring';cell=scoring/'round_000/S02_E'
    spec=dict(checkpoint='learner',starts=[dict(id=i) for i in 'abcd'],rendering_protocol='correct',
              observation_preprocessing='rotate180',split='development')
    def episodes(successes):
        return [dict(id=i,task=t,success=success,init_state_hash=i,reset_seed=0,repetition=0,
            condition='original',split='development',rendering_protocol='correct')
            for i,t,success in zip('abcd',['a','a','b','b'],successes)]
    atomic_json(scoring/'declaration.json',dict(supervisor=str(tmp_path/'teacher'),split='development',
        config=dict(training_mode='save_warm_50_50',seed=1000,batch_queries=5)))
    atomic_json(cell.parent/'learner/evaluation.json',dict(spec=spec,episodes=episodes([1,0,1,0])))
    cache=tmp_path/'supervisor'
    atomic_json(cache/'wrong/evaluation.json',dict(spec=dict(spec,checkpoint='teacher',rendering_protocol='old'),episodes=episodes([0,0,0,0])))
    atomic_json(cache/'correct/evaluation.json',dict(spec=dict(spec,checkpoint='teacher'),episodes=episodes([1,1,1,1])))
    atomic_json(cell/'kv/scores.json',dict(scores=dict(zip('abcd',[0,1,0,1])),pool_task_means={'a':1,'b':2}))
    matrix=[dict(directory=str(cell),round=0,recipe=dict(id='S02_E',method='kv'),metrics=dict(between_task_spearman=None))]
    sources={p:p.read_bytes() for p in tmp_path.rglob('*.json')}
    output=tmp_path/'report';result=referral_diagnostic_report(output,matrix,supervisor_cache=cache)
    assert result['unavailable']==[]
    assert len(result['records'])==2
    assert all(r['oracle_normalized_gain']==pytest.approx(1) for r in result['records'])
    assert all('/correct/' in r['supervisor_evaluation'] for r in result['records'])
    assert all(p.read_bytes()==contents for p,contents in sources.items())
    assert {'summary.csv','episode_correlations.csv','task_correlations.csv','referral_curves.csv','report.html'}<={p.name for p in output.iterdir()}
    assert all((output/name).exists() and (output/Path(name).with_suffix('.pdf')).exists() for name in result['figures'])
    assert 'data:image/png;base64' in (output/'report.html').read_text()
