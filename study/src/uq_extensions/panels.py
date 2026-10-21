"""Common-policy scoring and supervisor referral, separate from acquisition."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from .acquisition import variance_scores
from .baselines import METHODS, baseline_scores
from .common import atomic_json, identity, read_json, stream_seed
from .data import load_manifest, candidate_view
from .evaluation import evaluate, evaluation_view, OBSERVATION_PREPROCESSING, RENDERING_PROTOCOL
from .referral import pair_outcomes, referral_curve, score_metrics, clustered_referral_interval


def supervisor_cache_key(*, split, condition, model, manifest, max_steps, per_task):
    return identity(dict(split=split, condition=condition, model=str(Path(model).resolve()),
                         manifest=manifest, max_steps=max_steps, per_task=per_task,
                         observation_preprocessing=OBSERVATION_PREPROCESSING,
                         rendering_protocol=RENDERING_PROTOCOL))[:20]


def round_context(config, run_dir, round_index, *, allow_nonrandom=False):
    root=Path(run_dir); experiment=read_json(root/'experiment.json')
    if experiment['config']['arm']!='random' and not allow_nonrandom: raise ValueError('Principal score panel requires common Random histories')
    if experiment['config']['training_mode']!=config.training_mode: raise ValueError('Panel training-mode mismatch')
    if experiment['config']['eligibility_protocol']!=config.eligibility_protocol: raise ValueError('Panel eligibility mismatch')
    bases=experiment['base_models']; acquired=[]
    if round_index:
        fit=read_json(root/f'round_{round_index:03d}/policy_fit.json');models=fit['models']
        if not models:raise ValueError(f'Round {round_index} weights were retired; its recorded learning-curve results remain available')
        ledger=read_json(root/'acquisition_ledger.json')
        acquired=[id for r in ledger['rounds'][:round_index] for id in r['ids']]
    else:models=bases
    manifest=load_manifest(config.data_manifest)
    if manifest['manifest_hash']!=experiment['manifest_hash']: raise ValueError('Panel manifest mismatch')
    return models,manifest,candidate_view(manifest,sorted(set(manifest['pretraining_ids'])|set(acquired)))


def assess_scores(scores, pool_scores, query, learner, supervisor, directory):
    table={r['id']:r for r in learner}
    if set(table)!=set(query.ids):raise ValueError('Score/outcome IDs mismatch')
    tasks=query.tasks; success=[table[id]['success'] for id in query.ids]
    pool_means={task:float(np.mean([value for _,t,value in pool_scores if t==task])) for task in sorted(set(tasks))}
    h=None
    if supervisor is not None:
        pair_outcomes(learner,supervisor)
        byid={r['id']:r for r in supervisor};h=[byid[id]['success'] for id in query.ids]
    values=[scores[id] for id in query.ids]
    metrics=score_metrics(values,success,tasks,initial_pool_task_means=pool_means,supervisor=h)
    if h is not None:
        for allocation in ('global','within_task'):
            curve,summary=referral_curve(values,success,h,tasks,query.ids,allocation=allocation,
                                         clusters=[table[id]['init_state_hash'] for id in query.ids])
            metrics['referral_'+allocation]=summary
            summary['raw_gain_cluster_bootstrap_95']=clustered_referral_interval(values,success,h,tasks,
                [table[id]['init_state_hash'] for id in query.ids],allocation=allocation)
            atomic_json(Path(directory)/f'referral_{allocation}.json',curve)
            import pandas as pd
            pd.DataFrame(curve).to_csv(Path(directory)/f'referral_{allocation}.csv',index=False)
        mean_only={id:pool_means[task] for id,task in zip(query.ids,tasks)}
        _,metrics['task_mean_referral']=referral_curve([mean_only[id] for id in query.ids],success,h,tasks,query.ids,
                                                      clusters=[table[id]['init_state_hash'] for id in query.ids])
    atomic_json(Path(directory)/'metrics.json',metrics)
    atomic_json(Path(directory)/'scores.json',dict(scores=scores,pool_task_means=pool_means))
    return metrics


def score_panel(config, run_dir, round_index, directory, *, device, methods=('kv','kv_t',*METHODS),
                supervisor_model=None, supervisor_cache='storage/extensions/supervisor/evaluation',
                split=None, condition=None, allow_nonrandom=False, evaluation_directory=None):
    models,manifest,support=round_context(config,run_dir,round_index,allow_nonrandom=allow_nonrandom)
    directory=Path(directory);split=split or config.evaluation_split
    evaluation_directory=Path(evaluation_directory) if evaluation_directory else directory/'learner'
    learner=evaluate(models[0],manifest,evaluation_directory,split=split,device=device,
                     max_steps=config.evaluation_max_steps,condition=condition,batch_size=config.evaluation_batch_size,per_task=config.evaluation_per_task)
    query=evaluation_view(evaluation_directory);pool=candidate_view(manifest,manifest['target_ids'])
    if condition and condition.get('family')!='original':
        from .teaching import taught_support
        changed={c.id for c in pool.candidates if not condition.get('tasks') or c.task in condition['tasks']}
        pool=taught_support(manifest,pool.ids,condition,changed,directory/'condition_pool')
    supervisor=None
    if supervisor_model:
        cache=Path(supervisor_cache)/supervisor_cache_key(split=split,condition=condition,model=supervisor_model,
            manifest=manifest['manifest_hash'],max_steps=config.evaluation_max_steps,per_task=config.evaluation_per_task)
        supervisor=evaluate(supervisor_model,manifest,cache,split=split,device=device,
                            max_steps=config.evaluation_max_steps,condition=condition,batch_size=config.evaluation_batch_size,per_task=config.evaluation_per_task)
    results={}
    for method in methods:
        target=directory/method
        if method in ('kv','kv_t'):
            cfg=replace(config,arm=method)
            if config.frozen_recipe:
                from .config import FeatureConfig,GPConfig
                recipe=read_json(config.frozen_recipe)
                cfg=replace(cfg,features=FeatureConfig(**recipe['kv_t_features' if method=='kv_t' else 'kv_features']),
                    gp=GPConfig(**recipe.get('kv_t_gp' if method=='kv_t' else 'kv_gp',recipe.get('gp'))))
            scores,estimator,_=variance_scores(cfg,models[0],support,query,target,device,role=split)
            from .features import cache_features
            px=cache_features(models[0],pool,cfg.features,target/'pool_features',device=device,
                              batch_size=cfg.feature_batch_size,banks=(cfg.features.bank,),role='candidate',support_ids=support.ids)[cfg.features.bank]
            poolvalues=estimator.score(px)
            mapping=estimator.diagnostics.get('mapping_check')
        else:
            scores=baseline_scores(method,models,query,target/'episode_scores',device=device,seed=stream_seed(config.seed,round_index,'score'))
            pooled=baseline_scores(method,models,pool,target/'pool_scores',device=device,seed=stream_seed(config.seed,round_index,'pool'))
            poolvalues=[pooled[id] for id in pool.ids];mapping=None
        metrics=assess_scores(scores,list(zip(pool.ids,pool.tasks,poolvalues)),query,learner,supervisor,target)
        if mapping is not None:metrics['mapping_check']=mapping
        if method in ('kv','kv_t'):metrics['gp_diagnostics']=estimator.diagnostics
        atomic_json(target/'metrics.json',metrics)
        results[method]=metrics
    atomic_json(directory/'panel.json',dict(round=round_index,training_mode=config.training_mode,split=split,
                condition=condition,models=models,support_ids=support.ids,results=results))
    return results
