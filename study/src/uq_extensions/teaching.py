"""Matched five-demo relevant/random teaching branches with frozen conditions."""
from dataclasses import replace
from pathlib import Path
import os
import numpy as np
from .common import array_hash,atomic_json,atomic_npz,identity,read_json,stream_seed,preserve_environment
from .conditions import validate_condition,apply_condition
from .data import candidate_view
from .evaluation import evaluate,evaluation_view
from .panels import round_context
from .protocol import Candidate,CandidateView,select_save
from .training import fit_policies,retire_optimizers
from .acquisition import variance_scores


@preserve_environment('UQ_EXT_RUN_ROOT','UQ_EXT_BUDGET_GB')
def teach_selectors(config,run_dir,round_index,condition_path,directory,*,gpu_ids,device,supervisor_model=None,specialized_features=None):
    """Three selectors see the identical mixed, condition-consistent input pool."""
    from .config import FeatureConfig,GPConfig
    from .acquisition import acquire
    condition=read_json(condition_path);validate_condition(condition,final=config.phase=='final')
    if condition['family'] not in ('undertrained_language','undertrained_appearance'):raise ValueError('Select a language or visual teaching condition')
    if config.phase=='final':
        frozen=read_json(config.frozen_recipe)
        if identity(condition) not in [identity(c) for c in frozen.get('conditions',[])]:raise ValueError('Condition is absent from the frozen teaching recipe')
    models,manifest,support=round_context(config,run_dir,round_index)
    previous=read_json(Path(run_dir)/f'round_{round_index:03d}/complete.json')['acquired'] if round_index else []
    excluded=set(support.ids) if config.eligibility_protocol=='unique_demo' else set()
    eligible=candidate_view(manifest,sorted(set(manifest['target_ids'])-excluded))
    changed={c.id for c in eligible.candidates if c.task in condition['tasks']}
    if len(changed)<5:raise ValueError('Teaching condition needs at least five eligible affected demonstrations at this history')
    root=Path(directory).resolve()
    mixed=taught_support(manifest,eligible.ids,condition,changed,root/'candidate_inputs')
    integration=candidate_view(manifest,manifest['integration_ids'])
    integration=taught_support(manifest,integration.ids,condition,{c.id for c in integration.candidates if c.task in condition['tasks']},root/'reference_inputs')
    specialized=specialized_features or replace(config.features,bank='language' if condition['family']=='undertrained_language' else 'visual_patches',target_bank='joint_late')
    configurations={'specialized':replace(config,arm='kv',companion=False,features=specialized),
                    'overall':replace(config,arm='kv_t',companion=False),'random':replace(config,arm='random',companion=False)}
    request=dict(condition=condition,source_run=str(Path(run_dir).resolve()),round=round_index,eligible=mixed.ids,
        condition_applied_ids=sorted(changed),previous=previous,seed=config.seed,training_mode=config.training_mode,
        selector_configs={k:v.scientific_dict() for k,v in configurations.items()})
    from .common import plain
    if (root/'teaching.json').exists() and read_json(root/'teaching.json')!=plain(request):raise ValueError('Teaching experiment changed')
    atomic_json(root/'teaching.json',request)
    starts=[read_json(Path(run_dir)/'experiment.json')['base_models'][0]] if config.training_mode=='pristine_refit' else models[:1]
    kwargs=dict(split=config.evaluation_split,device=device,max_steps=config.evaluation_max_steps,batch_size=config.evaluation_batch_size,per_task=config.evaluation_per_task)
    before=evaluate(models[0],manifest,root/'before_condition',condition=condition,**kwargs)
    ordinary=evaluate(models[0],manifest,root/'before_ordinary',**kwargs)
    teacher=evaluate(supervisor_model,manifest,root/'supervisor',condition=condition,**kwargs) if supervisor_model else None
    outcomes={};scores={};batches={}
    import pandas as pd
    def macro(rows):return float(pd.DataFrame(rows).groupby('task').success.mean().mean())
    def unaffected(rows):
        other=[r for r in rows if r['task'] not in condition['tasks']]
        return macro(other) if other else None
    from .referral import referral_curve,pair_outcomes
    referral={}
    def referral_metrics(key,score,learner):
        if teacher is None:return
        pair_outcomes(learner,teacher);h={r['id']:r['success'] for r in teacher}
        curve,metrics=referral_curve([score[r['id']] for r in learner],[r['success'] for r in learner],
            [h[r['id']] for r in learner],[r['task'] for r in learner],[r['id'] for r in learner])
        referral[key]=metrics;atomic_json(root/'referral'/f'{key}.json',dict(curve=curve,metrics=metrics))
    scores['before']={}
    for head in ('overall','specialized'):
        scores['before'][head]=variance_scores(configurations[head],models[0],support,evaluation_view(root/'before_condition'),root/'before_scores'/head,device,role=config.evaluation_split)[0]
        referral_metrics('before_'+head,scores['before'][head],before)
    for branch,cfg in configurations.items():
        out=root/branch;out.mkdir(parents=True,exist_ok=True)
        ids,audit=acquire(cfg,models[:1],support,mixed,integration,out/'selection',round_index+1,device)
        # Persist commitment before any training indices/trajectory overlays are passed on.
        commitment=dict(ids=ids,eligible=mixed.ids,condition_applied_ids=sorted(set(ids)&changed),audit=audit)
        if (out/'commitment.json').exists() and read_json(out/'commitment.json')!=plain(commitment):
            # Wall-clock timing changes on resume, selected IDs and inputs may not.
            old=read_json(out/'commitment.json')
            if old['ids']!=ids or old['eligible']!=mixed.ids:raise ValueError('Teaching selection changed')
        atomic_json(out/'commitment.json',commitment);batches[branch]=ids
        overlay={str(manifest['episode_map'][id]):dict(task=next(c.task for c in mixed.candidates if c.id==id),condition=condition)
                 for id in ids if id in changed}
        os.environ['UQ_EXT_RUN_ROOT']=str(out);os.environ['UQ_EXT_BUDGET_GB']=str(cfg.budget_gb)
        fitted=fit_policies(cfg,manifest,starts,[*previous,*ids],run_dir=out,round_index=round_index+1,gpu_ids=gpu_ids,overlay=overlay or None)
        after=evaluate(fitted[0],manifest,out/'condition',condition=condition,**kwargs)
        control=evaluate(fitted[0],manifest,out/'ordinary',**kwargs)
        augmented=taught_support(manifest,sorted(set(support.ids)|set(ids)),condition,set(ids)&changed,out/'support_inputs')
        scores[branch]={}
        for head,head_cfg in (('overall',configurations['overall']),('specialized',configurations['specialized'])):
            scores[branch][head]=variance_scores(head_cfg,fitted[0],augmented,evaluation_view(out/'condition'),out/'scores'/head,device,role=config.evaluation_split)[0]
            referral_metrics(branch+'_'+head,scores[branch][head],after)
        retire_optimizers(fitted)
        outcomes[branch]=dict(condition=after,ordinary=control,success=macro(after),ordinary_success=macro(control),
            improvement=macro(after)-macro(before),ordinary_change=macro(control)-macro(ordinary),condition_demo_count=len(set(ids)&changed),
            unaffected_change=unaffected(control)-unaffected(ordinary) if unaffected(control) is not None else None,
            remaining_supervisor_gap=macro(teacher)-macro(after) if teacher else None)
    for branch in outcomes:outcomes[branch]['improvement_over_random']=outcomes[branch]['improvement']-outcomes['random']['improvement']
    overlap={a+'/'+b:sorted(set(batches[a])&set(batches[b])) for a,b in (('specialized','overall'),('specialized','random'),('overall','random'))}
    result=dict(before=before,before_ordinary=ordinary,supervisor=teacher,branches=outcomes,batches=batches,overlap=overlap,scores=scores,referral=referral)
    atomic_json(root/'teaching_results.json',result)
    pd.DataFrame([dict(branch=k,**{x:v for x,v in row.items() if x not in ('condition','ordinary')}) for k,row in outcomes.items()]).to_csv(root/'branch_comparison.csv',index=False)
    from .reporting import teaching_report
    teaching_report(root,result,mixed)
    from .storage import retire_feature_arrays
    retire_feature_arrays(root)
    return result


def taught_support(manifest,ids,condition,condition_ids,directory):
    view=candidate_view(manifest,ids);candidates=[]
    for c in view.candidates:
        candidates.append(c)
        if c.id not in condition_ids:continue
        observation=apply_condition(view.observation(c.id),condition,c.task)
        path=Path(directory)/(str(c.episode_index)+'.npz');atomic_npz(path,**observation)
        candidates[-1]=Candidate(id=c.id,episode_index=c.episode_index,task=c.task,observation_file=str(path),
            input_hash=identity({k:array_hash(v) for k,v in observation.items()}))
    return CandidateView(candidates)


@preserve_environment('UQ_EXT_RUN_ROOT', 'UQ_EXT_BUDGET_GB')
def teach(config,run_dir,round_index,condition_path,directory,*,gpu_ids,device,relevant_ids=None,supervisor_model=None):
    condition=read_json(condition_path);validate_condition(condition,final=config.phase=='final')
    if condition['family'] not in ('undertrained_language','undertrained_appearance'):
        raise ValueError('Teaching compares the frozen language and appearance gaps')
    if config.phase=='final':
        frozen=read_json(config.frozen_recipe)
        if identity(condition) not in [identity(c) for c in frozen['conditions']]:raise ValueError('Teaching condition was not frozen')
    models,manifest,support=round_context(config,run_dir,round_index)
    previous=read_json(Path(run_dir)/f'round_{round_index:03d}/complete.json')['acquired'] if round_index else []
    excluded=set(support.ids) if config.eligibility_protocol=='unique_demo' else set()
    eligible=candidate_view(manifest,sorted(set(manifest['target_ids'])-excluded))
    relevant_pool=eligible.subset([c.id for c in eligible.candidates if c.task in condition['tasks']])
    seed=stream_seed(config.seed,'teaching',condition['name'])
    if relevant_ids is None:
        relevant_ids,_=select_save(relevant_pool,dict.fromkeys(relevant_pool.ids,0.),count=5,temperature=0,seed=seed,random_within_task=True)
    if len(set(relevant_ids))!=5 or not set(relevant_ids).issubset(relevant_pool.ids):raise ValueError('Need five eligible condition demonstrations')
    random_ids,_=select_save(eligible,dict.fromkeys(eligible.ids,0.),count=5,temperature=0,seed=seed,random_within_task=True)
    root=Path(directory).resolve();request=dict(condition=condition,previous=previous,relevant=relevant_ids,random=random_ids,
        training_mode=config.training_mode,round=round_index,seed=config.seed,source_run=str(Path(run_dir).resolve()))
    if (root/'teaching.json').exists() and read_json(root/'teaching.json')!=request:raise ValueError('Teaching branch identity changed')
    atomic_json(root/'teaching.json',request)
    base=read_json(Path(run_dir)/'experiment.json')['base_models'][0]
    starts=[base] if config.training_mode=='pristine_refit' else models[:1]
    cfg=replace(config,arm='kv',companion=False)
    before=evaluate(models[0],manifest,root/'before',split=config.evaluation_split,device=device,condition=condition)
    outcomes={};scores={}
    if supervisor_model:
        outcomes['supervisor']=evaluate(supervisor_model,manifest,root/'supervisor',split=config.evaluation_split,device=device,condition=condition)
    for branch,new in (('relevant',relevant_ids),('random',random_ids)):
        out=root/branch;out.mkdir(parents=True,exist_ok=True)
        os.environ['UQ_EXT_RUN_ROOT']=str(out);os.environ['UQ_EXT_BUDGET_GB']=str(cfg.budget_gb)
        overlay={str(manifest['episode_map'][id]):dict(task=next(c.task for c in eligible.candidates if c.id==id),condition=condition)
                 for id in new} if branch=='relevant' else None
        fitted=fit_policies(cfg,manifest,starts,[*previous,*new],run_dir=out,round_index=round_index+1,gpu_ids=gpu_ids,overlay=overlay)
        outcomes[branch]=evaluate(fitted[0],manifest,out/'evaluation',split=config.evaluation_split,device=device,condition=condition)
        augmented=taught_support(manifest,sorted(set(support.ids)|set(new)),condition,set(new) if branch=='relevant' else set(),out/'support_inputs')
        for method in ('kv','kv_t'):
            method_config=replace(cfg,arm=method)
            if config.frozen_recipe:
                from .config import FeatureConfig
                recipe=read_json(config.frozen_recipe)
                method_config=replace(method_config,features=FeatureConfig(**recipe['kv_t_features' if method=='kv_t' else 'kv_features']))
            score,_,_=variance_scores(method_config,fitted[0],augmented,evaluation_view(out/'evaluation'),out/'scores'/method,device,role=config.evaluation_split)
            scores[branch+'/'+method]=score
        retire_optimizers(fitted)
    for method in ('kv','kv_t'):
        method_config=replace(cfg,arm=method)
        if config.frozen_recipe:
            from .config import FeatureConfig
            method_config=replace(method_config,features=FeatureConfig(**read_json(config.frozen_recipe)['kv_t_features' if method=='kv_t' else 'kv_features']))
        scores['before/'+method]=variance_scores(method_config,models[0],support,evaluation_view(root/'before'),root/'before_scores'/method,device,role=config.evaluation_split)[0]
    import pandas as pd
    def macro(rows):return float(pd.DataFrame(rows).groupby('task').success.mean().mean())
    summary=dict(before=macro(before),**{k:macro(v) for k,v in outcomes.items()})
    summary['relevant_minus_random']=summary['relevant']-summary['random']
    summary['relevant_improvement']=summary['relevant']-summary['before'];summary['random_improvement']=summary['random']-summary['before']
    atomic_json(root/'teaching_results.json',dict(summary=summary,scores=scores,episodes=dict(before=before,**outcomes)))
    return summary
