"""Bounded development screen and content-addressed scientific recipe freeze."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from .common import atomic_json, identity, plain, read_json, file_hash
from .config import OBSERVATION_BANKS
from .panels import score_panel
from .preprocessing import UninformativeBank


def screen(config, pilots, directory, *, device, supervisor_model=None):
    if config.phase!='development' or config.evaluation_split!='development':
        raise ValueError('Feature selection requires development data')
    directory=Path(directory);objective='raw_global_referral_gain' if supervisor_model else 'negative_macro_risk_coverage_area'
    declaration=dict(objective=objective,pilots=pilots,supervisor=supervisor_model,configuration=plain(config),
                     order=['joint_compression','six_KV_banks','two_KVT_pairs'],tie_tolerance=1e-4)
    declaration_path=directory/'screen_declaration.json'
    if declaration_path.exists() and read_json(declaration_path)!=declaration: raise ValueError('Screen declaration changed')
    atomic_json(declaration_path,declaration)
    records=[]
    def assess(label,feature,method):
        metrics=[];failure=None
        for i,pilot in enumerate(pilots):
            try:
                cfg=replace(config,features=feature,arm=method,training_mode=pilot['training_mode'])
                result=score_panel(cfg,pilot['run_dir'],pilot['round'],directory/label/f'pilot_{i}',device=device,
                                   methods=(method,),supervisor_model=supervisor_model)[method]
                metrics.append(result)
            except UninformativeBank as error:
                failure=str(error);break
        usable=not failure and all(m.get('mapping_check',{}).get('usable',True) for m in metrics)
        values=[m['referral_global']['raw_gain'] if supervisor_model else -m['macro_risk_coverage_area'] for m in metrics]
        record=dict(label=label,features=plain(feature),method=method,usable=usable,
                    objective=float(np.mean(values)) if usable else None,metrics=metrics,failure=failure)
        records.append(record);atomic_json(directory/'screen_results.json',records)
        return record
    def best(rows,preferred):
        valid=[r for r in rows if r['usable']]
        if not valid:raise RuntimeError('No usable development candidate; inspect screen_results.json')
        maximum=max(r['objective'] for r in valid)
        tied=[r for r in valid if maximum-r['objective']<=1e-4]
        return next((r for r in tied if r['label']==preferred),max(tied,key=lambda r:r['objective']))
    compression=[assess('compression_'+c,replace(config.features,bank='joint_intermediate',compression=c),'kv') for c in ('pca','random')]
    c=best(compression,'compression_pca')['features']['compression']
    banks=[assess('bank_'+b,replace(config.features,compression=c,bank=b),'kv') for b in OBSERVATION_BANKS]
    kv=best(banks,'bank_joint_intermediate')
    pairs=[assess('pair_joint',replace(config.features,compression=c,bank='joint_intermediate',target_bank='joint_late'),'kv_t'),
           assess('pair_visual',replace(config.features,compression=c,bank='visual_local',target_bank='visual_scene'),'kv_t')]
    target=best(pairs,'pair_joint')
    selection=dict(kv_features=kv['features'],kv_t_features=target['features'],gp=plain(config.gp),
                   objective=objective,selected_labels=[kv['label'],target['label']],declaration=declaration)
    atomic_json(directory/'selection.json',selection)
    return selection


def freeze(screen_directory, output, *, refit_evidence=None, conditions=()):
    directory=Path(screen_directory);selection=read_json(directory/'selection.json')
    results=read_json(directory/'screen_results.json')
    for label in selection['selected_labels']:
        row=next(r for r in results if r['label']==label)
        if not row['usable']: raise ValueError('Cannot freeze an unusable mapping/bank')
    budget=None;evidence=None
    if refit_evidence:
        evidence=read_json(refit_evidence)
        if evidence.get('split')!='development' or not evidence.get('early_budget') or not evidence.get('late_budget'):
            raise ValueError('Pristine budget needs early and late development fitting checks')
        budget=int(evidence['selected_refit_steps'])
        if budget<=0:raise ValueError('Invalid refit budget')
        for key in ('early_artifact','late_artifact'):
            if not Path(evidence[key]).is_file():raise ValueError('Missing refit adequacy artifact')
    from .conditions import validate_condition
    conditions_data=[]
    for path in conditions:
        condition=read_json(path);validate_condition(condition,final=True);conditions_data.append(condition)
    payload=dict(**selection,screen_files={p.name:file_hash(p) for p in directory.glob('*.json')},
                 refit_steps=budget,refit_evidence=evidence,conditions=conditions_data)
    atomic_json(output,dict(**payload,recipe_hash=identity(payload)))
