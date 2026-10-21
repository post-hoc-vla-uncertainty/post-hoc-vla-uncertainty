"""Declared, bounded common-policy comparisons and machine-readable matrices."""
from dataclasses import replace
from pathlib import Path
import time
import pandas as pd
from .common import atomic_json,identity,plain,read_json
from .config import FeatureConfig,GPConfig,feature_uses_flow
from .panels import score_panel
from .preprocessing import UninformativeBank


def recipe(label,method,features,gp,question,*,group=None):
    return dict(id=label,method=method,features=plain(features),gp=plain(gp),question=question,group=group or label.split('_')[0])


def reference_recipes(*, flow=None, preview=False):
    f=FeatureConfig(**(flow or {}));g=GPConfig()
    if preview:
        f=replace(f,components=32,flow_samples=2,support_flow_samples=2,reference_flow_samples=2,action_bins=5)
        g=replace(g,input_components=8,output_components=4,fit_steps=20)
    # 4 policy probes x 3 flow times x ~1,450 episodes requires sparse fitting.
    sparse=replace(g,inference='sparse',sparse_justification='Conditioned support has episodes x Gaussian starts x solver times; bounded 512 inducing locations')
    families={'E':replace(f,bank='joint_intermediate',target_bank='joint_late'),
              'V':replace(f,bank='action_expert',target_bank='velocity')}
    rows=[]
    for family,features in families.items():
        gp=sparse if family=='V' else g
        for compression in ('random','pca'):
            rows.append(recipe(f'S0{1 if compression=="random" else 2}_{family}','kv',replace(features,compression=compression),gp,'KV geometry at a common source'))
        for output in ('random','pca'):
            rows.append(recipe(f'S0{3 if output=="random" else 4}_{family}','kv_t',features,replace(gp,output_compression=output),'Target compression at a common source'))
    if preview:return [r for r in rows if r['id'] in ('S02_E','S04_E','S04_V')]+[
        recipe('S11_patch_top','kv',replace(f,bank='visual_patches'),g,'Inspect native patch maps')]
    return rows


def broad_recipes(*,flow=None):
    rows=reference_recipes(flow=flow);f=FeatureConfig(**(flow or {}));g=GPConfig()
    sg=replace(g,inference='sparse',sparse_justification='Multi-probe policy-conditioned support; bounded 512 inducing locations')
    for bank in ('language','visual_local','visual_scene','state','joint_late','contextual_language','contextual_vision'):
        rows.append(recipe('S07_'+bank,'kv',replace(f,bank=bank),g,'Location diagnostic'))
    rows.append(recipe('S08_visual_pair','kv_t',replace(f,bank='visual_local',target_bank='visual_scene'),g,'Noise-free visual mapping'))
    for distance in (1,3):
        rows.append(recipe(f'S09_distance_{distance}','kv_t',replace(f,bank='action_expert',target_bank='velocity',expert_distance=distance),sg,'Expert computation span'))
    for bank in ('language','visual_scene','contextual_language','contextual_vision','action_context'):
        rows.append(recipe('S10_'+bank,'kv_t',replace(f,bank=bank,target_bank='velocity'),sg,'Available conditioning versus mapping difficulty'))
    rows.append(recipe('S10_final_action','kv_t',replace(f,bank='joint_late',target_bank='action_chunk',flow_indices=(9,),flow_weights=(1,)),sg,'Context plus initial Gaussian chunk to final action chunk; one target per Gaussian start'))
    for aggregation in ('mean','max','top_mean'):
        rows.append(recipe('S11_patch_'+aggregation,'kv',replace(f,bank='visual_patches',patch_aggregation=aggregation),g,'Small local gaps versus outlier sensitivity'))
    for method in ('vfd','gu','action_l2','ace','decu','vlm_token_entropy','vlm_perplexity'):
        rows.append(recipe('S13_'+method,method,f,g,'Common paper/reference score'))
    return rows


def focused_recipes(selected,*,seeds=(42,43,44)):
    """Compression rechecks on retained pairs; dependence/PLS controls only on V."""
    rows=[]
    for row in selected:
        f=FeatureConfig(**row['features']);g=GPConfig(**row['gp'])
        for seed in seeds:
            feature=replace(f,seed=seed)
            for c in ('pca','random'):
                rows.append(recipe(f'{row["id"]}_KV_{c}_{seed}','kv',replace(feature,compression=c),g,'Compression repetition on retained source',group='S01/S02'))
                rows.append(recipe(f'{row["id"]}_T_{c}_{seed}','kv_t',feature,replace(g,output_compression=c,input_projection='pls',output_dependence='independent'),'Output compression repetition',group='S03/S04'))
            rows.append(recipe(f'{row["id"]}_matched_KV_{seed}','kv',replace(feature,components=g.input_components),g,'Matched input dimension KV control',group='S08'))
        if not feature_uses_flow(f):continue
        for projection in ('pca','random'):
            rows.append(recipe(f'{row["id"]}_input_{projection}','kv_t',f,replace(g,input_projection=projection,output_dependence='independent',output_compression='pca'),'Remove target-informed input projection',group='S05/S06'))
        for output in ('pca','random'):
            for dependence in ('independent','coregionalized'):
                gp=replace(g,output_compression=output,output_dependence=dependence,inference='sparse',
                    sparse_justification='Matched 512-inducing comparison of independent and LMC outputs')
                rows.append(recipe(f'{row["id"]}_{output}_{dependence}','kv_t',f,gp,'Focused independent versus LMC output dependence',group='S12'))
    return rows


def run_matrix(config,run_dir,rounds,recipes,directory,*,device,supervisor,allow_nonrandom=False,condition=None):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    split=config.evaluation_split+('_gain_panel' if condition and condition.get('family')=='familiar_diversity' else '')
    from .runner import source_identity
    declaration=dict(recipes=recipes,rounds=list(rounds),config=plain(config),run=str(Path(run_dir).resolve()),
        supervisor=supervisor,condition=condition,principal=not allow_nonrandom,split=split,source_hash=source_identity())
    path=root/'declaration.json'
    if path.exists() and read_json(path)!=plain(declaration):raise ValueError('Scoring declaration changed; use a new directory')
    atomic_json(path,declaration);records=[]
    existing={(r['round'],r['recipe']['id']):r for r in read_json(root/'matrix.json')} if (root/'matrix.json').exists() else {}
    for r in rounds:
        for row in recipes:
            if (r,row['id']) in existing:
                records.append(existing[(r,row['id'])]);continue
            started=time.perf_counter();out=root/f'round_{r:03d}'/row['id']
            cfg=replace(config,features=FeatureConfig(**row['features']),gp=GPConfig(**row['gp']),frozen_recipe=None)
            status='complete';metrics=None;reason=None
            print(f'Score matrix: round {r}, {row["id"]}',flush=True)
            try:
                metrics=score_panel(cfg,run_dir,r,out,device=device,methods=(row['method'],),supervisor_model=supervisor,
                    allow_nonrandom=allow_nonrandom,evaluation_directory=root/f'round_{r:03d}/learner',condition=condition,split=split)[row['method']]
                mapping=metrics.get('mapping_check',{})
                if mapping.get('performed') and not mapping.get('usable'):status='mapping_infeasible'
            except UninformativeBank as e:status='uninformative';reason=str(e)
            except ValueError as e:
                if 'Repeated inputs' not in str(e):raise
                status='missing_conditioning';reason=str(e)
            record=dict(round=r,recipe=row,status=status,reason=reason,metrics=metrics,seconds=time.perf_counter()-started,directory=str(out))
            records.append(record);atomic_json(root/'matrix.json',records)
            write_table(records,root)
        from .storage import retire_feature_arrays
        retire_feature_arrays(root/f'round_{r:03d}')
    from .reporting import experiment_report
    write_table(records,root)
    experiment_report(root,matrix=records)
    return records


def write_table(records,root):
    rows=[];heads=[]
    for r in records:
        m=r.get('metrics') or {};mapping=m.get('mapping_check',{})
        rows.append(dict(round=r['round'],recipe=r['recipe']['id'],method=r['recipe']['method'],status=r['status'],
            referral_gain=m.get('referral_global',{}).get('raw_gain'),mapping_usable=mapping.get('usable'),
            negative_spearman=-m['between_task_spearman'] if m.get('between_task_spearman') is not None else None,
            negative_pearson=-m['between_task_pearson'] if m.get('between_task_pearson') is not None else None,
            within_task_auroc=m.get('macro_auroc'),within_task_risk_coverage=m.get('macro_risk_coverage_area'),
            effective_input_dim=m.get('gp_diagnostics',{}).get('input_components',m.get('gp_diagnostics',{}).get('components')),
            effective_output_dim=m.get('gp_diagnostics',{}).get('output_count'),
            peak_gpu_bytes=m.get('gp_diagnostics',{}).get('peak_gpu_allocated_bytes'),
            process_peak_host_bytes=m.get('gp_diagnostics',{}).get('process_peak_host_bytes'),
            seconds=r['seconds'],metrics_json=str(Path(r['directory'])/r['recipe']['method']/'metrics.json')))
        p=Path(r['directory'])/r['recipe']['method']/'scores.json'
        if p.exists():
            for task,score in read_json(p)['pool_task_means'].items():heads.append(dict(round=r['round'],recipe=r['recipe']['id'],task=task,score=score))
    pd.DataFrame(rows).to_csv(Path(root)/'configuration_matrix.csv',index=False)
    pd.DataFrame(heads).to_csv(Path(root)/'task_head_scores.csv',index=False)


def comparison_report(root,records):
    """Equal-panel development summaries assist a decision without selecting it."""
    import numpy as np
    rows=[]
    for label in dict.fromkeys(r['recipe']['id'] for r in records):
        cells=[r for r in records if r['recipe']['id']==label];spec=cells[0]['recipe']
        def mean(key,negative=False):
            values=[]
            for cell in cells:
                value=cell.get('metrics') or {}
                for k in key.split('.'):value=value.get(k) if isinstance(value,dict) else None
                if value is not None:values.append(value)
            return float(np.mean(values))*(-1 if negative else 1) if values else None
        rows.append(dict(recipe=label,method=spec['method'],source=spec['features']['bank'],target=spec['features']['target_bank'],
            projection_seed=spec['features']['seed'],panels=len(cells),usable_panels=sum(c['status']=='complete' for c in cells),
            mean_referral_gain=mean('referral_global.raw_gain'),mean_negative_spearman=mean('between_task_spearman',True),
            mean_negative_pearson=mean('between_task_pearson',True),mean_within_task_auroc=mean('macro_auroc'),
            total_seconds=sum(c['seconds'] for c in cells),statuses=', '.join(sorted({c['status'] for c in cells}))))
    frame=pd.DataFrame(rows).sort_values('mean_referral_gain',ascending=False,na_position='last')
    root=Path(root);root.mkdir(parents=True,exist_ok=True);frame.to_csv(root/'comparison.csv',index=False)
    from .reporting import referral_diagnostic_report
    diagnostics=referral_diagnostic_report(root/'referral_diagnostics',records)
    (root/'report.html').write_text('<html><meta charset="utf-8"><title>Development comparison</title><body><h1>Development comparison</h1>'
        '<style>body{font:15px system-ui;max-width:1300px;margin:2rem auto;padding:1rem}img.plot{max-width:100%}</style>'
        '<p>Equal weight per declared checkpoint and mode. Sorted by mean referral gain for inspection; no recipe is selected automatically. '
        'Check usable-panel counts, mapping diagnostics, projection repetitions, and costs before choosing. Detailed reports remain in each mode directory.</p>'+frame.to_html(index=False)+diagnostics['html']+'</body></html>')
    return rows
