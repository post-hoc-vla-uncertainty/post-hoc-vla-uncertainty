"""Experiment stages with explicit scientific decision boundaries.

No work is submitted by import or --plan. Each numbered stage is resumable and
ends before the next user decision. The JSON decisions are editable artifacts.
"""
import argparse
import fcntl
from dataclasses import replace
import os
from pathlib import Path
from .common import atomic_json,read_json,plain,identity,file_hash
from .config import ExperimentConfig,FeatureConfig,GPConfig,ARMS,MODES,ELIGIBILITY_BY_MODE,feature_uses_flow
from .matrix import broad_recipes,reference_recipes,focused_recipes,run_matrix,comparison_report

STAGES={
 '01_preview':'One 100-update KV acquisition batch; paired scoring/referral and patch/flow examples. STOP: review outputs.',
 '02_histories':'Reuse completed reproduction Random seed 1000; early/later pristine-budget checks. STOP: choose refit budget.',
 '03_broad_scoring':'Common pristine Random history; both families and location/span/conditioning diagnostics. STOP: retain feasible E and V mappings.',
 '04_focused_scoring':'Compression repetitions, matched-dimension controls and focused output dependence. STOP: freeze the main recipe.',
 '05_active_learning':'Seven arms x two modes x three seeds; reuse SAVE Random/VFD reproduction and common pristine Random; frozen scoring panels. No further scientific choices inside this stage.',
 '06_condition_development':'Inspect proposed visual/language and control conditions on development starts. STOP: verify conditions and choose teaching round.',
 '07_teaching':'Frozen three-selector teaching comparisons and condition/control final score panels.',
}


def decision(root,name,evidence,template):
    path=Path(root)/'decisions'/f'{name}.json'
    payload=dict(evidence_hash=identity(evidence),**template)
    if not path.exists():atomic_json(path,payload)
    print(f'DECISION BOUNDARY: edit {path}. No later stage was started.',flush=True)
    return path


def require_decision(root,name,evidence,required):
    path=Path(root)/'decisions'/f'{name}.json'
    if not path.exists():raise ValueError(f'First complete the preceding stage and fill {path}')
    data=read_json(path)
    if data.get('evidence_hash')!=identity(evidence):raise ValueError(f'{path}: evidence changed; review the new report and update evidence_hash')
    missing=[k for k in required if data.get(k) is None or data.get(k) is False or data.get(k)==[]]
    if missing:raise ValueError(f'{path}: decisions still required: {", ".join(missing)}')
    if 'outputs_ready' in required and data['outputs_ready'] is not True:raise ValueError('outputs_ready must be the JSON boolean true after reviewing the preview')
    return data


def base_config(settings,root,*,arm='random',mode=MODES[0],seed=1000,phase='development',name='common'):
    base=2*(seed-1000);storage=Path(settings['storage_root'])
    return ExperimentConfig(name=name,arm=arm,training_mode=mode,seed=seed,phase=phase,
        eligibility_protocol=settings.get('eligibility_by_mode',ELIGIBILITY_BY_MODE)[mode],
        base_models=[str(storage/f'outputs/pretrain/libero_pretrain_leak3/seed_{i}/checkpoints/last/pretrained_model') for i in (base,base+1)],
        data_manifest=settings['data_manifest'],output_root=str(Path(root)/'runs'),companion=arm=='random',
        feature_batch_size=settings['feature_batch_size'],evaluation_batch_size=settings['evaluation_batch_size'],
        budget_gb=settings['trajectory_budget_gb'],features=FeatureConfig(**settings['features']))


def store_config(root,cfg,label):
    path=Path(root)/'configs'/f'{label}.json';atomic_json(path,cfg);return path


def supervisor(root,args,settings,device):
    path=Path(root)/'supervisor.json'
    if path.exists():
        record=read_json(path)
        if args.supervisor and str(Path(args.supervisor).resolve())!=record['model']:raise ValueError('The fixed supervisor changed')
        from .common import model_identity
        if model_identity(record['model'])!=record['model_hash']:raise ValueError('Supervisor weights changed')
        return record['model']
    if args.supervisor:
        from .common import model_identity
        model=str(Path(args.supervisor).resolve());record=dict(model=model,model_hash=model_identity(model),source='user-provided fixed supervisor')
    else:
        from .supervisor import train_supervisor
        from .data import load_manifest
        model=train_supervisor(Path(root)/'supervisor',load_manifest(settings['data_manifest']),device=device)
        record=read_json(Path(root)/'supervisor/supervisor.json')
    atomic_json(path,record);return model


def completed(root,name):return read_json(Path(root)/name/'result.json')


def finish(root,name,payload):
    atomic_json(Path(root)/name/'result.json',payload)
    from .storage import usage_bytes
    atomic_json(Path(root)/'storage_usage.json',dict(total_bytes=usage_bytes(root),shared_feature_bytes=usage_bytes(Path(root)/'feature_cache'),
        note='Extension workspace only; separate from reproduction; all round weights retained'))
    return payload


def stage_preview(args,settings,gpus,device):
    from .runner import run
    from .reporting import experiment_report
    root=Path(args.root);teacher=supervisor(root,args,settings,device)
    cfg=base_config(settings,root,arm='kv',phase='smoke',name='preview')
    cfg=replace(cfg,rounds=1,fit_steps=100,save_frequency=100,evaluation_per_task=2,features=FeatureConfig(components=32))
    store_config(root,cfg,'preview');run(cfg,gpu_ids=gpus,device=device)
    rows=run_matrix(cfg,cfg.run_dir(),(0,1),reference_recipes(preview=True),root/'01_preview/scoring',device=device,
        supervisor=teacher,allow_nonrandom=True)
    report=experiment_report(root/'01_preview',matrix=rows,runs=(cfg.run_dir(),))
    result=finish(root,'01_preview',dict(run=str(cfg.run_dir()),report=str(report),matrix_hash=identity(rows),smoke=True,supervisor=teacher))
    decision(root,'01_outputs',result,dict(outputs_ready=None,notes='Review the HTML, CSV/JSON, acquisition gallery, patch and flow outputs before proceeding.'))


def stage_histories(args,settings,gpus,device):
    from .reference import import_reproduction
    from .data import load_manifest
    from .refit_check import check_refit
    root=Path(args.root);require_decision(root,'01_outputs',completed(root,'01_preview'),['outputs_ready'])
    cfg=base_config(settings,root);store_config(root,cfg,'common_save')
    import_reproduction(cfg,Path(settings['storage_root'])/'outputs/reproduction/random/seed_1000',load_manifest(cfg.data_manifest))
    evidence=check_refit(cfg,cfg.run_dir(),root/'02_histories/refit',gpu_ids=gpus,device=device,
        rounds=tuple(settings['refit_rounds']),budgets=tuple(settings['refit_budgets']))
    result=finish(root,'02_histories',dict(save_run=str(cfg.run_dir()),refit_evidence=evidence))
    decision(root,'02_refit_budget',result,dict(refit_steps=None,recommendation=evidence['selected_refit_steps'],notes='Choose one of the tested budgets using both early and later comparisons.'))


def refit_budget(root):
    evidence=completed(root,'02_histories');choice=require_decision(root,'02_refit_budget',evidence,['refit_steps'])
    steps=int(choice['refit_steps'])
    tested=evidence['refit_evidence']['mean_success']
    if str(steps) not in tested:raise ValueError('Choose a tested refit budget; extend the budget study if necessary')
    return steps


def common_configs(settings,root):
    return [replace(base_config(settings,root,mode=mode),refit_steps=refit_budget(root)) for mode in MODES]


def stage_broad(args,settings,gpus,device):
    from .runner import run
    root=Path(args.root);teacher=supervisor(root,args,settings,device);configs=common_configs(settings,root)
    cfg=configs[1];store_config(root,cfg,'common_pristine');run(cfg,gpu_ids=gpus,device=device)
    recipes=broad_recipes(flow=settings['features']);all_rows=[]
    for config in configs:
        rows=run_matrix(config,config.run_dir(),settings['panel_rounds'],recipes,root/'03_broad_scoring'/config.training_mode,
            device=device,supervisor=teacher)
        all_rows.extend([dict(**row,training_mode=config.training_mode) for row in rows])
    comparison_report(root/'03_broad_scoring',all_rows)
    result=finish(root,'03_broad_scoring',dict(recipes=recipes,records=all_rows,objective='equal-checkpoint/equal-mode mean raw global referral gain'))
    decision(root,'03_mappings',result,dict(embedding_recipe=None,velocity_recipe=None,
        notes='Choose one feasible noise-free KV-T mapping and one velocity mapping for focused tests. Both families remain. Pure missing-conditioning controls cannot be the overall head.'))


def usable_recipe(result,label):
    row=next((r for r in result['recipes'] if r['id']==label),None)
    if row is None or row['method']!='kv_t':raise ValueError(f'Choose a listed KV-T recipe: {label}')
    measurements=[r for r in result['records'] if r['recipe']['id']==label]
    if not measurements or any(r['status']!='complete' for r in measurements):raise ValueError(f'{label} did not pass mapping feasibility at all declared panels; review diagnostics')
    return row


def stage_focused(args,settings,gpus,device):
    root=Path(args.root);broad=completed(root,'03_broad_scoring');choice=require_decision(root,'03_mappings',broad,['embedding_recipe','velocity_recipe'])
    e=usable_recipe(broad,choice['embedding_recipe']);v=usable_recipe(broad,choice['velocity_recipe'])
    if feature_uses_flow(FeatureConfig(**e['features'])) or v['features']['target_bank']!='velocity':raise ValueError('Retain one embedding and one velocity mapping')
    recipes=focused_recipes([e,v],seeds=tuple(settings['projection_seeds']));records=[]
    teacher=supervisor(root,args,settings,device)
    for cfg in common_configs(settings,root):
        rows=run_matrix(cfg,cfg.run_dir(),settings['panel_rounds'],recipes,root/'04_focused_scoring'/cfg.training_mode,device=device,supervisor=teacher)
        records.extend([dict(**row,training_mode=cfg.training_mode) for row in rows])
    comparison_report(root/'04_focused_scoring',records)
    result=finish(root,'04_focused_scoring',dict(recipes=recipes,records=records,retained=[e,v],refit_steps=refit_budget(root)))
    decision(root,'04_main_recipe',result,dict(kv_recipe=None,kv_t_recipe=None,notes='Select matched-source KV/KV-T recipes; projection seed is predeclared, not selected for favorable results.'))


def freeze_main(root,settings):
    result=completed(root,'04_focused_scoring');choice=require_decision(root,'04_main_recipe',result,['kv_recipe','kv_t_recipe'])
    kv=next((r for r in result['recipes'] if r['id']==choice['kv_recipe']),None);kt=usable_recipe(result,choice['kv_t_recipe'])
    if kv is None or kv['method']!='kv':raise ValueError('Choose a listed KV recipe')
    source_fields=('bank','target_bank','expert_distance','expert_context','action_bins','flow_indices','flow_weights','flow_seed','flow_samples','support_flow_samples','reference_flow_samples')
    if any(kv['features'][k]!=kt['features'][k] for k in source_fields):raise ValueError('KV and KV-T require the same source and flow conditioning')
    if kt['features']['bank'] not in ('joint_intermediate','joint_late','action_expert','action_context'):raise ValueError('Main overall head requires fused context')
    if kv['features']['seed']!=settings['projection_seeds'][0] or kt['features']['seed']!=settings['projection_seeds'][0]:raise ValueError('Use the predeclared first projection seed; other projection seeds are robustness repetitions')
    if any(r['status']!='complete' for r in result['records'] if r['recipe']['id']==kv['id']):raise ValueError('Selected KV bank was uninformative')
    declared_broad=completed(root,'03_broad_scoring')['recipes']
    diagnostics=[r for r in declared_broad if r['group'] in ('S07','S11') or r['id'] in ('S02_E','S02_V')]
    diagnostics.extend([kv,kt])
    evidence=completed(root,'02_histories')['refit_evidence'];evidence=dict(evidence,selected_refit_steps=result['refit_steps'])
    payload=dict(kv_features=kv['features'],kv_t_features=kt['features'],kv_gp=kv['gp'],kv_t_gp=kt['gp'],gp=kv['gp'],
        eligibility_by_mode=settings.get('eligibility_by_mode',ELIGIBILITY_BY_MODE),
        selected_labels=[kv['id'],kt['id']],refit_steps=result['refit_steps'],refit_evidence=evidence,conditions=[],
        diagnostic_recipes=diagnostics,principal_recipes=[*declared_broad,*result['recipes']],
        development_result_hash=identity(result),decision=choice,
        objective='mean raw global referral gain on common development histories; retain mapping feasibility and simpler tied constructions')
    path=Path(root)/'frozen_recipe.json';record=dict(**payload,recipe_hash=identity(payload))
    if path.exists() and read_json(path)!=plain(record):raise ValueError('Frozen recipe changed; start a new experiment workspace')
    atomic_json(path,record);return path,record


def selected_config(settings,root,recipe_path,recipe,*,arm,mode,seed,phase='final'):
    target=arm in ('kv_t','kv_t_ivr');cfg=base_config(settings,root,arm=arm,mode=mode,seed=seed,phase=phase,name='main')
    return replace(cfg,frozen_recipe=str(recipe_path),features=FeatureConfig(**recipe['kv_t_features' if target else 'kv_features']),
        gp=GPConfig(**recipe['kv_t_gp' if target else 'kv_gp']),refit_steps=recipe['refit_steps'],evaluation_split='final' if phase=='final' else 'development')


def stage_main(args,settings,gpus,device):
    from .runner import run
    from .evaluation import evaluate
    from .data import load_manifest
    from .panels import round_context
    from .reporting import summarize_suite
    from .reference import import_reproduction,available_panel_rounds,evaluate_reference,reference_curve
    root=Path(args.root);recipe_path,frozen=freeze_main(root,settings);teacher=supervisor(root,args,settings,device)
    records=[];new_runs=[]
    for mode in args.modes.split(','):
        for seed in map(int,args.seeds.split(',')):
            for arm in args.arms.split(','):
                cfg=selected_config(settings,root,recipe_path,frozen,arm=arm,mode=mode,seed=seed)
                cfg.validate();store_config(root,cfg,f'main_{mode}_{arm}_{seed}')
                if mode=='save_warm_50_50' and arm in ('random','vfd'):
                    source=import_reproduction(cfg,Path(settings['storage_root'])/f'outputs/reproduction/{arm}/seed_{seed}',load_manifest(cfg.data_manifest))
                    report_dir=source
                elif arm=='random' and seed==1000:
                    source=next(c for c in common_configs(settings,root) if c.training_mode==mode).run_dir()
                    # Reuse the completed, protocol-identical weights/history, then evaluate final starts separately.
                    out=root/'05_active_learning/reused_random'/mode
                    manifest=load_manifest(cfg.data_manifest)
                    for r in range(16):
                        models,_,_=round_context(cfg,source,r)
                        if reference_curve(cfg):evaluate_reference(cfg,models[0],out/f'round_{r:03d}/evaluation',r,device=device)
                        else:evaluate(models[0],manifest,out/f'round_{r:03d}/evaluation',split='final',device=device,batch_size=cfg.evaluation_batch_size)
                    atomic_json(out/'reuse.json',dict(source_run=str(source),recipe_hash=frozen['recipe_hash'],split='final',double_count_as_new_seed=False))
                    atomic_json(out/'experiment.json',dict(config=plain(cfg),source_run=str(source),base_models=read_json(source/'experiment.json')['base_models']))
                    atomic_json(out/'acquisition_ledger.json',read_json(source/'acquisition_ledger.json'))
                    for r in range(1,16):
                        atomic_json(out/f'round_{r:03d}/policy_fit.json',read_json(source/f'round_{r:03d}/policy_fit.json'))
                        atomic_json(out/f'round_{r:03d}/selection/acquisition.json',read_json(source/f'round_{r:03d}/selection/acquisition.json'))
                    report_dir=out
                else:
                    source=run(cfg,gpu_ids=gpus,device=device);new_runs.append(str(source));report_dir=source
                panel_rounds=available_panel_rounds(source,settings['panel_rounds'])
                missing=sorted(set(settings['panel_rounds'])-set(panel_rounds))
                if missing:
                    print(f'{mode}/{arm}/seed {seed}: diagnostic panels unavailable at retired rounds {missing}; all learning-curve outcomes retained.',flush=True)
                    atomic_json(root/'05_active_learning/panels'/mode/arm/f'seed_{seed}'/'unavailable.json',dict(rounds=missing,reason='reproduction weights retired; no retraining'))
                rows=run_matrix(cfg,source,panel_rounds,frozen['principal_recipes'] if arm=='random' else frozen['diagnostic_recipes'],
                    root/'05_active_learning/panels'/mode/arm/f'seed_{seed}',device=device,supervisor=teacher,allow_nonrandom=arm!='random')
                records.append(dict(arm=arm,seed=seed,mode=mode,source_run=str(source),report_dir=str(report_dir),panel_hash=identity(rows),unavailable_panel_rounds=missing))
    # Parts can be run separately and resumed without silently skipping an arm.
    chunk=identity(dict(modes=args.modes,seeds=args.seeds,arms=args.arms))[:12]
    atomic_json(root/'05_active_learning/parts'/f'{chunk}.json',records)
    combined={}
    for p in (root/'05_active_learning/parts').glob('*.json'):
        for r in read_json(p):combined[(r['mode'],r['arm'],r['seed'])]=r
    finish(root,'05_active_learning',dict(trajectories=list(combined.values()),expected=42,complete=len(combined)==42,recipe_hash=frozen['recipe_hash']))
    summarize_suite([r['report_dir'] for r in combined.values()],root/'05_active_learning/summary')
    print(f'Main stage has {len(combined)}/42 trajectories. Retained weights plus final evaluations are available; condition development is a separate stage.')


def stage_conditions(args,settings,gpus,device):
    from .conditions import validate_condition
    root=Path(args.root);recipe_path,frozen=freeze_main(root,settings)
    proposals=read_json(args.conditions) if args.conditions else []
    if not proposals:
        path=root/'decisions/condition_candidates.json'
        if not path.exists():atomic_json(path,[])
        raise ValueError(f'Populate {path} with paths to proposed condition JSON files, then pass --conditions {path}. See configs/extensions/conditions.example.json and EXPERIMENTS.md.')
    teacher=supervisor(root,args,settings,device);records=[]
    for path in proposals:
        condition=read_json(path);validate_condition(condition)
        for cfg in common_configs(settings,root):
            rows=run_matrix(cfg,cfg.run_dir(),settings['teaching_round_candidates'],frozen['diagnostic_recipes'],
                root/'06_condition_development'/condition['name']/cfg.training_mode,device=device,supervisor=teacher,condition=condition)
            records.append(dict(condition_path=path,condition=condition,mode=cfg.training_mode,records=rows))
    result=finish(root,'06_condition_development',dict(proposals=proposals,records=records))
    decision(root,'06_teaching',result,dict(visual_condition=None,language_condition=None,control_conditions=None,teaching_round=None,
        notes='Verify goals/attributes on development images, record development_verification artifacts, and choose one tested round. Control conditions must include harmless variation and familiar diversity.'))


def stage_teaching(args,settings,gpus,device):
    from .conditions import validate_condition
    from .teaching import teach_selectors
    from .runner import run
    root=Path(args.root);recipe_path,frozen=freeze_main(root,settings)
    result=completed(root,'06_condition_development');choice=require_decision(root,'06_teaching',result,
        ['visual_condition','language_condition','control_conditions','teaching_round'])
    paths=[choice['visual_condition'],choice['language_condition'],*choice['control_conditions']]
    conditions=[read_json(p) for p in paths]
    for c in conditions:validate_condition(c,final=True)
    if [c['family'] for c in conditions[:2]]!=['undertrained_appearance','undertrained_language']:raise ValueError('Select one visual and one language condition')
    if not {'harmless_variation','familiar_diversity'}.issubset({c['family'] for c in conditions[2:]}):raise ValueError('Include both declared control families')
    if choice['teaching_round'] not in settings['teaching_round_candidates']:raise ValueError('Choose a development-tested teaching round')
    tested={identity({k:v for k,v in r['condition'].items() if k!='development_verification'}) for r in result['records']}
    if any(identity({k:v for k,v in c.items() if k!='development_verification'}) not in tested for c in conditions):raise ValueError('A selected condition was not evaluated during development')
    payload={k:v for k,v in frozen.items() if k!='recipe_hash'};payload['conditions']=conditions;payload['teaching_decision']=choice
    teaching_recipe=root/'teaching_recipe.json';atomic_json(teaching_recipe,dict(**payload,recipe_hash=identity(payload)))
    teacher=supervisor(root,args,settings,device);records=[]
    for mode in args.modes.split(','):
        for seed in map(int,args.seeds.split(',')):
            source=next(c for c in common_configs(settings,root) if c.training_mode==mode).run_dir() if seed==1000 else base_config(settings,root,mode=mode,seed=seed,name='main').run_dir()
            if not (source/'complete.json').exists():raise ValueError(f'Complete the corresponding Random history in stage 05 first: {source}')
            cfg=selected_config(settings,root,teaching_recipe,payload,arm='kv_t',mode=mode,seed=seed)
            for path in paths[:2]:
                condition=read_json(path);out=root/'07_teaching'/mode/f'seed_{seed}'/condition['name']
                result=teach_selectors(cfg,source,choice['teaching_round'],path,out,gpu_ids=gpus,device=device,supervisor_model=teacher)
                records.append(dict(condition=condition['name'],mode=mode,seed=seed,path=str(out),result_hash=identity(result)))
            for path in paths:
                condition=read_json(path)
                run_matrix(cfg,source,[choice['teaching_round']],frozen['diagnostic_recipes'],root/'07_teaching/condition_panels'/mode/f'seed_{seed}'/condition['name'],
                    device=device,supervisor=teacher,condition=condition)
    finish(root,'07_teaching',dict(comparisons=records,branch_fits=3*len(records)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=STAGES);parser.add_argument('--root',default='storage/extensions/experiments/v1')
    parser.add_argument('--settings',default='configs/extensions/experiments/workflow.json');parser.add_argument('--gpus',default='6,7')
    parser.add_argument('--supervisor');parser.add_argument('--conditions');parser.add_argument('--plan',action='store_true')
    parser.add_argument('--keep-feature-cache',action='store_true',help='Retain regenerable raw arrays after each policy panel/round; requires more disk')
    parser.add_argument('--progress-steps',type=int,default=100,help='Training metric interval; epoch summaries are always emitted')
    parser.add_argument('--progress-seconds',type=float,default=60,help='Timed progress and worker heartbeat interval')
    parser.add_argument('--arms',default=','.join(ARMS));parser.add_argument('--modes',default=','.join(MODES));parser.add_argument('--seeds',default='1000,1001,1002')
    args=parser.parse_args();settings=read_json(args.settings)
    if args.progress_steps<1 or not 0<args.progress_seconds<float('inf'):parser.error('Progress intervals must be positive and finite')
    os.environ['VLA_PROGRESS_STEPS']=str(args.progress_steps);os.environ['VLA_PROGRESS_SECONDS']=str(args.progress_seconds)
    os.environ['PYTHONUNBUFFERED']='1'
    if any(x not in ARMS for x in args.arms.split(',')) or any(x not in MODES for x in args.modes.split(',')) or any(x not in ('1000','1001','1002') for x in args.seeds.split(',')):raise ValueError('Unknown arm, mode, or prepared seed')
    print(STAGES[args.stage],flush=True)
    if args.plan:
        print(f'Workspace: {args.root}\nSettings: {args.settings}\nNo GPU, training, file creation or decision changes.');return
    # Check decision requirements before initializing CUDA or training the supervisor.
    gates={'02_histories':('01_outputs','01_preview',['outputs_ready']),
           '03_broad_scoring':('02_refit_budget','02_histories',['refit_steps']),
           '04_focused_scoring':('03_mappings','03_broad_scoring',['embedding_recipe','velocity_recipe']),
           '05_active_learning':('04_main_recipe','04_focused_scoring',['kv_recipe','kv_t_recipe']),
           '06_condition_development':('04_main_recipe','04_focused_scoring',['kv_recipe','kv_t_recipe']),
           '07_teaching':('06_teaching','06_condition_development',['visual_condition','language_condition','control_conditions','teaching_round'])}
    if args.stage in gates:
        name,preceding,keys=gates[args.stage];require_decision(args.root,name,completed(args.root,preceding),keys)
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    with (root/'workflow.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another stage is using this workspace; shared caches and decision files require one stage at a time')
        from .cli import device_setup
        gpus,device=device_setup(args.gpus)
        os.environ['UQ_EXT_FEATURE_CACHE']=str((root/'feature_cache').resolve())
        os.environ['UQ_EXT_RETIRE_FEATURE_CACHE']='0' if args.keep_feature_cache else '1'
        stages=dict(zip(STAGES,(stage_preview,stage_histories,stage_broad,stage_focused,stage_main,stage_conditions,stage_teaching)))
        stages[args.stage](args,settings,gpus,device)


if __name__=='__main__':main()
