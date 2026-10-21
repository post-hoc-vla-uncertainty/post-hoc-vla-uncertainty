"""Matched early/later development checks of pristine fitting budgets."""
from dataclasses import replace
from pathlib import Path
import os
import pandas as pd
from .common import atomic_json,identity,read_json,preserve_environment
from .data import load_manifest
from .training import fit_policies,retire_optimizers
from .evaluation import evaluate


@preserve_environment('UQ_EXT_RUN_ROOT','UQ_EXT_BUDGET_GB')
def check_refit(config,run_dir,directory,*,rounds=(1,10),budgets=(4000,8000),gpu_ids,device):
    if config.phase!='development' or config.evaluation_split!='development':raise ValueError('Refit selection is development-only')
    if len(rounds)!=2 or min(rounds)<1 or rounds[0]>=rounds[1]:raise ValueError('Specify an early and a later completed acquisition round')
    if len(set(budgets))<2 or min(budgets)<1:raise ValueError('Compare at least two positive fitting budgets')
    root=Path(directory).resolve();experiment=read_json(Path(run_dir)/'experiment.json');base=experiment['base_models'][0]
    declaration=dict(source_run=str(Path(run_dir).resolve()),rounds=rounds,budgets=budgets,seed=config.seed,
                     objective='mean task-macro success across early and late development checks',
                     preference='lowest update budget within 0.01 absolute success of best mean',split='development')
    path=root/'declaration.json'
    from .common import plain
    declaration=plain(declaration)
    if path.exists() and read_json(path)!=declaration:raise ValueError('Refit-check declaration changed')
    atomic_json(path,declaration);results={};artifacts=[]
    for r in rounds:
        # Refitting only needs the base and acquired IDs, even when the source
        # trajectory's intermediate policy was retired by rolling retention.
        manifest=load_manifest(config.data_manifest)
        if experiment['manifest_hash']!=manifest['manifest_hash']:raise ValueError('Refit manifest mismatch')
        acquired=read_json(Path(run_dir)/f'round_{r:03d}/complete.json')['acquired'];record=[]
        for steps in budgets:
            cfg=replace(config,arm='random',companion=False,training_mode='pristine_refit',refit_steps=steps)
            out=root/f'budget_{steps}/at_round_{r:03d}';out.mkdir(parents=True,exist_ok=True)
            os.environ['UQ_EXT_RUN_ROOT']=str(out);os.environ['UQ_EXT_BUDGET_GB']=str(cfg.budget_gb)
            model=fit_policies(cfg,manifest,[base],acquired,run_dir=out,round_index=r,gpu_ids=gpu_ids)[0]
            episodes=evaluate(model,manifest,out/'evaluation',split='development',device=device,batch_size=cfg.evaluation_batch_size)
            success=float(pd.DataFrame(episodes).groupby('task').success.mean().mean())
            record.append(dict(steps=steps,success=success,model=model));results.setdefault(steps,[]).append(success)
            retire_optimizers([model])
        artifact=root/f'round_{r:03d}_comparison.json'
        atomic_json(artifact,dict(round=r,demonstrations=len(set(acquired)),results=record,split='development'))
        artifacts.append(str(artifact))
    means={steps:sum(values)/len(values) for steps,values in results.items()};best=max(means.values())
    selected=min(steps for steps,value in means.items() if value>=best-.01)
    evidence=dict(split='development',early_budget=read_json(artifacts[0])['demonstrations'],
                  late_budget=read_json(artifacts[1])['demonstrations'],early_artifact=artifacts[0],late_artifact=artifacts[1],
                  selected_refit_steps=selected,rationale=declaration['preference'],mean_success=means,
                  declaration_hash=identity(declaration),convergence_claim=False)
    atomic_json(root/'refit_evidence.json',evidence);return evidence
