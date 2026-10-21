"""Commands for preparing and executing extension experiments."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
from .common import atomic_json,plain,read_json
from .config import ARMS,MODES,ExperimentConfig,FeatureConfig,GPConfig


def configure(args):
    directory=Path(args.output);directory.mkdir(parents=True,exist_ok=True)
    configs=[]
    recipe=read_json(args.recipe) if args.recipe else None
    for mode in args.modes.split(','):
        for arm in args.arms.split(','):
            for seed in map(int,args.seeds.split(',')):
                if seed not in (1000,1001,1002):raise ValueError('Prepared base seed pairs are 1000,1001,1002')
                base=2*(seed-1000)
                paths=[str(Path(os.environ['STORAGE_ROOT'])/f'outputs/pretrain/libero_pretrain_leak3/seed_{i}/checkpoints/last/pretrained_model') for i in (base,base+1)]
                cfg=ExperimentConfig(name=args.name,phase=args.phase,arm=arm,training_mode=mode,seed=seed,
                    companion=arm=='random' and not args.no_companion,base_models=paths,rounds=args.rounds,
                    frozen_recipe=args.recipe,evaluation_split='final' if args.phase=='final' else 'development')
                if recipe:
                    cfg.features=FeatureConfig(**recipe['kv_t_features' if arm in ('kv_t','kv_t_ivr') else 'kv_features']);cfg.gp=GPConfig(**recipe.get('kv_t_gp' if arm in ('kv_t','kv_t_ivr') else 'kv_gp',recipe.get('gp'))) 
                    if mode=='pristine_refit':
                        if not recipe['refit_steps']:raise ValueError('Recipe has no validated pristine budget')
                        cfg.refit_steps=recipe['refit_steps']
                cfg.validate();path=directory/f'{mode}_{arm}_{seed}.json';atomic_json(path,cfg);configs.append(str(path))
    atomic_json(directory/'suite.json',dict(configs=configs));print(directory/'suite.json')


def device_setup(gpus):
    if gpus=='cpu':
        os.environ['CUDA_VISIBLE_DEVICES']='';return ['cpu'],'cpu'
    ids=gpus.split(',')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate GPU IDs')
    os.environ['CUDA_VISIBLE_DEVICES']=gpus
    from lerobot.utils.gpu_utils import configure_egl_for_visible_cuda
    configure_egl_for_visible_cuda(os.environ)
    ids=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in this process; check GPU group membership')
    return ids,'cuda:0'


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare-data');p.add_argument('--dataset-root',default=os.environ.get('STORAGE_ROOT','storage')+'/datasets/HuggingFaceVLA/libero')
    p.add_argument('--output',default='storage/extensions/data');p.add_argument('--revision',default='86958911c0f959db2bbbdb107eb3e17c5f9c798e')
    p=sub.add_parser('configure');p.add_argument('--output',default='configs/extensions/development');p.add_argument('--name',default='pilot')
    p.add_argument('--modes',default='save_warm_50_50');p.add_argument('--arms',default='random');p.add_argument('--seeds',default='1000')
    p.add_argument('--rounds',type=int,default=15);p.add_argument('--phase',choices=('development','final'),default='development')
    p.add_argument('--recipe');p.add_argument('--no-companion',action='store_true')
    p=sub.add_parser('run');p.add_argument('config');p.add_argument('--gpus',required=True);p.add_argument('--evaluation-limit',type=int)
    p=sub.add_parser('suite');p.add_argument('manifest');p.add_argument('--gpus',required=True)
    p=sub.add_parser('estimate');p.add_argument('config');p.add_argument('--weight-bytes',type=int,default=1217881621)
    p=sub.add_parser('evaluate');p.add_argument('model');p.add_argument('output');p.add_argument('--manifest',default='storage/extensions/data/data_manifest.json')
    p.add_argument('--split',choices=('development','final','development_gain_panel','final_gain_panel'),default='development')
    p.add_argument('--condition');p.add_argument('--gpus',required=True)
    p=sub.add_parser('score-panel');p.add_argument('config');p.add_argument('run_dir');p.add_argument('output');p.add_argument('--rounds',default='0,5,10,15')
    p.add_argument('--methods',default='kv,kv_t,vfd,gu,action_l2,ace,decu,vlm_token_entropy,vlm_perplexity')
    p.add_argument('--supervisor');p.add_argument('--condition');p.add_argument('--gain-panel',action='store_true');p.add_argument('--gpus',required=True)
    p=sub.add_parser('screen');p.add_argument('config');p.add_argument('pilots');p.add_argument('output');p.add_argument('--supervisor');p.add_argument('--gpus',required=True)
    p=sub.add_parser('freeze');p.add_argument('screen_dir');p.add_argument('output');p.add_argument('--refit-evidence');p.add_argument('--conditions',nargs='*',default=[])
    p=sub.add_parser('refit-check');p.add_argument('config');p.add_argument('run_dir');p.add_argument('output')
    p.add_argument('--rounds',default='1,10');p.add_argument('--budgets',default='4000,8000');p.add_argument('--gpus',required=True)
    p=sub.add_parser('supervisor');p.add_argument('output');p.add_argument('--manifest',default='storage/extensions/data/data_manifest.json');p.add_argument('--gpus',required=True)
    p.add_argument('--steps',type=int,default=30000);p.add_argument('--seed',type=int,default=991);p.add_argument('--smoke',action='store_true')
    p=sub.add_parser('teach');p.add_argument('config');p.add_argument('run_dir');p.add_argument('condition');p.add_argument('output')
    p.add_argument('--round',type=int,default=5);p.add_argument('--relevant-ids',nargs=5);p.add_argument('--supervisor');p.add_argument('--gpus',required=True)
    p=sub.add_parser('report');p.add_argument('runs',nargs='+');p.add_argument('--output',default='reports/extensions')
    p=sub.add_parser('_train-worker');p.add_argument('request')
    args=parser.parse_args()
    if args.command=='_train-worker':
        from .training import train_worker
        return train_worker(args.request)
    if hasattr(args,'gpus'):gpus,device=device_setup(args.gpus)
    if args.command=='prepare-data':
        from .data import prepare_data
        return prepare_data(args.dataset_root,args.output,source_revision=args.revision)
    if args.command=='configure':return configure(args)
    if args.command=='run':
        from .runner import run
        return run(ExperimentConfig.load(args.config),gpu_ids=gpus,device=device,evaluation_limit=args.evaluation_limit)
    if args.command=='suite':
        from .runner import run
        for path in read_json(args.manifest)['configs']:run(ExperimentConfig.load(path),gpu_ids=gpus,device=device)
    if args.command=='estimate':
        cfg=ExperimentConfig.load(args.config);count=cfg.policy_count
        print(f'{cfg.arm}, {cfg.training_mode}: {count} policy/policies, {cfg.rounds} completed rounds')
        print(f'Completed round weights: {args.weight_bytes*count*cfg.rounds/1e9:.2f} GB; shared starting weights: {args.weight_bytes*count/1e9:.2f} GB')
        print(f'Conservative transient training reserve: {8*count} GB; feature/GP/evaluation caches are additional.')
        print(f'Per-experiment output guard: {cfg.budget_gb:g} GB; shared data, environment, bases and supervisor are outside it.')
    if args.command=='evaluate':
        from .data import load_manifest
        from .evaluation import evaluate
        return evaluate(args.model,load_manifest(args.manifest),args.output,split=args.split,device=device,condition=read_json(args.condition) if args.condition else None)
    if args.command=='score-panel':
        from .panels import score_panel
        cfg=ExperimentConfig.load(args.config)
        for r in map(int,args.rounds.split(',')):
            score_panel(cfg,args.run_dir,r,Path(args.output)/f'round_{r:03d}',device=device,methods=args.methods.split(','),
                supervisor_model=args.supervisor,condition=read_json(args.condition) if args.condition else None,
                split=cfg.evaluation_split+('_gain_panel' if args.gain_panel else ''))
    if args.command=='screen':
        from .screening import screen
        return screen(ExperimentConfig.load(args.config),read_json(args.pilots),args.output,device=device,supervisor_model=args.supervisor)
    if args.command=='freeze':
        from .screening import freeze
        return freeze(args.screen_dir,args.output,refit_evidence=args.refit_evidence,conditions=args.conditions)
    if args.command=='refit-check':
        from .refit_check import check_refit
        return check_refit(ExperimentConfig.load(args.config),args.run_dir,args.output,rounds=tuple(map(int,args.rounds.split(','))),
                           budgets=tuple(map(int,args.budgets.split(','))),gpu_ids=gpus,device=device)
    if args.command=='supervisor':
        from .supervisor import train_supervisor
        from .data import load_manifest
        return train_supervisor(args.output,load_manifest(args.manifest),steps=args.steps,seed=args.seed,device=device,smoke=args.smoke)
    if args.command=='teach':
        from .teaching import teach
        return teach(ExperimentConfig.load(args.config),args.run_dir,args.round,args.condition,args.output,gpu_ids=gpus,device=device,
                     relevant_ids=args.relevant_ids,supervisor_model=args.supervisor)
    if args.command=='report':
        from .reporting import summarize_suite
        return summarize_suite(args.runs,args.output)


if __name__=='__main__':main()
