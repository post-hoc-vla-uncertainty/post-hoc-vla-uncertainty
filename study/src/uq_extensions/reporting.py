"""Task-macro learning curves and seed-paired contrasts with uncertainty intervals."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json,read_json
from .rollout_metrics import learning_summary


def automation_capacity(curve, tolerance=.02, *, column='utility'):
    """Maximum observed automation within a success deficit from all-supervisor.

    Use actual operating points, without interpolating fractional episodes or
    assuming monotonic success: the learner may beat the supervisor on a start.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Success tolerance must be finite and nonnegative')
    frame=pd.DataFrame(curve).sort_values('automated')
    if len(frame)<2 or not np.isfinite(frame[['automated',column]].to_numpy()).all():
        raise ValueError('Automation requires a finite complete referral curve')
    if not np.isclose(frame.iloc[0].automated,0) or not np.isclose(frame.iloc[-1].automated,1):
        raise ValueError('Referral curve must include zero and full automation')
    threshold=float(frame.iloc[0][column])-tolerance
    return float(frame.loc[frame[column]>=threshold-1e-12,'automated'].max())


def referral_report_statistics(scores,learner,supervisor,tasks,ids,*,allocation='global',curve=None):
    """Reference-repo random-to-oracle AUC normalization for success utility.

    Keep this derived analysis in reporting: changing plots or adding summaries
    must not invalidate trained policies, scores or their evidence hashes.
    """
    from .referral import referral_curve,safe_correlation
    scores,learner,supervisor=map(lambda v:np.asarray(v,dtype=float),(scores,learner,supervisor))
    if not (len(scores)==len(learner)==len(supervisor)==len(tasks)==len(ids)) or len(set(ids))!=len(ids):
        raise ValueError('Referral arrays must have matching unique identities')
    if not all(np.isfinite(v).all() for v in (scores,learner,supervisor)):
        raise ValueError('Referral scores and outcomes must be finite')
    if curve is None:
        curve,_=referral_curve(scores,learner,supervisor,tasks,ids,allocation=allocation)
    ranked=pd.DataFrame(curve).sort_values('automated')
    # Oracle refers by signed benefit, including cases where referral hurts.
    oracle,_=referral_curve(supervisor-learner,learner,supervisor,tasks,ids,allocation=allocation)
    oracle=pd.DataFrame(oracle).sort_values('automated')
    auc=float(np.trapz(ranked.utility,ranked.automated))
    random_auc=float(np.trapz(ranked.random_utility,ranked.automated))
    oracle_auc=float(np.trapz(oracle.utility,oracle.automated))
    gap=oracle_auc-random_auc
    normalized=(auc-random_auc)/gap if gap>1e-12 else None
    statistics=dict(allocation=allocation,collaborative_auc_success=auc,random_auc_success=random_auc,
        oracle_auc_success=oracle_auc,random_to_oracle_auc_gap=gap,
        oracle_normalized_gain=normalized,oracle_normalization_applicable=gap>1e-12,
        gain_over_random=auc-random_auc,regret_to_oracle=oracle_auc-auc,
        automation_success_tolerance=.02,
        automation_within_two_points=automation_capacity(ranked),
        random_automation_within_two_points=automation_capacity(ranked,column='random_utility'),
        oracle_automation_within_two_points=automation_capacity(oracle),
        signed_gain_spearman=safe_correlation(scores,supervisor-learner),
        episodes=len(ids),tasks=len(set(tasks)))
    return statistics,ranked,oracle


def referral_diagnostic_report(root,matrix,*,supervisor_cache=None):
    """Regenerable CPU-only plots from scores and matching paired outcomes.

    Original matrix/metrics/declaration files remain unchanged. New summaries,
    plot data, PNG/PDF files, and standalone HTML live in this report directory.
    """
    import base64,html,os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from scipy.stats import rankdata
    from .common import model_identity
    from .referral import pair_outcomes
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    project=Path(os.environ.get('VLA_PROJECT_ROOT',Path(__file__).resolve().parents[2]))
    cache=Path(supervisor_cache) if supervisor_cache is not None else project/'storage/extensions/supervisor/evaluation'
    cached=[(p,read_json(p)) for p in sorted(cache.glob('*/evaluation.json'))]
    model_hashes={};paired={};summaries=[];panels=[];unavailable=[];task_rows=[];episode_rows=[];curve_rows=[]
    for row in matrix:
        if not row.get('metrics'):continue
        folder=Path(row['directory']);scoring=folder.parent.parent
        key=(str(scoring),row['round'])
        try:
            if key not in paired:
                declaration=read_json(scoring/'declaration.json')
                model=str(Path(declaration['supervisor']).resolve())
                if model not in model_hashes:model_hashes[model]=model_identity(model)
                learner=read_json(folder.parent/'learner/evaluation.json')
                spec=learner['spec']
                matches=[(p,d) for p,d in cached if d['spec'].get('checkpoint')==model_hashes[model]
                    and {k:v for k,v in d['spec'].items() if k!='checkpoint'}=={k:v for k,v in spec.items() if k!='checkpoint'}]
                if len(matches)!=1:raise ValueError(f'Expected one supervisor matching checkpoint and complete evaluation protocol; found {len(matches)}')
                teacher_path,teacher=matches[0]
                pair_outcomes(learner['episodes'],teacher['episodes'])
                paired[key]=(learner,teacher,teacher_path,declaration)
            learner,teacher,teacher_path,declaration=paired[key]
            values=read_json(folder/row['recipe']['method']/'scores.json')
        except (FileNotFoundError,ValueError,KeyError) as error:
            unavailable.append(dict(recipe=row['recipe']['id'],round=row['round'],directory=str(folder),reason=str(error)))
            continue
        left={e['id']:e for e in learner['episodes']};right={e['id']:e for e in teacher['episodes']}
        ids=sorted(left)
        if sorted(values['scores'])!=ids:raise ValueError(f'Scores and outcomes differ in {folder}')
        score=np.array([values['scores'][i] for i in ids]);l=np.array([left[i]['success'] for i in ids]);h=np.array([right[i]['success'] for i in ids])
        tasks=[left[i]['task'] for i in ids];metrics=row['metrics'];cfg=declaration['config']
        common=dict(recipe=row['recipe']['id'],round=row['round'],training_mode=cfg['training_mode'],seed=cfg['seed'],
            split=declaration['split'],condition=declaration.get('condition'),source=str(scoring),
            query_slots=row['round']*cfg['batch_queries'],supervisor_evaluation=str(teacher_path))
        for task in sorted(set(tasks)):
            task_rows.append(dict(**common,task=task,pool_mean_uncertainty=values['pool_task_means'][task],
                success=float(np.mean([left[i]['success'] for i in ids if left[i]['task']==task]))))
        for i,s,a,b in zip(ids,score,l,h):
            episode_rows.append(dict(**common,id=i,task=left[i]['task'],uncertainty=float(s),learner_success=int(a),
                supervisor_success=int(b),signed_supervisor_benefit=int(b-a)))
        for allocation in ('global','within_task'):
            curve_file=folder/row['recipe']['method']/f'referral_{allocation}.csv'
            curve=pd.read_csv(curve_file) if curve_file.exists() else None
            statistics,ranked,oracle=referral_report_statistics(score,l,h,tasks,ids,allocation=allocation,curve=curve)
            stored=metrics.get('referral_'+allocation,{})
            if stored and not np.isclose(statistics['gain_over_random'],stored['raw_gain']):
                raise ValueError(f'Referral report disagrees with recorded gain in {folder}')
            record=dict(**common,**statistics,headroom_normalized_gain=stored.get('normalized_gain'),
                task_success_spearman=metrics.get('between_task_spearman'),task_success_pearson=metrics.get('between_task_pearson'))
            summaries.append(record)
            for name,frame in [('ranked',ranked),('oracle',oracle)]:
                curve_rows.extend(dict(**common,allocation=allocation,curve=name,**r) for r in frame.to_dict('records'))
            if allocation=='global':panels.append(dict(record=record,ranked=ranked,oracle=oracle))
    atomic_json(root/'summary.json',dict(normalization='(AUC_success - AUC_random) / (AUC_oracle - AUC_random)',
        oracle='Signed paired supervisor-minus-learner success; retrospective diagnostic only',
        automation='Maximum evaluated operating point with system success >= supervisor success - 0.02; no interpolation',
        task_correlation='Offline demonstration-pool mean uncertainty versus task success; negative is desirable',
        benefit_correlation='Evaluation-start uncertainty versus signed supervisor benefit; positive is desirable',
        records=summaries,unavailable=unavailable))
    for name,data in [('summary',summaries),('task_correlations',task_rows),('episode_correlations',episode_rows),('referral_curves',curve_rows)]:
        pd.DataFrame(data).to_csv(root/f'{name}.csv',index=False)
    sections=['<h2>Normalized referral area, correlations and automation</h2>',
        '<p><b>nAUC = (AUC − AUC<sub>random</sub>) / (AUC<sub>oracle</sub> − AUC<sub>random</sub>)</b>, '
        'using success utility over automated fraction. 0 = random referral; 1 = the paired oracle; negative = worse than random. '
        'An oracle uses observed signed supervisor benefit and is only a retrospective benchmark. Undefined when the random-to-oracle gap is zero. '
        'This matches the reference repository’s oracle_normalized_gain / fraction_possible_benefit. The earlier supervisor-headroom normalization remains separately in summary.csv.</p>',
        '<p>Automation means the maximum measured fraction handled by the learner while combined success is at least supervisor success minus 2 percentage points. '
        'Curves may be nonmonotonic because the supervisor can fail where the learner succeeds. No fractional-episode interpolation is used. '
        'Spearman coefficients use task-average pool scores versus success (negative is desirable), and actual episode-start scores versus signed supervisor benefit (positive is desirable). '
        'These descriptive estimates use one policy seed per panel; they are not confidence bounds or deployment guarantees.</p>']
    figure_paths=[]
    def save(fig,name):
        for extension in ('png','pdf'):fig.savefig(root/f'{name}.{extension}',dpi=140,bbox_inches='tight')
        plt.close(fig);figure_paths.append(name+'.png')
        sections.append('<img class="plot" alt="'+html.escape(name)+'" src="data:image/png;base64,'+
                        base64.b64encode((root/f'{name}.png').read_bytes()).decode()+'">')
    frame=pd.DataFrame([p['record'] for p in panels])
    if len(frame):
        display=frame[['recipe','round','training_mode','oracle_normalized_gain','signed_gain_spearman',
            'task_success_spearman','automation_within_two_points']]
        sections.append(display.to_html(index=False,float_format=lambda v:f'{v:.3f}',na_rep='undefined',escape=True))
    # Facet by source: never average different policies, conditions, or checkpoints.
    labels={'S02_E':'KV fused','S04_E':'KV-T embedding','S04_V':'KV-T velocity','S11_patch_top':'Patch KV'}
    for group_index,source in enumerate(dict.fromkeys(p['record']['source'] for p in panels)):
        group=[p for p in panels if p['record']['source']==source]
        recipes=list(dict.fromkeys(p['record']['recipe'] for p in group));first=group[0]['record']
        mode={'save_warm_50_50':'SAVE','pristine_refit':'Pristine refit'}.get(first['training_mode'],first['training_mode'])
        title=f"{mode} / seed {first['seed']} / {first['split']}"
        for page,start in enumerate(range(0,len(recipes),8)):
            selected=recipes[start:start+8];colors={r:plt.get_cmap('tab10')(i) for i,r in enumerate(selected)}
            subset=[p for p in group if p['record']['recipe'] in selected]
            fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
            plot_specs=[('oracle_normalized_gain','Oracle-normalized referral area','nAUC'),
                ('automation_within_two_points','Automation within 2 pp of supervisor','Automated fraction'),
                ('signed_gain_spearman','Uncertainty vs signed supervisor benefit','Spearman ρ (higher is better)'),
                ('task_success_spearman','Pool mean uncertainty vs task success','Spearman ρ (more negative is better)')]
            for ax,(metric,heading,ylabel) in zip(axes.flat,plot_specs):
                for recipe in selected:
                    measurements=sorted([p['record'] for p in subset if p['record']['recipe']==recipe],key=lambda r:r['round'])
                    ax.plot([r['query_slots'] for r in measurements],[np.nan if r[metric] is None else r[metric] for r in measurements],
                            marker='o',label=labels.get(recipe,recipe),color=colors[recipe])
                ax.set(title=heading,xlabel='Acquisition query slots',ylabel=ylabel);ax.grid(alpha=.2)
                if 'spearman' in metric:ax.set_ylim(-1.05,1.05)
                if metric=='automation_within_two_points':
                    baseline=sorted([p['record'] for p in subset if p['record']['recipe']==selected[0]],key=lambda r:r['round'])
                    ax.plot([r['query_slots'] for r in baseline],[r['random_automation_within_two_points'] for r in baseline],
                            'k--',label='Random referral')
                    ax.set_ylim(-.02,1.02);ax.yaxis.set_major_formatter(PercentFormatter(1))
                else:ax.axhline(0,color='grey',ls=':',lw=1)
            axes[0,0].legend(fontsize=7);axes[0,1].legend(fontsize=7)
            fig.suptitle(title);save(fig,f'overview_{group_index:02d}_{page:02d}')
            for round_ in sorted(set(p['record']['round'] for p in subset)):
                current=[p for p in subset if p['record']['round']==round_]
                fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
                for panel in current:
                    r=panel['record'];nauc=r['oracle_normalized_gain'];label=labels.get(r['recipe'],r['recipe'])+' | nAUC '+('undefined' if nauc is None else f'{nauc:.3f}')
                    curve=panel['ranked'];axes[0].plot(curve.automated,curve.utility,label=label,color=colors[r['recipe']])
                    ceilings=np.linspace(0,.2,41)
                    axes[1].step(ceilings,[automation_capacity(curve,v) for v in ceilings],where='post',label=labels.get(r['recipe'],r['recipe']),color=colors[r['recipe']])
                curve=current[0]['ranked'];oracle=current[0]['oracle']
                axes[0].plot(curve.automated,curve.random_utility,'k--',label='Expected random')
                axes[0].plot(oracle.automated,oracle.utility,color='grey',ls=':',label='Paired oracle')
                axes[1].step(ceilings,[automation_capacity(curve,v,column='random_utility') for v in ceilings],where='post',color='black',ls='--',label='Expected random')
                axes[1].axvline(.02,color='grey',ls=':',label='2 pp operating point')
                axes[0].set(title='Referral curves with normalized area',xlabel='Automated fraction',ylabel='Combined success',ylim=(0,1))
                axes[1].set(title='Automation capacity by allowed success deficit',xlabel='Allowed deficit from supervisor success',ylabel='Maximum automated fraction',ylim=(-.02,1.02))
                for ax in axes:
                    ax.grid(alpha=.2);ax.legend(fontsize=7);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
                fig.suptitle(f'{title} / round {round_}');save(fig,f'referral_automation_{group_index:02d}_{page:02d}_round_{round_:03d}')
                # Equal-weight task ranks make tied outcomes and tiny samples visible.
                fig,axes=plt.subplots(2,len(current),figsize=(max(7,3*len(current)),6.5),squeeze=False,layout='constrained')
                for col,panel in enumerate(current):
                    r=panel['record']
                    tr=[t for t in task_rows if t['source']==source and t['round']==round_ and t['recipe']==r['recipe']]
                    er=[t for t in episode_rows if t['source']==source and t['round']==round_ and t['recipe']==r['recipe']]
                    for ax,data,xkey,ykey,metric,ylabel in [
                        (axes[0,col],tr,'pool_mean_uncertainty','success','task_success_spearman','Task-success rank'),
                        (axes[1,col],er,'uncertainty','signed_supervisor_benefit','signed_gain_spearman','Supervisor-benefit rank')]:
                        x=rankdata([t[xkey] for t in data]);y=rankdata([t[ykey] for t in data])
                        ax.scatter(x,y,s=25,alpha=.75,color=colors[r['recipe']]);rho=r[metric]
                        ax.set(title=labels.get(r['recipe'],r['recipe'])+' / ρ '+('undefined' if rho is None else f'{rho:.3f}'),xlabel='Uncertainty rank',ylabel=ylabel);ax.grid(alpha=.2)
                        if data is tr:
                            for xi,yi,t in zip(x,y,data):ax.annotate(t['task'].split('/')[-1],(xi,yi),xytext=(3,3),textcoords='offset points',fontsize=7)
                fig.suptitle(f'Spearman rank plots / {title} / round {round_} (average ranks for ties)')
                save(fig,f'spearman_{group_index:02d}_{page:02d}_round_{round_:03d}')
    if unavailable:sections.append('<h3>Unavailable diagnostics</h3>'+pd.DataFrame(unavailable).to_html(index=False,escape=True))
    fragment=''.join(sections)
    (root/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Referral diagnostics</title>'
        '<style>body{font:15px system-ui;max-width:1300px;margin:2rem auto;padding:1rem}img.plot{max-width:100%}'
        'td,th{padding:.4rem;border:1px solid #ccc}table{border-collapse:collapse}</style>'+fragment+'</html>')
    return dict(html=fragment,figures=figure_paths,records=summaries,unavailable=unavailable)


def report_run(root, *, evaluation_paths=None, output_root=None):
    root=Path(root);experiment=read_json(root/'experiment.json');cfg=experiment['config'];rows=[]
    output=Path(output_root) if output_root is not None else root
    output.mkdir(parents=True,exist_ok=True)
    paths=({int(p.parent.parent.name.split('_')[1]):p for p in root.glob('round_*/evaluation/evaluation.json')}
           if evaluation_paths is None else {int(r):Path(p) for r,p in evaluation_paths.items()})
    manifest=read_json(cfg['data_manifest'])
    task_of={r['id']:r['task'] for r in manifest['observations']}
    trajectories=read_json(Path(manifest['evaluator_directory'])/'trajectory_index.json')
    frames={r['id']:r['dataset_to_index']-r['dataset_from_index'] for r in trajectories}
    ledger=read_json(root/'acquisition_ledger.json') if (root/'acquisition_ledger.json').exists() else {'rounds':[]}
    for r,path in sorted(paths.items()):
        episodes=pd.DataFrame(read_json(path)['episodes'])
        histories=ledger['rounds'][:r];ids=[id for h in histories for id in h['ids']]
        unique=set(ids)-set(ledger.get('pretraining_ids',[]))
        record=dict(round=r,new_demonstrations=len(unique),queries=len(ids),
                    prior_pretraining_queries=sum(len(h.get('prior_pretraining',[])) for h in histories),
                    prior_acquisition_queries=sum(len(h.get('prior_acquisition',[])) for h in histories),
                    revealed_frames=sum(frames[id] for id in unique),distinct_acquired_tasks=len({task_of[id] for id in unique}),
                    success=float(episodes.groupby('task').success.mean().mean()),
                    per_task_success=episodes.groupby('task').success.mean().to_dict())
        for metric in ('q_time','q_path','relative_time','relative_path'):
            means=episodes.groupby('task')[metric].mean().dropna()
            record[metric]=float(means.mean()) if len(means) else None
        fit=root/f'round_{r:03d}/policy_fit.json'
        record['training_seconds']=read_json(fit)['seconds'] if fit.exists() else 0
        record['policy_updates']=read_json(fit)['requested_steps_per_member']*read_json(fit).get('policy_count',len(read_json(fit)['models'])) if fit.exists() else 0
        selection=root/f'round_{r:03d}/selection/acquisition.json'
        record['acquisition_seconds']=read_json(selection)['seconds'] if selection.exists() else 0
        timing=path.parent/'timing.json'
        record['evaluation_seconds']=read_json(timing)['seconds'] if timing.exists() else None
        rows.append(record)
    if not rows:raise ValueError('No evaluations available')
    budget_key='queries' if cfg['eligibility_protocol']=='upstream_reference' else 'new_demonstrations'
    protocols={read_json(p).get('evaluation_protocol','paired_extension') for p in paths.values()}
    if len(protocols)!=1:raise ValueError('Mixed evaluation protocols in a learning curve')
    preprocessing={read_json(p).get('spec',{}).get('observation_preprocessing') for p in paths.values()}
    if len(preprocessing)!=1:raise ValueError('Mixed observation preprocessing in a learning curve')
    rendering={read_json(p).get('spec',{}).get('rendering_protocol') for p in paths.values()}
    if len(rendering)!=1:raise ValueError('Mixed rendering protocols in a learning curve')
    summary=dict(config=cfg,curve=rows,evaluation_protocol=protocols.pop(),**learning_summary(rows,budget_key=budget_key))
    if evaluation_paths is not None:
        summary.update(source_run=str(root.resolve()),evaluation_paths={str(r):str(p) for r,p in paths.items()},
                       observation_preprocessing=preprocessing.pop(),rendering_protocol=rendering.pop())
    allocations=[]
    for r,entry in enumerate(ledger['rounds'],1):
        for id in entry['ids']:allocations.append(dict(round=r,id=id,task=task_of[id]))
    pd.DataFrame(allocations).to_csv(output/'task_allocation.csv',index=False)
    pd.DataFrame([{k:v for k,v in r.items() if k!='per_task_success'} for r in rows]).to_csv(output/'learning_curve.csv',index=False)
    atomic_json(output/'summary.json',summary);return summary


def summarize_suite(roots,output,seed=42):
    summaries=[report_run(p) for p in roots];rows=[]
    for s in summaries:
        c=s['config'];rows.append(dict(mode=c['training_mode'],protocol=c['eligibility_protocol'],arm=c['arm'],seed=c['seed'],
                                        evaluation_protocol=s['evaluation_protocol'],budget_axis=s['budget_axis'],
                                        area=s['area'],final_success=s['final_success']))
    frame=pd.DataFrame(rows)
    if frame.duplicated(['mode','protocol','arm','seed']).any():raise ValueError('Duplicate seed trajectory')
    contrasts=[];rng=np.random.default_rng(seed)
    for (mode,protocol,evaluation_protocol),group in frame.groupby(['mode','protocol','evaluation_protocol']):
        for left,right in (('kv','vfd'),('kv_t','kv'),('kv_ivr','kv'),('kv_t_ivr','kv_t'),('kv_t_ivr','kv_ivr')):
            a=group[group.arm==left].set_index('seed');b=group[group.arm==right].set_index('seed')
            if set(a.index)!=set(b.index) or not len(a):continue
            for metric in ('area','final_success'):
                delta=(a[metric]-b[metric]).dropna().to_numpy()
                ci=np.quantile(delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1),[.025,.975]).tolist() if len(delta)>1 else None
                contrasts.append(dict(mode=mode,protocol=protocol,left=left,right=right,metric=metric,
                                       evaluation_protocol=evaluation_protocol,
                                       seeds=len(delta),paired_mean=float(delta.mean()),bootstrap_95=ci))
    for (protocol,arm,evaluation_protocol),group in frame.groupby(['protocol','arm','evaluation_protocol']):
        a=group[group['mode']=='save_warm_50_50'].set_index('seed');b=group[group['mode']=='pristine_refit'].set_index('seed')
        if set(a.index)!=set(b.index) or not len(a):continue
        for metric in ('area','final_success'):
            delta=(a[metric]-b[metric]).dropna().to_numpy()
            ci=np.quantile(delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1),[.025,.975]).tolist() if len(delta)>1 else None
            contrasts.append(dict(contrast='training_protocol',protocol=protocol,arm=arm,left='save_warm_50_50',right='pristine_refit',
                evaluation_protocol=evaluation_protocol,
                metric=metric,seeds=len(delta),paired_mean=float(delta.mean()),bootstrap_95=ci))
    output=Path(output);output.mkdir(parents=True,exist_ok=True);frame.to_csv(output/'seed_results.csv',index=False)
    atomic_json(output/'contrasts.json',contrasts)
    return contrasts


def experiment_report(root,*,matrix=None,runs=(),run_summaries=None):
    """Portable HTML plus existing JSON/CSV sources; never reads hidden labels."""
    import base64,html,io,math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    root=Path(root);root.mkdir(parents=True,exist_ok=True);sections=[]
    def figure(fig):
        stream=io.BytesIO();fig.savefig(stream,format='png',bbox_inches='tight',dpi=130);plt.close(fig)
        return '<img class="plot" src="data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()+'">'
    def picture(array):
        stream=io.BytesIO();Image.fromarray(np.asarray(array,dtype=np.uint8)).save(stream,format='PNG')
        return '<img width="220" src="data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()+'">'
    if matrix is not None:
        rows=[]
        for r in matrix:
            m=r.get('metrics') or {};label=f'{r["recipe"]["id"]} / round {r["round"]}'
            rows.append(dict(recipe=label,status=r['status'],referral_gain=m.get('referral_global',{}).get('raw_gain'),
                task_spearman=m.get('between_task_spearman'),task_pearson=m.get('between_task_pearson'),within_task_auroc=m.get('macro_auroc'),
                mapping_r2=m.get('mapping_check',{}).get('gp_r2'),seconds=round(r['seconds'],1),reason=r.get('reason')))
        sections.append('<h2>Configuration matrix</h2>'+pd.DataFrame(rows).to_html(index=False,escape=True))
        diagnostics=referral_diagnostic_report(root/'referral_diagnostics',matrix)
        sections.append(diagnostics['html'])
        for r in matrix:
            flow_files=list((Path(r['directory'])/r['recipe']['method']).glob('*_flow_scores.npz'))
            if not flow_files:continue
            with np.load(flow_files[0],allow_pickle=False) as a:
                times=a['flow_indices'];values=a['latent_variances'].reshape(len(a['ids']),-1,len(times))
                fig,ax=plt.subplots(figsize=(7,4));means=values.mean((0,1));spread=values.std((0,1))
                ax.errorbar(times,means,yerr=spread,marker='o');ax.set(xlabel='Solver step index',ylabel='Conditional latent GP variance',title=r['recipe']['id'])
                sections.append('<h2>Flow probe diagnostics</h2><p>Mean and spread of conditional latent variances across observations and fixed Gaussian probes. The spread is diagnostic; acquisition uses the declared weighted mean, not this spread.</p>'+figure(fig))
            break
        taskfile=root/'task_head_scores.csv'
        if taskfile.exists() and taskfile.stat().st_size>1:
            df=pd.read_csv(taskfile)
            if len(df):
                pivot=df.pivot_table(index='task',columns='recipe',values='score',aggfunc='mean')
                scale=(pivot-pivot.min())/(pivot.max()-pivot.min()).replace(0,np.nan)
                fig,ax=plt.subplots(figsize=(max(8,.35*len(pivot.columns)),5));im=ax.imshow(scale,aspect='auto',vmin=0,vmax=1)
                ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=90,fontsize=7);ax.set_yticks(range(len(pivot.index)),pivot.index,fontsize=7)
                fig.colorbar(im,ax=ax);sections.append('<h2>Task × head</h2><p>Color scaled separately per head; raw scores are in task_head_scores.csv. Missing or constant cells stay blank.</p>'+figure(fig))
        for r in matrix:
            patches=list((Path(r['directory'])/r['recipe']['method']).glob('*_patch_scores.npz'))
            if not patches:continue
            scorefile=patches[0]
            inputs=Path(r['directory']).parent/'learner'
            if not scorefile.exists() or not (inputs/'initial_inputs.json').exists():continue
            from .evaluation import evaluation_view
            view=evaluation_view(inputs)
            with np.load(scorefile,allow_pickle=False) as a:
                for index in np.argsort(a['variances'].max((1,2)))[-3:]:
                    id=str(a['ids'][index]);obs=view.observation(id);fig,axes=plt.subplots(2,2,figsize=(7,6))
                    for camera,key in enumerate(('image','image2')):
                        values=a['variances'][index,camera];side=math.isqrt(len(values))
                        axes[camera,0].imshow(obs[key]);axes[camera,1].imshow(obs[key]);axes[camera,1].imshow(values.reshape(side,side),extent=(0,obs[key].shape[1],obs[key].shape[0],0),alpha=.55,cmap='magma')
                        for axis in axes[camera]:axis.axis('off')
                    fig.suptitle(id);sections.append('<h2>Native patch variance</h2>'+figure(fig))
            break
    for run in runs:
        run=Path(run)
        summary=run_summaries[str(run)] if run_summaries is not None else report_run(run)
        curve=summary['curve'];cfg=summary['config']
        axis=summary['budget_axis'];fig,ax=plt.subplots(figsize=(7,4));ax.plot([v[axis] for v in curve],[v['success'] for v in curve],marker='o')
        ax.set(xlabel='Query slots' if axis=='queries' else 'New demonstrations',ylabel='Task-macro success',ylim=(0,1),title=f"{cfg['arm']} / {cfg['training_mode']}")
        sections.append('<h2>Active learning</h2>'+figure(fig))
        from .data import load_manifest,candidate_view
        manifest=load_manifest(cfg['data_manifest']);view=candidate_view(manifest)
        for path in sorted(run.glob('round_*/selection/acquisition.json')):
            audit=read_json(path);gains={x['id']:x['gain'] for x in audit.get('steps',[])}
            cards=[]
            for id in audit['selected_ids']:
                obs=view.observation(id);caption=f"{id} | score={audit.get('scores',{}).get(id)} | IVR gain={gains.get(id)}"
                cards.append('<figure>'+picture(obs['image'])+picture(obs['image2'])+'<figcaption>'+html.escape(caption)+'<br>'+html.escape(str(obs['instruction']))+'</figcaption></figure>')
            sections.append('<h3>'+html.escape(path.parent.parent.name)+' acquisition</h3>'+''.join(cards))
    allocated=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
    content='''<!doctype html><html><meta charset="utf-8"><title>VLA uncertainty experiments</title>
<style>body{font:15px system-ui;margin:2rem auto;max-width:1300px;padding:1rem;color:#182331}table{border-collapse:collapse;font-size:12px}td,th{padding:6px;border:1px solid #ccd3da}img.plot{max-width:100%}figure{display:inline-block;max-width:460px;vertical-align:top}figcaption{overflow-wrap:anywhere}h2{margin-top:2rem}</style>
<h1>VLA uncertainty experiment report</h1><p>Preview budgets are for inspecting outputs; scientific comparisons use the declared development/final recipes. Referral gain is measured against random referral at the same budget. JSON and CSV files retain full metrics and provenance.</p>'''
    content+=''.join(sections)+f'<p>Report directory: {allocated/1e9:.3f} GB before HTML generation. Shared feature caches and policy weights are accounted separately.</p></html>'
    (root/'report.html').write_text(content)
    return root/'report.html'


def teaching_report(root,result,candidates):
    import html,base64,io
    from PIL import Image
    root=Path(root);frame=pd.read_csv(root/'branch_comparison.csv')
    text='<html><meta charset="utf-8"><title>Teaching comparison</title><body><h1>Matched selector teaching</h1>'
    text+='<p>All selectors see the same mixed candidate pool and transformation rules. Improvements are paired to the common starting learner; branch overlap is allowed.</p>'+frame.to_html(index=False)
    for branch,ids in result['batches'].items():
        text+='<h2>'+html.escape(branch)+'</h2>'
        for id in ids:
            obs=candidates.observation(id);text+='<p>'+html.escape(id+': '+str(obs['instruction']))+'</p>'
            for camera in ('image','image2'):
                buffer=io.BytesIO();Image.fromarray(np.asarray(obs[camera],dtype=np.uint8)).save(buffer,format='PNG')
                text+='<img width="220" src="data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()+'">'
    text+='<h2>Batch overlap</h2><pre>'+html.escape(str(result['overlap']))+'</pre></body></html>'
    (root/'report.html').write_text(text)
