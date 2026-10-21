"""Original review's figure layout, adapted to paired three-policy-seed outcomes."""
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import rankdata

from . import ablation_plots

from uq_extensions.common import identity, atomic_npz, file_hash
from . import bootstrap
from .bootstrap import Plan, interpolate_rows
from .followup_data import DATASETS, POSITIONS, combine


def curve_samples(cohort, scores, cache):
    grid=np.linspace(0,1,101)
    digest=identity(dict(cohort={k:cohort[k] for k in ('ids','tasks','clusters','learner','supervisor')},
        scores=np.asarray(scores).tolist(), bootstrap_code=file_hash(bootstrap.__file__), version=1))
    path=Path(cache)/(digest+'.npz')
    if path.exists():
        with np.load(path) as saved:return {k:saved[k] for k in ('global','within_task')}
    plan=Plan(cohort,repetitions=2000);curves=plan.curves(scores)
    result={name:interpolate_rows(curves[key][0],curves[key][1],grid)
            for name,key in [('global','global_'),('within_task','within')]}
    atomic_npz(path,**result)
    return result


def generate(snapshot, review, selected, curve_heads, destination, cache):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    data=review['data'];valid=data[data.coverage=='complete'];balanced=review['balanced'];figures=[]
    conditions=sorted({p['condition'] for p in snapshot['policies']})
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    labels={r['head']:r['label'] for r in data.to_dict('records')}
    def save(fig,name,title,caption):
        for suffix in ('png','pdf'):fig.savefig(destination/(name+'.'+suffix),dpi=140,bbox_inches='tight')
        plt.close(fig);figures.append(dict(name=name,title=title,caption=caption))
    def intervals(ax,panel,metric,positions,color):
        for (_,row),y in zip(panel.iterrows(),positions):
            lo,hi=row.get(metric+'_total_low'),row.get(metric+'_total_high')
            if lo is not None and np.isfinite(lo):ax.hlines(y,lo,hi,color=color,lw=1,alpha=.7)
    for condition in conditions:
        panel=valid[(valid.condition==condition)&valid['head'].isin(selected)].sort_values('global_nauc',ascending=False)
        if not len(panel):continue
        fig,axes=plt.subplots(1,3,figsize=(16,max(5,len(panel)*.4)),layout='constrained',sharey=True)
        y=np.arange(len(panel));axes[0].set_yticks(y,[textwrap.fill(v,48) for v in panel.label],fontsize=8)
        for ax in axes:ax.axvline(0,color='.7',lw=.7);ax.grid(axis='x',alpha=.2);ax.set_ylim(len(panel)-.5,-.5)
        for ax,metrics in [(axes[0],[('neg_spearman','−Spearman'),('neg_pearson','−Pearson')]),
                           (axes[1],[('global_nauc','Global nAUC'),('within_nauc','Within-task nAUC')])]:
            for offset,(metric,name) in enumerate(metrics):
                color=('#2563eb','#d97706')[offset];positions=y+(-.12 if offset==0 else .12)
                intervals(ax,panel,metric,positions,color);ax.scatter(panel[metric],positions,c=color,s=25,label=name,marker=('o','s')[offset])
            ax.legend()
        axes[2].barh(y,panel.automation,color='#0d9488',height=.6);intervals(axes[2],panel,'automation',y,'black')
        axes[2].xaxis.set_major_formatter(PercentFormatter(1))
        for ax,title in zip(axes,('Task correlation','Oracle-normalized referral','Automation at ≤2 pp success loss')):ax.set_title(title)
        save(fig,'leaders_'+condition,DATASETS.get(condition,condition)+': selected heads versus paper baselines',
            'Means over three policy seeds, after averaging feature subsets within each policy. Lines are total 95% bootstrap intervals. Complete groups only; selection is exploratory.')
    if len(balanced):
        shown=balanced[balanced['head'].isin(selected)].head(25).iloc[::-1]
        fig,ax=plt.subplots(figsize=(12,max(4,len(shown)*.35)),layout='constrained')
        ax.barh(np.arange(len(shown)),shown.mean_global_nauc,color=['#2563eb' if m=='kv_t' else '#0d9488' if m=='kv' else '#dc6a38' for m in shown.method])
        intervals(ax,shown,'mean_global_nauc',np.arange(len(shown)),'black')
        ax.set_yticks(np.arange(len(shown)),[textwrap.fill(v,55) for v in shown.label],fontsize=8)
        ax.axvline(0,color='.5',lw=.7);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Mean global referral nAUC across the two training datasets')
        save(fig,'balanced','Heads complete for both training datasets','Average the datasets within each policy seed, then average three seeds. Intervals retain paired policies and evaluation samples; the two datasets are not six independent seeds.')
    positions=list(dict.fromkeys(data[data.scope!='paper']['group']))
    if positions:
        values=np.full((len(positions),len(conditions)*2),np.nan);counts={}
        for i,position in enumerate(positions):
            for j,(condition,method) in enumerate((c,m) for c in conditions for m in ('kv','kv_t')):
                panel=data[(data['group']==position)&(data.condition==condition)&(data.method==method)]
                complete=panel[panel.coverage=='complete'];counts[i,j]=(len(complete),len(panel))
                if len(complete):values[i,j]=complete.global_nauc.max()
        fig,ax=plt.subplots(figsize=(11,max(5,len(positions)*.5)),layout='constrained')
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eceff3');finite=values[np.isfinite(values)]
        im=ax.imshow(np.ma.masked_invalid(values),aspect='auto',cmap=cmap,vmin=min(0,finite.min()) if len(finite) else 0,vmax=max(.8,finite.max()) if len(finite) else .8)
        for (i,j),(n,total) in counts.items():
            text=f'{values[i,j]:.3f}\n{n}/{total}' if np.isfinite(values[i,j]) else f'—\n{n}/{total}' if total else 'not planned'
            ax.text(j,i,text,ha='center',va='center',fontsize=8,color='white' if np.isfinite(values[i,j]) and values[i,j]<.45 else 'black')
        ax.set_yticks(range(len(positions)),[POSITIONS.get(p,p) for p in positions]);ax.set_xticks(range(len(conditions)*2),[f'{c}\n{m.upper().replace("_","-")}' for c in conditions for m in ('kv','kv_t')])
        fig.colorbar(im,ax=ax,label='Best observed mean global nAUC')
        save(fig,'positions','Coverage and best observed referral by head location','Cells show the best complete three-policy-seed mean and complete/planned configuration-group counts. These selected maxima do not constitute matched layer comparisons.')
    if curve_heads and conditions:
        print('[report] Computing paired policy/evaluation confidence bands for selected referral curves',flush=True)
        fig,axes=plt.subplots(2,len(conditions),figsize=(7*len(conditions),10),squeeze=False,layout='constrained')
        af,aa=plt.subplots(1,len(conditions),figsize=(7*len(conditions),5),squeeze=False,layout='constrained')
        handles={};grid=np.linspace(0,1,101);losses=np.linspace(0,.2,81)
        for col,condition in enumerate(conditions):
            for index,head in enumerate(curve_heads):
                bundle=review['bundles'].get((condition,head))
                if bundle is None:continue
                print(f'[report] Referral bands: {condition} / {head}',flush=True)
                color=plt.get_cmap('tab10')(index);actual={k:[] for k in ('global','within_task')};samples={k:[] for k in actual};capacities=[]
                for seed in bundle['seeds']:
                    runs=bundle['runs'][seed];seed_actual={k:[] for k in actual};seed_samples={k:[] for k in actual};seed_caps=[]
                    for run in runs:
                        cohort=snapshot['cohorts'][run['row']['policy_id']];detail=run['detail'];boot=curve_samples(cohort,detail['scores'],cache)
                        for name in actual:
                            curve=detail['curves'][name];seed_actual[name].append(np.interp(grid,curve[:,0],curve[:,1]));seed_samples[name].append(boot[name])
                        curve=detail['curves']['global'];hbar=np.mean(cohort['supervisor'])
                        seed_caps.append([curve[curve[:,1]>=hbar-loss-1e-12,0].max() for loss in losses])
                    for name in actual:actual[name].append(np.mean(seed_actual[name],axis=0));samples[name].append(np.mean(seed_samples[name],axis=0))
                    capacities.append(np.mean(seed_caps,axis=0))
                for row,name in enumerate(actual):
                    evaluation,total=combine(samples[name]);band=np.quantile(total if total is not None else evaluation,[.025,.975],axis=0)
                    line=axes[row,col].plot(grid,np.mean(actual[name],axis=0),color=color,label=labels[head])[0];handles[head]=line
                    axes[row,col].fill_between(grid,band[0],band[1],color=color,alpha=.09,linewidth=0)
                mean=np.mean(capacities,axis=0);aa[0,col].step(losses*100,mean,where='post',color=color)
                if len(capacities)>1:
                    sd=np.std(capacities,axis=0,ddof=1);aa[0,col].fill_between(losses*100,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),color=color,alpha=.1,step='post')
            cohorts=[c for pid,c in snapshot['cohorts'].items() if pid.startswith(condition+'_seed')]
            for row,name in enumerate(('global','within_task')):
                ax=axes[row,col]
                if cohorts:
                    oracle=np.mean([np.interp(grid,c['oracles'][name][:,0],c['oracles'][name][:,1]) for c in cohorts],axis=0)
                    random=np.mean([np.interp(grid,c['oracles'][name][:,0],c['oracles'][name][:,2]) for c in cohorts],axis=0)
                    ax.plot(grid,oracle,'k:',label='Oracle');ax.plot(grid,random,'--',color='.5',label='Random')
                ax.set(title=DATASETS.get(condition,condition)+' · '+name.replace('_',' '),xlabel='Fraction handled by learner',ylabel='Combined success',ylim=(0,1.02));ax.grid(alpha=.2)
                ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
            aa[0,col].set(title=DATASETS.get(condition,condition),xlabel='Allowed success loss (percentage points)',ylabel='Maximum automated fraction',ylim=(0,1.02));aa[0,col].axvline(2,ls=':',color='.5');aa[0,col].grid(alpha=.2)
        for f in (fig,af):f.legend(handles.values(),[textwrap.fill(labels[h],65) for h in handles],loc='outside lower center',ncol=2,fontsize=8)
        save(fig,'referral','Referral curves on paired learner/supervisor outcomes','Lines average feature subsets within policies and then policies. Shading: pointwise total 95% bootstrap intervals, or evaluation-only for a single policy. Black dotted = hindsight oracle; grey dashed = random referral. These are not simultaneous confidence bands.')
        save(af,'automation','Automation as tolerated success loss changes','Mean automated fraction with ± sample policy-seed SD, after averaging subsets within each policy. The 2-pp operating point also has evaluation and total confidence intervals in the tables.')
    ablation_plots.generate(review['pairs'], save, observation='Each dot compares complete policy-seed means for one mapping and training dataset.')
    for condition in conditions:
        panel=valid[valid.condition==condition];heads=[]
        for m in ('kv','kv_t','vfd','gu'):
            part=panel[panel.method==m].sort_values('global_nauc',ascending=False)
            if len(part):heads.append(part.iloc[0]['head'])
        if not heads:continue
        fig,axes=plt.subplots(2,len(heads),figsize=(4*len(heads),7),squeeze=False,layout='constrained')
        for col,head in enumerate(heads):
            bundle=review['bundles'][(condition,head)];cell=panel[panel['head']==head].iloc[0]
            for i,seed in enumerate(bundle['seeds']):
                for j,run in enumerate(bundle['runs'][seed]):
                    detail=run['detail'];cohort=snapshot['cohorts'][run['row']['policy_id']];color=plt.get_cmap('tab10')(i)
                    axes[0,col].scatter(rankdata(detail['task_means']),rankdata(1-np.asarray(detail['success_rates'])),color=color,marker='o' if j==0 else 'x',alpha=.65,s=30,label=f'Policy seed {seed}' if j==0 else None)
                    gain=np.asarray(cohort['supervisor'])-np.asarray(cohort['learner'])
                    axes[1,col].scatter(rankdata(detail['scores'])/len(gain),gain,color=color,alpha=.2,s=8)
            axes[0,col].legend(fontsize=7);axes[0,col].set(title=textwrap.fill(labels[head],36)+f'\nMean −Spearman={cell.neg_spearman:.3f}',xlabel='Rank of task-average uncertainty',ylabel='Rank of task failure rate')
            axes[1,col].set(title=f'Mean benefit Spearman={cell.benefit_spearman:.3f}',xlabel='Uncertainty percentile',ylabel='Supervisor − learner mean success',ylim=(-1.1,1.1))
            for row in range(2):axes[row,col].grid(alpha=.15)
        save(fig,'correlations_'+condition,DATASETS.get(condition,condition)+': task and signed-benefit correlations','Colors identify policy-training seeds. Random feature subsets share their policy color; their scores are not averaged into an ensemble. Each initial condition uses the mean of three executions; negative supervisor benefit is retained. Titles average correlations within policies, then across policies.')
    return figures
