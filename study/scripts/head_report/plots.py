"""Standalone PNG/PDF figures generated from the captured report inputs."""
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import rankdata

from . import ablation_plots

from .data import POSITION_NAMES


def generate(snapshot, data, balanced, pairs, selected, curve_heads, destination):
    destination = Path(destination);destination.mkdir(parents=True, exist_ok=True)
    figures = [];rounds = snapshot['meta']['rounds'];valid = data[data.eligible]
    colors = {h:plt.get_cmap('tab10')(i) for i,h in enumerate(curve_heads)}
    labels = {r['head']:r['label']+f' · n={r.get("seed_count",1)}'+(' · fp64' if r.get('precision')=='float64' else '') for r in data.to_dict('records')}
    def ci_lines(ax, panel, metric, y, color):
        if metric+'_eval_low' not in panel:return
        for (_,row),position in zip(panel.iterrows(),y):
            kind='total' if np.isfinite(row.get(metric+'_total_low',np.nan)) else 'eval'
            lo,hi=row.get(metric+'_'+kind+'_low'),row.get(metric+'_'+kind+'_high')
            if lo is not None and np.isfinite(lo):ax.hlines(position,lo,hi,color=color,lw=1,alpha=.65)
    def save(fig, name, title, caption):
        fig.savefig(destination / (name + '.png'), dpi=140, bbox_inches='tight')
        fig.savefig(destination / (name + '.pdf'), bbox_inches='tight');plt.close(fig)
        figures.append(dict(name=name, title=title, caption=caption))
    if valid.empty:return figures
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})

    current = valid[(valid['round'] == max(rounds)) & valid['head'].isin(selected)].sort_values('global_nauc', ascending=False)
    if len(current):
        fig, axes = plt.subplots(1, 3, figsize=(16, max(5, len(current) * .34)), layout='constrained', sharey=True)
        y = np.arange(len(current));axes[0].set_yticks(y, [textwrap.fill(labels[h], 48) for h in current['head']], fontsize=8)
        for ax in axes:ax.axvline(0, color='.7', lw=.7);ax.grid(axis='x', alpha=.2);ax.set_ylim(len(current) - .5, -.5)
        for metric, name, color, marker in [('neg_spearman', '−Spearman', '#2563eb', 'o'), ('neg_pearson', '−Pearson', '#d97706', 's')]:
            positions=y+(-.12 if marker=='o' else .12)
            ci_lines(axes[0],current,metric,positions,color)
            axes[0].scatter(current[metric], positions, label=name, c=color, marker=marker, s=28)
        for metric, name, color, marker in [('global_nauc', 'Global nAUC', '#2563eb', 'o'), ('within_nauc', 'Within-task nAUC', '#d97706', 's')]:
            positions=y+(-.12 if marker=='o' else .12)
            ci_lines(axes[1],current,metric,positions,color)
            axes[1].scatter(current[metric], positions, label=name, c=color, marker=marker, s=28)
        axes[2].barh(y, current.automation, color='#0d9488', height=.6);axes[2].xaxis.set_major_formatter(PercentFormatter(1))
        ci_lines(axes[2],current,'automation',y,'black')
        axes[0].set_title('Task correlation');axes[1].set_title('Oracle-normalized referral');axes[2].set_title('Automation at ≤2 pp success loss')
        axes[0].legend();axes[1].legend()
        save(fig, 'leaders', f'Checkpoint {max(rounds)}: selected heads and all paper baselines',
             'Points/bars are means over feature seeds. Intervals are approximate total 95% CIs where multiple seeds exist; otherwise evaluation-only CIs. Numerical flags and seed counts are in the tables. Selection is exploratory.')

    if not balanced.empty:
        shown = balanced[balanced['head'].isin(selected)].head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(12, max(4, len(shown) * .30)), layout='constrained')
        ax.barh(np.arange(len(shown)), shown.mean_global_nauc, color=['#dc6a38' if m not in ('kv', 'kv_t') else '#2563eb' if m == 'kv_t' else '#0d9488' for m in shown.method])
        ci_lines(ax,shown,'mean_global_nauc',np.arange(len(shown)),'black')
        ax.set_yticks(np.arange(len(shown)), [textwrap.fill(f'{r.label} · n={r.get("seed_count",1)}', 55) for _,r in shown.iterrows()], fontsize=8)
        ax.axvline(0, color='.5', lw=.7);ax.set_xlabel('Mean global referral nAUC across all requested checkpoints');ax.grid(axis='x', alpha=.2)
        save(fig, 'balanced', 'Comparison restricted to heads complete at every checkpoint',
             'Checkpoints are averaged within common feature seeds, then seeds are averaged. Lines are total 95% bootstrap CIs when estimable, otherwise evaluation-only CIs. Different checkpoints are not independent training seeds.')

    positions = list(dict.fromkeys(data[data.scope != 'paper']['group']))
    if positions:
        values = np.full((len(positions), len(rounds) * 2), np.nan);counts = {}
        for i, position in enumerate(positions):
            for j, (round_, method) in enumerate((r,m) for r in rounds for m in ('kv', 'kv_t')):
                planned = data[(data['group'] == position) & (data['round'] == round_) & (data.method == method)]
                done = planned[planned.eligible];counts[i,j] = (len(done), len(planned))
                if len(done):values[i,j] = done.global_nauc.max()
        fig, ax = plt.subplots(figsize=(10, max(5, len(positions) * .45)), layout='constrained')
        cmap = plt.get_cmap('viridis').copy();cmap.set_bad('#eceff3')
        finite=values[np.isfinite(values)]
        image = ax.imshow(np.ma.masked_invalid(values), aspect='auto', cmap=cmap, vmin=min(0,finite.min()) if len(finite) else 0, vmax=max(.8, finite.max()) if len(finite) else .8)
        for (i,j), (done,total) in counts.items():
            label = f'{values[i,j]:.3f}\n{done}/{total}' if np.isfinite(values[i,j]) else f'—\n{done}/{total}' if total else 'not planned'
            ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if np.isfinite(values[i,j]) and values[i,j] < .45 else 'black')
        ax.set_yticks(range(len(positions)), [POSITION_NAMES.get(v, v.replace('_', ' ')) for v in positions])
        ax.set_xticks(range(len(rounds)*2), [f'Round {r}\n{m.upper().replace("_", "-")}' for r in rounds for m in ('kv', 'kv_t')])
        fig.colorbar(image, ax=ax, label='Best observed global nAUC')
        save(fig, 'positions', 'Coverage and best observed referral at every head position',
             'Each cell shows the best eligible seed-mean configuration and completed/planned configuration-group counts (numerical versions separate). This is a selected maximum, not a controlled layer comparison; coverage differs.')

    if curve_heads and snapshot['cohorts']:
        fig, axes = plt.subplots(2, len(rounds), figsize=(7*len(rounds), 10), squeeze=False, layout='constrained')
        handles = {}
        for col, round_ in enumerate(rounds):
            if round_ not in snapshot['cohorts']:continue
            cohort = snapshot['cohorts'][round_]
            for row, allocation in enumerate(('global', 'within_task')):
                ax = axes[row,col]
                for head in curve_heads:
                    detail = snapshot['payload'].get((round_,head))
                    if detail:
                        line = ax.plot(detail['curves'][allocation][:,0],detail['curves'][allocation][:,1],lw=1.5,color=colors[head],label=labels[head])[0]
                        band=detail.get('bands',{}).get(allocation)
                        if band:ax.fill_between(detail['curves'][allocation][:,0],band['low'],band['high'],color=colors[head],alpha=.08,linewidth=0)
                        handles[head] = line
                oracle = cohort['oracles'][allocation]
                ax.plot(oracle[:,0],oracle[:,1],'k:',lw=1.4,label='Hindsight oracle')
                ax.plot(oracle[:,0],oracle[:,2],color='.5',ls='--',lw=1.2,label='Random referral')
                ax.axhline(cohort['summary']['supervisor_success']-.02,color='.6',ls=':',lw=.8)
                ax.set(title=f'Round {round_} · {allocation.replace("_", " ")}',xlabel='Fraction handled by learner',ylabel='Combined success',ylim=(0,1.02))
                ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.15)
        fig.legend(handles.values(), [textwrap.fill(labels[h], 65) for h in handles], loc='outside lower center',ncol=2,fontsize=8)
        save(fig, 'referral', 'Referral curves on paired learner/supervisor outcomes',
             'Lines average referral success across seeds. Shading: pointwise total 95% bootstrap CI with multiple seeds, evaluation-only CI for n=1. Within-task curves impose the same referred fraction per task. Dotted black = oracle; dashed grey = random. Bands are not simultaneous confidence bands.')
        fig, axes = plt.subplots(1,len(rounds),figsize=(7*len(rounds),5),squeeze=False,layout='constrained')
        for col, round_ in enumerate(rounds):
            ax = axes[0,col];tolerances = np.linspace(0,.2,81)
            for head in curve_heads:
                detail = snapshot['payload'].get((round_,head))
                if not detail:continue
                members=detail.get('members',[dict(detail=detail)])
                capacities=[]
                for member in members:
                    actual=member['detail']['curves']['global']
                    capacities.append([actual[actual[:,1]>=actual[0,1]-t-1e-12,0].max() for t in tolerances])
                capacity=np.mean(capacities,axis=0)
                ax.step(tolerances*100,capacity,where='post',color=colors[head],label=labels[head])
                if len(capacities)>1:
                    sd=np.std(capacities,axis=0,ddof=1)
                    ax.fill_between(tolerances*100,np.clip(capacity-sd,0,1),np.clip(capacity+sd,0,1),step='post',color=colors[head],alpha=.10)
            ax.axvline(2,color='.5',ls=':');ax.set(title=f'Round {round_}',xlabel='Allowed success loss from supervisor (percentage points)',ylabel='Maximum observed automated fraction',ylim=(0,1.02));ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
        fig.legend(handles.values(),[textwrap.fill(labels[h],65) for h in handles],loc='outside lower center',ncol=2,fontsize=8)
        save(fig, 'automation', 'Automation as the tolerated success loss changes',
             'Compute automation separately for every seed, then average; shading is ± sample seed SD, clipped to [0,1] for display (no band for n=1). The vertical line marks the 2-pp tolerance whose evaluation/total CIs are in the tables. These are validation operating points.')

    ablation_plots.generate(pairs, save, observation='Each dot compares two configuration seed means at one checkpoint.')

    for round_ in rounds:
        panel = valid[valid['round']==round_]
        scatter_heads = []
        for method in ('kv','kv_t'):
            part = panel[panel.method==method].sort_values('global_nauc',ascending=False)
            if len(part):scatter_heads.append(part.iloc[0]['head'])
        for method in ('vfd','gu'):
            candidates=panel[panel.method==method]
            if len(candidates):scatter_heads.append(candidates.iloc[0]['head'])
        scatter_heads = list(dict.fromkeys(scatter_heads))[:4]
        if not scatter_heads or round_ not in snapshot['cohorts']:continue
        cohort = snapshot['cohorts'][round_]
        fig, axes = plt.subplots(2,len(scatter_heads),figsize=(4*len(scatter_heads),7),squeeze=False,layout='constrained')
        for col,head in enumerate(scatter_heads):
            detail = snapshot['payload'][(round_,head)];cell = panel[panel['head']==head].iloc[0]
            members=detail.get('members',[dict(seed=42,detail=detail)])
            gain = cohort['supervisor']-cohort['learner']
            for i,member in enumerate(members):
                values=member['detail'];color=plt.get_cmap('tab10')(i)
                axes[0,col].scatter(rankdata(values['task_means']),rankdata(1-np.asarray(values['success_rates'])),s=35,color=color,marker='o' if i%2==0 else 'x',alpha=.7,label=f'Seed {member["seed"]}')
                jitter=np.random.default_rng(12+i).uniform(-.07,.07,len(gain))
                axes[1,col].scatter(rankdata(values['scores'])/len(gain),gain+jitter,s=9,alpha=.3,color=color)
            axes[0,col].legend(fontsize=7)
            axes[0,col].set(title=textwrap.fill(labels[head],36)+f'\nMean −Spearman={cell.neg_spearman:.3f}',xlabel='Rank of task-average uncertainty',ylabel='Rank of task failure rate')
            axes[1,col].set(title=f'Mean signed-benefit Spearman={cell.benefit_spearman:.3f}',xlabel='Uncertainty percentile',ylabel='Supervisor success − learner success',yticks=[-1,0,1],ylim=(-1.2,1.2))
            axes[1,col].xaxis.set_major_formatter(PercentFormatter(1))
            for row in range(2):axes[row,col].grid(alpha=.15)
        save(fig,f'correlations_r{round_:03d}',f'Round {round_}: task and episode-level Spearman plots',
             'Each color is one feature seed; raw uncertainty scores are not averaged into an ensemble. Titles give the mean of per-seed correlations. Top: ten task points per seed. Bottom: episode uncertainty versus signed benefit, with display-only vertical jitter. Tables report seed SD and confidence intervals.')

    return figures
