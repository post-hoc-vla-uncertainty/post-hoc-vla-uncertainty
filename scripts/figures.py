"""Result figures from data/final_save_v1 and data/derived (run scripts/analysis.py first).

Requires numpy, pandas, scipy and matplotlib. Writes vector PDFs to figures/generated/.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import DATA, DERIVED, ROOT, SEEDS, TASKS, cohort, curves  # noqa: E402

OUT = ROOT/'figures/generated'
# Validated categorical order (dataviz reference palette, light mode, adjacent pairs);
# colour follows the method in every figure. Random is a neutral reference.
COLOR = {'h1': '#2a78d6', 'vfd': '#eb6834', 'h2': '#1baf7a', 'gu': '#eda100', 'h1_ivr': '#e87ba4',
         'random': '#898781', 'other': '#b5b3ab', 'oracle': '#52514e'}
MARKER = {'h1': 'o', 'vfd': 's', 'h2': 'D', 'gu': '^', 'h1_ivr': 'v', 'random': 'x'}
LABEL = {'h1': 'KV-Ctx', 'h2': 'KV-Dec', 'vfd': 'VFD', 'gu': 'GU', 'h1_ivr': 'KV-Ctx+IVR', 'random': 'Random',
         'action_l2': 'Action-L2', 'ace': 'ACE', 'decu': 'DECU', 'vlm_token_entropy': 'VLM entropy',
         'vlm_perplexity': 'VLM perplexity'}
INK, GRID = '#0b0b0b', '#e1e0d9'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8,
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'lines.linewidth': 1.6, 'lines.solid_capstyle': 'round', 'pdf.fonttype': 42, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02})
WIDTH = 6.2  # inches, the manuscript text width


def acquisition_curves():
    success = pd.read_csv(DATA/'acquisition_success.csv')
    fig, ax = plt.subplots(figsize=(WIDTH*.62, 2.5))
    # Ensemble scores dashed. KV-Dec nearly coincides with VFD, so it is drawn above it dotted
    # with open markers, which keeps the VFD line and markers visible underneath.
    for arm in ['random', 'gu', 'vfd', 'h2', 'h1']:
        g = success[success.arm == arm].groupby('query_slots').success
        x, m, s = g.mean().index.values, 100*g.mean().values, 100*g.std(ddof=1).values
        ax.fill_between(x, m-s, m+s, color=COLOR[arm], alpha=.10, linewidth=0)
        open_marker = arm == 'h2'
        ax.plot(x, m, color=COLOR[arm], marker=MARKER[arm], markersize=4.5,
                markerfacecolor='none' if open_marker else COLOR[arm],
                markeredgecolor=COLOR[arm] if open_marker else 'white', markeredgewidth=1.1 if open_marker else .8,
                linestyle={'vfd': '--', 'gu': '--', 'h2': ':'}.get(arm, '-'), label=LABEL[arm],
                zorder=3 if arm == 'h1' else 2)
    ax.set_xlabel('Requested demonstrations')
    ax.set_ylabel('LIBERO-10 success (%)')
    ax.set_xticks([0, 25, 50, 75]); ax.set_xlim(-2, 77); ax.set_ylim(20, 70)
    handles = dict(zip(*ax.get_legend_handles_labels()[::-1]))
    order = [LABEL[a] for a in ['h1', 'h2', 'vfd', 'gu', 'random']]
    ax.legend([handles[k] for k in order], order, loc='lower right', fontsize=7, handlelength=3)
    fig.savefig(OUT/'acquisition_curves.pdf'); plt.close(fig)


def referral_curves(rnd=10):
    conditions = pd.read_csv(DATA/'conditions.csv')
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.35), sharey=False)
    heads = ['gu', 'vfd', 'h2', 'h1']
    for k, (key, title) in enumerate([('global_', 'Global referral'), ('within', 'Within-task referral')]):
        ax = axes[k]; stacks = {h: [] for h in heads+['oracle']}
        for seed in SEEDS:
            d = cohort(conditions, seed, rnd); p, s, t = d.primary.values, d.supervisor.values, d.task.values
            o = curves(s-p, p, s, t)[key]
            grid = np.linspace(0, 1, 101)
            def gain(curve):
                x, y, ref = curve   # x: fraction kept by the primary
                q = 1-x[::-1]; g = (y-ref)[::-1]
                return 100*np.interp(grid, q, g)
            stacks['oracle'].append(gain(o))
            for h in heads: stacks[h].append(gain(curves(d[h].values, p, s, t)[key]))
        oracle = np.mean(stacks['oracle'], 0)
        ax.plot(grid, oracle, color=COLOR['oracle'], linewidth=1.0, label='Hindsight oracle')
        print(f'{title}: oracle peak {oracle.max():.1f} pp; best head peak', {h: round(float(np.mean(stacks[h], 0).max()), 2) for h in heads})
        ax.axhline(0, color=COLOR['random'], linewidth=1.0, label='Random referral')
        for h in heads:
            m = np.mean(stacks[h], 0)
            ax.plot(grid, m, color=COLOR[h], label=LABEL[h], zorder=3 if h == 'h1' else 2)
        ax.set_title(title, fontsize=8, color=INK, loc='left')
        ax.set_xlabel('Fraction referred to the supervisor')
        ax.set_xlim(0, 1)
        if k == 0: ax.set_ylabel('Success gain over random (pp)')
    axes[0].set_ylim(-1, 20); axes[1].set_ylim(-.2, 4)  # same relative height of y = 0
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(n) for n in ['KV-Ctx', 'KV-Dec', 'VFD', 'GU', 'Hindsight oracle', 'Random referral']]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper center', ncol=6,
               bbox_to_anchor=(.5, 1.07), fontsize=7, handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT/'referral_curves.pdf'); plt.close(fig)


def benchmark_structure():
    variance = pd.read_csv(DERIVED/'variance_decomposition.csv')
    within = pd.read_csv(DERIVED/'within_task.csv')
    summary = pd.read_csv(DERIVED/'within_task_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), gridspec_kw=dict(width_ratios=[1, 1.25]))
    # (a) Where success varies: between tasks, reproducibly between conditions, or between executions.
    ax = axes[0]
    groups = [('Primary, round 0', (variance.role == 'primary') & (variance['round'] == '0')),
              ('Primary, round 10', (variance.role == 'primary') & (variance['round'] == '10')),
              ('Supervisor', variance.role == 'supervisor')]
    parts = [('Between tasks', COLOR['h1']), ('Between conditions\n(reproducible)', COLOR['h2']),
             ('Between executions', '#d6d5ce')]
    for i, (name, mask) in enumerate(groups):
        v = variance[mask]
        task = (v.between_task/v.total).mean(); cond = (v.between_condition_reproducible/v.total).mean()
        left = 0
        for value, (label, color) in zip([task, cond, 1-task-cond], parts):
            ax.barh(i, value, left=left, height=.55, color=color, edgecolor='white', linewidth=1.5,
                    label=label if i == 0 else None)
            if value > .15:
                ax.text(left+value/2, i, f'{100*value:.0f}%', ha='center', va='center', fontsize=7,
                        color='white' if label != 'Between executions' else INK)
            left += value
    ax.set_yticks(range(3)); ax.set_yticklabels([g[0] for g in groups]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(['0', '50%', '100%'])
    ax.set_xlabel('Share of success variance'); ax.grid(axis='y', visible=False)
    ax.set_title('(a) Where success varies', fontsize=8, color=INK, loc='left')
    ax.legend(loc='upper left', bbox_to_anchor=(-.55, -.3), ncol=3, fontsize=6.5, handlelength=1, columnspacing=.8)
    # (b) Within-task failure ranking at round 10 (seed values and seed-and-condition 95% interval).
    ax = axes[1]
    heads = ['vlm_perplexity', 'vlm_token_entropy', 'ace', 'decu', 'action_l2', 'gu', 'vfd', 'h1', 'h2']
    for i, h in enumerate(heads):
        c = COLOR.get(h, COLOR['other']); row = summary[(summary['head'] == h) & (summary['round'] == 10)].iloc[0]
        ax.plot([row.lo, row.hi], [i, i], color=c, linewidth=2.2, alpha=.45, solid_capstyle='round')
        seeds = within[(within['head'] == h) & (within['round'] == 10)].within_failure_auroc
        ax.scatter(seeds, [i]*len(seeds), s=9, color=c, alpha=.8, linewidths=0, zorder=3)
        ax.scatter([row.within_failure_auroc], [i], s=34, color=c, edgecolors='white', linewidths=1, zorder=4)
    ax.axvline(.5, color=COLOR['random'], linewidth=1)
    ax.set_yticks(range(len(heads))); ax.set_yticklabels([LABEL[h] for h in heads])
    ax.set_xlabel('Within-task failure AUROC, round 10'); ax.grid(axis='y', visible=False)
    ax.set_xlim(.35, .7)
    ax.set_title('(b) Ranking conditions within a task', fontsize=8, color=INK, loc='left')
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT/'benchmark_structure.pdf'); plt.close(fig)


def ivr_supplement():
    coverage = pd.read_csv(DERIVED/'ivr_coverage.csv')
    success = pd.read_csv(DATA/'acquisition_success.csv')
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.2))
    ax = axes[0]
    for arm in ['h1_ivr', 'h1']:
        g = coverage[coverage.arm == arm].groupby('round')
        ax.plot(g.pool_mean_variance.mean(), color=COLOR[arm], label=f'{LABEL[arm]}: mean')
        ax.plot(g.pool_max_variance.mean(), color=COLOR[arm], linewidth=1.0, alpha=.7, marker=MARKER[arm],
                markersize=3, label=f'{LABEL[arm]}: maximum')
    ax.set_xlabel('Acquisition round'); ax.set_ylabel('KV-Ctx variance, candidate pool'); ax.set_xticks([1, 5, 10, 15])
    ax.set_title('(a) Remaining support variance (tasks 3-9)', fontsize=8, loc='left', color=INK)
    ax.legend(fontsize=6.5)
    ax = axes[1]
    tasks = [f'task_{t}' for t in range(10)]
    width = .38
    for k, arm in enumerate(['h1', 'h1_ivr']):
        for j, rnd in enumerate([5, 15]):
            v = success[(success.arm == arm) & (success['round'] == rnd)][tasks].mean().values
            if rnd == 15:
                ax.bar(np.arange(10)+(k-.5)*width, 100*v, width=width, color=COLOR[arm], label=LABEL[arm],
                       edgecolor='white', linewidth=1)
    ax.set_xticks(range(10)); ax.set_xlabel('LIBERO-10 task'); ax.set_ylabel('Success at round 15 (%)')
    ax.set_title('(b) Per-task success after 75 queries', fontsize=8, loc='left', color=INK)
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(fontsize=6.5, loc='upper center', ncol=2); ax.grid(axis='x', visible=False)
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT/'ivr_supplement.pdf'); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    acquisition_curves(); referral_curves(); benchmark_structure(); ivr_supplement()
    print(f'Figures written to {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
