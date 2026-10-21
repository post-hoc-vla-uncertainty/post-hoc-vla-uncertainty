"""Matched treatment effects for every reported performance metric."""
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# Keep figure IDs stable for both report renderers and standalone exports.
PANELS = (
    ('ablations', 'Matched ablations: referral and automation', (
        ('global_nauc', 'Global referral nAUC', 1),
        ('within_nauc', 'Within-task referral nAUC', 1),
        ('automation', 'Automation at ≤2 pp loss', 100))),
    ('ablations_correlations', 'Matched ablations: task and success-benefit correlations', (
        ('neg_spearman', 'Task −Spearman', 1),
        ('neg_pearson', 'Task −Pearson', 1),
        ('benefit_spearman', 'Success-benefit Spearman', 1),
        ('benefit_pearson', 'Success-benefit Pearson', 1))),
    ('ablations_continuous', 'Matched ablations: continuous-benefit correlations', (
        ('benefit_time_spearman', 'Time-benefit Spearman', 1),
        ('benefit_time_pearson', 'Time-benefit Pearson', 1),
        ('benefit_path_spearman', 'Path-benefit Spearman', 1),
        ('benefit_path_pearson', 'Path-benefit Pearson', 1))),
)


def generate(pairs, save, *, observation):
    if pairs.empty:
        return
    families = sorted(pairs.family.unique())
    rng = np.random.default_rng(4)
    # Use identical vertical offsets so each matched pair is traceable across metrics.
    groups = [(name, pairs[pairs.family == name]) for name in families]
    offsets = [i + rng.uniform(-.17, .17, len(group)) for i, (_, group) in enumerate(groups)]
    legend = [Line2D([], [], marker='o', linestyle='', color='#0d9488', label='Matching numerical settings'),
              Line2D([], [], marker='o', linestyle='', markerfacecolor='none', color='#c2410c', label='Different recorded settings'),
              Line2D([], [], marker='|', linestyle='', color='black', markersize=13, label='Median across defined pairs')]
    for name, title, metrics in PANELS:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 + 3.8 * len(metrics), max(5, len(families) * .55)),
                                 layout='constrained', sharey=True)
        for ax, (metric, label, scale) in zip(axes, metrics):
            for i, (_, group) in enumerate(groups):
                column = 'delta_' + metric
                values = group[column].to_numpy(dtype=float) * scale if column in group else np.full(len(group), np.nan)
                finite = np.isfinite(values)
                comparable = group.comparable_numerics.to_numpy(dtype=bool)
                good, mixed = finite & comparable, finite & ~comparable
                ax.scatter(values[good], offsets[i][good], c='#0d9488', s=17, alpha=.65)
                ax.scatter(values[mixed], offsets[i][mixed], facecolors='none', edgecolors='#c2410c', s=20, alpha=.75)
                if finite.any():
                    ax.plot([np.median(values[finite])], [i], 'k|', markersize=17, markeredgewidth=2)
                else:
                    ax.text(.98, i, 'unavailable', transform=ax.get_yaxis_transform(), ha='right', va='center', color='.5', fontsize=7)
            ax.set_title(label, fontsize=10)
            ax.axvline(0, color='.4', ls='--', lw=.8)
            ax.grid(axis='x', alpha=.2)
            ax.set_xlabel('Treatment − control' + (' (percentage points)' if scale == 100 else ''))
            ax.set_ylim(len(families) - .5, -.5)
        axes[0].set_yticks(range(len(families)), [textwrap.fill(family, 38) for family in families], fontsize=8)
        fig.legend(handles=legend, loc='outside lower center', ncol=3, fontsize=8)
        save(fig, name, title, observation +
             ' Positive changes favor the treatment; task correlations use the sign convention shown in their titles.'
             ' Black ticks are medians of the defined pairs for each metric; undefined values are omitted.'
             ' Orange dots include float32 KV versus float64 KV-T. The horizontal scales differ between panels.'
             ' Pairs share models and data; their spread is not a confidence interval. Pair-specific bootstrap intervals are exported in paired_comparisons.csv.')
