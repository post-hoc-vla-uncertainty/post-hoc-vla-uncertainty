"""Plot metric-specific paired effects, preserving signs and missing values."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from head_report.ablation_plots import generate, PANELS
from head_report.analysis import ABLATION_METRICS


def test_all_ablation_metrics_use_their_own_paired_values():
    assert {metric for _, _, panels in PANELS for metric, _, _ in panels} == set(ABLATION_METRICS)
    first = dict(family='One treatment', comparable_numerics=True)
    second = dict(family='One treatment', comparable_numerics=False)
    for index, metric in enumerate(ABLATION_METRICS):
        first['delta_' + metric] = .01 * (index + 1)
        second['delta_' + metric] = -.02 * (index + 1)
    # A correlation can be undefined without invalidating that pair's other metrics.
    second['delta_benefit_path_pearson'] = np.nan
    pairs = pd.DataFrame([first, second])
    saved = []
    def save(fig, name, title, caption):
        panels = next(p for n, _, p in PANELS if n == name)
        for ax, (metric, _, scale) in zip(fig.axes, panels):
            expected = pairs['delta_' + metric].dropna().to_numpy() * scale
            plotted = np.concatenate([scatter.get_offsets()[:, 0].compressed() for scatter in ax.collections])
            assert np.allclose(np.sort(plotted), np.sort(expected))
            assert np.allclose(ax.lines[0].get_xdata(), [np.median(expected)])
            assert ('percentage points' in ax.get_xlabel()) == (metric == 'automation')
        saved.append(name)
        plt.close(fig)
    generate(pairs, save, observation='A paired comparison.')
    assert len(saved) == 3
