"""Tables and figure of the context-head experiments (starting observations, events during execution, referral; handoff/q5).

Reads the tie-corrected summaries and paired differences written by scripts/q5_analysis.py (data/derived/q5_*.json),
and policies, the seed-0 executions of perturbed starting states and the referral conditions from data/q5. Requires
numpy, pandas, scipy and matplotlib. Writes tables/generated/context_*.tex and perturbation_library.tex, figures/generated/context_success.pdf and the plotted
values to data/derived/q5_detection_drop.csv and data/derived/q5_supervisor_gain.csv.

    python3 scripts/q5_results.py
"""
from pathlib import Path
import json
import sys

from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import COLOR, INK, OUT, WIDTH  # noqa: E402  (shared figure style)

ROOT = Path(__file__).resolve().parents[1]
DATA, TAB = ROOT/'data/q5', ROOT/'tables/generated'
DERIVED = ROOT/'data/derived'
E1 = json.loads((DERIVED/'q5_e1_summary.json').read_text())
E2 = json.loads((DERIVED/'q5_e2_summary.json').read_text())
E2_EXPORT = json.loads((DATA/'e2_summary.json').read_text())
E3 = json.loads((DERIVED/'q5_e3_summary.json').read_text())
PAIRED = json.loads((DERIVED/'q5_paired.json').read_text())
POLICIES = json.loads((DATA/'policies.json').read_text())
LIBRARY = json.loads((ROOT/'handoff/q5/perturbations.json').read_text())

# Diverging around AUROC 0.5 (no signal, neutral grey): blue above, red below, light enough for black text throughout.
CMAP = LinearSegmentedColormap.from_list('auroc', [(0, '#e8907e'), (.25, '#f4c9be'), (.5, '#f0efec'), (.75, '#c3daf7'),
                                                   (1, '#86b6f0')])
HEADS = ['pre_ws', 'pre_wrist', 'pre_lang', 'pre_state', 'ctx_ws', 'ctx_wrist', 'ctx_lang', 'ctx_state', 'kv_ctx', 'kv_dec', 'vfd']
DEG = r'$^\circ$'
# Rows of the two detection tables: (group, [(key, description)]), keys as in the research code.
START_ROWS = [
    ('Instruction', [('I1:P5', 'Strongly paraphrased'),
                     ('I2:all', 'All object words replaced by synonyms'),
                     ('I3:typo', 'One typo in a non-object word'),
                     ('I4:pronouns', 'Object nouns replaced by pronouns'),
                     ('I5:german', 'Translated into German')]),
    ('Camera images', [('V1:x0.4', r'Light at 40\,\% intensity'),
                       ('V2:sigma0.08', r'Gaussian noise on both images ($\sigma=0.08$)'),
                       ('V3:4cm6deg', rf'Workspace camera moved by 4\,cm and 6{DEG}'),
                       ('V4:75', r'Target 75\,\% occluded in the workspace image'),
                       ('V5:table', 'Table texture replaced'),
                       ('V6:blackout', 'Wrist image blacked out')]),
    ('Scene', [('S1:10cm', r'Target displaced by 10\,cm'),
               ('S2:near', 'Another object moved next to the target'),
               ('S3:90deg', f'Target rotated by 90{DEG}')]),
    ('Robot', [('R1:0.1rad', r'Arm joints offset by 0.1\,rad'),
               ('R2:3cm', r'Reported end-effector position offset by 3\,cm')]),
    # The other task's instruction in the same scene (M1:same_scene) is left out. It forms a valid training input by
    # construction, so no familiarity score can respond, and its success is checked against the original task.
    ('Instruction and scene', [('M1:other_scene', ("Other task's instruction, its objects absent", 'from the scene'))]),
]
EXECUTION_ROWS = [
    ('Instruction', [('I1', 'Strongly paraphrased'),
                     ('M1', "Switched to another task's instruction"),
                     ('I5', "Negated (``do not \\ldots'')")]),
    ('Camera images', [('V1', r'Light at 40\,\% intensity'),
                       ('V3', rf'Workspace camera bumped by 2\,cm and 3{DEG}'),
                       ('V4', r'Target 75\,\% occluded for 60 steps'),
                       ('V6', 'Wrist image frozen for 60 steps'),
                       ('V7', r'Heavy noise on both images ($\sigma=0.35$)')]),
    ('Scene', [('S1', r'Target displaced by 6\,cm'),
               ('S2', 'Another object moved next to the target'),
               ('S4', 'Target removed')]),
    # The heavier objects (D1) are left out. In most of their runs the object was nearly handled when the change set
    # in at step 100, so the run does not test the change.
    ('Robot', [('R3', r'Arm pushed (joints offset by 0.1\,rad)')]),
]
E1_ROWS = [key for _, rows in START_ROWS for key, _ in rows]
E3_ROWS = [key for _, rows in EXECUTION_ROWS for key, _ in rows]


def cell(value, ci):
    """AUROC on its colour (the interval is not shown)."""
    return rf'\cellcolor[HTML]{{{to_hex(CMAP(value))[1:].upper()}}}{value:.2f}'


def detection_table(name, groups, result, prefix):
    """Full-width table of detection AUROCs, rows grouped by the perturbed input, with the mean AUROC of the two paper
    heads and VFD; result(key, head) -> (auroc, ci). Rows are numbered with a prefix (S for starting observations, E for
    execution) so that the text can refer to them across both tables."""
    gap = r'!{\hspace{2pt}}'
    heads = ['Workspace', 'Wrist', 'Instruction', 'State'] * 2 + [r'\KVctx{}', r'\KVdec{}', 'VFD']
    lines = [rf'\begin{{tabularx}}{{\linewidth}}{{@{{}}r@{{\hspace{{4pt}}}}>{{\raggedright\arraybackslash}}X*{{4}}{{c}}{gap}*{{4}}{{c}}{gap}cc{gap}c@{{}}}}',
             r'\toprule',
             r'& & \multicolumn{4}{c}{Raw heads} & \multicolumn{4}{c}{Fused heads} & \multicolumn{3}{c}{Whole-policy}\\',
             r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(l){11-13}',
             '& Perturbation & ' + ' & '.join(rf'\rotatebox{{90}}{{{h}}}' for h in heads) + r'\\',
             r'\midrule']
    keys = []
    for i, (group, rows) in enumerate(groups):
        if i:
            lines.append(r'\addlinespace[2pt]')
        lines.append(rf'\multicolumn{{13}}{{@{{}}l}}{{\textit{{{group}}}}}\\')
        for key, label in rows:
            keys.append(key)
            # A label given as two lines continues in an uncoloured row, so that the coloured cells keep one line.
            first, rest = (label, None) if isinstance(label, str) else label
            lines.append(f'{prefix}{len(keys)} & ' + first + ' & ' + ' & '.join(cell(*result(key, h)) for h in HEADS) + r'\\')
            if rest:
                lines.append(' & ' + rest + ' &' * len(HEADS) + r'\\')
    means = [float(np.mean([result(k, h)[0] for k in keys])) for h in HEADS[-3:]]
    lines += [r'\midrule', r'& Mean over perturbations' + ' &' * 8 + ' & ' + ' & '.join(cell(m, (0, 1)) for m in means) + r'\\',
              r'\bottomrule', r'\end{tabularx}']
    (TAB/f'{name}.tex').write_text('\n'.join(lines) + '\n')


def detection_tables():
    start = E1['variants']['dense']
    detection_table('context_start', START_ROWS, lambda k, h: (start[k][h]['auroc'], start[k][h]['ci']), 'S')
    execution = E3['events']
    detection_table('context_execution', EXECUTION_ROWS,
                    lambda k, h: (execution[k]['heads'][h]['auroc'], execution[k]['heads'][h]['auroc_ci']), 'E')


def fmt(m, digits=2):
    return f"${m['mean']:.{digits}f}\\pm{m['sd']:.{digits}f}$"


def referral_table():
    s = E2['supports']['dense']
    rows = [('kv_ctx', r'\KVctx{}'), ('kv_dec', r'\KVdec{}'), ('vfd', 'VFD (two policies)'),
            ('cross_replicate', 'Cross-replicate reference')]
    lines = [r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
             r'Score & Within-task nAUC & Global nAUC & Automated & Failure AUROC\\', r'\midrule']
    for key, label in rows:
        m = s[key]
        w = m['within_nauc']
        lines.append(f"{label} & {fmt(w)} & {fmt(m['global_nauc'])} & "
                     f"{fmt(m['automation'])} & {fmt(m['failure_auroc'])}\\\\")
        if key == 'vfd':
            lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB/'context_referral.tex').write_text('\n'.join(lines) + '\n')


def policies_table():
    lines = [r'\begin{tabular}{@{}lccc@{}}', r'\toprule', r'Seed & Primary $P_s$ & Companion $C_s$ & Supervisor $S_s$\\', r'\midrule']
    for s in (0, 1, 2):
        lines.append(f"{s} & " + ' & '.join(f"{100 * POLICIES[f'{r}_s{s}']['clean_success_mean']:.1f}"
                                            for r in ('primary', 'companion', 'supervisor')) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB/'context_policies.tex').write_text('\n'.join(lines) + '\n')


# Levels of each library family that the results show, with the experiments that show them: E1 starting observations
# (Table 4 and Figure 5a), E3 during execution (Table 5) and E2 referral (Table 6 and Figure 5b). Level texts as in
# handoff/q5/perturbations.json. During execution, the instruction switch (M1) is the same-scene swap for tasks 0 and 1
# and an instruction naming absent objects for the other tasks. Families and levels the results do not show are left out.
USED_LEVELS = {
    'I1': {'P3': 'E1 E2', 'P5': 'E1 E2 E3'},
    'I2': {'all object words replaced': 'E1 E2'},
    'I3': {"'please ' prefix": 'E2', 'one typo in a non-object word': 'E1'},
    'I4': {"object nouns replaced by pronouns ('put it in the basket')": 'E1'},
    'I5': {"negated ('do not ...')": 'E3', 'German translation': 'E1'},
    'V1': {'intensity x0.8': 'E2', 'x0.4': 'E1 E2 E3'},
    'V2': {'Gaussian noise sigma 0.01': 'E2', 'sigma 0.08': 'E1 E2'},
    'V3': {'2 cm + 3 deg': 'E1 E2 E3', '4 cm + 6 deg': 'E1'},
    'V4': {'50%': 'E1 E2', '75%': 'E1 E3'},
    'V5': {'table texture swap': 'E1'},
    'V6': {'blackout': 'E1 E2', 'frozen frame (E3: repeat the frame at onset)': 'E3'},
    'V7': {'Gaussian noise sigma 0.35': 'E3', 'uniform grey image': 'E1'},
    'S1': {'6 cm': 'E1 E3', '10 cm': 'E1'},
    'S2': {'next to the target (< 5 cm)': 'E1 E3'},
    'S3': {'90 deg': 'E1'},
    'S4': {'vanish target': 'E3'},
    'R1': {'0.1 rad': 'E1'},
    'R2': {'3 cm': 'E1 E2'},
    'R3': {'0.1 rad': 'E3'},
    'M1': {'same scene: tasks 0 <-> 1 (LIVING_ROOM_SCENE2)': 'E3',
           'other scene: an instruction naming objects that are absent': 'E1 E3'},
}


def library_table():
    times, end, tick = r' $\times$ ', r'\\', r'$\checkmark$'
    lines = [r'\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{2.1cm}>{\raggedright\arraybackslash}Xcccc@{}}',
             r'\toprule',
             r'Input & Perturbation and levels & \shortstack{Starting\\observ.} & \shortstack{During\\execution} & Referral'
             r' & \shortstack{Label-\\preserving}' + end, r'\midrule']
    shown = {'E1': {k.split(':')[0] for k in E1_ROWS} | {k.split(':')[0] for p in DROP_POINTS.values() for k in p},
             'E3': set(E3_ROWS), 'E2': {k.split(':')[0] for k in GAIN_VARIANTS if k != 'clean'}}
    for e, families in shown.items():  # USED_LEVELS must cover exactly the families that the results show
        assert families == {f for f, kept in USED_LEVELS.items() if any(e in x.split() for x in kept.values())}, e
    for f in LIBRARY['families']:
        kept = USED_LEVELS.get(f['id'], {})
        if not kept:
            continue
        assert set(kept) <= set(f['levels']), f['id']
        levels = '; '.join(str(level).replace('E3: ', 'during execution: ') for level in f['levels'] if level in kept)
        levels = levels.replace(' <-> ', ' and ').replace(' (LIVING_ROOM_SCENE2)', '')
        levels = levels.replace('%', r'\%').replace('_', r'\_').replace('<', '$<$').replace('>', '$>$')
        # M1 also holds the other-scene level, so its research-code name ("... in the same scene") is too narrow.
        name = {'M1': "other task's instruction"}.get(f['id'], f['name']).replace('%', r'\%')
        modality = f['modality'].replace(' x ', times)
        preserving = {True: 'yes', 'approx.': 'approx.'}.get(f['label_preserving'], 'no')
        experiments = set(' '.join(kept.values()).split())
        used = ' & '.join(tick if e in experiments else '' for e in ('E1', 'E3', 'E2'))
        lines.append(f"{modality} & {name}: {levels} & {used} & {preserving}" + end)
    lines += [r'\bottomrule', r'\end{tabularx}']
    (TAB/'perturbation_library.tex').write_text('\n'.join(lines) + '\n')


def paired_table():
    """Paired differences of the paper heads against VFD (same draws): E2 metrics, and E1/E3 AUROC counts."""
    def signed(x):
        return f'{x:+.2f}'.replace('-0.00', '0.00').replace('+0.00', '0.00')

    def ci(d):
        return f"${signed(d['mean'])}$ {{\\scriptsize$[{signed(d['ci'][0])}, {signed(d['ci'][1])}]$}}"

    lines = [r'\begin{tabular}{@{}llcccc@{}}', r'\toprule',
             r'Experiment & Difference & \multicolumn{4}{c}{Mean over seeds, 95\,\% hierarchical interval}\\',
             r'\cmidrule(l){3-6}', r'Referral (Table~\ref{tab:context-referral}) & & Within-task nAUC & Global nAUC & Automated & Failure AUROC\\', r'\midrule']
    for h, label in (('kv_ctx', r'\KVctx{} $-$ VFD'), ('kv_dec', r'\KVdec{} $-$ VFD')):
        d = PAIRED['e2']['dense'][f'{h}-vfd']
        lines.append(f"& {label} & " + ' & '.join(ci(d[m]) for m in ('within_nauc', 'global_nauc', 'automation', 'failure_auroc')) + r'\\')
    lines += [r'\midrule', r'Detection AUROC & & Higher & Not different & Lower & Median difference\\', r'\midrule']
    for name, source, keys in (('Starting observations (Table~\\ref{tab:context-start})', PAIRED['e1']['dense'], E1_ROWS),
                               ('During execution (Table~\\ref{tab:context-execution})', PAIRED['e3'], E3_ROWS)):
        for h, label in (('kv_ctx', r'\KVctx{} $-$ VFD'), ('kv_dec', r'\KVdec{} $-$ VFD')):
            ds = [source[k][f'{h}-vfd'] for k in keys if f'{h}-vfd' in source.get(k, {})]
            higher = sum(d['ci'][0] > 0 for d in ds); lower = sum(d['ci'][1] < 0 for d in ds)
            median = float(np.median([d['mean'] for d in ds]))
            lines.append(f"{name} & {label} & {higher} & {len(ds)-higher-lower} & {lower} & ${signed(median)}$\\\\")
            name = ''
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB/'context_paired.tex').write_text('\n'.join(lines) + '\n')


# Perturbed inputs as in the context_start table, each with the marker that stands for it in both panels of the figure.
INPUT_MARKERS = {'Instruction': ('o', 30), 'Camera images': ('s', 26), 'Scene': ('^', 34), 'Robot': ('D', 22),
                 'Instruction and scene': ('X', 46)}
# (a) Perturbations executed with the seed-0 primary (one level per family), by perturbed input:
# key -> (label, label offset in points, horizontal alignment); overlapping points share a label.
DROP_POINTS = {
    'Instruction': {'I1:P3': ('Paraphrased', (5, 0), 'left'), 'I2:all': ('Object synonyms', (-6, 0), 'right'),
                    'I3:typo': ('Typo, joint offset', (-6, 0), 'right'),
                    'I4:pronouns': ('German, pronouns', (-8, -9), 'right'), 'I5:german': ('', (0, 0), 'right')},
    'Camera images': {'V1:x0.4': ('Light at 40%', (-5, 2), 'right'), 'V2:sigma0.08': ('Image noise', (5, -1), 'left'),
                      'V3:2cm3deg': ('Camera moved', (5, -1), 'left'), 'V4:50': ('Target occluded', (0, 7), 'center'),
                      'V5:table': ('Table texture', (5, 0), 'left'),
                      'V6:blackout': ('Wrist image blacked out, grey images', (-7, 5), 'right'),
                      'V7:grey': ('', (0, 0), 'right')},
    'Scene': {'S1:6cm': ('Target displaced', (-5, 0), 'right'), 'S2:near': ('Distractor', (5, 0), 'left'),
              'S3:90deg': ('Target rotated', (-5, 0), 'right')},
    'Robot': {'R1:0.1rad': ('', (0, 0), 'left'), 'R2:3cm': ('State offset', (-5, 0), 'right')},
}
# (b) Variants of the referral experiment: key -> (label, perturbed input, None for the clean conditions).
GAIN_VARIANTS = {
    'I1:P3': ('Moderately paraphrased', 'Instruction'), 'I1:P5': ('Strongly paraphrased', 'Instruction'),
    'I2:all': ('Object synonyms', 'Instruction'), 'I3:please': ("'Please' prefix", 'Instruction'),
    'V1:x0.8': ('Light at 80% intensity', 'Camera images'), 'V1:x0.4': ('Light at 40% intensity', 'Camera images'),
    'V2:sigma0.01': (r'Image noise ($\sigma=0.01$)', 'Camera images'),
    'V2:sigma0.08': (r'Image noise ($\sigma=0.08$)', 'Camera images'),
    'V3:2cm3deg': ('Camera moved by 2 cm, 3°', 'Camera images'), 'V4:50': ('Target 50% occluded', 'Camera images'),
    'V6:blackout': ('Wrist image blacked out', 'Camera images'), 'R2:3cm': ('State offset by 3 cm', 'Robot'),
    'clean': ('Clean', None)}
# Variants whose gain interval lies above zero take these colours from the top, as the highlighted rows of Figure 4b;
# the others are grey.
GAIN_COLORS = [COLOR['h2'], COLOR['h1'], COLOR['vfd'], COLOR['gu'], COLOR['h1_ivr'], '#8e6bbf']
BOOT = 2000  # bootstrap draws, as in scripts/q5_analysis.py


def detection_drop(seed=0):
    """Per perturbation: detection AUROCs and the drop in success it causes, both for the seed-0 primary. The drop is
    clean minus perturbed success on the same starting states (three executions each)."""
    runs = pd.read_csv(DATA/'e1_harm.csv')
    runs['success'] = runs[['primary_rep0', 'primary_rep1', 'primary_rep2']].astype(float).mean(axis=1)
    runs = runs[(runs.seed == seed) & runs.success.notna()].set_index(['task', 'init_state_id'])
    clean = runs[runs.variant == 'clean'].success
    detection = E1['variants']['dense']
    rows = []
    for group, points in DROP_POINTS.items():
        for key in points:
            perturbed = runs[runs.variant == key].success
            rows.append({'variant': key, 'group': group, 'conditions': len(perturbed),
                         'drop': float((clean.loc[perturbed.index]-perturbed).mean()),
                         **{h: detection[key][h]['per_seed'][seed] for h in ('kv_ctx', 'kv_dec', 'vfd')}})
    table = pd.DataFrame(rows)
    table.to_csv(DERIVED/'q5_detection_drop.csv', index=False, float_format='%.4f')
    for h in ('kv_dec', 'kv_ctx', 'vfd'):
        print(f'{h}: Spearman correlation of detection and drop over {len(table)} perturbations '
              f"{spearmanr(table[h], table['drop'])[0]:.2f}")
    return table


def supervisor_gain():
    """Per referral variant: supervisor minus primary success (three executions each), averaged over tasks and initial
    states for each seed, with the 95 % hierarchical bootstrap interval of scripts/q5_analysis.py (initial states
    resampled within tasks for each seed, then seeds resampled)."""
    c = pd.read_csv(DATA/'e2_conditions.csv')
    c['gain'] = (c[[f'supervisor_rep{i}' for i in range(3)]].mean(axis=1)
                 - c[[f'primary_rep{i}' for i in range(3)]].mean(axis=1))
    seeds, variants = sorted(c.seed.unique()), sorted(c.variant.unique())
    tasks, states = sorted(c.task.unique()), sorted(c.init_state_id.unique())
    g = c.set_index(['seed', 'variant', 'task', 'init_state_id']).gain
    cube = g.reindex(pd.MultiIndex.from_product([seeds, variants, tasks, states])).values.reshape(
        len(seeds), len(variants), len(tasks), len(states))
    assert not np.isnan(cube).any()
    rng = np.random.default_rng(0)
    draws = []
    for k in range(len(seeds)):
        idx = rng.integers(len(states), size=(BOOT, len(tasks), len(states)))
        draws.append(cube[k][:, np.arange(len(tasks))[None, :, None], idx].mean(axis=(2, 3)))  # variant x draw
    draws = np.stack(draws)
    pick = rng.integers(len(seeds), size=(BOOT, len(seeds)))
    total = np.mean([draws[pick[:, j], :, np.arange(BOOT)] for j in range(len(seeds))], axis=0)  # draw x variant
    per_seed = cube.mean(axis=(2, 3))
    table = pd.DataFrame({'mean': per_seed.mean(axis=0), 'sd': per_seed.std(axis=0, ddof=1),
                          'lo': np.quantile(total, .025, axis=0), 'hi': np.quantile(total, .975, axis=0),
                          **{f'seed{s}': per_seed[k] for k, s in enumerate(seeds)}}, index=variants)
    table['harmless_expected'] = c.groupby('variant').harmless_expected.first()
    for v, row in table.iterrows():  # same as the export of the research code
        exported = E2_EXPORT['families'][v]['gain']
        assert abs(row['mean']-exported) < 1e-9, (v, row['mean'], exported)
    table.to_csv(DERIVED/'q5_supervisor_gain.csv', float_format='%.4f')
    primary = c[[f'primary_rep{i}' for i in range(3)]].mean(axis=1)
    supervisor = c[[f'supervisor_rep{i}' for i in range(3)]].mean(axis=1)
    for name, mask in (('perturbed', c.variant != 'clean'), ('clean', c.variant == 'clean')):
        print(f'Referral experiment, {name} conditions: primary {100*primary[mask].mean():.1f}%, '
              f'supervisor {100*supervisor[mask].mean():.1f}%')
    return table


def success_figure(head='kv_dec'):
    """(a) Detection against the loss of success; (b) the gain from referral per variant, drawn like Figure 4b."""
    drop, gain = detection_drop(), supervisor_gain()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    blue, grey = COLOR['h1'], COLOR['other']
    ax = axes[0]
    ax.axvline(.5, color=COLOR['random'], linewidth=1)
    ax.axhline(0, color=COLOR['random'], linewidth=1)
    for group, points in DROP_POINTS.items():
        marker, size = INPUT_MARKERS[group]
        g = drop[drop.group == group]
        ax.scatter(g[head], 100*g['drop'], marker=marker, s=size, color=blue, edgecolors='white', linewidths=.6, zorder=3)
        for r in g.itertuples():
            text, offset, ha = points[r.variant]
            if text:
                ax.annotate(text, (getattr(r, head), 100*r.drop), xytext=offset, textcoords='offset points', ha=ha,
                            va='center', fontsize=6, color=INK)
    ax.set_xlim(.45, 1.06); ax.set_ylim(-12, 90); ax.set_xticks([.5, .6, .7, .8, .9, 1])
    ax.text(.03, .8, f"Spearman $\\rho$ = {spearmanr(drop[head], drop['drop'])[0]:.2f}", transform=ax.transAxes,
            ha='left', va='top', fontsize=6.5, color=INK,
            bbox=dict(boxstyle='round,pad=.3', facecolor='white', edgecolor='#c3c2b7', linewidth=.6))
    ax.set_xlabel('KV-Dec detection AUROC, perturbed against clean')
    ax.set_ylabel('Drop in success (pp)')
    ax.set_title('(a) Detection and loss of success', fontsize=8, color=INK, loc='left')
    ax = axes[1]
    rows = []  # sorted by the mean gain, largest at the top
    colors = iter(GAIN_COLORS)
    for y, (key, r) in enumerate(gain.loc[list(GAIN_VARIANTS)].sort_values('mean', ascending=False).iterrows()):
        c = next(colors) if r.lo > 0 else grey
        ax.plot([100*r.lo, 100*r.hi], [y, y], color=c, linewidth=2.2, alpha=.45, solid_capstyle='round')
        ax.scatter(100*r[['seed0', 'seed1', 'seed2']].astype(float), [y]*3, s=9, color=c, alpha=.8, linewidths=0, zorder=3)
        ax.scatter([100*r['mean']], [y], s=34, color=c, edgecolors='white', linewidths=1, zorder=4)
        rows.append((y, *GAIN_VARIANTS[key]))
    ax.axvline(0, color=COLOR['random'], linewidth=1)
    ax.set_yticks([r[0] for r in rows])
    ax.set_yticklabels(['    '+r[1] for r in rows], fontsize=6.5)  # leading space for the input's symbol
    ax.invert_yaxis(); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
    ax.set_xlim(-25, 65); ax.set_xticks([-20, 0, 20, 40, 60])
    ax.set_xlabel('Supervisor − primary (pp)')  # tight_layout ignores label width, so keep it short
    ax.set_title('(b) Gain from referral', fontsize=8, color=INK, loc='left')
    fig.tight_layout(w_pad=1.5)
    # Inputs: the same blue symbols in (a), in front of the labels of (b) and in a compact legend below both panels.
    symbol = {g: dict(marker=m, markersize=.7*np.sqrt(s), color=blue, markeredgecolor='white', markeredgewidth=.4,
                      linestyle='') for g, (m, s) in INPUT_MARKERS.items()}
    shown = [g for g in INPUT_MARKERS if g in DROP_POINTS or any(v[1] == g for v in GAIN_VARIANTS.values())]
    fig.legend([Line2D([], [], **symbol[g]) for g in shown], shown, loc='upper center',
               bbox_to_anchor=(.5, 0), ncol=len(shown), fontsize=6.5, handletextpad=.1, columnspacing=1, borderaxespad=0)
    fig.canvas.draw()
    renderer, to_axes = fig.canvas.get_renderer(), ax.transAxes.inverted()
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for (y, _, group), tick in zip(rows, ax.get_yticklabels()):
        if group is not None:
            x = to_axes.transform((tick.get_window_extent(renderer).x0 + 3.5*fig.dpi/72, 0))[0]
            ax.plot([x], [y], transform=transform, clip_on=False, **symbol[group])
    fig.savefig(OUT/'context_success.pdf'); plt.close(fig)


if __name__ == '__main__':
    detection_tables()
    referral_table()
    policies_table()
    library_table()
    paired_table()
    print('wrote tables/generated/context_{start,execution,referral,policies,paired}.tex and perturbation_library.tex')
    success_figure()
    print('wrote figures/generated/context_success.pdf')
