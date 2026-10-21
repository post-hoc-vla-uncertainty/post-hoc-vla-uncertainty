"""Balanced summaries and explicitly matched comparisons, without winner freezing."""
from itertools import combinations
import numpy as np
import pandas as pd

from uq_extensions.common import identity

METRICS = ['global_nauc', 'within_nauc', 'neg_spearman', 'neg_pearson', 'benefit_spearman', 'automation']
ABLATION_METRICS = [*METRICS, 'benefit_pearson', 'benefit_time_spearman', 'benefit_time_pearson',
                    'benefit_path_spearman', 'benefit_path_pearson']
DECODER_ABLATION = 'Earlier action-decoder embedding'
ALL_PAIRS = 'All available pairs'
MATCHED_PAIRS = 'Matching numerical settings and seed IDs (sensitivity check)'


def frame(snapshot):
    data = pd.DataFrame([{k:v for k,v in r.items() if k != 'milestones'} for r in snapshot['rows']])
    for key in METRICS:
        if key not in data:data[key] = np.nan
    return data


def balanced_summary(data, rounds):
    rows = []
    for head, group in data[data.eligible].groupby('head'):
        if set(group['round']) != set(rounds) or len(group) != len(rounds):continue
        record = dict(head=head, label=group.iloc[0].label, method=group.iloc[0].method,
            group=group.iloc[0]['group'], scope=group.iloc[0].scope, policies=len(group),
            precisions='/'.join(sorted(set(group.precision))), jitter_escalated=bool(group.jitter_escalated.any()))
        for metric in METRICS:
            values = group[metric]
            record['mean_' + metric] = float(values.mean()) if values.notna().all() else np.nan
            for _, cell in group.iterrows():record[f'{metric}_r{cell["round"]}'] = cell[metric]
        rows.append(record)
    return pd.DataFrame(rows).sort_values('mean_global_nauc', ascending=False) if rows else pd.DataFrame()


def select_heads(data, rounds, top=3):
    """Select by declared display rules each rerun, never a hard-coded winning ID."""
    chosen = {};current = data[data.eligible]
    def add(head, reason):chosen.setdefault(head, []).append(reason)
    for head in current[current.scope == 'paper']['head'].unique():add(head, 'Paper baseline')
    balanced = balanced_summary(data, rounds)
    if len(balanced):
        for method in ('kv', 'kv_t'):
            for head in balanced[balanced.method == method].head(top)['head']:
                add(head, f'Top {top} {method} by mean referral across all checkpoints')
    for round_ in rounds:
        panel = current[(current['round'] == round_) & (current.scope != 'paper')]
        for metric in METRICS:
            candidates = panel.dropna(subset=[metric]).sort_values([metric, 'head'], ascending=[False, True])
            if len(candidates):add(candidates.iloc[0]['head'], f'Best observed {metric}, round {round_}')
        for scope in ('observation', 'action'):
            for method in ('kv', 'kv_t'):
                part = panel[(panel.scope == scope) & (panel.method == method)].sort_values('global_nauc', ascending=False)
                if len(part):add(part.iloc[0]['head'], f'Best {scope} {method} referral, round {round_}')
        new = panel[(panel.method == 'kv_t') & (panel.precision == 'float64')].sort_values('global_nauc', ascending=False)
        if len(new):add(new.iloc[0]['head'], f'Best float64 KV-T referral, round {round_}')
    return chosen


def comparison_heads(data, rounds):
    """Small common set keeps the main curves readable."""
    current = data[data.eligible];heads = []
    for key in ('paper__vfd', 'paper__gu'):
        if key in set(current['head']):heads.append(key)
    balanced = balanced_summary(data, rounds)
    if len(balanced):
        for method in ('kv', 'kv_t'):
            part = balanced[balanced.method == method]
            heads.extend(part.head(2)['head'])
    newest = current[(current.method == 'kv_t') & (current.precision == 'float64') & (current['round'] == max(rounds))]
    if len(newest):heads.append(newest.sort_values('global_nauc', ascending=False).iloc[0]['head'])
    if len(heads) < 3:
        heads.extend(current.sort_values('global_nauc', ascending=False)['head'].head(5))
    return list(dict.fromkeys(heads))[:7]


def descriptor(recipe):
    f, g = recipe['features'], recipe['gp'];method = recipe['method']
    projection = g['input_projection'] if method == 'kv_t' else f['compression']
    features = {k:v for k,v in f.items() if k not in ('compression', 'components', 'seed', 'patch_aggregation')}
    gp = {k:v for k,v in g.items() if k not in ('precision', 'input_projection', 'input_components', 'output_compression',
        'output_components', 'output_seed', 'pls_variance_target', 'lengthscale_initialization')}
    return dict(method=method, group=recipe['group'], scope=recipe['scope'], extraction=identity(features),
        extraction_without_distance=identity({k:v for k,v in features.items() if k != 'expert_distance'}),
        kernel_setup=identity(gp), input=projection,
        dimensions=g['input_components'] if method == 'kv_t' else f['components'], seed=f['seed'],
        output=g.get('output_compression'), outputs=g.get('output_components'), output_seed=g.get('output_seed'),
        adaptive=g.get('pls_variance_target') if method == 'kv_t' else None,
        initialization=g.get('lengthscale_initialization', 'geometry'), aggregation=f.get('patch_aggregation'),
        distance=f.get('expert_distance', 0))


def equal_except(a, b, ignored):
    return {k:v for k,v in a.items() if k not in ignored} == {k:v for k,v in b.items() if k not in ignored}


def matched_pairs(snapshot, data):
    output = []
    for round_, panel in data[data.eligible & data.method.isin(['kv', 'kv_t'])].groupby('round'):
        cells = panel.to_dict('records')
        descriptions = {r['head']:descriptor(snapshot['recipes'][(round_, r['head'])]) for r in cells}
        for left, right in combinations(cells, 2):
            a, b = descriptions[left['head']], descriptions[right['head']]
            differences = []
            def match(family, ignore, predicate, reverse=False):
                if equal_except(a, b, ignore) and predicate:
                    differences.append((family, right, left) if reverse else (family, left, right))
            # The primary estimator comparison holds input coordinates fixed and
            # uses PCA16 targets; the kernel/likelihood fitting is the treatment.
            if {a['method'], b['method']} == {'kv', 'kv_t'}:
                target = b if b['method'] == 'kv_t' else a
                match('KV → KV-T (PCA16 targets)', {'method', 'kernel_setup'},
                      target['output'] == 'pca' and target['outputs'] == 16 and target['adaptive'] is None,
                      reverse=a['method'] == 'kv_t')
            if a['method'] == b['method']:
                for base in ('random', 'pca', 'pls'):
                    match('Covariance whitening', {'input'}, {a['input'], b['input']} == {base, base + '_whiten'},
                          reverse=a['input'].endswith('_whiten'))
                match('Random coordinates → PCA', {'input'}, {a['input'], b['input']} == {'random', 'pca'}, reverse=a['input'] == 'pca')
                match('PCA32 → PLS32', {'input'}, {a['input'], b['input']} == {'pca', 'pls'} and a['dimensions'] == b['dimensions'] == 32,
                      reverse=a['input'] == 'pls')
                match('Input dimension (smaller → larger)', {'dimensions'}, a['dimensions'] != b['dimensions'], reverse=a['dimensions'] > b['dimensions'])
                match('Random coordinate seed 42 → 43', {'seed'}, {a['seed'], b['seed']} == {42, 43} and a['input'].startswith('random'), reverse=a['seed'] == 43)
                match('Output PCA16 → random16', {'output'}, a['method'] == 'kv_t' and {a['output'], b['output']} == {'pca', 'random'}, reverse=a['output'] == 'random')
                match('PLS32 → adaptive PLS (cap 512)', {'adaptive'}, a['method'] == 'kv_t' and a['input'] == 'pls' and {a['adaptive'], b['adaptive']} == {None, .99}, reverse=a['adaptive'] is not None)
                match('Variance-informed ARD initialization', {'initialization'}, {a['initialization'], b['initialization']} == {'geometry', 'variance'}, reverse=a['initialization'] == 'variance')
                match(DECODER_ABLATION, {'group', 'extraction', 'distance'},
                      a['group'].startswith('expert_') and b['group'].startswith('expert_') and a['distance'] != b['distance'],
                      reverse=a['distance'] > b['distance'])
                if a['group'] == b['group'] == 'visual_patches' and a['aggregation'] != b['aggregation']:
                    order = {'mean':0, 'top_mean':1, 'max':2}
                    match('Patch aggregation', {'aggregation'}, True, reverse=order[a['aggregation']] > order[b['aggregation']])
            for family, control, treatment in differences:
                flags = []
                if control['precision'] != treatment['precision']:flags.append('different precision')
                if (control.get('posterior_precision') or control['precision']) != (treatment.get('posterior_precision') or treatment['precision']):
                    flags.append('different posterior precision')
                if control['inference'] != treatment['inference']:flags.append('different inference')
                if control['jitter_escalated'] or treatment['jitter_escalated']:flags.append('recorded jitter escalation')
                if control['actual_input_dim'] != treatment['actual_input_dim'] and family not in ('Input dimension (smaller → larger)', 'PLS32 → adaptive PLS (cap 512)'):
                    flags.append('different effective input dimension')
                note = 'PLS is refitted with the changed targets' if family == 'Output PCA16 → random16' and a['input'].startswith('pls') else ''
                cd, td = descriptions[control['head']], descriptions[treatment['head']]
                contrast = family
                if family == 'Input dimension (smaller → larger)':
                    contrast = f'Input dimensions: {cd["dimensions"]} → {td["dimensions"]}'
                elif family == DECODER_ABLATION:
                    def position(d):return 'before output projection' if d == 0 else f'{d} block{"s" if d != 1 else ""} earlier'
                    contrast = f'Decoder: {position(cd["distance"])} → {position(td["distance"])}'
                row = dict(family=family, round=round_, group=control['group'], control=control['head'], treatment=treatment['head'],
                    control_label=control['label'], treatment_label=treatment['label'],
                    contrast=contrast, control_method=control['method'], treatment_method=treatment['method'],
                    control_precision=control['precision'], treatment_precision=treatment['precision'],
                    comparable_numerics=not flags, caveats='; '.join(flags), note=note)
                for metric in ABLATION_METRICS:
                    x, y = control.get(metric), treatment.get(metric)
                    row['delta_' + metric] = y - x if x is not None and y is not None else np.nan
                output.append(row)
    return pd.DataFrame(output)


def ablation_interpretation(values):
    """Describe observed directions, without inferring significance or equivalence."""
    if values.empty:return 'No pairs meet this sensitivity filter; no conclusion is available.'
    parts = []
    d = values['delta_global_nauc'].dropna()
    if not len(d):
        parts.append('Global referral change is undefined for these pairs.')
    elif np.all(np.abs(d) < 1e-12):
        parts.append('Global referral was unchanged in all observed pairs.')
    else:
        median = d.median(); fraction = (d > 0).mean()
        if abs(median) < .001:
            conclusion = 'The median global referral change was below 0.001 in magnitude'
        elif fraction >= .75:
            conclusion = 'The treatment usually improved global referral'
        elif (d < 0).mean() >= .75:
            conclusion = 'The treatment usually reduced global referral'
        else:
            conclusion = 'Global referral effects were mixed, with a median favoring '+('the treatment' if median > 0 else 'the control')
        parts.append(f'{conclusion} ({median:+.3f}; {int((d > 0).sum())}/{len(d)} pairs improved).')
    def direction(metric):
        vals = values['delta_'+metric].dropna()
        if not len(vals):return 'undefined'
        median = vals.median()
        if abs(median) < .0005:return 'unchanged at three-decimal precision'
        return ('higher' if median > 0 else 'lower')+f' ({median:+.3f})'
    parts.append('Within-task referral median: '+direction('within_nauc')+'.')
    parts.append('Task correlation medians: −Spearman '+direction('neg_spearman')+', −Pearson '+direction('neg_pearson')+'.')
    per_round = values.groupby('round')['delta_global_nauc'].median().dropna()
    if (per_round > .0005).any() and (per_round < -.0005).any():
        parts.append('The global referral direction reverses between checkpoints.')
    return ' '.join(parts)


def ablation_summary(pairs):
    rows = []
    if pairs.empty:return pd.DataFrame()
    for (family, contrast), group in pairs.groupby(['family', 'contrast']):
        matched = group[group.comparable_numerics]
        for subset, values in [(ALL_PAIRS, group), (MATCHED_PAIRS, matched)]:
            row = dict(family=family, contrast=contrast, subset=subset, pairs=len(values),
                       interpretation=ablation_interpretation(values))
            for metric in ABLATION_METRICS:
                d = values['delta_' + metric].dropna()
                row['median_delta_' + metric] = float(d.median()) if len(d) else np.nan
                row['fraction_positive_' + metric] = float((d > 0).mean()) if len(d) else np.nan
                row['defined_pairs_' + metric] = len(d)
            rows.append(row)
    return pd.DataFrame(rows)
