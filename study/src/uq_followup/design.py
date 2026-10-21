"""The explicitly enumerated 133-recipe follow-up matrix."""
from copy import deepcopy
from dataclasses import replace

from uq_extensions.common import plain
from uq_extensions.config import FeatureConfig, GPConfig
from uq_extensions.head_study_design import study_recipes
from uq_extensions.baselines import METHODS

MAPPINGS = {
    'early_to_decoder': ('joint_early_flow', 'decoder_hidden', True),
    'middle_to_decoder': ('joint_intermediate_flow', 'decoder_hidden', True),
    'middle_to_final_context': ('joint_intermediate', 'joint_late', False),
    'decoder_to_velocity': ('action_expert', 'velocity', True),
}
VARIANTS = [('random512', 'random', 512, 42), ('random512', 'random', 512, 43),
            ('pca512', 'pca', 512, 42), ('pls512', 'pls', 512, 42),
            ('pca_adaptive', 'pca', 512, 42), ('pls_adaptive', 'pls', 32, 42)]


def recipes(fit_steps=10000, conditioning_mode='append'):
    if conditioning_mode not in ('append', 'compressed'): raise ValueError('Unknown conditioning mode')
    result = []
    for old in study_recipes():
        if (old['method'] != 'kv' or old['features']['components'] != 512 or
                old['group'] in ('visual_patches', 'state', 'expert_1', 'expert_3',
                                 'joint_intermediate', 'expert_0')):
            continue
        row = deepcopy(old)
        row['variant'] = row['features']['compression'].removesuffix('_whiten') + '512'
        row['whiten'] = row['features']['compression'].endswith('_whiten')
        row['gp']['precision'] = 'float32'
        row['gp']['fit_steps'] = fit_steps
        row.pop('matched_control', None)
        result.append(row)
    assert len(result) == 52
    for mapping, (bank, target, flow) in MAPPINGS.items():
        for variant, method, width, seed in VARIANTS:
            features = FeatureConfig(bank=bank, target_bank=target, compression=method,
                components=width, seed=seed, geometry_seed=2026, action_bins=50,
                flow_indices=(0, 5, 9), flow_samples=50, support_flow_samples=4)
            gp = GPConfig(fit_steps=fit_steps, input_projection=method, input_components=width,
                output_components=16, output_batch_size=16, output_seed=142,
                split_seed=242, training_seed=342, inference='sparse' if flow else 'exact',
                sparse_justification='Four support probes x three flow times; 512 inducing points' if flow else None,
                pls_variance_target=.99 if variant == 'pls_adaptive' else None)
            for estimator, whiten in [('kv', False), ('kv', True), ('kv_t', False)]:
                label = f'{mapping}__{estimator}_{variant}_s{seed}' + ('__whiten_lw' if whiten else '')
                result.append(dict(id=label, group=mapping, method=estimator, variant=variant,
                    whiten=whiten, scope='action' if flow else 'observation',
                    conditioning_columns=359 if conditioning_mode == 'append' and mapping in ('early_to_decoder', 'middle_to_decoder') else 0,
                    features=plain(features), gp=plain(replace(gp, precision='float64' if estimator == 'kv_t' else 'float32'))))
    control = next(r for r in result if r['id'] == 'decoder_to_velocity__kv_t_pca512_s42')
    random_output = deepcopy(control)
    random_output['id'] += '__random_output16'
    random_output['gp']['output_compression'] = 'random'
    result.append(random_output)
    control = next(r for r in result if r['id'] == 'middle_to_final_context__kv_t_random512_s42')
    whitened = deepcopy(control)
    whitened.update(id=control['id'] + '__whiten_lw', whiten=True)
    result.append(whitened)
    for method in METHODS:
        result.append(dict(id='paper__'+method, group='paper', method=method, scope='paper'))
    result.sort(key=lambda r: (r['scope'] == 'action', r['method'] == 'kv_t', r['id']))
    assert len(result) == len({r['id'] for r in result}) == 133
    return result
