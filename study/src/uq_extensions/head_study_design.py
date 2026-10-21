"""Single declared matrix for the frozen-policy train/validation head study."""
from dataclasses import replace

from .config import FeatureConfig, GPConfig
from .matrix import recipe

# Dimensions describe the actual pooled representations, not channel counts.
HEADS = [
    dict(id='language', title='Language before multimodal processing', width=7681, pca=(512,)),
    dict(id='visual_local', title='Early spatial vision', width=24576, pca=(32,128,512), target='visual_scene'),
    dict(id='visual_scene', title='Late spatial vision', width=6144, pca=(32,512)),
    dict(id='visual_patches', title='Native early visual patches, per patch', width=768, pca=(32,128,512)),
    dict(id='state', title='Raw robot state', width=8, pca=()),
    dict(id='joint_intermediate', title='Intermediate image/language context plus raw state', width=2888,
         random=(32,128,512), pca=(32,128,512), target='joint_late'),
    dict(id='joint_late', title='Final image/language context plus raw state', width=2888, pca=(32,512)),
    dict(id='contextual_language', title='Contextualized language tokens', width=7681, pca=(512,)),
    dict(id='contextual_vision', title='Contextualized image tokens', width=1920, pca=(512,)),
    dict(id='contextual_state_intermediate', title='Intermediate fused state token', width=960,
         pca=(32,512), target='contextual_state_late'),
    dict(id='contextual_state_late', title='Final fused state token', width=960, pca=(32,512)),
    *[dict(id=f'expert_{distance}', bank='action_expert', title=title, width=27239,
           pca=(32,128,512) if distance==1 else (512,), target='velocity', flow=True, distance=distance)
      for distance,title in [(0,'Action decoder immediately before linear output'),
                             (1,'Action decoder before its last transformer block'),
                             (3,'Action decoder before its last three transformer blocks')]],
    dict(id='action_context', title='Final context plus current noisy chunk and time', width=3239,
         pca=(32,512), target='velocity', flow=True),
    *[dict(id=bank+'_to_velocity', bank=bank, title=title+' plus current noisy chunk and time',
           width=width+351, target='velocity', flow=True, diagnostic_only=True)
      for bank,title,width in [('language','Pure language',7681),('visual_scene','Late spatial vision',6144),
         ('contextual_language','Contextualized language',7681),('contextual_vision','Contextualized vision',1920),
         ('contextual_state_late','Final fused state token',960)]],
    dict(id='final_action', bank='joint_late', title='Final context plus initial Gaussian chunk and time field',
         width=3239, target='action_chunk', flow=True, diagnostic_only=True, target_random=True),
]


def original_study_recipes(*, scope='all', fit_steps=10000, output_batch_size=16, flow_samples=50):
    rows=[]
    for head in HEADS:
        if scope=='observation' and head.get('flow'):continue
        f=FeatureConfig(bank=head.get('bank',head['id']), target_bank=head.get('target','joint_late'),
            geometry_seed=2026,action_bins=50,flow_samples=flow_samples,support_flow_samples=4,
            reference_flow_samples=4,flow_indices=(0,5,9),expert_distance=head.get('distance',0))
        if head['id']=='final_action':f=replace(f,flow_indices=(9,))
        g=GPConfig(fit_steps=fit_steps,output_batch_size=output_batch_size,output_components=16,
                   input_components=32,output_seed=142,split_seed=242,training_seed=342)
        if head.get('flow'):
            g=replace(g,inference='sparse',sparse_justification='Up to 1472 episodes x 4 Gaussian chunks x 3 generation steps; fixed 512 inducing points')
        inputs=[]
        if not head.get('diagnostic_only'):
            for n in head.get('random',(512,)):
                n=min(n,head['width'])
                inputs.extend(('random',n,seed) for seed in ((42,43) if n<head['width'] else (42,)))
            inputs.extend(('pca',n,42) for n in head['pca'])
            for projection,n,seed in inputs:
                for aggregation in (('mean','max','top_mean') if head['id']=='visual_patches' else ('top_mean',)):
                    suffix=f'_{aggregation}' if head['id']=='visual_patches' else ''
                    label=f'{head["id"]}__kv_{projection}{n}_s{seed}{suffix}'
                    row=recipe(label,'kv',replace(f,compression=projection,components=n,seed=seed,patch_aggregation=aggregation),
                               g,head['title'],group=head['id'])
                    row['scope']='action' if head.get('flow') else 'observation'
                    rows.append(row)
        if 'target' not in head:continue
        target_inputs=[*inputs,('pls',32,42),('pls_adaptive',32,42)]
        outputs=('pca','random') if not head.get('diagnostic_only') or head.get('target_random') else ('pca',)
        for output in outputs:
            for projection,n,seed in target_inputs:
                gp=replace(g,input_projection='pls' if projection=='pls_adaptive' else projection,
                           input_components=n,output_compression=output,
                           pls_variance_target=.99 if projection=='pls_adaptive' else None)
                label=f'{head["id"]}__kvt_{projection}{n}_s{seed}_{output}16'
                row=recipe(label,'kv_t',replace(f,seed=seed),gp,head['title']+' → '+head['target'],group=head['id'])
                row['scope']='action' if head.get('flow') else 'observation'
                if projection=='pls_adaptive':
                    row['conditional_on']=label.replace('pls_adaptive','pls')
                rows.append(row)
    # Predeclared focused output-dependence comparison; no result-based choice.
    for bank in ('joint_intermediate','expert_1'):
        if scope=='observation' and bank=='expert_1':continue
        for output in ('pca','random'):
            base=next(r for r in rows if r['id']==f'{bank}__kvt_pls32_s42_{output}16')
            for dependence in ('independent','coregionalized'):
                if bank=='expert_1' and dependence=='independent':continue # existing matched sparse control
                gp=GPConfig(**base['gp'])
                gp=replace(gp,inference='sparse',output_dependence=dependence,
                    sparse_justification='Matched 512-inducing independent versus rank-4 coregionalized comparison')
                row=recipe(f'{bank}__{dependence}_sparse_{output}16','kv_t',FeatureConfig(**base['features']),gp,
                    'Output dependence, matched inputs and inducing-point budget',group=bank)
                row['scope']=base['scope'];rows.append(row)
    from .baselines import METHODS
    for method in METHODS:
        row=recipe('paper__'+method,method,FeatureConfig(),GPConfig(),'Pinned paper/reference score',group='paper')
        row['scope']='paper';rows.append(row)
    # First get broad KV outputs and paper comparisons, then fit the longer GPs.
    rank={'observation':0,'paper':1,'action':2}
    rows.sort(key=lambda r:(rank[r['scope']],r['method']=='kv_t'))
    assert len({r['id'] for r in rows})==len(rows)
    return rows


def whitening_recipes(original):
    """Append matched geometry comparisons; never reorder historical recipes."""
    rows=[]
    for control in original:
        if control['method']!='kv':continue
        f=FeatureConfig(**control['features'])
        # Language is rank-limited; the focused random-width controls stay unchanged.
        if control['group']=='language' and f.compression=='random':continue
        if control['group']=='joint_intermediate' and f.compression=='random' and f.components!=512:continue
        row=recipe(control['id']+'__whiten_lw','kv',replace(f,compression=f.compression+'_whiten'),
            GPConfig(**control['gp']),'Regularized covariance whitening versus the same unwhitened KV',group=control['group'])
        row.update(scope=control['scope'],extension='whitening_v1',matched_control=control['id'])
        rows.append(row)
    for bank in ('visual_local','joint_intermediate'):
        control=next(r for r in original if r['id']==f'{bank}__kvt_pls32_s42_pca16')
        f=replace(FeatureConfig(**control['features']),compression='pls',components=32)
        label=f'{bank}__kv_pls32_s42_pca16'
        for whitening in (False,True):
            row=recipe(label+('__whiten_lw' if whitening else ''),'kv',
                replace(f,compression='pls_whiten' if whitening else 'pls'),GPConfig(**control['gp']),
                'Target-informed projection with fixed KV; isolate projection from learned ARD',group=bank)
            row.update(scope='observation',extension='whitening_v1',
                       matched_control=label if whitening else control['id'])
            rows.append(row)
    # Only three new expensive fits per policy, all observation heads with PCA16 targets.
    for bank in ('visual_local','joint_intermediate'):
        control=next(r for r in original if r['id']==f'{bank}__kvt_random512_s42_pca16')
        row=recipe(control['id']+'__whiten_lw','kv_t',FeatureConfig(**control['features']),
            replace(GPConfig(**control['gp']),input_projection='random_whiten'),
            'ARD in covariance principal directions of the same 512 selected coordinates',group=bank)
        row.update(scope='observation',extension='whitening_v1',matched_control=control['id'])
        rows.append(row)
    control=next(r for r in original if r['id']=='joint_intermediate__kvt_pls32_s42_pca16')
    row=recipe(control['id']+'__variance_init_lw','kv_t',FeatureConfig(**control['features']),
        replace(GPConfig(**control['gp']),lengthscale_initialization='variance'),
        'Unchanged PLS32 input and ARD model; start from regularized variance-informed length scales',group=control['group'])
    row.update(scope='observation',extension='whitening_v1',matched_control=control['id'])
    rows.append(row)
    return rows


def study_recipes(**kwargs):
    original=original_study_recipes(**kwargs)
    rows=original+whitening_recipes(original)
    assert len({r['id'] for r in rows})==len(rows)
    return rows
