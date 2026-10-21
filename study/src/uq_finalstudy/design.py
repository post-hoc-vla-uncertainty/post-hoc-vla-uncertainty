"""Frozen recipes and a dependency graph; no GPU work during preparation."""
from copy import deepcopy
from pathlib import Path
import os

from uq_extensions.common import atomic_json, file_hash, identity, model_identity, read_json
from uq_extensions.config import ExperimentConfig
from uq_extensions.data import load_manifest
from uq_followup.protocol import evaluation_manifest

ARMS = ('random', 'vfd', 'gu', 'h1', 'h2', 'h3', 'h4', 'h1_ivr')
SCORES = ('h1', 'h2', 'h3', 'h4', 'vfd', 'gu', 'action_l2', 'ace', 'decu',
          'vlm_token_entropy', 'vlm_perplexity')
ROUNDS = 15
ACQUISITION_EVALUATION_ROUNDS = (0, 5, 10, 15)
COMMON_EVALUATION_ROUNDS = (0, 10)
ENSEMBLE = ('random', 'vfd', 'gu')
COLLABORATOR_SOURCE = 'reference_src/vla-ua/src/lerobot/uncertainty/uncertainty_scoring/kernel_variance.py'


def evaluation_rounds(declaration, kind):
    """Read the frozen schedule; version-1 studies keep their original schedule."""
    defaults = dict(acquisition=ACQUISITION_EVALUATION_ROUNDS, common=COMMON_EVALUATION_ROUNDS)
    default = defaults[kind] if declaration.get('version', 1) >= 2 else range(ROUNDS+1)
    values = declaration.get(f'{kind}_evaluation_rounds', list(default))
    if (not values or any(type(r) is not int or not 0 <= r <= ROUNDS for r in values)
            or list(values) != sorted(set(values)) or values[0] != 0):
        raise ValueError(f'Invalid {kind} evaluation rounds: {values}')
    return tuple(values)


def recipes():
    from uq_followup.design import recipes as matrix
    rows = {r['id']: r for r in matrix()}
    selected = dict(h1='joint_late__kv_pca512_s42__whiten_lw',
                    h3='decoder_to_velocity__kv_pls512_s42__whiten_lw',
                    h4='decoder_to_velocity__kv_t_pca512_s42')
    result = {k: deepcopy(rows[v]) for k, v in selected.items()}
    result['h2'] = dict(id='collaborator_pooled_decoder_whiten', method='collaborator_kv',
        source=COLLABORATOR_SOURCE, location='joint_late', dimensions=480, ridge=.01,
        projection='whiten', lengthscale='median_pairwise', flow_level=9,
        flow_level_indexing='one_based', noise_seed=0, observations_per_batch=1)
    return result


def source_hash(project):
    from uq_followup.launcher import source_hash as existing
    project = Path(project)
    files = [p for p in (project/'src/uq_finalstudy').glob('*.py') if p.name != 'reporting.py']
    # The collaborator kernel is loaded from its own file, not its lerobot package.
    files += [project/COLLABORATOR_SOURCE,
              project/'upstream/uq_vla_extensions/scripts/calibration/calibration_comparison.py']
    files += sorted((project/'upstream/uq_vla_extensions/src/iterative_fine_tuning').glob('*.py'))
    return identity(dict(existing=existing(project), files={str(p.relative_to(project)): file_hash(p) for p in files}))


def model_record(path, provenance):
    path = Path(path).resolve()
    return dict(model=str(path), model_hash=model_identity(path), provenance=provenance)


def round_dir(root, arm, seed, round_):
    return Path(root)/'runs'/arm/f'seed_{seed}'/f'round_{round_:03d}'


def model_path(root, arm, seed, round_, member=0):
    return round_dir(root, arm, seed, round_)/f'model_{member}.json'


def save_immutable(path, value):
    if Path(path).exists() and read_json(path) != value:
        raise RuntimeError(f'Immutable artifact differs: {path}')
    atomic_json(path, value)


def prepare(root, project, manifest_path, *, budget_gb=1800):
    root, project = Path(root).resolve(), Path(project).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    root.mkdir(parents=True, exist_ok=True)
    path = root/'declaration.json'
    if path.exists():
        declaration = read_json(path)
        if (declaration['source_hash'] != source_hash(project) or
            declaration['manifest_hash'] != manifest['manifest_hash'] or
            declaration['recipes'] != recipes()):
            raise RuntimeError('Scientific source/data/recipes changed. Use a new study directory; do not mix results.')
        register_reuse(root, declaration)
        return declaration
    followup = project/'storage/extensions/followup_v1'
    groups = []
    # Preserve the disjoint reproduction ensembles; seed choice never uses performance.
    for index, pair in enumerate(((0, 1), (2, 3), (4, 5))):
        from uq_followup.policies import original_checkpoint
        from reproduction_storage import validate_checkpoint
        for seed in pair:
            if original_checkpoint(project, dict(condition='original', seed=seed)) is None:
                raise RuntimeError(f'Missing complete starting policy for seed {seed}')
        bases = [model_record(project/f'storage/outputs/pretrain/libero_pretrain_leak3/seed_{s}/checkpoints/030000/pretrained_model',
                              dict(kind='reused_pretraining', training_seed=s)) for s in pair]
        supervisor = read_json(followup/f'policies/supervisor_seed{index}/model.json')
        if set(supervisor['spec']['ids']) != set(manifest['pretraining_ids']) | set(manifest['target_ids']):
            raise ValueError('Supervisor does not cover the declared full demonstration pool')
        teacher = model_record(supervisor['model'], dict(kind='reused_supervisor', training_seed=index))
        if validate_checkpoint(Path(supervisor['model']).parent) != supervisor['spec']['steps']:
            raise RuntimeError('Supervisor training is incomplete')
        if teacher['model_hash'] != supervisor['model_hash']:
            raise RuntimeError('Supervisor checkpoint changed')
        groups.append(dict(seed=1000+index, selection_seed=42+index,
            primary_seed=pair[0], companion_seed=pair[1], supervisor_seed=index, bases=bases, supervisor=teacher))
    reuse = {}
    from uq_extensions.reference import validate_source
    for group in groups:
        seed = group['seed']
        for arm in ('random', 'vfd'):
            source = project/f'storage/outputs/reproduction/{arm}/seed_{seed}'
            if not (source/'run_state.json').exists(): continue
            completed = 0
            for r in range(ROUNDS):
                if not (source/f'round_{r:03d}/reproduction_commit.json').exists(): break
                completed += 1
            if not completed: continue
            config = ExperimentConfig(arm=arm, seed=seed, companion=True,
                base_models=[b['model'] for b in group['bases']])
            checked = validate_source(config, source, manifest, through_round=completed)
            reused = dict(source=str(source), rounds=completed, evidence=checked['evidence'],
                          available_rounds=checked['available'], selections={}, models={}, evaluations={})
            for r in range(completed+1):
                if r in checked['models']:
                    reused['models'][str(r)] = [model_record(p, dict(kind='reused_reproduction', source=str(source), round=r))
                                               for p in checked['models'][r]]
                ep = source/('initial_evaluation/member_00/member_evaluation.json' if r == 0 else
                            f'round_{r-1:03d}/training/member_00/member_evaluation.json')
                reused['evaluations'][str(r)] = str(ep)
                if r:
                    reused['selections'][str(r)] = [checked['mapping'][i] for i in checked['rows'][r-1]['selected_episode_ids']]
            # An interrupted ensemble round may contain a fully finished member.
            # Audit its request, seed, support history and checkpoint independently.
            partial = source/f'round_{completed:03d}'
            selection_path = partial/'selection_manifest.json'
            if completed < ROUNDS and selection_path.exists() and completed in checked['models']:
                selection = read_json(selection_path)
                indices = [v['episode_id'] for v in selection['selected_episodes']]
                if selection['round_index'] != completed or len(set(indices)) != 5:
                    raise RuntimeError('Invalid partial-round acquisition')
                ids = [checked['mapping'][i] for i in indices]
                if not set(ids) <= set(manifest['target_ids']): raise RuntimeError('Ineligible partial acquisition')
                cumulative = {i for row in checked['rows'] for i in row['selected_episode_ids']} | set(indices)
                retained = [None, None]
                from reproduction_storage import validate_checkpoint
                for member in range(2):
                    result_path = partial/f'training/member_{member:02d}/member_training_result.json'
                    request_path = partial/f'training/member_{member:02d}/member_training_request.json'
                    if not result_path.exists() or not request_path.exists(): continue
                    result, request = read_json(result_path), read_json(request_path)
                    previous = Path(checked['models'][completed][member]).resolve()
                    if (request['round_index'] != completed or request['member_index'] != member or
                        Path(request['config_snapshot_path']).resolve() != (source/'iterative_fine_tuning_config.json').resolve() or
                        Path(request['model_reference']).resolve() != previous or
                        Path(result['start_model_path']).resolve() != previous or
                        set(request['history_episode_ids']) != {manifest['episode_map'][i] for i in manifest['pretraining_ids']} or
                        set(request['selected_episode_ids']) != cumulative or
                        result['seed'] != seed+1000*completed+member):
                        raise RuntimeError('Partial training member does not match the agreed recipe')
                    model = Path(result['final_model_path']).resolve()
                    if validate_checkpoint(model.parent) != 4000: continue
                    retained[member] = model_record(model, dict(kind='reused_completed_member_of_interrupted_round',
                                                              source=str(result_path), round=completed+1))
                    for evidence in (request_path, result_path, selection_path, model.parent/'checkpoint_inventory.json'):
                        reused['evidence'][str(evidence)] = file_hash(evidence)
                if any(v is not None for v in retained):
                    reused['models'][str(completed+1)] = retained
                    reused['selections'][str(completed+1)] = ids
                    reused['partially_completed_round'] = completed+1
            reuse[f'{arm}/{seed}'] = reused
    declaration = dict(version=2, project=str(project), manifest=str(manifest_path),
        manifest_hash=manifest['manifest_hash'], source_hash=source_hash(project), groups=groups,
        arms=list(ARMS), scores=list(SCORES), recipes=recipes(), rounds=ROUNDS, steps=4000,
        batch_size=32, queries=5, eligibility='upstream_reference', replay=[.5, .5],
        reference_evaluation='30 per task, upstream seed100, batch15, horizon520',
        common_evaluation='50 conditions per task, three executions each, horizon520',
        acquisition_evaluation_rounds=list(ACQUISITION_EVALUATION_ROUNDS),
        common_evaluation_rounds=list(COMMON_EVALUATION_ROUNDS),
        supervisor_pairing='independent full-pool supervisor per replicate; numeric training seed need not match primary',
        common_policy='Random, rounds 0 and 10; retained historical models reused',
        bootstrap_repetitions=2000, budget_gb=budget_gb, reuse=reuse,
        followup=str(followup), pretraining_demos=len(manifest['pretraining_ids']),
        supervisor_demos=len(set(manifest['pretraining_ids']) | set(manifest['target_ids'])))
    save_immutable(root/'evaluation_manifest.json', evaluation_manifest(manifest))
    save_immutable(path, declaration)
    register_reuse(root, declaration)
    return declaration


def register_reuse(root, declaration):
    root = Path(root)
    for group in declaration['groups']:
        seed = group['seed']
        save_immutable(root/'supervisors'/f'{seed}.json', group['supervisor'])
        for arm in ARMS:
            for member, record in enumerate(group['bases'][:2 if arm in ENSEMBLE else 1]):
                save_immutable(model_path(root, arm, seed, 0, member), record)
    for key, reused in declaration['reuse'].items():
        arm, seed = key.split('/'); seed = int(seed)
        for path, checksum in reused['evidence'].items():
            if file_hash(path) != checksum: raise RuntimeError(f'Reproduction evidence changed: {path}')
        for r, records in reused['models'].items():
            if int(r) == 0: continue
            for member, record in enumerate(records):
                if record is not None: save_immutable(model_path(root, arm, seed, int(r), member), record)
        for r, ids in reused['selections'].items():
            save_immutable(round_dir(root, arm, seed, int(r))/'selection.json',
                dict(selected_ids=ids, provenance='reused_reproduction', source=reused['source']))
        for r, path in reused['evaluations'].items():
            save_immutable(round_dir(root, arm, seed, int(r))/'reference.json',
                dict(source=path, source_hash=file_hash(path), provenance='original_reproduction_outcomes'))


def jobs(root, declaration):
    """Fine-grained jobs let training/evaluation use any GPU; only KV-T needs A100."""
    root = Path(root); result = []
    reference_rounds = evaluation_rounds(declaration, 'acquisition')
    common_rounds = evaluation_rounds(declaration, 'common')
    def add(kind, id_, output, dependencies=(), **extra):
        result.append(dict(kind=kind, id=id_, output=str(output), dependencies=list(dependencies), **extra))
        return id_
    def tid(arm, seed, r, m): return f'train_{arm}_{seed}_{r:02d}_{m}'
    for group in declaration['groups']:
        seed = group['seed']
        reps = [add('evaluate', f'eval_supervisor_{seed}_{rep}', root/f'evaluations/supervisor_{seed}/rep_{rep}/evaluation.json',
                    policy_id=f'supervisor_{seed}', model_record=str(root/'supervisors'/f'{seed}.json'), repetition=rep)
                for rep in range(3)]
        sup = add('collect', f'collect_supervisor_{seed}', root/f'evaluations/supervisor_{seed}/conditions.json', reps,
                  policy_id=f'supervisor_{seed}')
        for arm in ARMS:
            count = 2 if arm in ENSEMBLE else 1
            for r in range(ROUNDS+1):
                rd = round_dir(root, arm, seed, r)
                trains = []
                if r:
                    prev = [tid(arm, seed, r-1, m) for m in range(count)] if r > 1 else []
                    if r > 1: prev.append(f'acquire_{arm}_{seed}_{r-1:02d}')
                    acq = add('acquire', f'acquire_{arm}_{seed}_{r:02d}', rd/'selection.json', prev,
                              arm=arm, seed=seed, round=r, a100=arm == 'h4')
                    trains = [add('train', tid(arm, seed, r, m), model_path(root, arm, seed, r, m), [acq, *prev],
                                  arm=arm, seed=seed, round=r, member=m) for m in range(count)]
                # All arms start at the same policy: one round-0 evaluation per seed.
                if r in reference_rounds and (r or arm == 'random'):
                    add('reference', f'reference_{arm}_{seed}_{r:02d}', rd/'reference.json', trains[:1],
                        arm=arm, seed=seed, round=r)
                if arm != 'random' or r not in common_rounds: continue
                pid = f'random_{seed}_r{r:02d}'
                deps = [trains[0]] if trains else []
                reps = [add('evaluate', f'eval_{pid}_{rep}', root/f'evaluations/{pid}/rep_{rep}/evaluation.json', deps,
                            policy_id=pid, model_record=str(model_path(root, arm, seed, r)), repetition=rep)
                        for rep in range(3)]
                collect = add('collect', f'collect_{pid}', root/f'evaluations/{pid}/conditions.json', reps, policy_id=pid)
                score_jobs = []
                for head in SCORES:
                    score_jobs.append(add('score', f'score_{pid}_{head}', root/f'scores/{pid}/{head}/result.json',
                        [collect, sup, f'quality_round_{r:02d}', *trains], policy_id=pid, arm=arm, seed=seed, round=r, head=head, a100=head=='h4'))
                cleanup_deps = score_jobs
                if r == 0:
                    cleanup_deps += [f'acquire_{a}_{seed}_01' for a in ARMS]
                add('cleanup', f'cleanup_{pid}', root/f'cleanup/{pid}.json', cleanup_deps,
                    model_record=str(model_path(root, arm, seed, r)))
            if arm != 'random':
                for r in range(1, ROUNDS):
                    add('cleanup', f'cleanup_{arm}_{seed}_{r:02d}', root/f'cleanup/{arm}_{seed}_{r:02d}.json',
                        [f'acquire_{arm}_{seed}_{r+1:02d}'], model_record=str(model_path(root, arm, seed, r)))
    for r in common_rounds:
        dependencies = [f'collect_random_{g["seed"]}_r{r:02d}' for g in declaration['groups']]
        dependencies += [f'collect_supervisor_{g["seed"]}' for g in declaration['groups']]
        add('quality', f'quality_round_{r:02d}', root/f'quality/round_{r:02d}.json', dependencies, round=r)
    return result
