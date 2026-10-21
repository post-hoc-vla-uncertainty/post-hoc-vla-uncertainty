"""One resumable unit of work on one explicitly assigned CUDA/EGL device."""
import argparse
import gc
import os
from pathlib import Path
import shutil
import time

import numpy as np

from uq_extensions.common import atomic_json, file_hash, identity, model_identity, read_json, stream_seed
from uq_extensions.data import candidate_view, load_manifest
from .design import ENSEMBLE, model_path, model_record, round_dir, source_hash


def group_for(declaration, seed):
    return next(g for g in declaration['groups'] if g['seed'] == seed)


def models(root, arm, seed, round_):
    return [read_json(model_path(root, arm, seed, round_, m))['model'] for m in range(2 if arm in ENSEMBLE else 1)]


def acquired(root, arm, seed, through):
    return [id_ for r in range(1, through+1)
            for id_ in read_json(round_dir(root, arm, seed, r)/'selection.json')['selected_ids']]


def support_view(root, declaration, job, through):
    manifest = load_manifest(declaration['manifest'])
    ids = set(manifest['pretraining_ids']) | set(acquired(root, job['arm'], job['seed'], through))
    return manifest, candidate_view(manifest, ids)


def upstream_select(pool, scores, arm, seed, round_):
    """Use the authors' task weighting, GU scaling, tie order and RNG streams."""
    from iterative_fine_tuning.config import IterativeFineTuningConfig
    from iterative_fine_tuning.selection import _TaskCandidatePool, _select_episode_candidates
    from iterative_fine_tuning.manifests import EpisodeSelection
    project = Path(os.environ['VLA_PROJECT_ROOT'])
    cfg = IterativeFineTuningConfig.from_pretrained(str(project/'configs/reproduction/random.json'), cli_args=[])
    cfg.selection.seed = seed
    cfg.selection.task_selection_temperature = 0. if arm == 'random' else 1. if arm == 'gu' else 2.5
    cfg.selection.task_weighted_episode_selection = 'uniform_random' if arm == 'random' else 'top_k'
    cfg.selection.scoring_metric = 'ensemble_terminal_variance' if arm == 'gu' else 'vfd_oneway'
    pools = {}
    for task in sorted(set(pool.tasks)):
        group, task_id = task.split('/'); rows = sorted((c for c in pool.candidates if c.task == task), key=lambda c:c.episode_index)
        candidates = [EpisodeSelection(episode_id=c.episode_index, task_group=group, task_id=int(task_id), uncertainty=scores[c.id]) for c in rows]
        pools[(group, int(task_id))] = _TaskCandidatePool(group, int(task_id), None, candidates,
            float(np.mean([scores[c.id] for c in rows])), scored_episode_count=len(rows))
    selected = _select_episode_candidates(cfg, task_pools=pools, round_index=round_-1)
    lookup = {c.episode_index:c.id for c in pool.candidates}
    return [lookup[c.episode_id] for c in selected]


def acquire(root, declaration, job, device):
    from .heads import paper_scores, score_head
    arm, seed, r = job['arm'], job['seed'], job['round']
    manifest, support = support_view(root, declaration, job, r-1)
    pool = candidate_view(manifest, manifest['target_ids'])
    previous = models(root, arm, seed, r-1)
    directory = round_dir(root, arm, seed, r)
    group = group_for(declaration, seed)
    diagnostics = {}; audit = {}
    if arm == 'random': scores = dict.fromkeys(pool.ids, 0.)
    elif arm in ('vfd', 'gu'):
        scores = paper_scores(arm, previous, pool, directory/'acquisition_scores', device=device, acquisition=True)
    else:
        head = 'h1' if arm == 'h1_ivr' else arm
        scored, diagnostics = score_head(root, declaration, head, previous[0], support, {'pool':pool}, device=device,
                                        ivr=arm=='h1_ivr', ivr_seed=stream_seed(group['selection_seed'], 'ivr', r))
        scores = scored['pool']
        if diagnostics.get('mapping_check', {}).get('usable') is False:
            atomic_json(directory/'infeasible_mapping.json', diagnostics)
            raise RuntimeError('KV-T mapping failed its held-out prediction check; inspect diagnostics before retrying')
        if arm == 'h1_ivr':
            ids = scored['ivr']['selected_ids']; audit = scored['ivr']['audit']
    if arm != 'h1_ivr': ids = upstream_select(pool, scores, arm, group['selection_seed'], r)
    if len(ids) != 5 or len(set(ids)) != 5 or not set(ids) <= set(pool.ids):
        raise ValueError('Acquisition must reveal exactly five distinct eligible demonstrations')
    old = set(acquired(root, arm, seed, r-1)); known = set(manifest['pretraining_ids'])
    atomic_json(directory/'acquisition_diagnostics.json', diagnostics)
    atomic_json(job['output'], dict(selected_ids=ids, scores=scores, audit=audit,
        support_ids=support.ids, policy_hashes=[model_identity(p) for p in previous],
        new_unique_ids=[i for i in ids if i not in old | known],
        prior_acquisition=[i for i in ids if i in old], prior_pretraining=[i for i in ids if i in known],
        selection_seed=group['selection_seed'], provenance='new_acquisition', rule='IVR' if arm=='h1_ivr' else 'upstream_SAVE'))
    print(f'ACQUIRED {arm} seed {seed} round {r}: {ids}; genuinely new={len(set(ids)-old-known)}', flush=True)


def train(root, declaration, job):
    from uq_extensions.config import ExperimentConfig
    from uq_extensions.training import make_training_config, train_worker, retire_optimizers
    arm, seed, r, m = job['arm'], job['seed'], job['round'], job['member']
    manifest = load_manifest(declaration['manifest'])
    previous = models(root, arm, seed, r-1)
    history = acquired(root, arm, seed, r)
    # Separate member work directories prevent two concurrent config writers.
    run_dir = Path(root)/'training'/arm/f'seed_{seed}'/f'member_{m}'
    run_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(arm='random' if arm in ENSEMBLE else 'kv', seed=seed,
        companion=arm in ENSEMBLE, fit_steps=4000, train_batch_size=32,
        num_workers=4, save_frequency=1000, budget_gb=declaration['budget_gb'])
    path = make_training_config(config, manifest, previous, run_dir, r)
    directory = run_dir/f'round_{r:03d}/training/member_{m:02d}'
    request = dict(config_path=str(path), round_index=r, member_index=m, run_dir=str(run_dir),
        model_reference=previous[m], history_episode_ids=sorted(manifest['episode_map'][i] for i in manifest['pretraining_ids']),
        selected_episode_ids=sorted({manifest['episode_map'][i] for i in history}),
        output=str(directory/'member_training_result.json'), expected_steps=4000)
    request_path = directory/'request.json'; atomic_json(request_path, request)
    os.environ['UQ_EXT_RUN_ROOT'] = str(Path(root).resolve())
    os.environ['UQ_EXT_BUDGET_GB'] = str(declaration['budget_gb'])
    print(f'TRAIN {arm} seed {seed}, acquisition round {r}/15, member {m}: '
          f'4,000 updates, batch32, 6.4 replay-sampling epochs; {len(set(history))} unique acquired demos.', flush=True)
    train_worker(request_path)
    result = read_json(request['output'])
    final = str(Path(result['final_model_path']).resolve())
    # Only this newly owned model is cleaned; reused checkpoints remain untouched.
    retire_optimizers([final])
    reused = declaration['reuse'].get(f'{arm}/{seed}', {})
    reconstructed = r <= reused.get('rounds', 0)
    record = model_record(final, dict(kind='reconstructed_deleted_round' if reconstructed else 'new_SAVE_training',
        start_model_hash=model_identity(previous[m]), acquired_ids=sorted(set(history)),
        steps=4000, sampling_epochs=6.4, training_seed=seed+1000*(r-1)+m, member=m,
        reconstruction_note='Same recorded acquisitions and nearest retained prior policy; numerical identity is not assumed' if reconstructed else None))
    atomic_json(job['output'], record)


def reference(root, declaration, job, device):
    from uq_extensions.reference import evaluate_reference
    directory = round_dir(root, job['arm'], job['seed'], job['round'])/'reference_evaluation'
    model = read_json(model_path(root, job['arm'], job['seed'], job['round']))['model']
    evaluate_reference(None, model, directory, job['round'], device=device)
    source = directory/'reference_evaluation.json'
    atomic_json(job['output'], dict(source=str(source), source_hash=file_hash(source), provenance='new_SAVE_evaluation'))


def evaluate(root, declaration, job, device):
    from uq_extensions.evaluation import evaluate as run
    model = read_json(job['model_record'])
    manifest = read_json(Path(root)/'evaluation_manifest.json')
    repetition = job['repetition']; directory = Path(job['output']).parent
    # Reuse only identical policy weights and the exact original evaluation identity.
    followup = Path(declaration['followup'])
    for record_path in sorted((followup/'policies').glob('*/model.json')):
        record = read_json(record_path)
        source = followup/'evaluations'/record_path.parent.name/f'rep_{repetition}/evaluation.json'
        if record['model_hash'] != model['model_hash'] or not source.exists(): continue
        data = read_json(source); spec = data['spec']
        if (spec['checkpoint'] != model['model_hash'] or spec['manifest'] != manifest['manifest_hash'] or
            spec['starts'] != manifest['evaluation_splits'][f'study_rep{repetition}'] or spec['max_steps'] != 520): continue
        # Existing evaluator performs the complete identity check before any simulation.
        run(model['model'], manifest, source.parent, split=f'study_rep{repetition}', device=device, batch_size=15, max_steps=520)
        if len(data['episodes']) != 500: raise ValueError('Incomplete source evaluation')
        atomic_json(directory/'reuse.json', dict(source=str(source), checksum=file_hash(source)))
        atomic_json(job['output'], data)
        print(f'REUSED evaluation {job["policy_id"]} repetition {repetition}: {source}', flush=True)
        return
    run(model['model'], manifest, directory, split=f'study_rep{repetition}', device=device, batch_size=15, max_steps=520)


def collect(root, declaration, job):
    from uq_followup.protocol import aggregate_executions
    from uq_followup.observation_pairing import ObservationPairing
    canonical_path = Path(declaration['followup'])/'evaluations/original_seed0/rep_0/evaluation.json'
    canonical = {(r['task'], r['init_state_hash']):r for r in read_json(canonical_path)['episodes']}
    directory = Path(job['output']).parent
    raw = [r for rep in range(3) for r in read_json(directory/f'rep_{rep}/evaluation.json')['episodes']]
    pairing = ObservationPairing('reused_original_seed0')
    rows = aggregate_executions(raw, canonical=canonical, pairing=pairing, policy_id=job['policy_id'])
    if len(rows) != 500: raise ValueError('Expected 500 conditions and 1,500 executions')
    atomic_json(directory/'observation_pairing.json', pairing.summary())
    atomic_json(directory/'initial_inputs.json', [r['initial_input'] for r in rows])
    quality = dict(success=float(np.mean([r['success'] for r in rows])),
        per_task={task:float(np.mean([r['success'] for r in rows if r['task']==task])) for task in sorted({r['task'] for r in rows})})
    quality['warnings'] = ['Zero rollout success: inspect this policy before interpreting its heads'] if quality['success'] == 0 else []
    atomic_json(directory/'quality.json', quality)
    atomic_json(job['output'], rows)
    print(f'POLICY QUALITY {job["policy_id"]}: success {quality["success"]:.1%}; per-task={quality["per_task"]}', flush=True)


def quality(root, declaration, job):
    """Compare all policy seeds before any head is evaluated at this round."""
    r = job['round']; records = {}; warnings = []
    for g in declaration['groups']:
        seed = g['seed']
        primary = read_json(Path(root)/f'evaluations/random_{seed}_r{r:02d}/quality.json')
        teacher = read_json(Path(root)/f'evaluations/supervisor_{seed}/quality.json')
        records[str(seed)] = dict(primary=primary, supervisor=teacher, supervisor_advantage=teacher['success']-primary['success'])
        if primary['success'] == 0: warnings.append(f'Seed {seed}: zero primary success; inspect training')
        if teacher['success'] <= primary['success']: warnings.append(f'Seed {seed}: supervisor has no overall success advantage')
    values = [v['primary']['success'] for v in records.values()]
    if max(values)-min(values) > .25: warnings.append('Primary success spread exceeds 25 percentage points')
    tasks = next(iter(records.values()))['primary']['per_task']
    for task in tasks:
        values = [v['primary']['per_task'][task] for v in records.values()]
        if max(values)-min(values) > .6: warnings.append(f'{task}: primary success spread exceeds 60 percentage points')
    print(f'QUALITY round {r}: '+str({s:v['primary']['success'] for s,v in records.items()}), flush=True)
    for warning in warnings: print('QUALITY WARNING: '+warning, flush=True)
    # A valid weaker policy stays in the study. The audit is available before scores;
    # warnings are not an outcome-based seed exclusion or an automatic retraining rule.
    atomic_json(job['output'], dict(round=r, policies=records, warnings=warnings, auto_excluded=[]))


def score(root, declaration, job, device):
    from uq_extensions.protocol import Candidate, CandidateView
    from .heads import paper_scores, score_head, reusable_head
    from .reporting import cohort, point_metrics
    root = Path(root); directory = Path(job['output']).parent
    manifest, support = support_view(root, declaration, job, job['round'])
    pool = candidate_view(manifest, manifest['target_ids'])
    query = CandidateView([Candidate(**r) for r in read_json(root/f'evaluations/{job["policy_id"]}/initial_inputs.json')])
    paths = models(root, 'random', job['seed'], job['round'])
    head = job['head']; diagnostics = {}
    score_path = directory/'scores.json'
    if score_path.exists(): record = read_json(score_path)
    else:
        reused = reusable_head(declaration, model_identity(paths[0]), support.ids, declaration['recipes'][head]) if head in ('h1','h3','h4') else None
        if reused is not None and (reused/'scores.json').exists():
            previous = read_json(reused/'scores.json')
            # Follow-up scoring used canonical original_seed0 observations, audited in collect().
            old_inputs = read_json(Path(declaration['followup'])/'evaluations'/reused.parent.name/'initial_inputs.json')
            matching_inputs = {(x['id'],x['input_hash']) for x in old_inputs} == {(c.id,c.input_hash) for c in query.candidates}
            if set(previous['ids']) == set(query.ids) and matching_inputs:
                atomic_json(directory/'diagnostics.json', read_json(reused/'diagnostics.json'))
                atomic_json(score_path, previous)
                atomic_json(directory/'reuse.json', dict(source=str(reused), checksum=file_hash(reused/'scores.json')))
                print(f'REUSED raw {head} scores: {reused}; referral recomputed with this replicate\'s supervisor.', flush=True)
                return score(root, declaration, job, device)
        if head.startswith('h'):
            outputs, diagnostics = score_head(root, declaration, head, paths[0], support, {'query':query, 'pool':pool}, device=device)
        else:
            outputs = {role:paper_scores(head, paths, view, directory/role, device=device) for role, view in [('query', query), ('pool', pool)]}
        record = dict(ids=query.ids, scores=[outputs['query'][i] for i in query.ids], tasks=query.tasks,
            pool_task_means={task:float(np.mean([outputs['pool'][i] for i, t in zip(pool.ids, pool.tasks) if t==task])) for task in sorted(set(pool.tasks))})
        atomic_json(directory/'diagnostics.json', diagnostics)
        atomic_json(score_path, record)
    diagnostics = read_json(directory/'diagnostics.json') if (directory/'diagnostics.json').exists() else {}
    status = 'mapping_infeasible' if diagnostics.get('mapping_check', {}).get('usable') is False else 'complete'
    metrics = point_metrics(cohort(root, job['policy_id'], job['seed']), record)
    atomic_json(job['output'], dict(status=status, head=head, seed=job['seed'], round=job['round'],
                                  policy_id=job['policy_id'], metrics=metrics, model_hash=model_identity(paths[0])))
    print(f'SCORED {job["policy_id"]} {head}: {status}; {metrics}', flush=True)


def cleanup(root, declaration, job):
    # Only generated features owned by this study; models, fitted heads and raw scores remain.
    record = read_json(job['model_record'])
    path = Path(root)/'features'/record['model_hash']
    if path.is_symlink(): raise ValueError('Refusing symlinked feature cache cleanup')
    size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0
    if path.exists(): shutil.rmtree(path)
    atomic_json(job['output'], dict(model_hash=record['model_hash'], retired_feature_bytes=size, regenerable=True))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('request', type=Path)
    args = parser.parse_args(); request = read_json(args.request)
    if identity(request['spec']) != request['identity']: raise RuntimeError('Job request checksum mismatch')
    spec = request['spec']; root = Path(spec['study']); declaration = read_json(root/'declaration.json')
    if source_hash(declaration['project']) != declaration['source_hash']: raise RuntimeError('Scientific source changed')
    job = spec['job']; started = time.monotonic()
    if job['kind'] in ('acquire', 'train', 'evaluate', 'score'):
        from uq_extensions.storage import check_budget
        check_budget(root, declaration['budget_gb'], reserve_gb=20)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Each worker must see exactly one selected GPU')
    if job.get('a100') and 'A100' not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError('KV-T float64 requires an A100')
    torch.set_num_threads(4)
    memory_budget = None
    if job['kind'] in ('acquire', 'score'):
        torch.backends.cuda.matmul.allow_tf32 = False
        from uq_extensions.gpu_memory import configure_memory
        memory_budget = configure_memory('cuda:0')
    os.environ['UQ_EXT_PROJECTION_CACHE'] = str(root/'projection_cache')
    print(f'FINAL STUDY START {job["id"]}: GPU={torch.cuda.get_device_name(0)}, '
          f'CUDA={os.environ.get("CUDA_VISIBLE_DEVICES")}, EGL={os.environ.get("MUJOCO_EGL_DEVICE_ID")}', flush=True)
    if job['kind'] in ('acquire', 'evaluate', 'reference', 'score'):
        globals()[job['kind']](root, declaration, job, 'cuda:0')
    else: globals()[job['kind']](root, declaration, job)
    gc.collect(); torch.cuda.empty_cache()
    if memory_budget is not None:
        from uq_extensions.gpu_memory import memory_record
        atomic_json(args.request.with_name('gpu_memory.json'), memory_record('cuda:0', memory_budget))
    atomic_json(args.request.with_name('complete.json'), dict(identity=request['identity'],
        output_hash=file_hash(job['output']), seconds=time.monotonic()-started))
    print(f'FINAL STUDY DONE {job["id"]}; wall={(time.monotonic()-started)/60:.1f} min', flush=True)


if __name__ == '__main__': main()
