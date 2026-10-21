"""Immutable datasets, equal-epoch budgets and condition-level outcomes."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import math
import numpy as np

from uq_extensions.common import identity, read_json

SEEDS = (0, 1, 2)
CONDITIONS = ('original', 'plus35')
BASE_STEPS = 30000
BATCH_SIZE = 32


def sampled_additions(manifest, seed):
    rng = np.random.default_rng(20261001 + seed)
    original = set(manifest['pretraining_ids'])
    selected = []
    for task in range(3, 10):
        candidates = sorted(i for i in manifest['target_ids']
                            if i.startswith(f'libero_10/{task}/') and i not in original)
        if len(candidates) < 5:
            raise ValueError(f'Not enough new demonstrations for task {task}')
        selected.extend(rng.choice(candidates, 5, replace=False).tolist())
    assert len(set(selected)) == 35 and not original.intersection(selected)
    return sorted(selected)


def frame_counts(manifest):
    return {r['id']: r['dataset_to_index'] - r['dataset_from_index']
            for r in read_json(Path(manifest['evaluator_directory']) / 'trajectory_index.json')}


def policy_specs(manifest, seeds=SEEDS):
    lengths = frame_counts(manifest)
    original = sorted(manifest['pretraining_ids'])
    original_frames = sum(lengths[i] for i in original)
    specs = []
    for seed in seeds:
        added = sampled_additions(manifest, seed)
        for condition, ids in [('original', original), ('plus35', sorted(original + added)),
                               ('supervisor', sorted(set(original) | set(manifest['target_ids'])))]:
            frames = sum(lengths[i] for i in ids)
            steps = math.ceil(BASE_STEPS * frames / original_frames)
            specs.append(dict(id=f'{condition}_seed{seed}', seed=seed, condition=condition,
                ids=ids, episode_indices=sorted(manifest['episode_map'][i] for i in ids),
                added_ids=added if condition == 'plus35' else [], frames=frames,
                steps=steps, batch_size=BATCH_SIZE, effective_epochs=steps*BATCH_SIZE/frames,
                target_epochs=BASE_STEPS*BATCH_SIZE/original_frames,
                warmup_steps=round(1000*frames/original_frames)))
    return specs


def evaluation_manifest(manifest):
    result = deepcopy(manifest)
    starts = manifest['evaluation_splits']['development'] + manifest['evaluation_splits']['final']
    unique = {(s['task'], s['init_state_hash']): s for s in starts}
    if len(unique) != 500 or set(Counter(s['task'] for s in unique.values()).values()) != {50}:
        raise ValueError('Follow-up requires exactly 50 unique conditions for each of ten tasks')
    result['evaluation_splits'] = {}
    for repetition in range(3):
        values = []
        for _, start in sorted(unique.items(), key=lambda item: (item[1]['task'], item[1]['init_state_id'])):
            row = dict(start, repetition=repetition)
            row['id'] = start['id'].rsplit('/rep_', 1)[0] + f'/rep_{repetition}'
            values.append(row)
        result['evaluation_splits'][f'study_rep{repetition}'] = values
    result.pop('manifest_hash', None)
    result['followup_source_manifest'] = manifest['manifest_hash']
    result['manifest_hash'] = identity(result)
    return result


def aggregate_executions(records, repetitions=3, *, canonical=None, pairing=None, policy_id=''):
    groups = defaultdict(list)
    for row in records:
        groups[(row['task'], row['init_state_hash'])].append(row)
    result = []
    for _, values in sorted(groups.items(), key=lambda item: (item[0][0], item[1][0]['init_state_id'])):
        if len(values) != repetitions or {v['repetition'] for v in values} != set(range(repetitions)):
            raise ValueError('Missing or duplicate execution repetition')
        values.sort(key=lambda v: v['repetition'])
        reference = canonical[(values[0]['task'], values[0]['init_state_hash'])] if canonical is not None else values[0]
        if canonical is not None and pairing is None:
            raise ValueError('Canonical observations require an observation-pairing audit')
        if pairing is not None:
            for value in values:
                pairing.validate(reference, value, policy_id=policy_id)
        elif len({v['initial_input']['input_hash'] for v in values}) != 1:
            raise ValueError('Repeated executions must start from identical observations')
        row = deepcopy(next(v for v in values if v['repetition'] == 0))
        if pairing is not None:
            # Keep the actual rollout inputs as provenance. All heads and paired
            # supervisors score the same unchanged reference image, not an average.
            row['execution_initial_inputs'] = [deepcopy(v['initial_input']) for v in values]
            row['initial_input'] = deepcopy(reference['initial_input'])
            row['observation_pairing_protocol'] = pairing.summary()['protocol']
        row['id'] = row['id'].rsplit('/rep_', 1)[0]
        row['initial_input']['id'] = row['id']
        for key in ('success', 'q_time', 'q_path', 'time_seconds', 'path_meters', 'steps'):
            nums = [v[key] for v in values]
            row[key] = float(np.mean(nums)) if all(v is not None for v in nums) else None
        row.update(repetition=0, split='study_all50_mean3', execution_ids=[v['id'] for v in values],
                   execution_successes=[v['success'] for v in values])
        result.append(row)
    return result


def quality_summary(outcomes):
    policies = {}
    for name, rows in outcomes.items():
        bytask = defaultdict(list)
        for r in rows:
            bytask[r['task']].append(r['success'])
        policies[name] = dict(success=float(np.mean([np.mean(v) for v in bytask.values()])),
            per_task={t: float(np.mean(v)) for t, v in sorted(bytask.items())}, conditions=len(rows))
    warnings = []
    for condition in (*CONDITIONS, 'supervisor'):
        names = [n for n in policies if n.startswith(condition+'_seed')]
        if names:
            rates = [policies[n]['success'] for n in names]
            if max(rates)-min(rates) > .25 or min(rates) == 0:
                warnings.append(f'{condition}: inspect seed success rates {dict(zip(names, rates))}')
            for task in policies[names[0]]['per_task']:
                values = [policies[n]['per_task'][task] for n in names]
                if max(values)-min(values) > .6:
                    warnings.append(f'{condition}, {task}: large per-task spread {dict(zip(names, values))}')
    return dict(policies=policies, warnings=warnings,
        interpretation='Investigate flagged policies; never automatically discard or retrain a weaker seed.')
