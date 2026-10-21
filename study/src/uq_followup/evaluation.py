"""Three noise repetitions, shared cached outcomes and policy-quality screening."""
from pathlib import Path
from uq_extensions.common import atomic_json, read_json
from uq_extensions.evaluation import evaluate
from .protocol import aggregate_executions, quality_summary


def execute(study, policy_id, repetition, *, device, batch_size=15):
    study = Path(study)
    manifest = read_json(study/'evaluation_manifest.json')
    model = read_json(study/'policies'/policy_id/'model.json')['model']
    return evaluate(model, manifest, study/'evaluations'/policy_id/f'rep_{repetition}',
        split=f'study_rep{repetition}', device=device, batch_size=batch_size, max_steps=520)


def collect(study, policy_specs):
    from .observation_pairing import ObservationPairing
    study = Path(study); outcomes = {}
    reference_policy = min((p for p in policy_specs if p['condition'] == 'original'), key=lambda p: p['seed'])['id']
    canonical_rows = read_json(study/'evaluations'/reference_policy/'rep_0/evaluation.json')['episodes']
    canonical = {(r['task'], r['init_state_hash']): r for r in canonical_rows}
    if len(canonical_rows) != 500 or len(canonical) != 500:
        raise ValueError('Expected 500 unique canonical starting conditions')
    pairing = ObservationPairing(reference_policy)
    for policy in policy_specs:
        if policy['condition'] == 'companion': continue
        directory = study/'evaluations'/policy['id']
        records = []
        for repetition in range(3):
            records.extend(read_json(directory/f'rep_{repetition}/evaluation.json')['episodes'])
        if {(r['task'], r['init_state_hash']) for r in records} != set(canonical):
            raise ValueError(f'Policy evaluated different starting conditions: {policy["id"]}')
        rows = aggregate_executions(records, canonical=canonical, pairing=pairing, policy_id=policy['id'])
        if len(rows) != 500: raise RuntimeError('Expected 500 evaluated conditions per policy')
        outcomes[policy['id']] = rows
        print(f'Observation audit: {policy["id"]}; 1,500 executions checked', flush=True)
    # Publish derived inputs only after every policy/repetition passed the audit.
    # Raw episode JSON, evaluation JSON and observation NPZ files remain untouched.
    for policy_id, rows in outcomes.items():
        directory = study/'evaluations'/policy_id
        atomic_json(directory/'conditions.json', rows)
        atomic_json(directory/'initial_inputs.json', [r['initial_input'] for r in rows])
    atomic_json(study/'observation_pairing.json', pairing.summary())
    quality = quality_summary(outcomes)
    quality['supervisor_advantage'] = {
        policy['id']: quality['policies'][f'supervisor_seed{policy["seed"]}']['success'] - quality['policies'][policy['id']]['success']
        for policy in policy_specs if policy['condition'] in ('original', 'plus35')}
    atomic_json(study/'policy_quality.json', quality)
    lines = ['# Policy quality before uncertainty-head fitting', '',
             '| Policy | Overall success |', '|---|---:|']
    for name, row in quality['policies'].items(): lines.append(f'| {name} | {row["success"]:.1%} |')
    lines += ['', 'Per-task success and paired supervisor advantages: `policy_quality.json`.', '',
              'Initial observations: exact state/instruction/reset metadata; RGB render-roundoff audit in `observation_pairing.json`.',
              f'All heads score the saved repetition-0 observations from `{reference_policy}`; raw rollout inputs remain unchanged.', '',
              'Warnings trigger investigation; no policy is automatically excluded or retrained.']
    lines += ['- '+w for w in quality['warnings']]
    (study/'policy_quality.md').write_text('\n'.join(lines)+'\n')
    return quality
