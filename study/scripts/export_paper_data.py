"""Freeze compact final-study tables for the manuscript repository.

Reads storage/extensions/final_save_v1 (report, evaluations, scores, runs, jobs) and
writes CSV/JSON tables to paper/data/final_save_v1. No model, simulator or GPU is used.
Every source file is hashed in the output manifest; rerunning overwrites only that
output directory.

    .venv/bin/python scripts/export_paper_data.py
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADS = ['h1', 'h2', 'h3', 'h4', 'vfd', 'gu', 'action_l2', 'ace', 'decu', 'vlm_token_entropy', 'vlm_perplexity']
ARMS = ['random', 'vfd', 'gu', 'h1', 'h2', 'h3', 'h4', 'h1_ivr']
SEEDS = [1000, 1001, 1002]
COMMON_ROUNDS = [0, 10]
ACQ_ROUNDS = [0, 5, 10, 15]
METRICS = ['global_nauc', 'within_nauc', 'neg_spearman', 'neg_pearson', 'benefit_spearman',
           'benefit_pearson', 'automation']


class Sources:
    def __init__(self, study):
        self.study = study; self.files = {}

    def read(self, path):
        path = Path(path); data = path.read_bytes()
        self.files[str(path.resolve().relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return json.loads(data) if path.suffix == '.json' else data.decode()



def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for row in rows: writer.writerow({k: row.get(k, '') for k in fields})


def fmt(value):
    return '' if value is None else repr(float(value)) if isinstance(value, float) else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=ROOT/'storage/extensions/final_save_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'paper/data/final_save_v1')
    args = parser.parse_args()
    study, out = args.study.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    src = Sources(study)
    summary = src.read(study/'report/summary.json')
    declaration = src.read(study/'declaration.json')
    manifest = json.loads(Path(declaration['manifest']).read_text())
    preparatory = set(manifest['pretraining_ids'])

    # 1. Head metrics per seed and round, exactly as computed by the study report.
    rows = list(csv.DictReader(src.read(study/'report/all_rounds.csv').splitlines()))
    write_csv(out/'head_metrics_by_seed.csv', rows, list(rows[0]))

    # 2. Seed means, sample SD and hierarchical 95% intervals (per round and round-averaged).
    rows = []
    for block, label in [(summary['per_round'], None), (summary['round_averaged'], 'mean_0_10')]:
        for entry in block:
            rnd = label or str(entry['round'])
            for metric, value in entry['metrics'].items():
                rows.append(dict(head=entry['head'], round=rnd, metric=metric, seeds=len(entry['seeds']),
                                 mean=fmt(value['mean']), seed_sd=fmt(value['seed_std']),
                                 eval_lo=fmt(value['evaluation_ci95'][0]), eval_hi=fmt(value['evaluation_ci95'][1]),
                                 total_lo=fmt(value['total_ci95'][0]), total_hi=fmt(value['total_ci95'][1])))
    write_csv(out/'head_metrics_summary.csv', rows, list(rows[0]))

    rows = []
    for entry in summary['paired_differences']:
        for metric, value in entry['metrics'].items():
            rows.append(dict(head=entry['head'], baseline=entry['baseline'], rounds='0,10', metric=metric,
                             mean=fmt(value['mean']), seed_sd=fmt(value['seed_std']),
                             eval_lo=fmt(value['evaluation_ci95'][0]), eval_hi=fmt(value['evaluation_ci95'][1]),
                             total_lo=fmt(value['total_ci95'][0]), total_hi=fmt(value['total_ci95'][1])))
    write_csv(out/'paired_differences.csv', rows, list(rows[0]))

    # 3. Acquisition: success per arm/seed/round (overall and per task) and selections.
    acq_rows, sel_rows = [], []
    for arm in ARMS:
        for seed in SEEDS:
            seen = set()
            for rnd in range(0, 16):
                directory = study/f'runs/{arm}/seed_{seed}/round_{rnd:03d}'
                if rnd in ACQ_ROUNDS:
                    # All arms share the starting policy; its evaluation is recorded under Random.
                    reference = src.read((study/f'runs/random/seed_{seed}/round_000' if rnd == 0 else directory)/'reference.json')
                    path = Path(reference['source'])
                    if not path.resolve().is_relative_to(ROOT/'storage'):
                        raise ValueError(f'Reference outside research storage: {path}')
                    evaluation = src.read(path)
                    row = dict(arm=arm, seed=seed, round=rnd, query_slots=5*rnd,
                               success=fmt(evaluation['macro_avg_success_rate']),
                               rollouts_per_task=evaluation['n_rollouts_per_task'], provenance=reference['provenance'])
                    for task in evaluation['per_task']:
                        row[f'task_{task["task_id"]}'] = fmt(task['success_rate'])
                    acq_rows.append(row)
                if rnd == 0: continue
                selection = src.read(directory/'selection.json')
                for rank, demo in enumerate(selection['selected_ids']):
                    sel_rows.append(dict(arm=arm, seed=seed, round=rnd, rank=rank, demo_id=demo,
                                         task=demo.split('/')[1], preparatory=int(demo in preparatory),
                                         previously_acquired=int(demo in seen),
                                         score=fmt(selection.get('scores', {}).get(demo))))
                    seen.add(demo)
    write_csv(out/'acquisition_success.csv', acq_rows,
              ['arm', 'seed', 'round', 'query_slots', 'success', *[f'task_{t}' for t in range(10)],
               'rollouts_per_task', 'provenance'])
    write_csv(out/'acquisition_selections.csv', sel_rows,
              ['arm', 'seed', 'round', 'rank', 'demo_id', 'task', 'preparatory', 'previously_acquired', 'score'])

    # 4. Per-round acquisition diagnostics from each arm's own candidate scores (scored arms only).
    diag_rows = []
    for arm in ['vfd', 'gu', 'h1', 'h2', 'h3', 'h4', 'h1_ivr']:
        for seed in SEEDS:
            for rnd in range(1, 16):
                directory = study/f'runs/{arm}/seed_{seed}/round_{rnd:03d}'
                selection = src.read(directory/'selection.json'); scores = selection['scores']
                following = src.read(study/f'runs/{arm}/seed_{seed}/round_{rnd+1:03d}/selection.json')['scores'] if rnd < 15 else {}
                by_task = collections.defaultdict(list)
                for demo, value in scores.items(): by_task[demo.split('/')[1]].append(value)
                row = dict(arm=arm, seed=seed, round=rnd)
                for task in range(10):
                    values = by_task[str(task)]
                    row[f'task_mean_{task}'] = fmt(sum(values)/len(values))
                    row[f'task_max_{task}'] = fmt(max(values))
                for rank, demo in enumerate(selection['selected_ids']):
                    row[f'selected_score_{rank}'] = fmt(scores[demo])
                    row[f'selected_next_score_{rank}'] = fmt(following.get(demo))
                if 'audit' in selection and 'steps' in selection['audit']:
                    for rank, step in enumerate(selection['audit']['steps']):
                        row[f'ivr_gain_{rank}'] = fmt(step['gain'])
                diag_rows.append(row)
    fields = ['arm', 'seed', 'round', *[f'task_{k}_{t}' for k in ('mean', 'max') for t in range(10)]]
    fields = ['arm', 'seed', 'round', *[f'task_mean_{t}' for t in range(10)], *[f'task_max_{t}' for t in range(10)],
              *[f'selected_score_{r}' for r in range(5)], *[f'selected_next_score_{r}' for r in range(5)],
              *[f'ivr_gain_{r}' for r in range(5)]]
    write_csv(out/'acquisition_diagnostics.csv', diag_rows, fields)

    # 5. Common-policy conditions: per-execution outcomes and every head's raw score.
    cond_rows, pool_rows = [], []
    supervisors = {seed: {c['id']: c for c in src.read(study/f'evaluations/supervisor_{seed}/conditions.json')} for seed in SEEDS}
    for seed in SEEDS:
        for rnd in COMMON_ROUNDS:
            pid = f'random_{seed}_r{rnd:02d}'
            learner = src.read(study/f'evaluations/{pid}/conditions.json')
            scores = {}
            for head in HEADS:
                record = src.read(study/f'scores/{pid}/{head}/scores.json')
                scores[head] = dict(zip(record['ids'], record['scores']))
                for task, value in sorted(record['pool_task_means'].items()):
                    pool_rows.append(dict(seed=seed, round=rnd, head=head, task=task, pool_mean=fmt(value)))
            for c in learner:
                s = supervisors[seed][c['id']]
                if s['init_state_hash'] != c['init_state_hash'] or s['initial_input']['input_hash'] != c['initial_input']['input_hash']:
                    raise ValueError(f'Unpaired condition {c["id"]}')
                row = dict(seed=seed, round=rnd, condition=c['id'], task=c['task'], init_state_id=c['init_state_id'],
                           primary=fmt(c['success']), supervisor=fmt(s['success']))
                for i, value in enumerate(c['execution_successes']): row[f'primary_rep{i}'] = fmt(float(value))
                for i, value in enumerate(s['execution_successes']): row[f'supervisor_rep{i}'] = fmt(float(value))
                for head in HEADS: row[head] = fmt(scores[head][c['id']])
                cond_rows.append(row)
    write_csv(out/'conditions.csv', cond_rows,
              ['seed', 'round', 'condition', 'task', 'init_state_id', 'primary', 'supervisor',
               *[f'primary_rep{i}' for i in range(3)], *[f'supervisor_rep{i}' for i in range(3)], *HEADS])
    write_csv(out/'pool_task_means.csv', pool_rows, ['seed', 'round', 'head', 'task', 'pool_mean'])
    counts = collections.Counter(i.split('/')[1] for i in manifest['pretraining_ids'] if i.startswith('libero_10/'))
    pool = collections.Counter(i.split('/')[1] for i in manifest['target_ids'])
    write_csv(out/'libero10_pools.csv', [dict(task=t, preparatory_demos=counts.get(str(t), 0), candidate_pool=pool[str(t)])
                                         for t in range(10)], ['task', 'preparatory_demos', 'candidate_pool'])

    # 6. Measured job costs (wall-clock seconds of completed jobs, one GPU per job).
    cost_rows = []
    pattern = re.compile(r'(acquire|train|score|eval|collect|reference)_(\w+?)_(\d{4})_(?:r)?(\d+)(?:_(.+))?$')
    for job in sorted(os.listdir(study/'jobs')):
        match = pattern.match(job)
        complete = study/'jobs'/job/'complete.json'
        if not match or not complete.exists(): continue
        record = src.read(complete)
        if record.get('seconds') is None: continue
        kind, arm, seed, rnd, rest = match.groups()
        memory = study/'jobs'/job/'gpu_memory.json'
        gpu = src.read(memory) if memory.exists() else {}
        cost_rows.append(dict(job=job, kind=kind, arm=arm, head=rest or '', seed=seed, round=int(rnd),
                              seconds=fmt(record['seconds']), device=gpu.get('device', ''),
                              peak_reserved_gib=fmt(gpu['peak_reserved_bytes']/2**30) if gpu else ''))
    write_csv(out/'job_costs.csv', cost_rows, ['job', 'kind', 'arm', 'head', 'seed', 'round', 'seconds', 'device', 'peak_reserved_gib'])

    # 7. Exploratory head studies (supplement only): within-task referral of every configuration.
    exploratory = []
    protocols = {'original_study': 'one policy pair; 20 conditions per task, one execution each; 30,000-update supervisor',
                 'followup_study': 'three policies per condition; 50 conditions per task, three executions each; 39,993-update supervisors'}
    for name in protocols:
        path = ROOT/f'shared_results/{name}/report/configuration_summary.csv'
        for r in csv.DictReader(src.read(path).splitlines()):
            exploratory.append(dict(study=name, condition=r.get('condition') or f"round_{r['round']}", head=r['head'],
                                    label=r['label'], method=r['method'], group=r['group'], seed_coverage=r['seed_coverage'],
                                    **{k: r[k] for k in ('within_nauc', 'within_nauc_total_low', 'within_nauc_total_high',
                                                         'global_nauc', 'neg_spearman', 'neg_pearson')}))
    write_csv(out/'exploratory_head_studies.csv', exploratory, list(exploratory[0]))

    # 8. Recipes and provenance.
    (out/'recipes.json').write_text(json.dumps({k: declaration['recipes'][k] for k in ('h1', 'h2', 'h3', 'h4')}, indent=1, sort_keys=True)+'\n')
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', 'src', 'scripts'], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob('*')) if p.name != 'manifest.json'}
    (out/'manifest.json').write_text(json.dumps(dict(
        exported_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        study='storage/extensions/final_save_v1', report_note=summary.get('note'), progress=summary.get('progress'),
        declaration_source_hash=declaration.get('source_hash'), research_repository_head=revision,
        exploratory_protocols=protocols,
        research_source_has_uncommitted_changes=dirty, exporter='scripts/export_paper_data.py',
        outputs=outputs, sources=src.files), indent=1, sort_keys=True)+'\n')
    print(f'Wrote {len(outputs)} tables to {out}; {len(src.files)} source files hashed.')


if __name__ == '__main__':
    main()
