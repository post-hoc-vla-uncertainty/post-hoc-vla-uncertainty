#!/usr/bin/env python3
"""Audited, restartable migration from every-round to sparse evaluations.

Run only after stopping the study. --snapshot must contain the original
declaration.json and source/*.py captured before editing the scheduling code.
Weights, optimizer states, feature batches, fits and measured outputs are untouched.
"""
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
from pathlib import Path
import time

from uq_extensions.common import atomic_json, file_hash, identity, read_json
from uq_finalstudy.design import (ACQUISITION_EVALUATION_ROUNDS, COMMON_EVALUATION_ROUNDS,
    COLLABORATOR_SOURCE, jobs, recipes, source_hash)
from uq_finalstudy.launcher import completed, validate_graph


def verify_source(project, snapshot, original):
    from uq_followup.launcher import source_hash as library_hash
    project, snapshot = Path(project), Path(snapshot)
    package = project/'src/uq_finalstudy'
    current = {p.name:p for p in package.glob('*.py')}
    saved = {p.name:p for p in (snapshot/'source').glob('*.py')}
    if set(current) != set(saved):
        raise RuntimeError('Source file inventory changed; schedule-only migration refused')
    for name, path in current.items():
        if name not in ('design.py', 'launcher.py', 'reporting.py') and file_hash(path) != file_hash(saved[name]):
            raise RuntimeError(f'Numerical source changed: {name}')
    external = [project/COLLABORATOR_SOURCE,
                project/'upstream/uq_vla_extensions/scripts/calibration/calibration_comparison.py']
    external += sorted((project/'upstream/uq_vla_extensions/src/iterative_fine_tuning').glob('*.py'))
    files = {str((package/name).relative_to(project)):file_hash(path)
             for name,path in saved.items() if name != 'reporting.py'}
    files.update({str(path.relative_to(project)):file_hash(path) for path in external})
    if identity(dict(existing=library_hash(project), files=files)) != original['source_hash']:
        raise RuntimeError('Original scientific source cannot be verified against the snapshot')
    if original['recipes'] != recipes():
        raise RuntimeError('Head recipes changed; schedule-only migration refused')


def build_plan(root, snapshot):
    original = read_json(snapshot/'declaration.json')
    if read_json(root/'declaration.json') != original or original.get('version') != 1:
        raise RuntimeError('Expected the unchanged version-1 declaration')
    verify_source(original['project'], snapshot, original)
    old_jobs = jobs(root, original)
    if old_jobs != read_json(root/'job_graph.json') or old_jobs != read_json(snapshot/'job_graph.json'):
        raise RuntimeError('Original dependency graph changed')
    revised = deepcopy(original)
    revised.update(version=2, source_hash=source_hash(original['project']),
        acquisition_evaluation_rounds=list(ACQUISITION_EVALUATION_ROUNDS),
        common_evaluation_rounds=list(COMMON_EVALUATION_ROUNDS),
        common_policy='Random, rounds 0 and 10; retained historical models reused')
    new_jobs = jobs(root, revised); validate_graph(new_jobs)
    old_map = {j['id']:j for j in old_jobs}
    writes = []; done = []; identities = 0
    def change(path, after):
        before = read_json(path) if path.exists() else None
        if before != after:
            writes.append(dict(path=str(path.relative_to(root)), before=before, after=after))
    for job in new_jobs:
        if old_map.get(job['id']) != job:
            raise RuntimeError(f'Retained job semantics changed: {job["id"]}')
        directory = root/'jobs'/job['id']; path = directory/'request.json'
        if completed(job, root): done.append(job['id'])
        old_spec = dict(study=str(root), declaration=identity(original), job=job)
        old_request = dict(identity=identity(old_spec), spec=old_spec)
        if path.exists() and read_json(path) != old_request:
            raise RuntimeError(f'Original request differs: {job["id"]}')
        spec = dict(study=str(root), declaration=identity(revised), job=job)
        request = dict(identity=identity(spec), spec=spec)
        change(path, request)
        marker = directory/'complete.json'
        if marker.exists():
            record = read_json(marker)
            if record['identity'] != old_request['identity'] or not completed(job, root):
                raise RuntimeError(f'Invalid original completion: {job["id"]}')
            record['identity'] = request['identity']; change(marker, record); identities += 1
    new_ids = {j['id'] for j in new_jobs}
    change(root/'job_graph.json', new_jobs)
    change(root/'declaration.json', revised)
    change(root/'queue.json', dict(total=len(new_jobs), complete=len(done), failed={}, running=[],
        pending=[j['id'] for j in new_jobs if j['id'] not in done], updated_at=time.time()))
    return dict(version=1, study=str(root), old_source_hash=original['source_hash'],
        new_source_hash=revised['source_hash'], new_declaration_hash=identity(revised),
        before_job_counts=dict(Counter(j['kind'] for j in old_jobs)),
        after_job_counts=dict(Counter(j['kind'] for j in new_jobs)),
        preserved_completed_jobs=done, rebound_completion_records=identities,
        omitted_jobs=[j['id'] for j in old_jobs if j['id'] not in new_ids], writes=writes)


def migrate(root, snapshot):
    root, snapshot = Path(root).resolve(), Path(snapshot).resolve()
    with (root/'.launcher.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Stop the running final study before migrating')
        plan_path = snapshot/'migration_plan.json'
        if plan_path.exists(): plan = read_json(plan_path)
        else:
            plan = build_plan(root, snapshot)
            atomic_json(plan_path, plan)  # Journal old and new metadata before writing anything.
        original = read_json(snapshot/'declaration.json')
        verify_source(original['project'], snapshot, original)
        if plan['study'] != str(root) or source_hash(original['project']) != plan['new_source_hash']:
            raise RuntimeError('Migration target/source changed')
        # Validate the entire journal first. This permits resuming a partially applied migration.
        for item in plan['writes']:
            path = root/item['path']
            current = read_json(path) if path.exists() else None
            if current != item['before'] and current != item['after']:
                raise RuntimeError(f'Metadata changed during migration: {path}')
        for item in plan['writes']:
            atomic_json(root/item['path'], item['after'])
        audit = {k:v for k,v in plan.items() if k != 'writes'}
        audit['completed_at'] = time.time()
        atomic_json(snapshot/'complete.json', audit)
        print(f'Migrated {root}: {plan["after_job_counts"]}; '
              f'preserved {len(plan["preserved_completed_jobs"])} completed/imported jobs. '
              f'Audit: {snapshot}', flush=True)
        return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('storage/extensions/final_save_v1'))
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    migrate(args.output, args.snapshot)
