#!/usr/bin/env python3
"""Finish the current velocity fits and one final-action recipe at both checkpoints.

Scheduling-only wrapper: preserve historical declarations, requests, numerical
source identities, optimizer checkpoints, and every existing result. Never stop
another launcher or launch the follow-up study automatically.
"""
import argparse
import json
from pathlib import Path
import sys


VELOCITY = 'action_context__kvt_pls_adaptive32_s42_pca16'
FINAL_ACTION = 'final_action__kvt_pls_adaptive32_s42_pca16'
SELECTED = (VELOCITY, FINAL_ACTION)


def select_recipes(recipes):
    by_id = {row['id']: row for row in recipes}
    missing = set(SELECTED) - by_id.keys()
    if missing:
        raise ValueError(f'Selected recipes are absent from this study: {sorted(missing)}')
    # Keep the original recipe dictionaries, including their scientific identities.
    return [by_id[id_] for id_ in SELECTED]


def selected_readiness(request, original_ready):
    if request['spec']['recipe']['id'] == FINAL_ACTION:
        # The old PLS32 prerequisite only avoided redundant adaptive fits when
        # 32 already attained 99%. Adaptive PLS independently checks 32/128/512
        # on training data. It can run directly without a separate fixed32 GP.
        return 'ready'
    return original_ready(request)


def finish_in_order(jobs, slots, original_queue, **kwargs):
    """Drain both current velocity fits before starting either final-action fit."""
    if any(j['spec']['recipe']['id'] not in SELECTED for j in jobs):
        raise ValueError('Unexpected job in restricted finish queue')
    failures = []
    for index, name in enumerate(SELECTED):
        phase = [j for j in jobs if j['spec']['recipe']['id'] == name]
        later = [j for j in jobs if j['spec']['recipe']['id'] in SELECTED[index+1:]]
        options = dict(kwargs)
        on_update = options['on_update']
        options['on_update'] = lambda running, pending: on_update(running, pending + later)
        failures.extend(original_queue(phase, slots, **options))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', default='4,5', help='Selected A100 GPU indices or UUIDs')
    parser.add_argument('--root', type=Path, default=Path('storage/extensions/head_ablation_v1'))
    parser.add_argument('--plan', action='store_true', help='Show selected jobs; do not launch or modify the study')
    args = parser.parse_args()
    root = args.root.resolve()
    declaration = json.loads((root/'declaration.json').read_text())
    if declaration['rounds'] != [0, 10]:
        raise ValueError('This finish plan expects the existing round-0/round-10 study')
    select_recipes(declaration['recipes'])
    print('Restricted finish: adaptive PLS (99% target, cap 512), PCA16 targets.\n'
          'First: context + noisy action → velocity at rounds 0 and 10.\n'
          'Then: context + initial noise → final action chunk at rounds 0 and 10.\n'
          'Other unfinished heads remain pending; completed selected heads are reused.', flush=True)
    for name in SELECTED:
        for round_ in declaration['rounds']:
            path = root/f'round_{round_:03d}'/name/'result.json'
            status = json.loads(path.read_text())['status'] if path.exists() else 'unfinished (resume saved fit if present)'
            print(f'  round {round_:2d} {name}: {status}', flush=True)
    if args.plan:
        return
    from uq_extensions import head_study
    original_jobs = head_study.queued_jobs
    original_ready = head_study.job_ready
    original_queue = head_study.run_queue

    def restricted_jobs(root, recipes, *arguments, **options):
        return original_jobs(root, select_recipes(recipes), *arguments, **options)

    head_study.queued_jobs = restricted_jobs
    head_study.job_ready = lambda request: selected_readiness(request, original_ready)
    head_study.run_queue = lambda jobs, slots, **options: finish_in_order(jobs, slots, original_queue, **options)
    argv = sys.argv
    try:
        sys.argv = [argv[0], '--gpus', args.gpus, '--root', str(root)]
        # Existing study lock prevents overlap; normal GPU routing, identity
        # verification, failure isolation and graceful interruption remain active.
        head_study.main()
    finally:
        sys.argv = argv
        head_study.queued_jobs = original_jobs
        head_study.job_ready = original_ready
        head_study.run_queue = original_queue


if __name__ == '__main__':
    main()
