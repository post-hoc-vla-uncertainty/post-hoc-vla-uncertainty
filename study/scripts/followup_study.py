#!/usr/bin/env python3
"""Head-first scheduling and on-demand reports, outside scientific run identity."""
from pathlib import Path
import sys

from uq_extensions.common import atomic_json, file_hash, read_json


def head_family(recipe):
    # Match the report's coordinate-seed aggregation exactly.
    return recipe['id'].replace('_s42', '').replace('_s43', '')


def head_first(jobs, declaration):
    """Keep each family's datasets, policy seeds and random-feature subsets close.

    No barriers between families: the dispatcher still fills any idle compatible
    GPU. Observation and action phases remain separate in the original launcher.
    """
    ranks = {}
    for recipe in declaration['recipes']:
        ranks.setdefault(head_family(recipe), len(ranks))
    policies = {p['id']: p for p in declaration['policies']}
    def key(job):
        spec = job['request']['spec']; recipe = spec['recipe']; policy = policies[spec['policy_id']]
        return (ranks[head_family(recipe)], ('original', 'plus35').index(policy['condition']),
                policy['seed'], recipe['id'])
    return sorted((j for j in jobs if j is not None), key=key)


def run(forwarded):
    from uq_followup import launcher, reporting
    argv = sys.argv
    if '--report-only' in forwarded:
        try:
            sys.argv = [argv[0], *forwarded]; launcher.main()
        finally: sys.argv = argv
        return

    original_drain, original_report = launcher.drain, reporting.aggregate_report

    def drain(root, jobs, slots, stage):
        if stage not in ('observation-heads', 'action-heads'):
            return original_drain(root, jobs, slots, stage)
        root = Path(root).resolve(); declaration = read_json(root/'declaration.json')
        jobs = head_first(jobs, declaration)
        atomic_json(root/'scheduling.json', dict(order='head_family, condition, policy_seed, coordinate_seed',
            action_heads_last=True, gpu_routing='unchanged', source=str(Path(__file__).resolve()),
            scheduler_hash=file_hash(__file__), aggregate_report='on_request_only'))
        print('Queue order: head → training dataset → policy-training seed → random-feature subset. '
              'Completed jobs are skipped; compatible GPUs remain busy. '
              'The combined report is regenerated only with --report-only.', flush=True)
        return original_drain(root, jobs, slots, stage)

    def report_on_request(root):
        # The scientific launcher also asks for reports at phase boundaries.
        # Disable those calls here without changing its pinned source identity.
        return 'not regenerated automatically; use --report-only to update the combined report'

    try:
        launcher.drain, reporting.aggregate_report = drain, report_on_request
        sys.argv = [argv[0], *forwarded]; launcher.main()
    finally:
        sys.argv = argv
        launcher.drain, reporting.aggregate_report = original_drain, original_report


if __name__ == '__main__': run(sys.argv[1:])
