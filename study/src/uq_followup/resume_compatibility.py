"""Narrow reuse of completed pre-head jobs after an audited pairing repair."""
from .observation_pairing import PAIRING_PROTOCOL
from uq_extensions.common import identity, read_json


def reusable_completed_request(root, previous, current):
    path = root/'source_migration.json'
    if not path.exists():
        return False
    migration = read_json(path)
    old, new = previous['spec'], current['spec']
    if (migration.get('protocol') != PAIRING_PROTOCOL or new['kind'] not in ('train', 'evaluate')
            or old.get('source_hash') != migration.get('from_source_hash')
            or new.get('source_hash') != migration.get('to_source_hash')):
        return False
    if previous['identity'] != identity(old) or current['identity'] != identity(new):
        return False
    if dict(old, source_hash=new['source_hash']) != new:
        return False
    # Only the specific completed jobs inspected by the repair are reusable.
    return migration.get('completed_requests', {}).get(new['id']) == previous['identity']
