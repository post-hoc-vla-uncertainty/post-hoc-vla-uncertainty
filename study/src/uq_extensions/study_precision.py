"""Audited, per-cell precision migration without relabeling completed results."""
from copy import deepcopy
from pathlib import Path

from .common import atomic_json, identity, read_json

COMPATIBILITY_PATH = Path(__file__).resolve().parents[2] / 'configs/experiments/head_study_precision_compatibility.json'
KIND = 'head_study_kvt_float64_v1'


def compatibility_record():
    if not COMPATIBILITY_PATH.exists():return None
    record = read_json(COMPATIBILITY_PATH)
    return record if record.get('kind') == KIND else None


def migration_record(root):
    path = Path(root) / 'precision_migration.json'
    if not path.exists():return None
    record = read_json(path)
    compatibility = compatibility_record()
    if (record.get('kind') != KIND or not compatibility
            or record.get('compatibility_hash') != identity(compatibility)):
        raise RuntimeError('Unrecognized precision migration; preserving existing study')
    return record


def resume_precision_declaration(root, stored, requested):
    migration = migration_record(root)
    if not migration:return None
    compatibility = compatibility_record()
    if requested.get('source_hash') != compatibility['source_after']:return None
    original = {k:v for k,v in stored.items() if k != 'precision_update'}
    if identity(original) != migration['declaration_before_hash']:return None
    from .study_compatibility import scheduler_declaration
    previous_request = dict(requested, source_hash=compatibility['source_before'])
    accepted = original if original == previous_request else scheduler_declaration(original, previous_request)
    if accepted != original:return None
    provenance = dict(kind=KIND, execution_source_hash=compatibility['source_after'],
                      migration_hash=identity(migration), cells=len(migration['cells']))
    if stored.get('precision_update', provenance) != provenance:return None
    return dict(original, precision_update=provenance)


def precision_request(root, request):
    """Only cells explicitly unfinished at migration acquire a new fit identity."""
    migration = migration_record(root)
    if not migration:return request
    spec = request['spec']
    key = f'round_{spec["round"]:03d}/{spec["recipe"]["id"]}'
    cell = migration['cells'].get(key)
    if cell is None:return request
    if request['identity'] != cell['request_before_identity']:
        raise RuntimeError(f'Precision migration input changed: {key}')
    if spec['recipe']['method'] != 'kv_t' or spec['recipe']['gp']['output_dependence'] != 'independent':
        raise RuntimeError('Precision migration is restricted to independent KV-T')
    updated = deepcopy(spec)
    updated['recipe']['gp']['precision'] = 'float64'
    updated['source_hash'] = compatibility_record()['source_after']
    updated['precision_migration'] = KIND
    return dict(identity=identity(updated), spec=updated)


def archive_previous_request(root, request):
    """Keep the old request and cache pointers before publishing the new request."""
    if request['spec'].get('precision_migration') != KIND:return
    out = Path(request['spec']['directory'])
    previous = out / 'request.json'
    if not previous.exists():return
    old = read_json(previous)
    if old == request:return
    key = f'round_{request["spec"]["round"]:03d}/{request["spec"]["recipe"]["id"]}'
    cell = migration_record(root)['cells'][key]
    if old['identity'] != cell['request_before_identity'] or identity(old['spec']) != old['identity']:
        raise RuntimeError(f'Cannot archive unexpected request: {key}')
    archive = Path(root) / 'precision_archive' / key
    for relative in ('request.json', 'failure.json', 'kv_t/estimator/cache_link.json', 'panel.json'):
        path = out / relative
        if path.exists() and not (archive / relative).exists():
            atomic_json(archive / relative, read_json(path))


def validate_worker_request(root, request):
    migration = migration_record(root)
    if not migration:return
    from .runner import source_identity
    if source_identity() != compatibility_record()['source_after']:
        raise RuntimeError('Precision worker source differs from the audited implementation')
    spec = request['spec']
    key = f'round_{spec["round"]:03d}/{spec["recipe"]["id"]}'
    if key not in migration['cells']:return
    original = deepcopy(spec)
    original.pop('precision_migration', None)
    original['recipe']['gp']['precision'] = 'float32'
    original['source_hash'] = migration['request_source_before']
    expected = precision_request(root, dict(identity=identity(original), spec=original))
    if request != expected:
        raise RuntimeError('Unfinished KV-T must use its new float64 request')


def assert_worker_hardware(request, device):
    if request['spec'].get('precision_migration') != KIND:return
    import torch
    if not str(device).startswith('cuda') or 'A100' not in torch.cuda.get_device_name(device).upper():
        raise RuntimeError('Migrated float64 KV-T is restricted to NVIDIA A100 GPUs')
