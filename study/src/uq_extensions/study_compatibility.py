"""Exact, audited compatibility for the append-only whitening study extension.

This is deliberately not a general switch for ignoring source/config changes.
The manifest pins both source revisions, both matrices, and eligible cache recipes.
"""
from pathlib import Path
from .common import identity, plain, read_json

COMPATIBILITY_PATH = Path(__file__).resolve().parents[2]/'configs/experiments/head_study_whitening_compatibility.json'
SCHEDULER_COMPATIBILITY_PATH = COMPATIBILITY_PATH.with_name('head_study_scheduler_compatibility.json')


def scheduler_record():
    if not SCHEDULER_COMPATIBILITY_PATH.exists():return None
    record=read_json(SCHEDULER_COMPATIBILITY_PATH)
    return record if record.get('kind')=='head_study_scheduler_v1' else None


def compatible_execution_source(actual,expected):
    if actual==expected:return True
    from .study_precision import compatibility_record as precision_record
    precision=precision_record()
    if precision and actual==precision['source_after']:
        actual=precision['source_before']
        if actual==expected:return True
    record=scheduler_record()
    return bool(record and record['source_after']==actual and record['base_source_hash']==expected)


def scheduler_declaration(stored,requested):
    """Bridge through the existing whitening approval without rewriting its provenance."""
    record=scheduler_record()
    if not record or requested.get('source_hash')!=record['source_after']:return None
    provenance=dict(kind=record['kind'],execution_source_hash=record['source_after'],
                    compatibility_manifest_hash=identity(record))
    if 'scheduler_update' in stored and stored['scheduler_update']!=provenance:return None
    previous={k:v for k,v in stored.items() if k!='scheduler_update'}
    expected=dict(requested,source_hash=record['base_source_hash'])
    accepted=previous if previous==expected else append_declaration(previous,expected)
    return dict(accepted,scheduler_update=provenance) if accepted is not None else None


def compatibility_record():
    if not COMPATIBILITY_PATH.exists():return None
    record=read_json(COMPATIBILITY_PATH)
    return record if record.get('kind')=='head_study_whitening_append_v1' else None


def append_declaration(stored, requested):
    record=compatibility_record()
    if not record or requested.get('source_hash')!=record['source_after']:return None
    if stored.get('source_hash') not in record['source_before']:return None
    before_hash=identity(stored.get('recipes',[]));after_hash=identity(requested.get('recipes',[]))
    if after_hash!=record['recipes_after_hash']:return None
    if before_hash not in (record['recipes_before_hash'],after_hash):return None
    count=record['original_recipe_count']
    if identity(requested['recipes'][:count])!=record['recipes_before_hash']:return None
    if stored['recipes']!=requested['recipes'][:len(stored['recipes'])]:return None
    ignored={'recipes','source_hash','append_only_extension'}
    if {k:v for k,v in stored.items() if k not in ignored}!={k:v for k,v in requested.items() if k not in ignored}:return None
    provenance=dict(kind=record['kind'],execution_source_hash=record['source_after'],
        original_recipe_count=count,appended_recipe_count=len(requested['recipes'])-count,
        recipes_hash=after_hash,compatibility_manifest_hash=identity(record))
    if 'append_only_extension' in stored and stored['append_only_extension']!=provenance:return None
    return dict(stored,recipes=requested['recipes'],append_only_extension=provenance)


def legacy_estimator_codes(spec):
    from .study_precision import compatibility_record as precision_record
    precision=precision_record()
    if precision and spec['code']==precision['estimator_code_after']:
        # Float32 KV algebra is unchanged. KV-T must get a fresh cache and Adam
        # state; never treat an old fit as a new float64 fit.
        from .runner import source_identity
        if spec['method']!='kv' or source_identity()!=precision['source_after']:return []
        previous=dict(spec,code=precision['estimator_code_before'])
        return [previous['code'],*legacy_estimator_codes(previous)]
    record=compatibility_record()
    if not record or spec['code']!=record['estimator_code_after']:return []
    scientific={k:plain(spec[k]) for k in ('method','features','gp')}
    if identity(scientific) not in record['legacy_estimator_config_hashes']:return []
    from .runner import source_identity
    if not compatible_execution_source(source_identity(),record['source_after']):return []
    return record['estimator_code_before']
