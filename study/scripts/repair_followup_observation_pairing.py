#!/usr/bin/env python3
"""CPU-only repair of the known pre-head render-roundoff aggregation failure.

Audit every observation, rebuild derived condition averages, and record explicit
source compatibility for completed policy/evaluation jobs. Never launch GPU work
or relabel the source identities of the original requests/completions.
"""
import argparse
import ast
import fcntl
from pathlib import Path
import time

from uq_extensions.common import atomic_json, file_hash, identity, model_identity, read_json
from uq_extensions.runner import source_identity
from uq_followup.launcher import source_hash
from uq_followup.observation_pairing import PAIRING_PROTOCOL

OLD_SOURCE = '630455f27545c4b4d30daba8119d2d9020c3fca42b7c86af67bb1919401f5fdf'
OLD_LIBRARY = 'a1d1f3591a4119a9987eb6634b1e2b994215d78faac869baff274039dcb5b376'
UNCHANGED_FILES = {
    '__init__.py': '58e1775c8284beee34e496a7115fa161d7b35da6b412463db90f67602444b0a3',
    'design.py': 'ddfc31fc9a48455e53625b09509ea6cac9cd860ea3c698ac357a23a74f4923e9',
    'estimators.py': 'b210fb14fc99dc3a644ca9e5c24b576d389755c799037fc8d9e3fa532b9e3ad5',
    'features.py': '0424635240ff941d553ad435e9848bb471c5eae1a8702eafa954fdfc02ab50ff',
    'policies.py': 'bd2f4458e34e29aabaefcd7dafb0833276109a8fa00a3adbc88434dc226c61a5',
    'scoring.py': '94656f46478efbc44b20c81f597955d224f92157fa5d351fc82429b7cfa4cc38',
    'worker.py': 'dee04d33cf7bf482a741bccdf501c34a291414f980a2b40cf170f99b06238195',
}
PRESERVED_AST = {
    'protocol.py': ('aggregate_executions', '1c3eecad5a1e9fd7d351adcb8b2d783563d5e58c0188b1523bebd6c7caa64ee2'),
    'evaluation.py': ('collect', 'ee22d56db3c5964bf123f9aefac70701e8087b0920b1daf7cf5f62ee28bf4b9e'),
}


def verify_compatible_science(project):
    """Require the original training, data selection, rollout and GP code."""
    if source_identity() != OLD_LIBRARY:
        raise RuntimeError('Underlying scientific library changed; this repair cannot certify reuse')
    for name, expected in UNCHANGED_FILES.items():
        if file_hash(project/'src/uq_followup'/name) != expected:
            raise RuntimeError(f'Unrelated scientific change in {name}; refusing migration')
    for name, (allowed_function, expected) in PRESERVED_AST.items():
        tree = ast.parse((project/'src/uq_followup'/name).read_text())
        tree.body = [n for n in tree.body if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == allowed_function)]
        if identity(ast.dump(tree, include_attributes=False)) != expected:
            raise RuntimeError(f'Changes outside {allowed_function} in {name}; refusing migration')


def repair(study):
    study = Path(study).resolve()
    with (study/'study.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Stop the study launcher before repairing its derived observations')
        declaration = read_json(study/'declaration.json')
        project = Path(declaration['project'])
        current = source_hash(project)
        transition_path = study/'source_migration.json'
        if transition_path.exists():
            transition = read_json(transition_path)
            if (transition.get('protocol') == PAIRING_PROTOCOL and transition.get('to_source_hash') == current
                    and transition.get('to_declaration_hash') == identity(declaration)):
                print('Pairing repair already applied; saved work remains reusable.', flush=True)
                return
            # A crash after writing the transition but before declaration commit
            # can safely repeat the audit while the original declaration remains.
            if not (transition.get('from_declaration_hash') == identity(declaration)
                    and transition.get('to_source_hash') == current):
                raise RuntimeError('A different source migration exists; refusing to overwrite it')
        if declaration['source_hash'] != OLD_SOURCE:
            raise RuntimeError('This repair only migrates the known completed-rollout source version')
        verify_compatible_science(project)
        if any((study/'heads').glob('*/*/result.json')) or any((study/'jobs').glob('*__*/request.json')):
            raise RuntimeError('Heads have already started; refusing to change their evaluation inputs')
        for policy in declaration['policies']:
            saved = read_json(study/'policies'/policy['id']/'model.json')
            if saved['spec'] != policy or model_identity(saved['model']) != saved['model_hash']:
                raise RuntimeError(f'Policy weights or dataset/budget changed: {policy["id"]}')
            if policy['condition'] != 'companion':
                for rep in range(3):
                    directory = study/'jobs'/f'evaluate_{policy["id"]}_rep{rep}'
                    if not (directory/'complete.json').exists():
                        raise RuntimeError(f'Finish all evaluation jobs before repair: {directory.name}')
        completed = {}
        for path in sorted((study/'jobs').glob('*/request.json')):
            request = read_json(path); spec = request['spec']
            if spec['kind'] not in ('train', 'evaluate') or spec['source_hash'] != OLD_SOURCE:
                raise RuntimeError(f'Unexpected prior job: {path}')
            if request['identity'] != identity(spec) or not path.with_name('complete.json').exists():
                raise RuntimeError(f'Invalid or unfinished prior job: {path}')
            if read_json(path.with_name('complete.json'))['identity'] != request['identity']:
                raise RuntimeError(f'Completion checksum mismatch: {path}')
            completed[spec['id']] = request['identity']
        from uq_followup.evaluation import collect
        quality = collect(study, declaration['policies'])
        audit = read_json(study/'observation_pairing.json')
        updated = dict(declaration, source_hash=current)
        backup = study/'migrations'/PAIRING_PROTOCOL
        backup.mkdir(parents=True, exist_ok=True)
        for name in ('declaration.json', 'started.json'):
            target = backup/name
            content = (study/name).read_bytes()
            if target.exists() and target.read_bytes() != content:
                raise RuntimeError(f'Original metadata backup differs: {target}')
            target.write_bytes(content)
        atomic_json(transition_path, dict(protocol=PAIRING_PROTOCOL, migrated_at=time.time(),
            from_source_hash=OLD_SOURCE, to_source_hash=current,
            from_declaration_hash=identity(declaration), to_declaration_hash=identity(updated),
            completed_requests=completed, unchanged_scientific_files=UNCHANGED_FILES,
            current_source_files={p.name: file_hash(p) for p in (project/'src/uq_followup').glob('*.py') if p.name != 'reporting.py'},
            observation_audit_hash=identity(audit),
            note='Only derived observation pairing changes. Original job requests, completions, raw rollouts and weights are preserved.'))
        # Final commit; prepare() must still match every other declaration field.
        atomic_json(study/'declaration.json', updated)
        print(f'Reused all policies and {audit["compared_executions"]:,} rollouts. '
              f'Accepted {audit["accepted_render_differences"]:,} bounded render differences; '
              f'max delta {audit["observed_max_image_delta"]}/255; '
              f'max {audit["observed_max_changed_channel_values"]} channel values per image.', flush=True)
        print('Quality warnings:', quality['warnings'], flush=True)
        print('Repair complete. Restart the usual study command; no GPU work was launched.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('storage/extensions/followup_v1'))
    repair(parser.parse_args().study)
