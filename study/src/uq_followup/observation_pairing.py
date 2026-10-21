"""Audit common simulator starts without requiring bit-identical GPU renders."""
from pathlib import Path
import numpy as np

from uq_extensions.common import array_hash, identity

PAIRING_PROTOCOL = 'bounded_render_roundoff_v1'
MAX_IMAGE_DELTA = 2             # intensity levels in an 8-bit RGB image
MAX_CHANGED_FRACTION = .001    # at most 0.1% of channel values, per camera
CONDITION_FIELDS = ('task', 'init_state_id', 'init_state_hash', 'reset_seed',
                    'condition', 'condition_hash', 'observation_spec',
                    'observation_preprocessing', 'rendering_protocol',
                    'max_steps', 'control_hz', 'action_steps', 'flow_steps')


def condition_id(row):
    return row['id'].rsplit('/rep_', 1)[0]


class ObservationPairing:
    """Validate every file's original hash, then compare to a fixed reference.

    Only RGB roundoff is tolerated. State, instruction, reset seed, simulator
    state hash, preprocessing, and rollout settings must agree exactly. The
    reference is chosen before looking at outcomes; no image is averaged or
    modified. Keep diagnostics for every accepted non-identical render.
    """

    def __init__(self, reference_policy):
        self.reference_policy = reference_policy
        self._references = {}
        self.comparisons = 0
        self.differences = []
        self.max_delta = 0
        self.max_changed = 0
        self.max_fraction = 0.

    @staticmethod
    def _load(candidate):
        path = Path(candidate['observation_file'])
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        if identity({k: array_hash(v) for k, v in arrays.items()}) != candidate['input_hash']:
            raise ValueError(f'Saved observation checksum mismatch: {path}')
        if set(arrays) != {'image', 'image2', 'state', 'instruction'}:
            raise ValueError(f'Unexpected observation fields: {path}')
        for camera in ('image', 'image2'):
            if arrays[camera].shape != (256, 256, 3) or arrays[camera].dtype != np.uint8:
                raise ValueError(f'Expected uint8 256×256 RGB {camera}: {path}')
        if arrays['state'].shape != (8,) or not np.isfinite(arrays['state']).all():
            raise ValueError(f'Invalid robot state: {path}')
        if arrays['instruction'].shape != () or arrays['instruction'].dtype.kind != 'U':
            raise ValueError(f'Invalid language instruction: {path}')
        return arrays

    def validate(self, reference, row, *, policy_id=''):
        label = f'{policy_id}/{row["id"]}'
        if condition_id(reference) != condition_id(row):
            raise ValueError(f'Different condition IDs: {label}')
        for key in CONDITION_FIELDS:
            if key not in reference or key not in row or reference[key] != row[key]:
                raise ValueError(f'Different initial condition or evaluation setting ({key}): {label}')
        for candidate, record in ((reference['initial_input'], reference), (row['initial_input'], row)):
            if candidate['task'] != record['task'] or candidate['id'] != record['id']:
                raise ValueError(f'Observation does not belong to its execution: {label}')
        key = (reference['initial_input']['observation_file'], reference['initial_input']['input_hash'])
        if key not in self._references:
            self._references[key] = self._load(reference['initial_input'])
        expected = self._references[key]
        actual = self._load(row['initial_input'])
        for name in ('state', 'instruction'):
            if expected[name].dtype != actual[name].dtype or not np.array_equal(expected[name], actual[name]):
                raise ValueError(f'Different initial {name}: {label}')
        images = {}
        for camera in ('image', 'image2'):
            delta = np.abs(expected[camera].astype(np.int16) - actual[camera].astype(np.int16))
            changed = int(np.count_nonzero(delta))
            if not changed:
                continue
            maximum = int(delta.max()); fraction = changed / delta.size
            if maximum > MAX_IMAGE_DELTA or fraction > MAX_CHANGED_FRACTION:
                raise ValueError(f'Initial {camera} difference exceeds render-roundoff bounds: {label}; '
                                 f'max delta={maximum}, changed={changed}/{delta.size} ({fraction:.4%})')
            images[camera] = dict(max_delta=maximum, changed_channel_values=changed, changed_fraction=fraction)
            self.max_delta = max(self.max_delta, maximum)
            self.max_changed = max(self.max_changed, changed)
            self.max_fraction = max(self.max_fraction, fraction)
        self.comparisons += 1
        if images:
            self.differences.append(dict(policy=policy_id, execution=row['id'],
                original_input_hash=row['initial_input']['input_hash'],
                canonical_input_hash=reference['initial_input']['input_hash'], images=images))

    def summary(self):
        return dict(protocol=PAIRING_PROTOCOL, reference_policy=self.reference_policy, reference_repetition=0,
            max_allowed_image_delta=MAX_IMAGE_DELTA, max_allowed_changed_fraction=MAX_CHANGED_FRACTION,
            compared_executions=self.comparisons, exact_observations=self.comparisons-len(self.differences),
            accepted_render_differences=len(self.differences), observed_max_image_delta=self.max_delta,
            observed_max_changed_channel_values=self.max_changed, observed_max_changed_fraction=self.max_fraction,
            state_instruction_and_condition_metadata='exact', differences=self.differences)
