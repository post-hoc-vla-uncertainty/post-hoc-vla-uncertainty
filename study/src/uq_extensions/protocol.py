"""Inputs-only candidates, immutable acquisitions, and strict cache roles."""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .common import atomic_json, identity, read_json, stream_seed


@dataclass(frozen=True)
class Candidate:
    id: str
    episode_index: int
    task: str
    observation_file: str
    input_hash: str


class CandidateView:
    """No trajectory dataset, action, length, future frame or outcome is exposed."""
    def __init__(self, candidates: Sequence[Candidate]):
        self.candidates = tuple(sorted(candidates, key=lambda c: c.id))
        if len({c.id for c in self.candidates}) != len(self.candidates):
            raise ValueError("Duplicate candidate identity")

    @property
    def ids(self):
        return [c.id for c in self.candidates]

    @property
    def tasks(self):
        return [c.task for c in self.candidates]

    def subset(self, ids):
        ids = set(ids)
        if not ids.issubset(self.ids):
            raise ValueError("Unknown candidate IDs")
        return CandidateView([c for c in self.candidates if c.id in ids])

    def observation(self, id):
        from .common import array_hash
        item = next(c for c in self.candidates if c.id == id)
        with np.load(item.observation_file, allow_pickle=False) as data:
            observation = {k: data[k] for k in data.files}
        allowed = {"image", "image2", "state", "instruction"}
        if set(observation) != allowed:
            raise ValueError("Candidate observation contains undeclared fields")
        if identity({k: array_hash(v) for k, v in observation.items()}) != item.input_hash:
            raise RuntimeError(f"Candidate inputs changed: {id}")
        return observation


class AcquisitionLedger:
    def __init__(self, path, *, manifest_hash, protocol, pretraining_ids, candidates):
        if protocol not in ('unique_demo','upstream_reference'):
            raise ValueError('Unknown eligibility protocol')
        self.path = Path(path)
        self.pretraining = frozenset(pretraining_ids)
        self.candidates = frozenset(candidates)
        expected = dict(version=1, manifest_hash=manifest_hash, protocol=protocol,
                        pretraining_ids=sorted(self.pretraining), candidate_ids=sorted(self.candidates))
        if self.path.exists():
            self.data = read_json(self.path)
            if any(self.data.get(k) != v for k, v in expected.items()):
                raise RuntimeError("Acquisition ledger identity changed")
        else:
            self.data = {**expected, "rounds": []}
            atomic_json(self.path, self.data)

    @property
    def acquired(self):
        return [id for r in self.data["rounds"] for id in r["ids"]]

    @property
    def support(self):
        return sorted(self.pretraining | set(self.acquired))

    def eligible(self):
        excluded = self.pretraining | set(self.acquired) if self.data["protocol"] == "unique_demo" else set()
        return sorted(self.candidates - excluded)

    def commit(self, round_index, ids, *, audit, expected_count=5):
        ids = list(ids)
        if round_index < len(self.data["rounds"]):
            old = self.data["rounds"][round_index]
            if old["ids"] != ids:
                raise RuntimeError("Cannot change a committed acquisition")
            return old
        if round_index != len(self.data["rounds"]):
            raise ValueError("Acquisition rounds must be committed in order")
        if len(ids) != expected_count or len(set(ids)) != len(ids):
            raise ValueError("Batch must contain the requested count of distinct IDs")
        if not set(ids).issubset(self.eligible()):
            raise ValueError("Attempted acquisition of ineligible demonstrations")
        record = dict(round_index=round_index, ids=ids, audit=audit,
                      prior_pretraining=[id for id in ids if id in self.pretraining],
                      prior_acquisition=[id for id in ids if id in self.acquired])
        self.data["rounds"].append(record)
        atomic_json(self.path, self.data)
        return record

    def reveal_indices(self, round_index, episode_map):
        if round_index >= len(self.data["rounds"]):
            raise RuntimeError("Commit must precede reveal")
        return [int(episode_map[id]) for id in self.data["rounds"][round_index]["ids"]]


def select_save(view, scores, *, count, temperature, seed, random_within_task=False):
    if len(view.ids) < count:
        raise ValueError("Insufficient eligible demonstrations; never fill with repeats")
    if set(scores) != set(view.ids) or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError("Scores must cover exactly the eligible finite-valued candidates")
    pools = {task: [c.id for c in view.candidates if c.task == task] for task in sorted(set(view.tasks))}
    means = {task: float(np.mean([scores[id] for id in ids])) for task, ids in pools.items()}
    if min(means.values()) < 0:
        shift = abs(min(means.values())) + 1e-12
        means = {k: v + shift for k, v in means.items()}
    rng = np.random.default_rng(seed)
    # Seeded ties are keyed by stable ID, independent of input array order.
    tie = {id: rng.random() for id in sorted(view.ids)}
    chosen = []
    for _ in range(count):
        tasks = sorted(task for task, ids in pools.items() if ids)
        values = np.array([means[t] for t in tasks], dtype=float)
        if temperature == 0 or not np.any(values > 0):
            probabilities = np.ones(len(tasks)) / len(tasks)
        else:
            logp = temperature * np.log(np.maximum(values, np.finfo(float).tiny))
            probabilities = np.exp(logp - logp.max())
            probabilities /= probabilities.sum()
        task = tasks[rng.choice(len(tasks), p=probabilities)]
        ids = pools[task]
        id = ids[rng.integers(len(ids))] if random_within_task else min(ids, key=lambda id: (-scores[id], tie[id], id))
        chosen.append(id)
        ids.remove(id)
    return chosen, {"task_means": means, "temperature": temperature, "seed": seed,
                    "rule": "uniform_task_uniform_episode" if random_within_task else "SAVE"}


def task_weights(tasks):
    tasks = list(tasks)
    unique = sorted(set(tasks))
    return np.array([1 / (len(unique) * tasks.count(t)) for t in tasks])


def cache_identity(*, role, checkpoint_hash, support_ids, feature_spec, split, settings):
    if role not in ("support", "candidate", "integration", "development", "final", "supervisor", "normalizer"):
        raise ValueError("Unknown cache role")
    return identity(dict(role=role, checkpoint=checkpoint_hash, support=sorted(support_ids),
                         features=feature_spec, split=split, settings=settings))
