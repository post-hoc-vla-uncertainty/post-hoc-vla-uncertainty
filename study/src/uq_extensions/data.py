"""One-time separation of candidate inputs from private trajectory/evaluator data."""
from collections import defaultdict
from io import BytesIO
from pathlib import Path
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from PIL import Image

from .common import array_hash, atomic_json, atomic_npz, file_hash, identity, read_json
from .protocol import Candidate, CandidateView


def load_manifest(path):
    data = read_json(path)
    claimed = data.pop("manifest_hash")
    if identity(data) != claimed:
        raise RuntimeError("Data manifest content hash mismatch")
    data["manifest_hash"] = claimed
    return data


def candidate_view(manifest, ids=None):
    allowed = set(ids) if ids is not None else None
    return CandidateView([Candidate(**record) for record in manifest["observations"] if allowed is None or record["id"] in allowed])


def prepare_data(dataset_root, output, source_revision):
    from lerobot.envs.libero import get_task_instruction, get_task_init_states
    from libero.libero import benchmark
    dataset_root, output = Path(dataset_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "data_manifest.json"
    if destination.exists():
        manifest = load_manifest(destination)
        if manifest["source_revision"] != source_revision or manifest["dataset_root"] != str(dataset_root):
            raise RuntimeError("Prepared extension data belongs to a different source")
        return manifest
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    instruction_map = {get_task_instruction(suite, task): f"{suite}/{task}" for suite in suites for task in range(10)}
    table = ds.dataset(str(dataset_root / "meta/episodes"), format="parquet").to_table(
        columns=["episode_index", "tasks", "dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index"])
    episodes = table.to_pylist()
    episode_info = {int(e["episode_index"]): e for e in episodes}
    public, private, normalizer_rows = [], [], defaultdict(list)
    file_hashes, episode_files, motion_parts = {}, defaultdict(list), defaultdict(list)
    files = sorted((dataset_root / "data").rglob("*.parquet"))
    for n, path in enumerate(files):
        source_hash = file_hash(path)
        file_hashes[str(path.relative_to(dataset_root))] = source_hash
        starts = pq.read_table(path, columns=["episode_index", "frame_index", "observation.images.image",
                                              "observation.images.image2", "observation.state"],
                               filters=[("frame_index", "=", 0)]).to_pylist()
        motion = pq.read_table(path, columns=["episode_index", "index", "timestamp", "observation.state"]).to_pandas()
        for ep, rows in motion.groupby("episode_index", sort=False):
            motion_parts[int(ep)].append((rows["index"].to_numpy(), rows.timestamp.to_numpy(),
                                         np.stack(rows["observation.state"].to_numpy())[:, :3]))
            episode_files[int(ep)].append({"file": str(path), "sha256": source_hash})
        for row in starts:
            ep = int(row["episode_index"])
            episode = episode_info[ep]
            if len(episode["tasks"]) != 1 or episode["tasks"][0] not in instruction_map:
                raise RuntimeError(f"Unknown task identity in episode {ep}")
            instruction = episode["tasks"][0]
            task = instruction_map[instruction]
            id = f"{task}/episode_{ep:06d}"
            observation = {"state": np.array(row["observation.state"], dtype=np.float32),
                           "instruction": np.asarray(instruction)}
            for camera in ("image", "image2"):
                encoded = row[f"observation.images.{camera}"]
                with Image.open(BytesIO(encoded["bytes"]) if encoded["bytes"] is not None else dataset_root / encoded["path"]) as image:
                    observation[camera] = np.asarray(image.convert("RGB"))
            input_hash = identity({k: array_hash(v) for k, v in observation.items()})
            observation_path = output / "initial_inputs" / f"{ep:06d}.npz"
            atomic_npz(observation_path, **observation)
            public.append(dict(id=id, episode_index=ep, task=task,
                               observation_file=str(observation_path), input_hash=input_hash))
        if n % 10 == 0:
            print(f"Prepared inputs and evaluator metadata from {n + 1}/{len(files)} data files", flush=True)
    if len(public) != len(episode_info) or len({r["episode_index"] for r in public}) != len(public):
        raise RuntimeError("Missing or duplicate initial observations in the dataset")
    for record in public:
        ep = record["episode_index"]; episode = episode_info[ep]
        parts = motion_parts[ep]
        indices = np.concatenate([p[0] for p in parts]); order = np.argsort(indices)
        timestamps = np.concatenate([p[1] for p in parts])[order]
        xyz = np.concatenate([p[2] for p in parts])[order]
        if len(indices) != episode["dataset_to_index"] - episode["dataset_from_index"]:
            raise RuntimeError(f"Incomplete demonstration frames: {ep}")
        private.append({"id": record["id"], **episode, "files": episode_files[ep],
                        "content_hash": identity([episode_files[ep], ep, indices[order].tolist()])})
        normalizer_rows[record["task"]].append({"id": record["id"], "time": float(timestamps[-1] - timestamps[0]),
                                              "path": float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())})
    public.sort(key=lambda r: r["id"])
    pretraining = [r["id"] for r in public if not r["task"].startswith("libero_10/") or int(r["task"].split("/")[1]) < 3]
    target = [r["id"] for r in public if r["task"].startswith("libero_10/")]
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    splits = {"development": [], "final": [], "development_gain_panel": [], "final_gain_panel": []}
    for task in range(10):
        states = np.asarray(get_task_init_states(suite, task))
        unique = []
        seen = set()
        for i, state in enumerate(states):
            key = array_hash(state)
            if key not in seen:
                unique.append((i, key)); seen.add(key)
        if len(unique) < 40:
            raise RuntimeError("Need at least 10 disjoint development and 30 final initial states per task")
        dev, final = unique[:-30], unique[-30:]
        for split, values in (("development", dev), ("final", final)):
            for i, key in values:
                record = dict(task=f"libero_10/{task}", init_state_id=i, init_state_hash=key,
                              reset_seed=10000 + task * 100 + i, repetition=0,
                              id=f"libero_10/{task}/start_{i:03d}/rep_0")
                splits[split].append(record)
            for i, key in values[:10]:
                for repetition in range(3):
                    splits[split + "_gain_panel"].append(dict(task=f"libero_10/{task}", init_state_id=i,
                        init_state_hash=key, reset_seed=10000 + task * 100 + i, repetition=repetition,
                        id=f"libero_10/{task}/start_{i:03d}/rep_{repetition}"))
    atomic_json(output / "evaluator" / "trajectory_index.json", private)
    normalizers = {task: dict(count=len(rows), mean_time=float(np.mean([r["time"] for r in rows])),
                             std_time=float(np.std([r["time"] for r in rows])),
                             mean_path=float(np.mean([r["path"] for r in rows])),
                             std_path=float(np.std([r["path"] for r in rows]))) for task, rows in normalizer_rows.items()}
    atomic_json(output / "evaluator" / "normalizers.json", {
        "tasks": normalizers, "episode_measurements": normalizer_rows,
        "convention": "released successful demonstrations; annotated endpoints, no first-success replay",
        "position": "unnormalized observation.state[:3], world end-effector translation in meters",
        "time": "recorded timestamp seconds; zero/missing baselines flagged by evaluator",
        "source_revision": source_revision})
    manifest = dict(version=1, source_revision=source_revision, dataset_root=str(dataset_root),
                    source_files=file_hashes, observations=public, pretraining_ids=pretraining,
                    target_ids=target, integration_ids=target, evaluation_splits=splits,
                    episode_map={r["id"]: r["episode_index"] for r in public},
                    evaluator_directory=str(output / "evaluator"))
    manifest["manifest_hash"] = identity(manifest)
    atomic_json(destination, manifest)
    return manifest


def verify_dataset(manifest):
    """Verify training payloads against the prepared manifest, caching unchanged file stats."""
    cache_path=Path(manifest['evaluator_directory'])/'verified_source_files.json'
    previous=read_json(cache_path) if cache_path.exists() else {}
    verified={}
    for relative,digest in manifest['source_files'].items():
        path=Path(manifest['dataset_root'])/relative;stat=path.stat()
        signature=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
        expected=dict(signature=signature,sha256=digest)
        if previous.get(relative)!=expected and file_hash(path)!=digest:
            raise RuntimeError(f'Training dataset content changed: {path}')
        verified[relative]=expected
    atomic_json(cache_path,verified)
