"""Checkpoint transactions and retention for the local reproduction launcher.

Only the launcher enables this policy. Dataset caches and other experiments are
never pruning targets. Checkpoints are published before older weights are retired.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import time

GB = 10**9
POLICY_ENV = "VLA_REPRODUCTION_POLICY"
VERSION = 1
CHECKPOINT = re.compile(
    r"(?:outputs/pretrain/libero_pretrain_leak3/seed_[0-5]|"
    r"outputs/reproduction/(?:random|vfd|gu|calibration_random)/seed_100[0-2]/"
    r"round_\d{3}/training/member_0[01])/checkpoints/\d{6}"
)


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_under(root):
    """Do not follow symlinks (notably checkpoints/last)."""
    for directory, _, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            path = Path(directory) / name
            if not path.is_symlink():
                yield path


def tree_bytes(root):
    total = 0
    for path in files_under(root):
        try:
            total += path.stat().st_size
        except FileNotFoundError:  # another worker has just retired a file
            pass
    return total


def validate_safetensors(path):
    """Validate file framing without loading tensors or initializing CUDA."""
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < size < 100_000_000:
            raise RuntimeError(f"Invalid safetensors header: {path}")
        header = json.loads(handle.read(size))
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    if path.stat().st_size != 8 + size + end:
        raise RuntimeError(f"Truncated checkpoint tensor file: {path}")


def validate_checkpoint(checkpoint, *, resumable=False):
    checkpoint = Path(checkpoint)
    model = checkpoint / "pretrained_model"
    required = [model / name for name in (
        "model.safetensors", "config.json", "train_config.json",
        "policy_preprocessor.json", "policy_postprocessor.json",
    )]
    required.append(checkpoint / "training_state/training_step.json")
    if resumable:
        required.extend(checkpoint / "training_state" / name for name in (
            "optimizer_state.safetensors", "optimizer_param_groups.json",
            "rng_state.safetensors", "scheduler_state.json",
        ))
    for path in required:
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Missing checkpoint component: {path}")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for item in read_json(model / name).get("steps", []):
            if item.get("state_file") and not (model / item["state_file"]).is_file():
                raise RuntimeError(f"Missing processor state: {model / item['state_file']}")
    for path in model.glob("*.safetensors"):
        validate_safetensors(path)
    if resumable:
        validate_safetensors(checkpoint / "training_state/optimizer_state.safetensors")
    return int(read_json(checkpoint / "training_state/training_step.json")["step"])


def inventory(checkpoint):
    """Persist hashes and JSON configs even when a checkpoint is later retired."""
    return {
        "version": VERSION,
        "files": {str(p.relative_to(checkpoint)): {
            "bytes": p.stat().st_size, "sha256": sha256(p),
        } for p in files_under(checkpoint) if p.name != "checkpoint_inventory.json"},
        "configs": {p.name: read_json(p) for p in (checkpoint / "pretrained_model").glob("*.json")},
    }


def set_last(checkpoint):
    link = checkpoint.parent / "last"
    temporary = checkpoint.parent / f".last-{os.getpid()}"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(checkpoint.name)
    temporary.replace(link)


class StoragePolicy:
    def __init__(self, path):
        self.path = Path(path)
        self.data = read_json(path)
        if self.data["version"] != VERSION:
            raise RuntimeError("Unsupported reproduction storage policy")
        self.storage = Path(self.data["storage_root"]).resolve()
        self.rolling = self.data["retention"] == "rolling"
        self.budget = int(self.data["budget_bytes"])

    @classmethod
    def create(cls, storage, *, budget_gb=100, retention="rolling"):
        storage = Path(storage).resolve()
        path = storage / "logs/reproduction/storage_policy.json"
        atomic_json(path, {
            "version": VERSION, "storage_root": str(storage),
            "budget_bytes": int(budget_gb * GB), "retention": retention,
            "random_completed_updates": [5, 10, 15], "other_completed_updates": [15],
        })
        return cls(path)

    @contextmanager
    def lock(self):
        with self.path.with_suffix(".lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            yield

    def usage(self):
        roots = [self.storage / p for p in (
            "outputs/pretrain/libero_pretrain_leak3", "outputs/reproduction", "logs/reproduction",
        )]
        return sum(tree_bytes(p) for p in roots)

    def check_budget(self, reserve=8 * GB):
        used = self.usage()
        # A guard, not a filesystem quota. Reserve also covers small logs/metadata.
        if used + reserve > self.budget:
            raise RuntimeError(
                f"Reproduction storage guard: {used/GB:.2f} GB used + {reserve/GB:.2f} GB "
                f"reserved exceeds {self.budget/GB:g} GB. No next operation was started. "
                "Inspect storage_usage.json; increase --budget-gb only if intended."
            )
        if shutil.disk_usage(self.storage).free < reserve + GB:
            raise RuntimeError("Insufficient filesystem space for the next checkpoint operation")
        atomic_json(self.path.parent / "storage_usage.json", {
            "used_bytes": used, "reserved_bytes": reserve, "budget_bytes": self.budget,
            "time": time.time(), "excludes": "shared inputs, environment, setup smoke outputs, later extensions",
        })
        return used

    def owned_checkpoint(self, path):
        path = Path(path)
        # Reject external targets and symlinked parents before any deletion.
        lexical = path.absolute()
        resolved = path.resolve()
        try:
            relative = str(resolved.relative_to(self.storage))
        except ValueError:
            raise RuntimeError(f"Checkpoint is outside reproduction storage: {path}")
        if lexical != resolved or not CHECKPOINT.fullmatch(relative):
            raise RuntimeError(f"Checkpoint is not an owned numeric directory: {path}")
        return resolved

    def retire(self, checkpoint, *, weights, reason):
        checkpoint = self.owned_checkpoint(checkpoint)
        if any(p.is_symlink() for p in (checkpoint / "pretrained_model", checkpoint / "training_state")):
            raise RuntimeError(f"Refusing to prune symlinked checkpoint contents: {checkpoint}")
        record_path = checkpoint / "retention.json"
        if not (checkpoint / "checkpoint_inventory.json").exists():
            validate_checkpoint(checkpoint)
            atomic_json(checkpoint / "checkpoint_inventory.json", inventory(checkpoint))
        # Write the deletion intent first; a crash can then be recovered idempotently.
        old = read_json(record_path) if record_path.exists() else {}
        weights = weights or old.get("weights_retired", False)
        atomic_json(record_path, {
            "version": VERSION, "weights_retired": weights,
            "optimizer_retired": True, "reason": reason, "time": time.time(),
        })
        optimizer = checkpoint / "training_state/optimizer_state.safetensors"
        optimizer.unlink(missing_ok=True)
        if weights:
            model = checkpoint / "pretrained_model"
            if model.is_symlink():
                raise RuntimeError(f"Refusing to prune symlinked model directory: {model}")
            if model.exists():
                shutil.rmtree(model)

    def maintain_pretrain(self, output, *, complete=False, finalize_at=None):
        """Adopt committed saves; keep one full resume point until training succeeds."""
        checkpoints = Path(output) / "checkpoints"
        candidates = sorted(p for p in checkpoints.glob("[0-9]" * 6) if p.is_dir())
        if not candidates:
            return None
        last = checkpoints / "last"
        chosen = last.resolve() if last.exists() else None
        # A committed transaction may precede a crash before updating `last`.
        for candidate in candidates:
            if (candidate / "checkpoint_inventory.json").exists() and (candidate / "pretrained_model").exists():
                if chosen is None or candidate.name > chosen.name:
                    chosen = candidate
            elif chosen is None or candidate.name > chosen.name:
                # Migrate saves from the earlier launcher, whose last link was
                # updated only after writing every checkpoint component.
                try:
                    validate_checkpoint(candidate, resumable=True)
                except (RuntimeError, ValueError, OSError, struct.error):
                    continue
                chosen = candidate
        if chosen is None:
            raise RuntimeError(f"No committed pretraining checkpoint: {output}")
        self.owned_checkpoint(chosen)
        step = int(read_json(chosen / "training_state/training_step.json")["step"])
        complete = complete or (finalize_at is not None and step == finalize_at)
        step = validate_checkpoint(chosen, resumable=not complete)
        set_last(chosen)
        if not self.rolling:
            return step
        # Validate the replacement before retiring any previous resume state.
        for candidate in candidates:
            if candidate < chosen:
                self.retire(candidate, weights=True, reason=f"superseded by step {step}")
            elif candidate > chosen:
                # An incomplete legacy write is not a resume point. Its earlier
                # committed checkpoint has been validated above and stays intact.
                self.owned_checkpoint(candidate)
                atomic_json(checkpoints / f"discarded_{candidate.name}.json", {
                    "reason": "incomplete legacy pretraining save; resume preceding checkpoint",
                    "files": {str(p.relative_to(candidate)): sha256(p) for p in files_under(candidate)},
                })
                shutil.rmtree(candidate)
        if complete:
            self.retire(chosen, weights=False, reason="preparatory training complete; keep immutable starting policy")
        return step


@contextmanager
def checkpoint_destination(checkpoint):
    """Serialize checkpoint writes, reserve space, and publish a complete directory."""
    policy_path = os.environ.get(POLICY_ENV)
    if not policy_path:
        yield checkpoint
        return
    policy = StoragePolicy(policy_path)
    checkpoint = policy.owned_checkpoint(checkpoint)
    with policy.lock():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        pending = checkpoint.with_name(f".pending-{checkpoint.name}")
        marker = pending / "transaction.json"
        if pending.exists():
            if pending.is_symlink() or not marker.exists() or read_json(marker).get("target") != str(checkpoint):
                raise RuntimeError(f"Unrecognized partial checkpoint: {pending}")
            shutil.rmtree(pending)
        if checkpoint.exists():
            raise RuntimeError(f"Checkpoint already exists; recover it before training: {checkpoint}")
        policy.check_budget(reserve=5 * GB)
        pending.mkdir()
        atomic_json(marker, {"target": str(checkpoint), "version": VERSION})
        try:
            yield pending
            validate_checkpoint(pending, resumable=True)
            atomic_json(pending / "checkpoint_inventory.json", inventory(pending))
            pending.rename(checkpoint)
        except BaseException:
            # Keep the previous committed checkpoint and last link untouched.
            if pending.exists():
                shutil.rmtree(pending)
            raise


def checkpoint_published(checkpoint):
    policy_path = os.environ.get(POLICY_ENV)
    if policy_path:
        policy = StoragePolicy(policy_path)
        checkpoint = policy.owned_checkpoint(checkpoint)
        if str(checkpoint.relative_to(policy.storage)).startswith("outputs/pretrain/"):
            with policy.lock():
                policy.maintain_pretrain(checkpoint.parent.parent)
