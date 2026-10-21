"""Content identities, atomic artifacts, and isolated random streams."""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from functools import wraps

import numpy as np


def preserve_environment(*keys):
    """Prevent library entry points from leaking checkpoint ownership into callers."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            previous = {key: os.environ.get(key) for key in keys}
            try:
                return function(*args, **kwargs)
            finally:
                for key, value in previous.items():
                    if value is None: os.environ.pop(key, None)
                    else: os.environ[key] = value
        return wrapped
    return decorate


def plain(value):
    if is_dataclass(value):
        # New optional settings must not change identities of historical runs.
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)
                if not (f.metadata.get('omit_default') and getattr(value, f.name) == f.default)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def identity(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(plain(value), f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def stream_seed(seed, *parts):
    return int(identity([seed, *parts])[:8], 16) % (2**31)


@contextmanager
def isolated_rng(seed):
    import torch
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def model_identity(path):
    path = Path(path).resolve()
    files = sorted(p for p in path.iterdir() if p.is_file() and p.name != "train_config.json")
    if not any(p.name == "model.safetensors" for p in files):
        raise FileNotFoundError(f"No policy weights: {path}")
    return identity({p.name: file_hash(p) for p in files})
