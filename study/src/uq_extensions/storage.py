"""All-round model retention with rolling state inside the active fit."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil

from .common import atomic_json, file_hash, read_json


def usage_bytes(root):
    total = 0
    for directory, _, files in os.walk(root, followlinks=False):
        for filename in files:
            p = Path(directory) / filename
            if not p.is_symlink():
                try: total += p.stat().st_size
                except FileNotFoundError: pass
    return total


def retire_feature_arrays(directory):
    """Retire only owned, regenerable shared arrays after a completed policy panel.

    Specifications, GP models, scores, patch maps and policy weights remain.
    A cache is marked retired before unlinking, so an interruption regenerates it.
    """
    if os.environ.get('UQ_EXT_RETIRE_FEATURE_CACHE')!='1':return
    cache=Path(os.environ['UQ_EXT_FEATURE_CACHE']).resolve();directory=Path(directory)
    retired={}
    for link in directory.rglob('cache_link.json'):
        record=read_json(link);path=Path(record['path'])
        if path.parent.resolve()!=cache or path.is_symlink() or path.name!=record['identity']:
            raise ValueError('Unowned feature-cache link')
        specpath=path/'feature_spec.json'
        if not specpath.exists():continue
        spec=read_json(specpath)
        if spec['identity']!=record['identity']:raise ValueError('Feature-cache identity mismatch')
        spec['arrays_retired']=True;atomic_json(specpath,spec)
        removed=0
        for filename in spec.get('array_files',{}).values():
            if Path(filename).name!=filename or not filename.endswith('.npy'):raise ValueError('Unsafe feature array filename')
            array=path/filename
            if array.is_symlink():raise ValueError('Symlinked feature array')
            if array.exists():removed+=array.stat().st_size;array.unlink()
        if removed:retired[str(path)]=removed
    if retired:atomic_json(directory/'feature_retirement.json',dict(arrays=retired,retired_bytes=sum(retired.values()),regenerable=True))


def check_budget(root, budget_gb, reserve_gb=8):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    used = usage_bytes(root)
    report = dict(used_bytes=used, reserved_bytes=int(reserve_gb*1e9), budget_bytes=int(budget_gb*1e9))
    atomic_json(root / "storage_usage.json", report)
    if used + reserve_gb*1e9 > budget_gb*1e9 or shutil.disk_usage(root).free < (reserve_gb + 1)*1e9:
        raise RuntimeError(f"Extension storage guard: {used/1e9:.2f} GB + {reserve_gb:g} GB reserve, budget {budget_gb:g} GB")


def owned_checkpoint(path):
    root = Path(os.environ["UQ_EXT_RUN_ROOT"]).resolve()
    path = Path(path)
    if path.absolute() != path.resolve() or not path.resolve().is_relative_to(root):
        raise RuntimeError("Checkpoint is outside this extension experiment")
    if path.parent.name != "checkpoints" or not path.name.isdigit():
        raise RuntimeError("Not a numeric training checkpoint")
    return path.resolve()


@contextmanager
def checkpoint_destination(path):
    from reproduction_storage import validate_checkpoint, inventory
    path = owned_checkpoint(path)
    root = Path(os.environ["UQ_EXT_RUN_ROOT"])
    with (root / "checkpoint_write.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        check_budget(root, float(os.environ.get("UQ_EXT_BUDGET_GB", 100)), 5)
        temporary = path.with_name(".writing-" + path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            marker = temporary / "transaction.json"
            if not marker.exists() or read_json(marker).get("target") != str(path):
                raise RuntimeError("Unrecognized interrupted checkpoint")
            shutil.rmtree(temporary)
        if path.exists():
            raise RuntimeError(f"Checkpoint already committed: {path}")
        temporary.mkdir()
        atomic_json(temporary / "transaction.json", {"target": str(path)})
        try:
            yield temporary
            validate_checkpoint(temporary, resumable=True)
            atomic_json(temporary / "checkpoint_inventory.json", inventory(temporary))
            temporary.rename(path)
        except BaseException:
            if temporary.exists(): shutil.rmtree(temporary)
            raise


def checkpoint_published(path):
    from reproduction_storage import set_last, validate_checkpoint
    path = owned_checkpoint(path)
    validate_checkpoint(path, resumable=True)
    set_last(path)
    # These are saves *within one fit*, not checkpoints from preceding rounds.
    for older in sorted(path.parent.glob("[0-9]" * len(path.name))):
        if older >= path: continue
        owned_checkpoint(older)
        if not (older / "checkpoint_inventory.json").exists():
            raise RuntimeError("Refusing to retire unrecognized training state")
        atomic_json(older / "retention.json", {"weights_retired": True, "reason": "superseded within-fit checkpoint", "replacement": str(path)})
        for child in (older / "pretrained_model", older / "training_state/optimizer_state.safetensors"):
            if child.is_symlink(): raise RuntimeError("Symlinked checkpoint payload")
            if child.is_dir(): shutil.rmtree(child)
            elif child.exists(): child.unlink()


def finalize_model(path):
    from reproduction_storage import validate_checkpoint
    checkpoint = Path(path).resolve().parent
    owned_checkpoint(checkpoint)
    validate_checkpoint(checkpoint)
    atomic_json(checkpoint / "retention.json", {"weights_retired": False, "optimizer_retired": True,
                                               "reason": "completed fit; retain this round for later analyses"})
    (checkpoint / "training_state/optimizer_state.safetensors").unlink(missing_ok=True)


def recover_checkpoint(parent):
    from reproduction_storage import set_last, validate_checkpoint
    parent = Path(parent)
    committed = [p for p in parent.glob("[0-9]" * 6) if (p / "checkpoint_inventory.json").exists() and (p / "pretrained_model").exists()]
    if committed:
        chosen = max(committed)
        validate_checkpoint(chosen)
        set_last(chosen)
        return chosen
    return None
