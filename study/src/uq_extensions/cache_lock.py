"""Process locks for shared caches; writers commit completion markers last."""
from contextlib import contextmanager
from functools import wraps
import fcntl
import inspect
from pathlib import Path
import time


@contextmanager
def cache_lock(path, *, label='shared cache', poll_seconds=.25, report_seconds=60):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Keep the inode after release: unlinking permits two independent lock owners.
    with path.open('a') as handle:
        waited=False;last_report=0.
        while True:
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now=time.monotonic()
                if not waited or now-last_report>=report_seconds:
                    print(f'[progress] Waiting for {label}: {path}',flush=True)
                    last_report=now
                waited=True
                time.sleep(poll_seconds)
        try:
            if waited:print(f'[progress] Shared {label} ready; checking cached work: {path}',flush=True)
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def lock_directory(argument, filename, *, label):
    """Serialize a directory's read/resume/write transaction across processes."""
    def decorate(function):
        signature=inspect.signature(function)
        @wraps(function)
        def wrapped(*args,**kwargs):
            directory=signature.bind(*args,**kwargs).arguments[argument]
            with cache_lock(Path(directory)/filename,label=label):
                return function(*args,**kwargs)
        return wrapped
    return decorate
