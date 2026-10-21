"""Apply the project-scoped device-identity fix to pinned robosuite 1.4.0."""
import importlib.metadata
import importlib.util
from pathlib import Path

ORIGINAL = '''CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if CUDA_VISIBLE_DEVICES != "":
    MUJOCO_EGL_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    if MUJOCO_EGL_DEVICE_ID is not None:
        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (
            MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES
        ), "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"
'''

REPLACEMENT = '''# VLA_PROJECT_EGL_UUID_FIX: CUDA and EGL device indices are different namespaces.
if os.environ.get("VLA_PROJECT_ROOT") and os.environ.get("MUJOCO_GL") == "egl":
    from lerobot.utils.gpu_utils import configure_egl_for_visible_cuda
    configure_egl_for_visible_cuda()
else:
''' + "\n".join("    " + line if line else "" for line in ORIGINAL.rstrip().splitlines()) + "\n"


def binding_path():
    return Path(importlib.util.find_spec("robosuite").origin).parent / "utils/binding_utils.py"


def check_patch():
    if REPLACEMENT not in binding_path().read_text():
        raise RuntimeError("Simulator GPU patch missing. Run: .venv/bin/python scripts/patch_simulator.py")


def apply_patch():
    if importlib.metadata.version("robosuite") != "1.4.0":
        raise RuntimeError("Reassess the EGL compatibility fix before changing robosuite versions")
    path = binding_path()
    content = path.read_text()
    if REPLACEMENT in content:
        return
    if content.count(ORIGINAL) != 1:
        raise RuntimeError(f"Unexpected simulator code; refusing to overwrite {path}")
    # Atomic replacement leaves any already-running process and its imported code alone.
    temporary = path.with_suffix(".py.vla-tmp")
    temporary.write_text(content.replace(ORIGINAL, REPLACEMENT))
    temporary.replace(path)
    print(f"Installed UUID-based EGL selection: {path}")


if __name__ == "__main__":
    apply_patch()
