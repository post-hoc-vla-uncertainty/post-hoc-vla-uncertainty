"""Initialize LIBERO without interactive prompts or changes outside the project."""
import importlib.util
import json
import os
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from patch_simulator import apply_patch as apply_gpu_patch

ROOT = Path(__file__).resolve().parents[1]

def main():
    apply_gpu_patch()
    lock = json.loads((ROOT / "assets.lock.json").read_text())["libero_assets"]
    assets = Path(os.environ["STORAGE_ROOT"]) / "libero_assets"
    snapshot_download(repo_id=lock["repo_id"], repo_type="dataset", revision=lock["revision"],
                      local_dir=assets, max_workers=8, token=False)
    package = Path(importlib.util.find_spec("libero").origin).parent / "libero"
    link = package / "assets"
    if not link.exists():
        link.symlink_to(assets, target_is_directory=True)
    config = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
              "init_states": str(package / "init_files"), "datasets": os.environ["HF_LEROBOT_HOME"],
              "assets": str(assets)}
    destination = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config))
    # Robosuite's shared /tmp/robosuite.log may belong to another machine user.
    # Its supported private-macro override disables just that file handler.
    robosuite = Path(importlib.util.find_spec("robosuite").origin).parent
    (robosuite / "macros_private.py").write_text("FILE_LOGGING_LEVEL = None\n")
    print(f"LIBERO configured: {destination}")

if __name__ == "__main__":
    main()
