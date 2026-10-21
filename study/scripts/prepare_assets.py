"""Download the immutable public dataset and VLM snapshots; never train a model."""
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]

def main():
    lock = json.loads((ROOT / "assets.lock.json").read_text())
    data = lock["dataset"]
    model = lock["vlm"]
    destination = Path(os.environ["HF_LEROBOT_HOME"]) / data["repo_id"]
    snapshot_download(repo_id=model["repo_id"], revision=model["revision"],
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
                      max_workers=4, token=False)
    snapshot_download(repo_id=data["repo_id"], repo_type="dataset", revision=data["revision"],
                      local_dir=destination, max_workers=8, token=False)
    print(f"Pinned assets ready: {destination}", flush=True)

if __name__ == "__main__":
    main()
