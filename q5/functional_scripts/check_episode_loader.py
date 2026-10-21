#!/usr/bin/env python
"""Check that load_nested_dataset's memory-mapped episode filter returns the same rows as the old in-RAM path."""

from pathlib import Path

import pyarrow.dataset as pa_ds

from lerobot.datasets.utils import load_nested_dataset

DATA = Path("storage/huggingface/lerobot/HuggingFaceVLA/libero/data")
EPISODES = [1500, 1501]

new = load_nested_dataset(DATA, episodes=EPISODES)
old = pa_ds.dataset(sorted(map(str, DATA.glob("*/*.parquet"))), format="parquet").to_table(
    filter=pa_ds.field("episode_index").isin(EPISODES))
print("rows", len(new), old.num_rows)
assert len(new) == old.num_rows
for column in ("episode_index", "frame_index", "index"):
    assert new.data.column(column).to_pylist() == old.column(column).to_pylist(), column
print("LOADER OK")
