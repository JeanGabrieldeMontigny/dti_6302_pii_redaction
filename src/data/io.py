"""Reading and writing processed splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_split(path: Path) -> pd.DataFrame:
    """Read a processed split; ``spans`` comes back as plain ``list[dict]`` with int offsets."""
    df = pd.read_parquet(path)
    df["spans"] = [
        [{"start": int(s["start"]), "end": int(s["end"]), "label": str(s["label"])} for s in spans]
        for spans in df["spans"]
    ]
    return df
