"""Leakage-aware splitting.

Each ``uid`` is a persona-grounded document generated twice (a ``us`` and an ``intl`` variant that
share e.g. the date of birth), so records are grouped by ``uid``. The official Hugging Face
``test`` split is our held-out test set; validation is carved out of ``train`` by hashing the
``uid``, which is deterministic and independent of row order.
"""

from __future__ import annotations

import hashlib

import pandas as pd

_BUCKETS = 10_000


def in_validation(uid: str, val_fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest()
    return int(digest, 16) % _BUCKETS < round(val_fraction * _BUCKETS)


def split_train_val(
    train: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = train["uid"].map(lambda u: in_validation(u, val_fraction, seed))
    return train[~mask].reset_index(drop=True), train[mask].reset_index(drop=True)


def remove_cross_split_leakage(
    splits: dict[str, pd.DataFrame], reference: str = "test"
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Drop rows from non-reference splits whose uid or exact text appears in the reference split."""
    ref = splits[reference]
    ref_uids, ref_texts = set(ref["uid"]), set(ref["text"])
    removed, out = {}, {reference: ref}
    for name, df in splits.items():
        if name == reference:
            continue
        leak = df["uid"].isin(ref_uids) | df["text"].isin(ref_texts)
        removed[name] = int(leak.sum())
        out[name] = df[~leak].reset_index(drop=True)
    return out, removed


def leakage_report(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    """Pairwise uid/text overlap counts between splits (all should be zero)."""
    names = sorted(splits)
    report = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            report[f"{a}|{b}"] = {
                "uid_overlap": len(set(splits[a]["uid"]) & set(splits[b]["uid"])),
                "text_overlap": len(set(splits[a]["text"]) & set(splits[b]["text"])),
            }
    return report
