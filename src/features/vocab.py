"""Character vocabulary: Unicode code points -> integer ids.

The model works on code points, not on a tokenizer, so the browser can reproduce the exact same
input with ``Array.from(text, c => c.codePointAt(0))``. The vocabulary is fit on the training split
only and saved inside ``preprocessing.json`` next to the model.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable

import numpy as np

PAD_ID = 0
UNK_ID = 1


class CharVocab:
    """Id ``i + 2`` is ``code_points[i]``; ids 0 and 1 are reserved for PAD and UNK."""

    def __init__(self, code_points: Iterable[int]):
        self.code_points = tuple(int(c) for c in code_points)
        self._index = {cp: i + 2 for i, cp in enumerate(self.code_points)}

    @classmethod
    def fit(cls, texts: Iterable[str], min_freq: int = 1) -> CharVocab:
        counts: Counter = Counter()
        for text in texts:
            counts.update(text)
        # Most frequent first; ties broken by code point so the result is deterministic.
        kept = [c for c, n in counts.items() if n >= min_freq]
        kept.sort(key=lambda c: (-counts[c], c))
        return cls(ord(c) for c in kept)

    @property
    def size(self) -> int:
        return len(self.code_points) + 2

    def encode(self, text: str) -> np.ndarray:
        get = self._index.get
        return np.fromiter((get(ord(c), UNK_ID) for c in text), dtype=np.int32, count=len(text))

    def to_dict(self) -> dict:
        return {"pad_id": PAD_ID, "unk_id": UNK_ID, "code_points": list(self.code_points)}

    @classmethod
    def from_dict(cls, d: dict) -> CharVocab:
        if d.get("pad_id") != PAD_ID or d.get("unk_id") != UNK_ID:
            raise ValueError("unsupported vocabulary: PAD/UNK ids differ from this code version")
        return cls(d["code_points"])

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict()).encode()).hexdigest()[:12]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CharVocab) and self.code_points == other.code_points

    def __hash__(self) -> int:
        return hash(self.code_points)
