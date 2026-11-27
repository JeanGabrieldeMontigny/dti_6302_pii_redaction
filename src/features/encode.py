"""Character-level BIO tags and fixed-size windows.

Tag ids: ``0 = O``; for label index ``k`` (order of ``labels``), ``B = 1 + 2k`` and ``I = 2 + 2k``.
Documents longer than the window are cut into overlapping windows; at inference each character
takes its prediction from the window in which it is most central (furthest from either edge).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.data.schema import Span

O_TAG = 0


def n_tags(labels: Sequence[str]) -> int:
    return 1 + 2 * len(labels)


def spans_to_tags(n_chars: int, spans: Sequence[Span], labels: Sequence[str]) -> np.ndarray:
    """Per-character BIO tag ids. Spans must not overlap (preprocessing guarantees this)."""
    index = {label: k for k, label in enumerate(labels)}
    tags = np.full(n_chars, O_TAG, dtype=np.int32)
    for s in spans:
        k = index[s["label"]]
        tags[s["start"] : s["end"]] = 2 + 2 * k
        tags[s["start"]] = 1 + 2 * k
    return tags


def tags_to_spans(tags: Sequence[int], labels: Sequence[str]) -> list[Span]:
    """Inverse of ``spans_to_tags``. An ``I`` tag after ``O`` or another label starts a new span."""
    spans: list[Span] = []
    current: Span | None = None
    for i, t in enumerate(tags):
        t = int(t)
        if t == O_TAG:
            current = None
            continue
        k, is_begin = (t - 1) // 2, (t - 1) % 2 == 0
        if current is not None and not is_begin and current["label"] == labels[k]:
            current["end"] = i + 1
        else:
            current = {"start": i, "end": i + 1, "label": labels[k]}
            spans.append(current)
    return spans


def window_starts(n_chars: int, window: int, stride: int) -> list[int]:
    """Start offsets of windows covering ``[0, n_chars)``; the last window ends at ``n_chars``."""
    if not 0 < stride <= window:
        raise ValueError(f"need 0 < stride <= window, got stride={stride}, window={window}")
    if n_chars <= window:
        return [0]
    starts = list(range(0, n_chars - window, stride))
    starts.append(n_chars - window)
    return starts


def make_windows(ids: np.ndarray, window: int, stride: int, pad_id: int = 0) -> np.ndarray:
    """Cut one encoded document into ``(n_windows, window)``, right-padding short documents."""
    starts = window_starts(len(ids), window, stride)
    out = np.full((len(starts), window), pad_id, dtype=ids.dtype)
    for row, start in enumerate(starts):
        chunk = ids[start : start + window]
        out[row, : len(chunk)] = chunk
    return out


def stitch(window_probs: np.ndarray, n_chars: int, window: int, stride: int) -> np.ndarray:
    """Combine per-window outputs ``(n_windows, window, C)`` into per-char ``(n_chars, C)``."""
    starts = window_starts(n_chars, window, stride)
    if len(starts) != len(window_probs):
        raise ValueError(f"expected {len(starts)} windows, got {len(window_probs)}")
    out = np.zeros((n_chars, window_probs.shape[-1]), dtype=window_probs.dtype)
    best = np.full(n_chars, -1, dtype=np.int64)
    for start, probs in zip(starts, window_probs, strict=True):
        end = min(start + window, n_chars)
        pos = np.arange(start, end)
        centrality = np.minimum(pos - start, start + window - 1 - pos)
        take = centrality > best[start:end]
        out[pos[take]] = probs[: end - start][take]
        best[start:end] = np.where(take, centrality, best[start:end])
    return out
