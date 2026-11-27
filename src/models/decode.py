"""Turn per-character tag probabilities into redaction spans.

The same rules are implemented in the browser, so keep this module small and explicit:

1. A character is redacted when ``1 - P(O) >= threshold``. The threshold is the recall/precision
   knob and is saved with the model.
2. Consecutive redacted characters form a span; a new span starts where the most likely tag is a
   ``B`` tag. The span's label is the label with the highest summed probability over its characters.
3. Leading/trailing whitespace is trimmed from each span.
4. Optional word snapping extends spans to whole alphanumeric words (a partly redacted word still
   leaks the rest of the word), then merges spans that now overlap.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def decode(
    text: str,
    probs: np.ndarray,
    labels: Sequence[str],
    threshold: float = 0.5,
    word_snap: bool = False,
) -> list[dict]:
    if probs.shape != (len(text), 1 + 2 * len(labels)):
        raise ValueError(f"probs shape {probs.shape} does not match text/labels")
    p_entity = 1.0 - probs[:, 0]
    begin, inside = probs[:, 1::2], probs[:, 2::2]
    label_scores = begin + inside
    best_tag = probs.argmax(axis=1)
    starts_new = (best_tag > 0) & ((best_tag - 1) % 2 == 0)  # most likely tag is a B tag

    runs: list[list[int]] = []
    for i in np.flatnonzero(p_entity >= threshold):
        if runs and runs[-1][1] == i and not starts_new[i]:
            runs[-1][1] = i + 1
        else:
            runs.append([int(i), int(i) + 1])

    spans = []
    for start, end in runs:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append([start, end])

    if word_snap:
        for span in spans:
            while span[0] > 0 and text[span[0] - 1].isalnum():
                span[0] -= 1
            while span[1] < len(text) and text[span[1]].isalnum():
                span[1] += 1
        merged: list[list[int]] = []
        for span in sorted(spans):
            if merged and span[0] < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], span[1])
            else:
                merged.append(span)
        spans = merged

    out = []
    for start, end in spans:
        k = int(label_scores[start:end].sum(axis=0).argmax())
        score = float(p_entity[start:end].mean())
        out.append({"start": start, "end": end, "label": labels[k], "score": round(score, 4)})
    return out
