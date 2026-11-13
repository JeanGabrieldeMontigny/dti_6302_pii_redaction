"""Span-level metrics shared by every model (baselines and Keras models).

- **Detection recall (primary):** share of gold entities whose characters are covered by
  predicted spans of *any* type (at least ``min_coverage`` of them). Reported overall and per label.
- **Detection precision:** share of predicted (redacted) characters that fall inside gold spans.
- **Type macro-F1 (secondary):** conditional on detection. Over detected gold entities only, the
  predicted type is the label of the predicted span with the largest overlap; F1 is computed per
  label over the labels that occur among those (gold or predicted) and macro-averaged. Labels with
  no detected entities get ``type_f1 = None`` and are excluded, so this metric does not double-count
  detection misses (those are measured by detection recall). With no detections it is 0.0.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from src.data.schema import Span


def _merge(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _covered(start: int, end: int, merged: list[tuple[int, int]]) -> int:
    return sum(max(0, min(end, e) - max(start, s)) for s, e in merged)


class SpanMetrics:
    """Accumulates counts document by document; call ``result()`` at the end."""

    def __init__(self, min_coverage: float = 1.0):
        self.min_coverage = min_coverage
        self.gold_by_label: Counter = Counter()
        self.detected_by_label: Counter = Counter()
        self.pred_chars = 0
        self.pred_chars_correct = 0
        self.gold_chars = 0
        self.gold_chars_covered = 0
        self.type_pairs: Counter = Counter()  # (gold_label, pred_label) for detected entities
        self.n_docs = 0

    def update(self, gold: Sequence[Span], pred: Sequence[Span]) -> None:
        self.n_docs += 1
        gold_merged = _merge((g["start"], g["end"]) for g in gold)
        pred_merged = _merge((p["start"], p["end"]) for p in pred)

        self.pred_chars += sum(e - s for s, e in pred_merged)
        self.pred_chars_correct += sum(_covered(s, e, gold_merged) for s, e in pred_merged)
        self.gold_chars += sum(e - s for s, e in gold_merged)
        self.gold_chars_covered += sum(_covered(s, e, pred_merged) for s, e in gold_merged)

        for g in gold:
            length = g["end"] - g["start"]
            self.gold_by_label[g["label"]] += 1
            covered = _covered(g["start"], g["end"], pred_merged)
            if covered == 0 or covered / length < self.min_coverage:
                continue
            self.detected_by_label[g["label"]] += 1
            best = max(
                pred,
                key=lambda p: min(g["end"], p["end"]) - max(g["start"], p["start"]),
            )
            self.type_pairs[(g["label"], best["label"])] += 1

    def result(self) -> dict:
        gold_total = sum(self.gold_by_label.values())
        detected_total = sum(self.detected_by_label.values())
        per_label = {}
        for label in sorted(self.gold_by_label):
            support = self.gold_by_label[label]
            per_label[label] = {
                "support": support,
                "detection_recall": self.detected_by_label[label] / support,
                "type_f1": self._type_f1(label) if label in self._type_labels() else None,
            }
        f1s = [self._type_f1(label) for label in sorted(self._type_labels())]
        return {
            "n_docs": self.n_docs,
            "n_entities": gold_total,
            "detection_recall": _safe(detected_total, gold_total),
            "char_precision": _safe(self.pred_chars_correct, self.pred_chars),
            "char_recall": _safe(self.gold_chars_covered, self.gold_chars),
            "type_macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "type_accuracy": _safe(
                sum(n for (g, p), n in self.type_pairs.items() if g == p), detected_total
            ),
            "per_label": per_label,
        }

    def _type_labels(self) -> set[str]:
        return {g for g, _ in self.type_pairs} | {p for _, p in self.type_pairs}

    def _type_f1(self, label: str) -> float:
        tp = self.type_pairs[(label, label)]
        fn = sum(n for (g, p), n in self.type_pairs.items() if g == label and p != label)
        fp = sum(n for (g, p), n in self.type_pairs.items() if p == label and g != label)
        return _safe(2 * tp, 2 * tp + fp + fn)


def _safe(num: float, den: float) -> float:
    return num / den if den else 0.0


def evaluate_spans(
    golds: Iterable[Sequence[Span]],
    preds: Iterable[Sequence[Span]],
    slices: dict[str, Sequence[str]] | None = None,
    min_coverage: float = 1.0,
) -> dict:
    """Overall metrics plus the same metrics per value of each slice column."""
    golds, preds = list(golds), list(preds)
    overall = SpanMetrics(min_coverage)
    by_slice: dict[str, dict[str, SpanMetrics]] = defaultdict(dict)
    for i, (g, p) in enumerate(zip(golds, preds, strict=True)):
        overall.update(g, p)
        for col, values in (slices or {}).items():
            key = values[i]
            if key not in by_slice[col]:
                by_slice[col][key] = SpanMetrics(min_coverage)
            by_slice[col][key].update(g, p)

    result = overall.result()
    result["slices"] = {
        col: {
            key: {k: v for k, v in m.result().items() if k != "per_label"}
            for key, m in sorted(groups.items())
        }
        for col, groups in by_slice.items()
    }
    return result
