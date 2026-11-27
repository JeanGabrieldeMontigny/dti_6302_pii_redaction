"""Turn raw records into clean, typed records with canonical spans.

Decisions (each is counted and reported, see ``PreprocessStats``):
- ``spans`` is parsed from its Python-literal string into ``[{start, end, label}]``.
- Character offsets are the source of truth. The raw span ``text`` field is dropped because it is
  sometimes lower-cased (e.g. ``Spanish`` -> ``spanish``) or stored as an int.
- Documents with overlapping spans are dropped: a flat tagging scheme cannot represent them and
  they are rare (<0.1%).
- Exact-duplicate texts within a split are dropped (keep first).
- ``text_tagged`` and ``document_description`` are dropped: derivable or unused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from src.data.schema import PROCESSED_COLUMNS, Span, parse_raw_spans


@dataclass
class PreprocessStats:
    input_docs: int = 0
    output_docs: int = 0
    dropped_overlap_docs: int = 0
    dropped_duplicate_docs: int = 0
    span_text_case_fixed: int = 0
    span_text_non_string: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def canonical_spans(text: str, raw_spans: list[dict], stats: PreprocessStats) -> list[Span] | None:
    """Return sorted canonical spans, or ``None`` if the document must be dropped."""
    spans: list[Span] = []
    for s in raw_spans:
        start, end = int(s["start"]), int(s["end"])
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"span out of bounds: {s!r} for text of length {len(text)}")
        raw_text = s.get("text")
        if not isinstance(raw_text, str):
            stats.span_text_non_string += 1
        elif raw_text != text[start:end]:
            stats.span_text_case_fixed += 1
        spans.append({"start": start, "end": end, "label": str(s["label"])})
    spans.sort(key=lambda s: (s["start"], s["end"]))
    for a, b in zip(spans, spans[1:], strict=False):
        if b["start"] < a["end"]:
            return None
    return spans


def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, PreprocessStats]:
    stats = PreprocessStats(input_docs=len(raw))
    keep_rows, spans_col = [], []
    for idx, text, raw_spans in zip(raw.index, raw["text"], raw["spans"], strict=True):
        spans = canonical_spans(text, parse_raw_spans(raw_spans), stats)
        if spans is None:
            stats.dropped_overlap_docs += 1
            continue
        keep_rows.append(idx)
        spans_col.append(spans)

    df = raw.loc[keep_rows].copy()
    df["spans"] = spans_col
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    stats.dropped_duplicate_docs = before - len(df)
    df = df[PROCESSED_COLUMNS].reset_index(drop=True)
    stats.output_docs = len(df)
    return df, stats
