"""Run a span model over a split and compute metrics, plus a per-label report."""

from __future__ import annotations

import time
from typing import Protocol

import pandas as pd

from src.data.schema import Span
from src.models.metrics import evaluate_spans


class SpanModel(Protocol):
    name: str

    def predict_batch(self, texts) -> list[list[Span]]: ...

    def params(self) -> dict: ...


def evaluate_model(
    model: SpanModel, df: pd.DataFrame, slice_columns: list[str], min_coverage: float = 1.0
) -> dict:
    t0 = time.perf_counter()
    preds = model.predict_batch(df["text"].tolist())
    elapsed = time.perf_counter() - t0
    result = evaluate_spans(
        df["spans"].tolist(),
        preds,
        slices={c: df[c].tolist() for c in slice_columns},
        min_coverage=min_coverage,
    )
    # Throughput on the training machine only; browser latency is measured separately.
    result["ms_per_doc"] = 1000 * elapsed / max(len(df), 1)
    return result


def per_label_table(result: dict) -> pd.DataFrame:
    rows = [{"label": k, **v} for k, v in result["per_label"].items()]
    return pd.DataFrame(rows).sort_values("detection_recall", ascending=False)


def render_markdown(result: dict, model_name: str, split: str) -> str:
    lines = [
        f"# {model_name} on {split}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "n_docs",
        "n_entities",
        "detection_recall",
        "char_precision",
        "char_recall",
        "type_macro_f1",
        "type_accuracy",
        "ms_per_doc",
    ]:
        value = result[key]
        lines.append(
            f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |"
        )
    lines += [
        "",
        "## Per label",
        "",
        "| Label | Support | Detection recall | Type F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, r in per_label_table(result).iterrows():
        type_f1 = "-" if pd.isna(r.type_f1) else f"{r.type_f1:.3f}"
        lines.append(f"| {r.label} | {r.support} | {r.detection_recall:.3f} | {type_f1} |")
    for col, groups in result["slices"].items():
        lines += [
            "",
            f"## Slice: {col}",
            "",
            "| Value | Docs | Detection recall | Char precision |",
            "| --- | ---: | ---: | ---: |",
        ]
        for key, m in groups.items():
            recall, precision = m["detection_recall"], m["char_precision"]
            lines.append(f"| {key} | {m['n_docs']} | {recall:.3f} | {precision:.3f} |")
    return "\n".join(lines) + "\n"
