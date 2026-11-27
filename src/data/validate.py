"""Data validation: schema checks plus statistical/domain rules, run before training.

Each rule produces a ``CheckResult``. ``error`` checks that fail stop the pipeline
(``DataValidationError``); ``warning`` checks are reported but do not block. Every run writes a
JSON report (machine-readable evidence) and a Markdown data-quality summary.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import (
    DOCUMENT_FORMATS,
    LABEL_SET,
    LABELS,
    LOCALES,
    PROCESSED_COLUMNS,
    RAW_COLUMNS,
    parse_raw_spans,
)


class DataValidationError(RuntimeError):
    pass


@dataclass
class CheckResult:
    name: str
    split: str
    passed: bool
    severity: str  # "error" | "warning"
    observed: Any
    threshold: Any = None
    detail: str = ""


@dataclass
class ValidationReport:
    stage: str
    checks: list[CheckResult] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def failed_errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def failed_warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def passed(self) -> bool:
        return not self.failed_errors

    def add(self, name, split, passed, observed, threshold=None, severity="error", detail=""):
        self.checks.append(
            CheckResult(name, split, bool(passed), severity, observed, threshold, detail)
        )

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "n_errors": len(self.failed_errors),
            "n_warnings": len(self.failed_warnings),
            "checks": [asdict(c) for c in self.checks],
            "stats": self.stats,
        }

    def raise_if_failed(self) -> None:
        if not self.passed:
            lines = [f"{c.split}/{c.name}: observed={c.observed}" for c in self.failed_errors]
            raise DataValidationError(f"{self.stage} validation failed:\n  " + "\n  ".join(lines))


# --------------------------------------------------------------------------- helpers


def js_distance(p: dict[str, int], q: dict[str, int]) -> float:
    """Jensen-Shannon distance (base 2, in [0, 1]) between two count distributions."""
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0) for k in keys], dtype=float)
    b = np.array([q.get(k, 0) for k in keys], dtype=float)
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2

    def kl(x, y):
        nz = x > 0
        return float(np.sum(x[nz] * np.log2(x[nz] / y[nz])))

    return math.sqrt(max(0.0, (kl(a, m) + kl(b, m)) / 2))


def _rate(n: int, d: int) -> float:
    return round(n / d, 6) if d else 0.0


def _text_stats(text: pd.Series) -> dict:
    lengths = text.str.len()
    q = lengths.quantile([0.0, 0.05, 0.5, 0.95, 1.0])
    return {f"p{int(k * 100)}": int(v) for k, v in q.items()} | {"mean": round(lengths.mean(), 1)}


def _split_stats(df: pd.DataFrame, n_spans: int, label_counts: Counter) -> dict:
    return {
        "docs": len(df),
        "unique_uids": int(df["uid"].nunique()),
        "spans": n_spans,
        "spans_per_doc": round(n_spans / max(len(df), 1), 2),
        "locale": df["locale"].value_counts().to_dict(),
        "document_format": df["document_format"].value_counts().to_dict(),
        "domains": int(df["domain"].nunique()),
        "document_types": int(df["document_type"].nunique()),
        "text_chars": _text_stats(df["text"]),
        "label_counts": dict(label_counts.most_common()),
    }


def _check_frame_basics(
    report: ValidationReport, name: str, df: pd.DataFrame, columns: list[str], cfg: dict
) -> bool:
    """Schema/type/null/category checks shared by raw and processed stages. False = stop."""
    missing = [c for c in columns if c not in df.columns]
    report.add("schema.required_columns", name, not missing, missing or "ok", columns)
    if missing:
        return False

    str_cols = [c for c in columns if c != "spans" or report.stage == "raw"]
    non_str = [c for c in str_cols if not pd.api.types.is_string_dtype(df[c])]
    report.add("schema.string_dtypes", name, not non_str, non_str or "ok")
    if non_str:  # string checks below would crash on other dtypes
        return False

    nulls = {c: int(df[c].isna().sum()) for c in columns}
    nulls = {c: n for c, n in nulls.items() if n}
    report.add("missing.no_nulls", name, not nulls, nulls or 0, 0)
    if nulls:
        return False

    blank = {c: int((df[c].str.strip() == "").sum()) for c in str_cols if c != "spans"}
    blank = {c: n for c, n in blank.items() if n}
    report.add("missing.no_blank_strings", name, not blank, blank or 0, 0)

    bad_locale = sorted(set(df["locale"]) - LOCALES)
    report.add("domain.locale_values", name, not bad_locale, bad_locale or "ok", sorted(LOCALES))
    bad_fmt = sorted(set(df["document_format"]) - DOCUMENT_FORMATS)
    report.add(
        "domain.document_format_values",
        name,
        not bad_fmt,
        bad_fmt or "ok",
        sorted(DOCUMENT_FORMATS),
    )

    lengths = df["text"].str.len()
    lo, hi = cfg["min_text_chars"], cfg["max_text_chars"]
    out_of_range = int(((lengths < lo) | (lengths > hi)).sum())
    report.add("stat.text_length_in_range", name, out_of_range == 0, out_of_range, [lo, hi])

    per_uid = int(df.groupby("uid").size().max())
    report.add(
        "domain.records_per_uid",
        name,
        per_uid <= cfg["max_records_per_uid"],
        per_uid,
        cfg["max_records_per_uid"],
    )
    return True


# --------------------------------------------------------------------------- raw stage


def _scan_raw_spans(df: pd.DataFrame) -> dict:
    """Single pass over the raw span strings collecting every span-level quality signal."""
    c = Counter()
    labels = Counter()
    unknown_labels = Counter()
    examples: dict[str, list] = {}

    def example(kind, item):
        examples.setdefault(kind, [])
        if len(examples[kind]) < 3:
            examples[kind].append(item)

    for uid, text, raw in zip(df["uid"], df["text"], df["spans"], strict=True):
        try:
            spans = parse_raw_spans(raw)
        except (ValueError, SyntaxError) as e:
            c["unparseable_docs"] += 1
            example("unparseable", {"uid": uid, "error": str(e)[:100]})
            continue
        if not spans:
            c["empty_docs"] += 1
        bounds = []
        for s in spans:
            c["spans"] += 1
            if not isinstance(s, dict) or not {"start", "end", "label"} <= set(s):
                c["malformed_spans"] += 1
                continue
            start, end, label = s["start"], s["end"], s["label"]
            if not (isinstance(start, int) and isinstance(end, int)):
                c["malformed_spans"] += 1
                continue
            if isinstance(label, str) and label in LABEL_SET:
                labels[label] += 1
            else:
                unknown_labels[str(label)] += 1
            if not (0 <= start < end <= len(text)):
                c["out_of_bounds_spans"] += 1
                example("out_of_bounds", {"uid": uid, "span": s})
                continue
            bounds.append((start, end))
            sub = text[start:end]
            span_text = s.get("text")
            if not isinstance(span_text, str):
                c["non_string_span_text"] += 1
                span_text = str(span_text)
            if span_text != sub:
                if span_text.lower() == sub.lower():
                    c["case_mismatch_spans"] += 1
                    example("case_mismatch", {"in_text": sub, "span_text": span_text})
                else:
                    c["text_mismatch_spans"] += 1
                    example("text_mismatch", {"uid": uid, "in_text": sub, "span_text": span_text})
            if sub != sub.strip():
                c["edge_whitespace_spans"] += 1
        bounds.sort()
        if any(b[0] < a[1] for a, b in zip(bounds, bounds[1:], strict=False)):
            c["overlap_docs"] += 1
            example("overlap", {"uid": uid})
    return {"counts": c, "labels": labels, "unknown_labels": unknown_labels, "examples": examples}


def validate_raw(splits: dict[str, pd.DataFrame], cfg: dict) -> ValidationReport:
    """Validate raw Hugging Face splits. ``cfg`` is the ``validation`` config section."""
    report = ValidationReport(stage="raw")
    label_counts: dict[str, Counter] = {}
    for name, df in splits.items():
        if not _check_frame_basics(report, name, df, RAW_COLUMNS, cfg):
            continue
        scan = _scan_raw_spans(df)
        c, n_docs, n_spans = scan["counts"], len(df), scan["counts"]["spans"]
        label_counts[name] = scan["labels"]

        report.add(
            "schema.spans_parseable", name, c["unparseable_docs"] == 0, c["unparseable_docs"], 0
        )
        report.add("schema.span_fields", name, c["malformed_spans"] == 0, c["malformed_spans"], 0)
        report.add(
            "domain.known_labels",
            name,
            not scan["unknown_labels"],
            dict(scan["unknown_labels"]) or "ok",
            f"{len(LABELS)} labels",
        )
        report.add(
            "domain.span_in_bounds",
            name,
            c["out_of_bounds_spans"] == 0,
            c["out_of_bounds_spans"],
            0,
        )
        report.add(
            "domain.span_text_matches_offsets",
            name,
            c["text_mismatch_spans"] == 0,
            c["text_mismatch_spans"],
            0,
            detail="span text must equal text[start:end] up to letter case",
        )
        rate = _rate(c["case_mismatch_spans"], n_spans)
        report.add(
            "stat.case_mismatch_span_rate",
            name,
            rate <= cfg["max_case_mismatch_span_rate"],
            rate,
            cfg["max_case_mismatch_span_rate"],
            detail="fixed in preprocessing (offsets win)",
        )
        rate = _rate(c["overlap_docs"], n_docs)
        report.add(
            "stat.overlap_doc_rate",
            name,
            rate <= cfg["max_overlap_doc_rate"],
            rate,
            cfg["max_overlap_doc_rate"],
            detail="dropped in preprocessing",
        )
        rate = _rate(c["empty_docs"], n_docs)
        report.add(
            "stat.empty_doc_rate",
            name,
            rate <= cfg["max_empty_doc_rate"],
            rate,
            cfg["max_empty_doc_rate"],
        )
        rate = _rate(int(df["text"].duplicated().sum()), n_docs)
        report.add(
            "stat.duplicate_text_rate",
            name,
            rate <= cfg["max_duplicate_text_rate"],
            rate,
            cfg["max_duplicate_text_rate"],
            detail="dropped in preprocessing",
        )

        stats = _split_stats(df, n_spans, scan["labels"])
        stats["span_quality"] = dict(c)
        stats["examples"] = scan["examples"]
        report.stats[name] = stats

    if "train" in label_counts:
        train = label_counts["train"]
        rare = {
            lbl: train.get(lbl, 0)
            for lbl in LABELS
            if train.get(lbl, 0) < cfg["min_spans_per_label"]
        }
        report.add(
            "stat.min_spans_per_label", "train", not rare, rare or "ok", cfg["min_spans_per_label"]
        )
    _cross_split_checks(report, splits, label_counts, cfg)
    return report


def _cross_split_checks(report, splits, label_counts, cfg) -> None:
    if "train" in splits and "test" in splits:
        tr, te = splits["train"], splits["test"]
        if "uid" in tr and "uid" in te:
            n = len(set(tr["uid"]) & set(te["uid"]))
            report.add("leakage.uid_overlap", "train|test", n == 0, n, 0)
            n = len(set(tr["text"]) & set(te["text"]))
            report.add("leakage.text_overlap", "train|test", n == 0, n, 0)
    if "train" in label_counts and "test" in label_counts:
        d = round(js_distance(label_counts["train"], label_counts["test"]), 4)
        report.add(
            "drift.label_distribution_train_vs_test",
            "train|test",
            d <= cfg["warn_label_js_distance"],
            d,
            cfg["warn_label_js_distance"],
            severity="warning",
            detail="Jensen-Shannon distance of label shares",
        )


# --------------------------------------------------------------------------- processed stage


def validate_processed(
    splits: dict[str, pd.DataFrame], cfg: dict, val_fraction: float | None = None
) -> ValidationReport:
    """Strict checks on cleaned splits: canonical spans and no leakage between splits."""
    report = ValidationReport(stage="processed")
    label_counts = {}
    for name, df in splits.items():
        if not _check_frame_basics(report, name, df, PROCESSED_COLUMNS, cfg):
            continue
        bad_bounds = overlaps = unknown = n_spans = 0
        labels = Counter()
        for text, spans in zip(df["text"], df["spans"], strict=True):
            prev_end = 0
            for s in spans:
                n_spans += 1
                labels[s["label"]] += 1
                unknown += s["label"] not in LABEL_SET
                bad_bounds += not (0 <= s["start"] < s["end"] <= len(text))
                overlaps += s["start"] < prev_end
                prev_end = max(prev_end, s["end"])
        label_counts[name] = labels
        report.add("domain.known_labels", name, unknown == 0, unknown, 0)
        report.add("domain.span_in_bounds", name, bad_bounds == 0, bad_bounds, 0)
        report.add("domain.spans_sorted_non_overlapping", name, overlaps == 0, overlaps, 0)
        dups = int(df["text"].duplicated().sum())
        report.add("stat.no_duplicate_texts", name, dups == 0, dups, 0)
        report.stats[name] = _split_stats(df, n_spans, labels)

    from src.data.split import leakage_report

    for pair, overlap in leakage_report(splits).items():
        report.add(
            "leakage.uid_overlap", pair, overlap["uid_overlap"] == 0, overlap["uid_overlap"], 0
        )
        report.add(
            "leakage.text_overlap", pair, overlap["text_overlap"] == 0, overlap["text_overlap"], 0
        )

    if val_fraction is not None and {"train", "validation"} <= set(splits):
        n_tr, n_va = splits["train"]["uid"].nunique(), splits["validation"]["uid"].nunique()
        observed = round(n_va / max(n_tr + n_va, 1), 4)
        ok = abs(observed - val_fraction) <= max(0.02, val_fraction * 0.2)
        report.add("stat.validation_fraction", "train|validation", ok, observed, val_fraction)

    if "train" in label_counts and "validation" in label_counts:
        d = round(js_distance(label_counts["train"], label_counts["validation"]), 4)
        report.add(
            "drift.label_distribution_train_vs_validation",
            "train|validation",
            d <= cfg["warn_label_js_distance"],
            d,
            cfg["warn_label_js_distance"],
            severity="warning",
        )
    if "train" in label_counts and "test" in label_counts:
        d = round(js_distance(label_counts["train"], label_counts["test"]), 4)
        report.add(
            "drift.label_distribution_train_vs_test",
            "train|test",
            d <= cfg["warn_label_js_distance"],
            d,
            cfg["warn_label_js_distance"],
            severity="warning",
        )
    return report


# --------------------------------------------------------------------------- reporting


def write_report(report: ValidationReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{report.stage}_validation.json"
    md_path = out_dir / f"{report.stage}_validation.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    md_path.write_text(render_markdown(report))
    return json_path, md_path


def render_markdown(report: ValidationReport) -> str:
    icon = {True: "pass", False: "FAIL"}
    lines = [
        f"# Data validation: {report.stage}",
        "",
        f"**Result:** {'PASSED' if report.passed else 'FAILED'} "
        f"({len(report.failed_errors)} errors, {len(report.failed_warnings)} warnings)",
        "",
        "## Checks",
        "",
        "| Check | Split | Severity | Result | Observed | Threshold |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in report.checks:
        result = icon[c.passed] if c.passed or c.severity == "error" else "warn"
        lines.append(
            f"| `{c.name}` | {c.split} | {c.severity} | {result} | {_fmt(c.observed)} | "
            f"{_fmt(c.threshold)} |"
        )

    splits = list(report.stats)
    if splits:
        lines += ["", "## Split summary", "", "| | " + " | ".join(splits) + " |"]
        lines.append("| --- |" + " --- |" * len(splits))
        for key in ["docs", "unique_uids", "spans", "spans_per_doc", "domains", "document_types"]:
            lines.append(
                f"| {key} | " + " | ".join(str(report.stats[s][key]) for s in splits) + " |"
            )
        for key in ["locale", "document_format", "text_chars"]:
            lines.append(
                f"| {key} | " + " | ".join(_fmt(report.stats[s][key]) for s in splits) + " |"
            )
        if report.stage == "raw":
            lines.append(
                "| span_quality | "
                + " | ".join(_fmt(report.stats[s]["span_quality"]) for s in splits)
                + " |"
            )

        lines += ["", "## Label distribution (share of spans)", ""]
        lines.append("| Label | " + " | ".join(splits) + " |")
        lines.append("| --- |" + " ---: |" * len(splits))
        totals = {s: sum(report.stats[s]["label_counts"].values()) or 1 for s in splits}
        order = sorted(LABELS, key=lambda lbl: -report.stats[splits[0]]["label_counts"].get(lbl, 0))
        for lbl in order:
            cells = []
            for s in splits:
                n = report.stats[s]["label_counts"].get(lbl, 0)
                cells.append(f"{n:,} ({100 * n / totals[s]:.1f}%)")
            lines.append(f"| {lbl} | " + " | ".join(cells) + " |")

        examples = {
            s: report.stats[s].get("examples") for s in splits if report.stats[s].get("examples")
        }
        if examples:
            lines += ["", "## Examples of quality issues", ""]
            for s, ex in examples.items():
                for kind, items in ex.items():
                    lines.append(f"- **{s} / {kind}:** `{json.dumps(items, default=str)[:300]}`")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items()) or "-"
    if isinstance(value, list):
        return ", ".join(map(str, value)) or "-"
    return "-" if value is None else str(value)
