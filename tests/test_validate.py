import pandas as pd
import pytest

from src.data.preprocess import preprocess
from src.data.split import split_train_val
from src.data.validate import (
    DataValidationError,
    js_distance,
    render_markdown,
    validate_processed,
    validate_raw,
)


def _failed(report):
    return {c.name for c in report.failed_errors}


def test_clean_raw_data_passes(raw_train, raw_test, validation_cfg):
    report = validate_raw({"train": raw_train, "test": raw_test}, validation_cfg)
    assert report.passed, _failed(report)
    assert "Data validation: raw" in render_markdown(report)


def test_missing_column_fails(raw_train, validation_cfg):
    report = validate_raw({"train": raw_train.drop(columns=["locale"])}, validation_cfg)
    assert "schema.required_columns" in _failed(report)
    with pytest.raises(DataValidationError):
        report.raise_if_failed()


def test_out_of_bounds_and_unknown_label_fail(raw_train, validation_cfg):
    df = raw_train.copy()
    df.loc[0, "spans"] = repr([{"start": 0, "end": 10_000, "text": "x", "label": "first_name"}])
    df.loc[1, "spans"] = repr(
        [{"start": 0, "end": 2, "text": df.loc[1, "text"][:2], "label": "favourite_colour"}]
    )
    report = validate_raw({"train": df}, validation_cfg)
    assert {"domain.span_in_bounds", "domain.known_labels"} <= _failed(report)


def test_case_mismatch_rate_threshold(raw_train, validation_cfg):
    df = raw_train.copy()
    for i in range(len(df) // 2):  # make many spans differ only by case
        spans = eval(df.loc[i, "spans"])  # noqa: S307 - trusted fixture
        spans[0]["text"] = spans[0]["text"].lower()
        df.loc[i, "spans"] = repr(spans)
    report = validate_raw({"train": df}, validation_cfg)
    assert "stat.case_mismatch_span_rate" in _failed(report)
    assert "domain.span_text_matches_offsets" not in _failed(report)


def test_non_string_column_reports_instead_of_crashing(raw_train, validation_cfg):
    df = raw_train.copy()
    df["locale"] = 1
    assert "schema.string_dtypes" in _failed(validate_raw({"train": df}, validation_cfg))


def test_unhashable_label_is_reported(raw_train, validation_cfg):
    df = raw_train.copy()
    df.loc[0, "spans"] = repr(
        [{"start": 0, "end": 2, "text": df.loc[0, "text"][:2], "label": ["x"]}]
    )
    assert "domain.known_labels" in _failed(validate_raw({"train": df}, validation_cfg))


def test_unknown_locale_fails(raw_train, validation_cfg):
    df = raw_train.copy()
    df.loc[0, "locale"] = "ca"
    assert "domain.locale_values" in _failed(validate_raw({"train": df}, validation_cfg))


def test_processed_detects_leakage(raw_train, validation_cfg):
    df, _ = preprocess(raw_train)
    train, val = split_train_val(df, 0.25, seed=42)
    assert validate_processed({"train": train, "validation": val}, validation_cfg).passed
    leaky = {"train": train, "validation": pd.concat([val, train.iloc[[0]]], ignore_index=True)}
    assert "leakage.uid_overlap" in _failed(validate_processed(leaky, validation_cfg))


def test_js_distance_bounds():
    assert js_distance({"a": 5, "b": 5}, {"a": 1, "b": 1}) == pytest.approx(0.0)
    assert js_distance({"a": 1}, {"b": 1}) == pytest.approx(1.0)
