"""End-to-end smoke test: raw files -> validation -> splits -> baseline logged to MLflow."""

import json
from pathlib import Path

import mlflow
import pytest

from pipelines import steps
from src.config import load_config
from src.data.ingest import write_manifest
from src.data.validate import DataValidationError
from tests.conftest import make_raw_split


@pytest.fixture
def cfg(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    make_raw_split(60).to_parquet(raw / "train.parquet", index=False)
    make_raw_split(20, offset=1000).to_parquet(raw / "test.parquet", index=False)
    write_manifest(raw, "fixture", "local", ["train", "test"])
    return load_config(
        overrides={
            "data": {
                "raw_dir": str(raw),
                "processed_dir": str(tmp_path / "processed"),
                "val_fraction": 0.2,
            },
            "validation": {
                "report_dir": str(tmp_path / "reports"),
                "min_spans_per_label": 0,
                "warn_label_js_distance": 1.0,
            },
            "evaluation": {"report_dir": str(tmp_path / "baseline")},
            "tracking": {
                "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                "artifact_location": str(tmp_path / "mlartifacts"),
                "experiment_name": "smoke",
            },
        }
    )


def test_pipeline_end_to_end(cfg, tmp_path):
    steps.validate_raw(cfg)
    out = steps.preprocess(cfg)
    assert out["counts"]["train"] > 0 and out["counts"]["validation"] > 0
    steps.validate_processed(cfg)
    result = steps.evaluate_baseline(cfg)

    assert (tmp_path / "reports" / "raw_validation.md").exists()
    assert (tmp_path / "baseline" / "regex_baseline_validation.md").exists()
    summary = json.loads((tmp_path / "baseline" / "regex_baseline_validation.json").read_text())
    assert summary["run_id"] == result["run_id"]
    assert summary["git_dirty"] in {"true", "false", "unknown"}
    manifest = json.loads((tmp_path / "processed" / "manifest.json").read_text())
    assert manifest["data_version"] == steps.data_version(cfg)

    run = mlflow.MlflowClient(cfg["tracking"]["tracking_uri"]).get_run(result["run_id"])
    assert run.data.tags["data_version"] == manifest["data_version"]
    assert 0.0 < run.data.metrics["validation.detection_recall"] <= 1.0


def test_pipeline_stops_on_bad_data(cfg):
    raw = Path(cfg["data"]["raw_dir"])
    make_raw_split(10).assign(locale="ca").to_parquet(raw / "train.parquet", index=False)
    write_manifest(raw, "fixture", "local", ["train", "test"])
    with pytest.raises(DataValidationError):
        steps.validate_raw(cfg)


def test_raw_files_must_match_manifest(cfg):
    raw = Path(cfg["data"]["raw_dir"])
    make_raw_split(30).to_parquet(raw / "train.parquet", index=False)  # swapped, valid data
    with pytest.raises(RuntimeError, match="does not match"):
        steps.validate_raw(cfg)


def test_stale_processed_data_is_rejected(cfg):
    steps.preprocess(cfg)
    cfg["data"]["val_fraction"] = 0.3  # config changed without re-running preprocess
    with pytest.raises(RuntimeError, match="re-run preprocess"):
        steps.evaluate_baseline(cfg)
