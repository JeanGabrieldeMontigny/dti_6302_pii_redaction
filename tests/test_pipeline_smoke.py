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


def test_train_step_produces_versioned_model(cfg, tmp_path):
    """raw -> splits -> train a tiny Keras tagger -> evaluated, logged and reloadable."""
    from src.models.predict import KerasTagger

    cfg["features"] |= {"min_char_freq": 1, "window": 32, "stride": 24}
    cfg["model"] |= {"name": "smoke_cnn", "arch": "cnn_small", "filters": 8, "embed_dim": 8}
    cfg["train"] |= {"epochs": 1, "batch_size": 16, "report_dir": str(tmp_path / "experiments")}
    cfg["artifacts"] = {"dir": str(tmp_path / "artifacts")}
    steps.preprocess(cfg)
    result = steps.train(cfg)

    model_dir = Path(result["model_dir"])
    assert result["model_version"].startswith("smoke_cnn-")
    card = json.loads((model_dir / "model_card.json").read_text())
    assert card["mlflow_run_id"] == result["run_id"]
    assert (tmp_path / "experiments" / "smoke_cnn_validation.md").exists()

    run = mlflow.MlflowClient(cfg["tracking"]["tracking_uri"]).get_run(result["run_id"])
    assert run.data.tags["model_version"] == result["model_version"]
    assert run.data.tags["hardware"] in {"cpu"} or run.data.tags["hardware"].startswith("gpu:")
    assert "validation.detection_recall" in run.data.metrics

    # Decoding sweep: curve + table, and the summary carries what `compare` needs.
    exp = tmp_path / "experiments"
    assert (exp / "smoke_cnn_decode_sweep.png").exists()
    sweep = (exp / "smoke_cnn_decode_sweep.csv").read_text()
    assert sweep.startswith("threshold,word_snap,detection_recall")
    summary = json.loads((exp / "smoke_cnn_validation.json").read_text())
    for key in ("model_version", "n_params", "model_size_bytes", "train_seconds", "epochs_run"):
        assert key in summary
    assert "decode_sweep_best" in summary and summary["hardware"] == run.data.tags["hardware"]

    # The standalone steps work on the trained model.
    steps.decode_sweep(cfg)
    cfg["evaluation"]["report_dir"] = str(tmp_path / "baseline")
    steps.evaluate_baseline(cfg)
    assert steps.compare(cfg)["models"] == 2

    # The saved artifact alone is enough to run inference.
    tagger = KerasTagger.load(model_dir)
    text = "Hi, I'm Maria. Email me at user1us@example.com or call 201-555-1001. " * 3
    spans = tagger.predict(text)  # longer than one window
    assert all(0 <= s["start"] < s["end"] <= len(text) for s in spans)
    assert tagger.params()["vocab_sha256"] == card["vocab_sha256"]


def _tiny_training(cfg, tmp_path, epochs):
    cfg["features"] |= {"min_char_freq": 1, "window": 32, "stride": 24}
    cfg["model"] |= {"name": "smoke_cnn", "arch": "cnn_small", "filters": 8, "embed_dim": 8}
    cfg["train"] |= {
        "epochs": epochs,
        "batch_size": 16,
        "early_stopping_patience": epochs,  # never stop early in these tests
        "report_dir": str(tmp_path / "experiments"),
    }
    cfg["artifacts"] = {"dir": str(tmp_path / "artifacts")}


def test_interrupted_training_resumes_in_the_same_run(cfg, tmp_path, monkeypatch):
    """Stop after epoch 1 of 3, re-run the step: it continues at epoch 2 in the same run."""
    import keras

    from src.models import train as train_mod

    _tiny_training(cfg, tmp_path, epochs=3)
    steps.preprocess(cfg)
    real_logger = train_mod.mlflow_epoch_logger

    class Crash(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.inner = real_logger()

        def on_epoch_end(self, epoch, logs=None):
            self.inner.on_epoch_end(epoch, logs)
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(train_mod, "mlflow_epoch_logger", Crash)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        steps.train(cfg)
    ckpt_root = tmp_path / "artifacts" / "checkpoints"
    (ckpt_dir,) = ckpt_root.iterdir()
    first_run_id = json.loads((ckpt_dir / "run.json").read_text())["run_id"]

    monkeypatch.setattr(train_mod, "mlflow_epoch_logger", real_logger)
    result = steps.train(cfg)

    client = mlflow.MlflowClient(cfg["tracking"]["tracking_uri"])
    assert result["run_id"] == first_run_id
    assert result["model_version"] == f"smoke_cnn-{first_run_id[:8]}"
    # Epoch 0 from the first attempt + epochs 1-2 after resuming, each logged once.
    steps_logged = [m.step for m in client.get_metric_history(first_run_id, "epoch.loss")]
    assert steps_logged == [0, 1, 2]
    run = client.get_run(first_run_id)
    assert run.data.tags["resumed"] == "true"
    assert run.data.params["epochs_run"] == "3"
    assert run.info.status == "FINISHED"
    assert not any(ckpt_root.iterdir()), "backup must be removed after a completed run"


def test_resume_can_be_disabled(cfg, tmp_path):
    _tiny_training(cfg, tmp_path, epochs=1)
    steps.preprocess(cfg)
    stale = tmp_path / "artifacts" / "checkpoints" / "stale"
    stale.mkdir(parents=True)
    (stale / "run.json").write_text(json.dumps({"run_id": "nope", "git_commit": "x"}))
    from src.models.train import checkpoint_key

    real = stale.parent / checkpoint_key(cfg, steps.data_version(cfg))
    stale.rename(real)
    cfg["train"]["resume"] = False
    result = steps.train(cfg)  # must not try to reopen MLflow run "nope"
    assert result["run_id"] != "nope"


def test_checkpoint_key_tracks_training_settings_only(cfg):
    from src.models.train import checkpoint_key

    key = checkpoint_key(cfg, "v1")
    assert checkpoint_key(cfg, "v1") == key
    assert checkpoint_key(cfg, "v2") != key  # other data
    changed = {**cfg, "train": {**cfg["train"], "learning_rate": 0.1}}
    assert checkpoint_key(changed, "v1") != key
    moved = {**cfg, "train": {**cfg["train"], "report_dir": "elsewhere", "resume": False}}
    assert checkpoint_key(moved, "v1") == key  # output location / resume flag don't matter


def test_source_hash_ignores_line_endings(tmp_path):
    lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert steps.source_sha256(lf) == steps.source_sha256(crlf)
    lf.write_bytes(b"x = 1\ny = 3\n")
    assert steps.source_sha256(lf) != steps.source_sha256(crlf)


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
