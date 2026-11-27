"""Decode sweep, comparison report, progress/status, and the reference redactor."""

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pipelines.status import describe
from src.models.compare import build_comparison
from src.models.sweep import decode_sweep, recommend
from src.serving.redactor import RedactionError, Redactor

LABELS = ["first_name", "email"]


class FakeTagger:
    """Predicts 'Ann' as a name with high confidence and 'x' with low confidence."""

    name = "fake"
    labels = LABELS
    threshold, word_snap = 0.5, False

    def predict_probs(self, texts):
        for text in texts:
            p = np.zeros((len(text), 5), dtype=np.float32)
            p[:, 0] = 1.0
            for i, ch in enumerate(text):
                if ch in "Ann":
                    p[i] = [0.1, 0.9 if i == text.index("A") else 0.0, 0.0, 0, 0]
                    p[i, 2] = 0.0 if i == text.index("A") else 0.9
                if ch == "x":
                    p[i] = [0.7, 0.3, 0, 0, 0]
            yield p

    def predict_batch(self, texts, threshold=None, word_snap=None):
        from src.models.decode import decode

        t = self.threshold if threshold is None else threshold
        return [
            decode(x, p, LABELS, t, bool(word_snap))
            for x, p in zip(texts, self.predict_probs(texts), strict=True)
        ]


def test_decode_sweep_and_recommendation():
    df = pd.DataFrame(
        {
            "text": ["Hi Ann x"],
            "spans": [[{"start": 3, "end": 6, "label": "first_name"}]],
        }
    )
    table = decode_sweep(FakeTagger(), df, [0.2, 0.5], [False])
    low, high = table.set_index("threshold").loc[0.2], table.set_index("threshold").loc[0.5]
    assert low["detection_recall"] == high["detection_recall"] == 1.0
    assert low["char_precision"] < high["char_precision"] == 1.0  # 'x' redacted only at 0.2
    assert recommend(table, 0.9)["threshold"] == 0.5
    assert recommend(table, 1.1) is None


def _summary(d, name, recall, **extra):
    (d / f"{name}_validation.json").write_text(
        json.dumps(
            {
                "detection_recall": recall,
                "char_precision": 0.9,
                "char_recall": 0.8,
                "type_macro_f1": 0.5,
                "data_version": "v1",
                "git_dirty": "false",
                "git_commit": "abc",
                **extra,
            }
        )
    )
    pd.DataFrame(
        {"label": ["email", "city"], "support": [10, 5], "detection_recall": [recall, 0.0]}
    ).to_csv(d / f"{name}_validation_per_label.csv", index=False)


def test_comparison_report(tmp_path):
    base, exp = tmp_path / "baseline", tmp_path / "experiments"
    base.mkdir()
    exp.mkdir()
    _summary(base, "regex_baseline", 0.3)
    _summary(exp, "e1", 0.5, n_params=1000, decode_sweep_best=None)
    _summary(exp, "e2", 0.8, git_dirty="true")
    out = build_comparison(base, exp, "validation", 0.85)
    assert out["models"] == 3
    md = (exp / "comparison.md").read_text()
    assert "| regex_baseline |" in md and "| e2 |" in md
    assert "no: uncommitted code" in md  # e2 is flagged as not citable
    assert "**0.800**" in md  # best model per label is bolded
    table = pd.read_csv(exp / "comparison.csv")
    assert list(table["model"]) == ["regex_baseline", "e1", "e2"]


def test_status_flags_stalled_runs():
    now = datetime(2026, 11, 27, 20, 0, tzinfo=UTC)
    state = {
        "run": "e2-abc",
        "status": "training",
        "epoch": 3,
        "epochs": 10,
        "step": 100,
        "steps": 400,
        "progress": 0.225,
        "loss": 0.03,
        "last_epoch": {"loss": 0.03, "val_loss": 0.031},
        "epoch_seconds": [600, 620],
        "eta_total_s": 4800,
        "eta_finish": "2026-11-27T21:20:00+00:00",
        "updated": (now - timedelta(seconds=30)).isoformat(),
    }
    text = describe(state, now)
    local = datetime(2026, 11, 27, 21, 20, tzinfo=UTC).astimezone().strftime("%H:%M")
    assert "epoch 3/10" in text and f"ETA: 1h20m (around {local})" in text
    assert "updated 0m30s ago" in text  # a UTC writer (container) and a local reader agree
    state["updated"] = (now - timedelta(hours=1)).isoformat()
    assert "NOT RUNNING" in describe(state, now)


def test_progress_reporter_writes_eta(tmp_path):
    import keras

    from src.models.train import progress_reporter

    path = tmp_path / "progress.json"
    model = keras.Sequential([keras.Input((2,)), keras.layers.Dense(1)])
    model.compile("sgd", "mse")
    x, y = np.zeros((32, 2)), np.zeros((32, 1))
    model.fit(x, y, epochs=2, batch_size=8, verbose=0, callbacks=[progress_reporter(path, "t")])
    state = json.loads(path.read_text())
    assert state["status"] == "finished" and state["epoch"] == 2 and state["epochs"] == 2
    assert state["progress"] == 1.0 and len(state["epoch_seconds"]) == 2


def test_redactor_contract():
    r = Redactor(FakeTagger(), "fake-1")
    out = r.redact("Hi Ann")
    assert out["redacted_text"] == "Hi [FIRST_NAME]"
    assert out["model_version"] == "fake-1" and out["offset_unit"] == "codepoint"
    assert out["spans"][0]["label"] == "first_name"
    for bad, code in [
        ("", "EMPTY_INPUT"),
        ("   ", "EMPTY_INPUT"),
        (42, "INVALID_TYPE"),
        ("a" * 20_001, "INPUT_TOO_LONG"),
    ]:
        with pytest.raises(RedactionError) as exc:
            r.redact(bad)
        assert exc.value.code == code
