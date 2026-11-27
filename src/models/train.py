"""Training for the Keras character taggers: windowed arrays, tf.data, fit."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.encode import make_windows, n_tags, spans_to_tags
from src.features.vocab import PAD_ID, CharVocab

log = logging.getLogger(__name__)


def subset_by_uid(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Deterministic share of persona groups (both locale variants of a uid stay together)."""
    if fraction >= 1.0:
        return df
    buckets = 10_000

    def keep(uid: str) -> bool:
        h = int(hashlib.sha256(f"subset:{seed}:{uid}".encode()).hexdigest(), 16)
        return h % buckets < fraction * buckets

    return df[df["uid"].map(keep)].reset_index(drop=True)


def build_arrays(
    df: pd.DataFrame,
    vocab: CharVocab,
    labels: Sequence[str],
    window: int,
    stride: int,
    o_class_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows of char ids, BIO tags and per-character loss weights (0 on padding)."""
    xs, ys = [], []
    for text, spans in zip(df["text"], df["spans"], strict=True):
        xs.append(make_windows(vocab.encode(text), window, stride, PAD_ID))
        ys.append(make_windows(spans_to_tags(len(text), spans, labels), window, stride, 0))
    x, y = np.concatenate(xs), np.concatenate(ys)
    w = (x != PAD_ID).astype(np.float32)
    if o_class_weight != 1.0:
        w *= np.where(y == 0, o_class_weight, 1.0).astype(np.float32)
    return x, y, w


def make_dataset(x, y, w, batch_size: int, shuffle: bool, seed: int):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y, w))
    if shuffle:
        ds = ds.shuffle(min(len(x), 50_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def mlflow_epoch_logger():
    """Keras callback logging each epoch's losses to the active MLflow run."""
    import keras
    import mlflow

    class _Logger(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            metrics = {f"epoch.{k}": float(v) for k, v in (logs or {}).items()}
            mlflow.log_metrics(metrics, step=epoch)

    return _Logger()


def hardware_info() -> str:
    """``gpu:<name>`` when TensorFlow sees a GPU, else ``cpu`` (recorded on every training run)."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "cpu"
    name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "unknown")
    return f"gpu:{name}"


def progress_reporter(path: Path, run_label: str, every_s: float = 15, log_every_s: float = 120):
    """Keras callback that keeps ``path`` (JSON) up to date with the epoch, step, losses and ETA,
    and logs a progress line every ``log_every_s`` seconds. Read by ``pipelines.status``."""
    import time
    from datetime import datetime, timedelta

    import keras

    class _Progress(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.epoch_seconds: list[float] = []
            self.last_logs: dict = {}
            self.epoch = 0
            self.t_epoch = self.t_write = self.t_log = time.monotonic()

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch, self.t_epoch = epoch, time.monotonic()

        def on_train_batch_end(self, batch, logs=None):
            now = time.monotonic()
            if now - self.t_write >= every_s:
                self._write(batch + 1, logs, now)

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_seconds.append(time.monotonic() - self.t_epoch)
            self.last_logs = {k: round(float(v), 5) for k, v in (logs or {}).items()}
            self._write(self.params.get("steps") or 0, logs, time.monotonic(), epoch_done=True)

        def on_train_end(self, logs=None):
            self._write(self.params.get("steps") or 0, logs, time.monotonic(), status="finished")

        def _write(self, step, logs, now, epoch_done=False, status="training"):
            epochs, steps = self.params.get("epochs") or 0, self.params.get("steps") or 0
            elapsed = now - self.t_epoch
            remaining_epochs = epochs - self.epoch - 1
            eta_epoch = 0.0 if epoch_done else elapsed / max(step, 1) * max(steps - step, 0)
            per_epoch = (
                sum(self.epoch_seconds) / len(self.epoch_seconds)
                if self.epoch_seconds
                else elapsed + eta_epoch
            )
            eta_total = 0.0 if status == "finished" else eta_epoch + remaining_epochs * per_epoch
            done = (self.epoch + (1 if epoch_done else step / max(steps, 1))) / max(epochs, 1)
            state = {
                "run": run_label,
                "status": status,
                "epoch": self.epoch + 1,
                "epochs": epochs,
                "step": step,
                "steps": steps,
                "progress": round(min(done, 1.0), 4),
                "loss": round(float((logs or {}).get("loss", float("nan"))), 5),
                "last_epoch": self.last_logs,
                "epoch_seconds": [round(s) for s in self.epoch_seconds],
                "eta_epoch_s": round(eta_epoch),
                "eta_total_s": round(eta_total),
                # UTC with offset: a container's clock (UTC) and the host's (local) agree.
                "eta_finish": (datetime.now(UTC) + timedelta(seconds=eta_total)).isoformat(
                    timespec="seconds"
                ),
                "updated": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            tmp.replace(path)  # atomic: a reader never sees a half-written file
            self.t_write = now
            if now - self.t_log >= log_every_s or epoch_done:
                self.t_log = now
                log.info(
                    "[%s] epoch %d/%d, step %d/%d (%.0f%% overall), loss %.4f, ETA %s (%s)",
                    run_label,
                    state["epoch"],
                    epochs,
                    step,
                    steps,
                    100 * state["progress"],
                    state["loss"],
                    _fmt_duration(eta_total),
                    (datetime.now() + timedelta(seconds=eta_total)).strftime("%H:%M"),
                )

    return _Progress()


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def checkpoint_key(cfg: dict, data_version: str) -> str:
    """Identifies one training configuration, so an interrupted run resumes only from its own
    backup: same experiment, settings, seed, data and MLflow store (the backup refers to a run in
    that store). Output paths and the resume flag don't count.
    """
    key = {
        "data_version": data_version,
        "tracking_uri": cfg["tracking"]["tracking_uri"],
        "seed": cfg["seed"],
        "val_fraction": cfg["data"]["val_fraction"],
        "features": cfg["features"],
        "model": cfg["model"],
        "train": {k: v for k, v in cfg["train"].items() if k not in ("report_dir", "resume")},
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    return f"{cfg['model']['name']}-{digest}"


def train_model(
    cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vocab: CharVocab,
    labels: Sequence[str],
    callbacks: list | None = None,
    backup_dir: Path | None = None,
):
    """Fit the configured model.

    With ``backup_dir``, the full training state (weights, optimizer, epoch) is saved after every
    epoch and restored when the same training is started again, so an interrupted run continues
    from its last completed epoch. The backup is deleted when training completes.
    """
    import keras

    from src.models.keras_tagger import build_model

    feat, mcfg, tcfg, seed = cfg["features"], cfg["model"], cfg["train"], cfg["seed"]
    keras.utils.set_random_seed(seed)

    x, y, w = build_arrays(
        train_df, vocab, labels, feat["window"], feat["stride"], tcfg["o_class_weight"]
    )
    vx, vy, vw = build_arrays(val_df, vocab, labels, feat["window"], feat["stride"])
    log.info("Training windows: %d, validation windows: %d", len(x), len(vx))

    model = build_model(
        mcfg["arch"],
        vocab.size,
        n_tags(labels),
        **{k: v for k, v in mcfg.items() if k not in ("arch", "name")},
    )
    model.compile(
        optimizer=keras.optimizers.Adam(tcfg["learning_rate"]),
        loss=keras.losses.SparseCategoricalCrossentropy(),
    )
    backup = []
    if backup_dir is not None:
        # First, so the epoch is saved before any later callback can stop training.
        # double_checkpoint keeps the previous backup if the process dies while writing.
        backup = [keras.callbacks.BackupAndRestore(str(backup_dir), double_checkpoint=True)]
    history = model.fit(
        make_dataset(x, y, w, tcfg["batch_size"], shuffle=True, seed=seed),
        validation_data=make_dataset(vx, vy, vw, tcfg["batch_size"], shuffle=False, seed=seed),
        epochs=tcfg["epochs"],
        shuffle=False,  # the tf.data pipeline already shuffles (seeded)
        callbacks=[
            *backup,
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=tcfg["early_stopping_patience"],
                restore_best_weights=True,
            ),
            *(callbacks or []),
        ],
        verbose=2,
    )
    return model, history.history, {"train_windows": len(x), "val_windows": len(vx)}
