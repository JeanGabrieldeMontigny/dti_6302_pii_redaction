"""Training for the Keras character taggers: windowed arrays, tf.data, fit."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence

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


def train_model(
    cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vocab: CharVocab,
    labels: Sequence[str],
    callbacks: list | None = None,
):
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
    history = model.fit(
        make_dataset(x, y, w, tcfg["batch_size"], shuffle=True, seed=seed),
        validation_data=make_dataset(vx, vy, vw, tcfg["batch_size"], shuffle=False, seed=seed),
        epochs=tcfg["epochs"],
        shuffle=False,  # the tf.data pipeline already shuffles (seeded)
        callbacks=[
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
