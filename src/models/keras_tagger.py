"""Keras character taggers.

All architectures map ``int32[batch, chars]`` -> ``float32[batch, chars, n_tags]`` (softmax over
BIO tags) and accept any sequence length, so the same model serves windows of any size.
They use only layers that browser runtimes support well (Embedding, Conv1D, LayerNormalization,
Dense, Add, Dropout); recurrent variants are added only once export has been verified for them.
"""

from __future__ import annotations

import keras
from keras import layers

ARCHITECTURES = ("cnn_small", "dilated_cnn")


def build_model(arch: str, vocab_size: int, n_tags: int, **params) -> keras.Model:
    if arch == "cnn_small":
        return _cnn_small(vocab_size, n_tags, **params)
    if arch == "dilated_cnn":
        return _dilated_cnn(vocab_size, n_tags, **params)
    raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHITECTURES}")


def _cnn_small(
    vocab_size: int,
    n_tags: int,
    embed_dim: int = 32,
    filters: int = 64,
    kernel_size: int = 5,
    n_layers: int = 3,
    dropout: float = 0.1,
    **_unused,
) -> keras.Model:
    """Plain stack of same-padded convolutions: the "small model" baseline."""
    inputs = keras.Input(shape=(None,), dtype="int32", name="char_ids")
    x = layers.Embedding(vocab_size, embed_dim, name="embed")(inputs)
    for i in range(n_layers):
        x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu", name=f"conv{i}")(
            x
        )
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_tags, activation="softmax", name="tags")(x)
    return keras.Model(inputs, outputs, name="cnn_small")


def _dilated_cnn(
    vocab_size: int,
    n_tags: int,
    embed_dim: int = 64,
    filters: int = 128,
    kernel_size: int = 3,
    dilations: tuple[int, ...] = (1, 2, 4, 8, 1, 2, 4, 8),
    dropout: float = 0.1,
    **_unused,
) -> keras.Model:
    """Residual dilated convolutions: wide context (receptive field grows with the dilations)."""
    inputs = keras.Input(shape=(None,), dtype="int32", name="char_ids")
    x = layers.Embedding(vocab_size, embed_dim, name="embed")(inputs)
    x = layers.Conv1D(filters, 5, padding="same", activation="relu", name="stem")(x)
    for i, d in enumerate(dilations):
        h = layers.Conv1D(
            filters, kernel_size, padding="same", dilation_rate=d, activation="relu", name=f"dil{i}"
        )(x)
        h = layers.Dropout(dropout)(h)
        x = layers.LayerNormalization(name=f"norm{i}")(layers.Add()([x, h]))
    outputs = layers.Dense(n_tags, activation="softmax", name="tags")(x)
    return keras.Model(inputs, outputs, name="dilated_cnn")
