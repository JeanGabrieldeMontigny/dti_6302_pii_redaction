"""Inference wrapper: text -> spans with a trained Keras tagger.

Everything needed to reproduce the model's input and output handling lives in
``preprocessing.json`` (vocabulary, label order, window/stride, decoding settings). Training,
evaluation, the Python reference redactor and the browser all read that one file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.features.encode import make_windows, stitch
from src.features.vocab import PAD_ID, CharVocab
from src.models.decode import decode

PREPROCESSING_FILE = "preprocessing.json"
MODEL_FILE = "model.keras"
FORMAT_VERSION = 1


def preprocessing_spec(
    vocab: CharVocab, labels: Sequence[str], window: int, stride: int, decode_cfg: dict
) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "input_unit": "unicode_code_point",
        "vocab": vocab.to_dict(),
        "vocab_sha256": vocab.sha256(),
        "labels": list(labels),
        "tag_scheme": "BIO: 0=O, B=1+2k, I=2+2k for labels[k]",
        "window": window,
        "stride": stride,
        "decode": {
            "threshold": float(decode_cfg["threshold"]),
            "word_snap": bool(decode_cfg["word_snap"]),
        },
    }


class KerasTagger:
    """Implements the ``SpanModel`` protocol used by ``src.models.evaluate``."""

    def __init__(self, model, spec: dict, name: str = "keras_tagger", batch_windows: int = 256):
        if spec.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported preprocessing format {spec.get('format_version')}")
        self.model = model
        self.spec = spec
        self.name = name
        self.vocab = CharVocab.from_dict(spec["vocab"])
        self.labels = list(spec["labels"])
        self.window = int(spec["window"])
        self.stride = int(spec["stride"])
        self.threshold = float(spec["decode"]["threshold"])
        self.word_snap = bool(spec["decode"]["word_snap"])
        self.batch_windows = batch_windows

    # ------------------------------------------------------------------ persistence

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(out_dir / MODEL_FILE)
        (out_dir / PREPROCESSING_FILE).write_text(json.dumps(self.spec, indent=1))

    @classmethod
    def load(cls, model_dir: Path, name: str | None = None) -> KerasTagger:
        import keras

        spec = json.loads((model_dir / PREPROCESSING_FILE).read_text())
        model = keras.models.load_model(model_dir / MODEL_FILE)
        return cls(model, spec, name=name or model_dir.name)

    # ------------------------------------------------------------------ inference

    def predict_probs(self, texts: Sequence[str]):
        """Yield per-character tag probabilities ``(len(text), n_tags)`` for each text, in order."""
        pending: list[tuple[int, np.ndarray]] = []  # (n_chars, windows) per document
        n_pending = 0
        for text in texts:
            windows = make_windows(self.vocab.encode(text), self.window, self.stride, PAD_ID)
            pending.append((len(text), windows))
            n_pending += len(windows)
            if n_pending >= self.batch_windows:
                yield from self._run(pending)
                pending, n_pending = [], 0
        if pending:
            yield from self._run(pending)

    def _run(self, pending: list[tuple[int, np.ndarray]]):
        batch = np.concatenate([w for _, w in pending])
        probs = np.asarray(self.model.predict_on_batch(batch))
        offset = 0
        for n_chars, windows in pending:
            doc = probs[offset : offset + len(windows)]
            offset += len(windows)
            yield stitch(doc, n_chars, self.window, self.stride)

    def predict_batch(
        self, texts: Sequence[str], threshold: float | None = None, word_snap: bool | None = None
    ) -> list[list[dict]]:
        threshold = self.threshold if threshold is None else threshold
        word_snap = self.word_snap if word_snap is None else word_snap
        return [
            decode(text, probs, self.labels, threshold, word_snap)
            for text, probs in zip(texts, self.predict_probs(texts), strict=True)
        ]

    def predict(self, text: str) -> list[dict]:
        return self.predict_batch([text])[0]

    def params(self) -> dict:
        return {
            "model": self.name,
            "window": self.window,
            "stride": self.stride,
            "threshold": self.threshold,
            "word_snap": self.word_snap,
            "vocab_size": self.vocab.size,
            "vocab_sha256": self.spec["vocab_sha256"],
        }
