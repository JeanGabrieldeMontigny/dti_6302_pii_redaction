"""Decoding sweep: the recall/precision trade-off of one trained model, without retraining.

The model's per-character probabilities are computed once per document; every
``(threshold, word_snap)`` setting is then decoded and scored from those same probabilities.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.models.decode import decode
from src.models.metrics import SpanMetrics

COLUMNS = ["threshold", "word_snap", "detection_recall", "char_precision", "char_recall"]


def decode_sweep(
    tagger,
    df: pd.DataFrame,
    thresholds: Sequence[float],
    word_snaps: Sequence[bool],
    min_coverage: float = 1.0,
) -> pd.DataFrame:
    """One row per decoding setting with the span metrics on ``df``."""
    grid = [(float(t), bool(s)) for s in word_snaps for t in thresholds]
    metrics = {g: SpanMetrics(min_coverage) for g in grid}
    texts = df["text"].tolist()
    for text, gold, probs in zip(texts, df["spans"], tagger.predict_probs(texts), strict=True):
        for (threshold, snap), m in metrics.items():
            m.update(gold, decode(text, probs, tagger.labels, threshold, snap))
    rows = []
    for (threshold, snap), m in metrics.items():
        r = m.result()
        rows.append(
            {
                "threshold": threshold,
                "word_snap": snap,
                "detection_recall": r["detection_recall"],
                "char_precision": r["char_precision"],
                "char_recall": r["char_recall"],
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def recommend(sweep: pd.DataFrame, min_char_precision: float) -> dict | None:
    """Highest detection recall whose char precision meets the target (None if none does)."""
    ok = sweep[sweep["char_precision"] >= min_char_precision]
    if ok.empty:
        return None
    best = ok.sort_values(["detection_recall", "char_precision"], ascending=False).iloc[0]
    return {
        "threshold": float(best["threshold"]),
        "word_snap": bool(best["word_snap"]),
        "detection_recall": float(best["detection_recall"]),
        "char_precision": float(best["char_precision"]),
    }


# Reference palette slots 1-2 (validated: CVD dE 26.6). Orange is < 3:1 on white, so series are
# also told apart by marker shape and direct labels, and the CSV table sits next to the PNG.
_SERIES = {False: ("#257de0", "o", "word snap off"), True: ("#f47230", "s", "word snap on")}
_TEXT, _MUTED, _GRID = "#191919", "#666666", "#e6e6e6"


def plot_sweep(
    sweep: pd.DataFrame,
    path: Path,
    title: str,
    current: tuple[float, bool],
    recommended: dict | None,
    min_char_precision: float,
) -> None:
    """Recall (y) vs precision (x), one line per word-snap setting, thresholds labelled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    for snap, (color, marker, label) in _SERIES.items():
        s = sweep[sweep["word_snap"] == snap].sort_values("threshold")
        if s.empty:
            continue
        ax.plot(
            s["char_precision"],
            s["detection_recall"],
            color=color,
            linewidth=1.5,
            marker=marker,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1,
            label=label,
            zorder=3,
        )
        # Direct label mid-curve. The curves fall to the right, so "on" (the upper curve) is
        # labelled above-right of its line and "off" (the lower one) below-left: never between.
        mid = s.iloc[len(s) // 3]
        ax.annotate(
            label,
            (mid["char_precision"], mid["detection_recall"]),
            xytext=(8, 8) if snap else (-8, -14),
            textcoords="offset points",
            ha="left" if snap else "right",
            color=_TEXT,
            fontsize=8,
        )
        if not snap:  # thresholds run the same way on both lines; label one line's ends
            for row in (s.iloc[0], s.iloc[-1]):
                ax.annotate(
                    f"t={row['threshold']:g}",
                    (row["char_precision"], row["detection_recall"]),
                    xytext=(6, -4),
                    textcoords="offset points",
                    color=_MUTED,
                    fontsize=7,
                )

    cur = sweep[(sweep["threshold"] == current[0]) & (sweep["word_snap"] == current[1])]
    if not cur.empty:
        ax.scatter(
            cur["char_precision"],
            cur["detection_recall"],
            s=140,
            facecolors="none",
            edgecolors=_TEXT,
            linewidths=1.5,
            zorder=4,
            label=f"configured (t={current[0]:g}, snap {'on' if current[1] else 'off'})",
        )
    if recommended:
        rec_snap = "on" if recommended["word_snap"] else "off"
        ax.scatter(
            [recommended["char_precision"]],
            [recommended["detection_recall"]],
            s=160,
            marker="*",
            color=_TEXT,
            zorder=5,
            label=(
                f"best recall at precision >= {min_char_precision:g} "
                f"(t={recommended['threshold']:g}, snap {rec_snap})"
            ),
        )
    ax.axvline(min_char_precision, color=_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.annotate(
        f"precision target {min_char_precision:g}",
        (min_char_precision, ax.get_ylim()[1]),
        xytext=(4, -10),
        textcoords="offset points",
        color=_MUTED,
        fontsize=7,
    )

    ax.set_xlabel("Character precision (share of redacted text that is PII)", color=_TEXT)
    ax.set_ylabel("Detection recall (share of PII entities fully redacted)", color=_TEXT)
    ax.set_title(title, color=_TEXT, fontsize=10, loc="left")
    ax.grid(color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower left", labelcolor=_TEXT)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
