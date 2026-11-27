"""Show the progress of training runs (epoch, step, losses, ETA).

    uv run python -m pipelines.status            # make status / scripts\\run.cmd status

Reads the ``progress.json`` that every training run keeps up to date in its checkpoint folder
(``artifacts/checkpoints/<experiment>-<hash>/``). A run whose file stops updating was interrupted;
running the same train command again resumes it.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.config import load_config, resolve_path

STALE_AFTER_S = 10 * 60


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def _aware(stamp: str) -> datetime:
    """Parse a timestamp; ones without an offset (older files) are taken as local time."""
    t = datetime.fromisoformat(stamp)
    return t if t.tzinfo else t.astimezone()


def describe(state: dict, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    age = (now - _aware(state["updated"])).total_seconds()
    last = state.get("last_epoch") or {}
    lines = [
        f"{state['run']}",
        f"  epoch {state['epoch']}/{state['epochs']}, step {state['step']}/{state['steps']}"
        f"  ({100 * state['progress']:.0f}% overall), current loss {state['loss']}",
    ]
    if last:
        lines.append(
            "  last completed epoch: "
            + ", ".join(f"{k} {v}" for k, v in last.items() if k in ("loss", "val_loss"))
        )
    if state.get("epoch_seconds"):
        avg = sum(state["epoch_seconds"]) / len(state["epoch_seconds"])
        lines.append(f"  epoch time: ~{_duration(avg)}")
    if age > STALE_AFTER_S and state["status"] == "training":
        lines.append(
            f"  NOT RUNNING: no update for {_duration(age)}. It was interrupted; run the same "
            "train command again to resume from the last completed epoch."
        )
    else:
        lines.append(
            f"  ETA: {_duration(state['eta_total_s'])} "
            f"(around {_aware(state['eta_finish']).astimezone().strftime('%H:%M')})"
            f"  [updated {_duration(age)} ago]"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="config whose artifacts dir to inspect")
    args = parser.parse_args(argv)
    root = resolve_path(load_config(args.config)["artifacts"]["dir"]) / "checkpoints"
    files = sorted(Path(root).glob("*/progress.json")) if Path(root).exists() else []
    if not files:
        print(f"No training in progress (nothing in {root}).")
        return
    for f in files:
        print(describe(json.loads(f.read_text())))
        print()


if __name__ == "__main__":
    main()
