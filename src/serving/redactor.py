"""Python reference redactor: text in, redacted text + spans out.

Implements the prediction contract the browser app will follow (docs plan, section 5.1), with the
same model files and ``preprocessing.json``:

    {"redacted_text": ..., "spans": [{"start", "end", "label", "score"}],
     "offset_unit": "codepoint", "model_version": ..., "latency_ms": ...}

Command line (make redact TEXT="..." / scripts\\run.cmd redact -Text "..."):

    uv run python -m src.serving.redactor "Hi, I'm Joseph Zarecki, call me at 727-360-0442"
    uv run python -m src.serving.redactor --model e2 --word-snap --file prompt.txt
    echo "some text" | uv run python -m src.serving.redactor --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.data.schema import redact as apply_redaction

MAX_CHARS = 20_000  # matches the data-validation upper bound on document length


class RedactionError(ValueError):
    """Invalid input. ``code`` is one of the contract's error codes."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class Redactor:
    def __init__(self, tagger, model_version: str):
        self.tagger = tagger
        self.model_version = model_version

    @classmethod
    def load(cls, ref: str | None = None, models_root: Path | None = None) -> Redactor:
        """``ref``: model directory, model version, or prefix (``e2``); default = newest model."""
        from src.config import load_config, resolve_path
        from src.models.predict import KerasTagger, find_model_dir

        root = models_root or resolve_path(load_config()["artifacts"]["dir"]) / "models"
        model_dir = find_model_dir(root, ref)
        return cls(KerasTagger.load(model_dir), model_dir.name)

    def redact(self, text, threshold: float | None = None, word_snap: bool | None = None) -> dict:
        if not isinstance(text, str):
            kind = type(text).__name__
            raise RedactionError("INVALID_TYPE", f"text must be a string, got {kind}")
        if not text.strip():
            raise RedactionError("EMPTY_INPUT", "text is empty")
        if len(text) > MAX_CHARS:
            raise RedactionError(
                "INPUT_TOO_LONG", f"{len(text):,} characters; the limit is {MAX_CHARS:,}"
            )
        t0 = time.perf_counter()
        spans = self.tagger.predict_batch([text], threshold=threshold, word_snap=word_snap)[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "redacted_text": apply_redaction(text, spans),
            "spans": spans,
            "offset_unit": "codepoint",
            "model_version": self.model_version,
            "latency_ms": round(latency_ms, 1),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Redact PII from text with a trained model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("text", nargs="?", help="text to redact (default: read stdin)")
    parser.add_argument("--file", type=Path, help="read the text from a file")
    parser.add_argument("--model", help="model dir, version or prefix (default: newest model)")
    parser.add_argument("--threshold", type=float, help="override the model's threshold")
    snap = parser.add_mutually_exclusive_group()
    snap.add_argument("--word-snap", dest="word_snap", action="store_true", default=None)
    snap.add_argument("--no-word-snap", dest="word_snap", action="store_false")
    parser.add_argument("--json", action="store_true", help="print the full JSON response")
    args = parser.parse_args(argv)
    # Windows consoles default to a legacy code page; PII text often has accents or emoji.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()

    redactor = Redactor.load(args.model)
    try:
        out = redactor.redact(text, args.threshold, args.word_snap)
    except RedactionError as exc:
        print(f"error {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    print(out["redacted_text"])
    print()
    print(f"model {out['model_version']}, {len(out['spans'])} spans, {out['latency_ms']} ms")
    for s in out["spans"]:
        snippet = text[s["start"] : s["end"]].replace("\n", " ")
        print(f"  {s['label']:<28} {s['score']:.2f}  {snippet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
