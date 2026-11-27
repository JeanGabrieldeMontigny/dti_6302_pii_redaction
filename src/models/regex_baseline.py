"""Baseline 1: rules-based PII detector using regular expressions.

Covers fixed-format entities only (emails, URLs, network addresses, SSNs, card numbers, phone
numbers, dates/times, coordinates, postal codes). It cannot find names, street addresses or
free-text attributes; per-label recall makes that gap explicit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.data.schema import Span

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)

# (label, pattern). Order is the tie-break priority when two matches have the same extent.
PATTERNS: list[tuple[str, str]] = [
    ("email", r"\b[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"),
    ("url", r"\bhttps?://[^\s<>\"')\]]+[^\s<>\"')\].,;:!?]"),
    ("ipv4", r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    ("mac_address", r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    (
        "ipv6",
        r"(?<![\w:])(?:[0-9A-Fa-f]{1,4}:){2,7}(?::?[0-9A-Fa-f]{1,4}|:)"
        r"(?:(?::[0-9A-Fa-f]{1,4})*)(?![\w:])",
    ),
    ("date_time", r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?\b"),
    ("ssn", r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    ("credit_debit_card", r"\b(?:\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}|\d{4}[ -]?\d{6}[ -]?\d{5})\b"),
    (
        "phone_number",
        r"(?<![\w-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\d{2,4}[ .-])\d{3,4}[ .-]?\d{3,4}\b",
    ),
    (
        "date",
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|"
        rf"{_MONTH} \d{{1,2}}(?:st|nd|rd|th)?,? \d{{4}}|\d{{1,2}} {_MONTH},? \d{{4}})\b",
    ),
    ("time", r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s?[AaPp]\.?[Mm]\.?)?(?!\w)"),
    ("coordinate", r"\(?-?\d{1,3}\.\d{3,},\s?-?\d{1,3}\.\d{3,}\)?"),
    ("postcode", r"\b(?:[A-Z]\d[A-Z] ?\d[A-Z]\d|\d{5}-\d{4})\b"),
]


@dataclass
class RegexBaseline:
    """Predict spans with the regex table above, resolving overlaps longest-match-first."""

    patterns: tuple[tuple[str, re.Pattern], ...] = tuple(
        (label, re.compile(p)) for label, p in PATTERNS
    )

    name = "regex_baseline"

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self.patterns]

    def predict(self, text: str) -> list[Span]:
        candidates = []
        for priority, (label, pattern) in enumerate(self.patterns):
            for m in pattern.finditer(text):
                if m.end() > m.start():
                    candidates.append((m.start(), m.end(), priority, label))
        # Longest first, then earlier priority; greedily keep non-overlapping matches.
        candidates.sort(key=lambda c: (-(c[1] - c[0]), c[2], c[0]))
        taken: list[tuple[int, int]] = []
        spans: list[Span] = []
        for start, end, _, label in candidates:
            if all(end <= s or start >= e for s, e in taken):
                taken.append((start, end))
                spans.append({"start": start, "end": end, "label": label})
        return sorted(spans, key=lambda s: s["start"])

    def predict_batch(self, texts) -> list[list[Span]]:
        return [self.predict(t) for t in texts]

    def params(self) -> dict:
        return {
            "model": self.name,
            "n_patterns": len(self.patterns),
            "labels": ",".join(self.labels),
        }
