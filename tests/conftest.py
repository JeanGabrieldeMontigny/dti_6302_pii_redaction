"""Shared fixtures: a tiny synthetic dataset in the exact raw Nemotron-PII format."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config

TEMPLATES = [
    (
        "Hi, I'm {first}. Email me at {email} or call {phone}.",
        ["first_name", "email", "phone_number"],
    ),
    (
        "Patient {first} {last}, SSN {ssn}, born on {dob}.",
        ["first_name", "last_name", "ssn", "date_of_birth"],
    ),
    ("Server {ip} was accessed by {first} on {dob}.", ["ipv4", "first_name", "date"]),
]
FIRST = ["Maria", "Joseph", "Aiko", "Liam", "Fatima", "Noah", "Chen", "Olga"]
LAST = ["Zarecki", "Nguyen", "Okafor", "Silva"]


def _record(i: int, locale: str, template_idx: int) -> dict:
    tpl, labels = TEMPLATES[template_idx]
    values = {
        "first": FIRST[(i + (locale == "intl")) % len(FIRST)],
        "last": LAST[i % len(LAST)],
        "email": f"user{i}{locale}@example.com",
        "phone": f"{200 + i:03d}-555-{1000 + i:04d}",
        "ssn": f"{100 + i:03d}-45-{6000 + i:04d}",
        "dob": f"19{50 + i % 40:02d}-0{1 + i % 9}-1{i % 9}",
        "ip": f"10.0.{i % 250}.{(i * 7) % 250}",
    }
    keys = [k.split("}")[0] for k in tpl.split("{")[1:]]
    text, spans, _cursor = "", [], 0
    for chunk, key, label in zip(tpl.split("{")[1:], keys, labels, strict=True):
        prefix = tpl.split("{")[0] if not spans else ""
        text += prefix
        start = len(text)
        text += values[key]
        spans.append({"start": start, "end": len(text), "text": values[key], "label": label})
        text += chunk.split("}", 1)[1]
    return {
        "uid": f"uid-{i:04d}",
        "domain": ["Healthcare", "Finance", "IT"][template_idx],
        "document_type": "Form",
        "document_description": "synthetic test record",
        "document_format": "unstructured" if i % 2 else "structured",
        "locale": locale,
        "text": text,
        "spans": repr(spans),  # raw data stores spans as a Python literal string
        "text_tagged": text,
    }


def make_raw_split(n_uids: int, offset: int = 0) -> pd.DataFrame:
    rows = [
        _record(offset + i, locale, (offset + i) % len(TEMPLATES))
        for locale in ("us", "intl")
        for i in range(n_uids)
    ]
    return pd.DataFrame(rows).astype("string")


@pytest.fixture
def raw_train() -> pd.DataFrame:
    return make_raw_split(40)


@pytest.fixture
def raw_test() -> pd.DataFrame:
    return make_raw_split(20, offset=1000)


@pytest.fixture
def validation_cfg() -> dict:
    cfg = load_config()["validation"]
    # The fixture only covers a handful of labels.
    return cfg | {"min_spans_per_label": 0, "warn_label_js_distance": 1.0}
