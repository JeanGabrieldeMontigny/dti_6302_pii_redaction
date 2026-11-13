"""Dataset schema and the canonical span representation.

Spans are half-open character intervals ``[start, end)`` over the Python ``str`` (Unicode code
points) of the document text. The browser runtime must convert from UTF-16 indices accordingly.
"""

from __future__ import annotations

import ast
from typing import TypedDict

RAW_COLUMNS = [
    "uid",
    "domain",
    "document_type",
    "document_description",
    "document_format",
    "locale",
    "text",
    "spans",
    "text_tagged",
]
PROCESSED_COLUMNS = ["uid", "domain", "document_type", "document_format", "locale", "text", "spans"]

LOCALES = {"us", "intl"}
DOCUMENT_FORMATS = {"structured", "unstructured"}

# The 55 entity types in Nemotron-PII (revision pinned in configs/default.yaml).
LABELS = (
    "account_number", "age", "api_key", "bank_routing_number", "biometric_identifier",
    "blood_type", "certificate_license_number", "city", "company_name", "coordinate", "country",
    "county", "credit_debit_card", "customer_id", "cvv", "date", "date_of_birth", "date_time",
    "device_identifier", "education_level", "email", "employee_id", "employment_status",
    "fax_number", "first_name", "gender", "health_plan_beneficiary_number", "http_cookie",
    "ipv4", "ipv6", "language", "last_name", "license_plate", "mac_address",
    "medical_record_number", "national_id", "occupation", "password", "phone_number", "pin",
    "political_view", "postcode", "race_ethnicity", "religious_belief", "sexuality", "ssn",
    "state", "street_address", "swift_bic", "tax_id", "time", "unique_id", "url", "user_name",
    "vehicle_identifier",
)
LABEL_SET = frozenset(LABELS)


class Span(TypedDict):
    start: int
    end: int
    label: str


def parse_raw_spans(raw: str) -> list[dict]:
    """Parse the raw ``spans`` column, which is a Python-literal string (single quotes, not JSON).

    ``ast.literal_eval`` only evaluates literals, so this is safe on untrusted input.
    """
    value = ast.literal_eval(raw)
    if not isinstance(value, list):
        raise ValueError(f"spans must be a list, got {type(value).__name__}")
    return value


def placeholder(label: str) -> str:
    """Typed placeholder used in redacted output, e.g. ``first_name`` -> ``[FIRST_NAME]``."""
    return f"[{label.upper()}]"


def redact(text: str, spans: list[Span]) -> str:
    """Replace each span with its typed placeholder. Spans must not overlap."""
    out, cursor = [], 0
    for s in sorted(spans, key=lambda s: s["start"]):
        out.append(text[cursor : s["start"]])
        out.append(placeholder(s["label"]))
        cursor = s["end"]
    out.append(text[cursor:])
    return "".join(out)
