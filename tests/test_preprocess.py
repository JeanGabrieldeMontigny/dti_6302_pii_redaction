import pandas as pd

from src.data.preprocess import PreprocessStats, canonical_spans, preprocess
from src.data.schema import parse_raw_spans, redact


def test_parse_raw_spans_handles_python_literals():
    spans = parse_raw_spans("[{'start': 0, 'end': 5, 'text': 'Maria', 'label': 'first_name'}]")
    assert spans == [{"start": 0, "end": 5, "text": "Maria", "label": "first_name"}]


def test_offsets_are_source_of_truth():
    text = "Maria is Spanish, PIN 1234."
    raw = [
        {"start": 9, "end": 16, "text": "spanish", "label": "race_ethnicity"},  # case quirk
        {"start": 22, "end": 26, "text": 1234, "label": "pin"},  # int span text
        {"start": 0, "end": 5, "text": "Maria", "label": "first_name"},
    ]
    stats = PreprocessStats()
    spans = canonical_spans(text, raw, stats)
    assert [s["label"] for s in spans] == ["first_name", "race_ethnicity", "pin"]  # sorted
    assert all(set(s) == {"start", "end", "label"} for s in spans)
    assert stats.span_text_case_fixed == 1
    assert stats.span_text_non_string == 1


def test_overlapping_spans_drop_document():
    raw = [
        {"start": 0, "end": 5, "text": "Maria", "label": "first_name"},
        {"start": 3, "end": 8, "text": "ia Lo", "label": "last_name"},
    ]
    assert canonical_spans("Maria Lopez", raw, PreprocessStats()) is None


def test_preprocess_drops_overlaps_and_duplicates(raw_train):
    df = pd.concat([raw_train, raw_train.iloc[[0]].assign(uid="dup")], ignore_index=True)
    df.loc[1, "spans"] = repr(
        [
            {"start": 0, "end": 4, "text": "x", "label": "first_name"},
            {"start": 2, "end": 6, "text": "y", "label": "last_name"},
        ]
    )
    out, stats = preprocess(df)
    assert stats.dropped_overlap_docs == 1
    assert stats.dropped_duplicate_docs == 1
    assert len(out) == len(raw_train) - 1
    assert "text_tagged" not in out.columns
    assert isinstance(out.loc[0, "spans"], list)


def test_redact_replaces_spans_with_typed_placeholders():
    text = "Hi, I'm Joseph. Call 727-360-0442."
    spans = [
        {"start": 8, "end": 14, "label": "first_name"},
        {"start": 21, "end": 33, "label": "phone_number"},
    ]
    assert redact(text, spans) == "Hi, I'm [FIRST_NAME]. Call [PHONE_NUMBER]."
