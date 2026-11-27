import pytest

from src.models.metrics import evaluate_spans

GOLD = [
    [
        {"start": 0, "end": 5, "label": "first_name"},
        {"start": 10, "end": 20, "label": "phone_number"},
    ]
]


def test_perfect_prediction():
    r = evaluate_spans(GOLD, GOLD)
    assert r["detection_recall"] == 1.0
    assert r["char_precision"] == 1.0
    assert r["type_macro_f1"] == 1.0


def test_no_prediction():
    r = evaluate_spans(GOLD, [[]])
    assert r["detection_recall"] == 0.0
    assert r["char_precision"] == 0.0
    assert r["per_label"]["phone_number"]["detection_recall"] == 0.0


def test_wrong_type_still_counts_as_detected():
    pred = [
        [
            {"start": 0, "end": 5, "label": "last_name"},
            {"start": 10, "end": 20, "label": "phone_number"},
        ]
    ]
    r = evaluate_spans(GOLD, pred)
    assert r["detection_recall"] == 1.0
    assert r["type_accuracy"] == 0.5
    assert r["per_label"]["first_name"]["type_f1"] == 0.0


def test_type_macro_f1_is_conditional_on_detection():
    # phone_number detected and typed correctly; first_name missed entirely.
    pred = [[{"start": 10, "end": 20, "label": "phone_number"}]]
    r = evaluate_spans(GOLD, pred)
    assert r["detection_recall"] == 0.5
    assert r["type_macro_f1"] == 1.0  # the miss is not counted again as a typing error
    assert r["per_label"]["first_name"]["type_f1"] is None
    assert evaluate_spans(GOLD, [[]])["type_macro_f1"] == 0.0


def test_partial_coverage_and_over_redaction():
    pred = [
        [
            {"start": 0, "end": 3, "label": "first_name"},
            {"start": 8, "end": 22, "label": "phone_number"},
        ]
    ]
    r = evaluate_spans(GOLD, pred)
    assert r["per_label"]["first_name"]["detection_recall"] == 0.0  # 3/5 chars < min_coverage=1
    assert r["per_label"]["phone_number"]["detection_recall"] == 1.0
    assert r["char_precision"] == pytest.approx(13 / 17)
    assert evaluate_spans(GOLD, pred, min_coverage=0.5)["detection_recall"] == 1.0


def test_slices():
    golds = GOLD * 2
    preds = [GOLD[0], []]
    r = evaluate_spans(golds, preds, slices={"locale": ["us", "intl"]})
    assert r["slices"]["locale"]["us"]["detection_recall"] == 1.0
    assert r["slices"]["locale"]["intl"]["detection_recall"] == 0.0


def test_per_label_table_order_is_deterministic():
    from src.models.evaluate import per_label_table

    per_label = {
        "ssn": {"support": 5, "detection_recall": 1.0, "type_f1": 1.0},
        "email": {"support": 5, "detection_recall": 0.5, "type_f1": 1.0},
        "mac_address": {"support": 5, "detection_recall": 1.0, "type_f1": 1.0},
        "date_of_birth": {"support": 5, "detection_recall": 1.0, "type_f1": 0.0},
    }
    expected = ["date_of_birth", "mac_address", "ssn", "email"]  # recall desc, then name
    assert per_label_table({"per_label": per_label})["label"].tolist() == expected
    reordered = dict(reversed(list(per_label.items())))
    assert per_label_table({"per_label": reordered})["label"].tolist() == expected
