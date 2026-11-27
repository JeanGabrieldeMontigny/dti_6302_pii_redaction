import numpy as np
import pytest

from src.models.decode import decode

LABELS = ("first_name", "email")
O_, B_NAME, I_NAME, B_EMAIL, I_EMAIL = range(5)


def probs_for(tags, confidence=0.9):
    """One-hot-ish probabilities: ``confidence`` on the given tag, the rest spread on others."""
    p = np.full((len(tags), 5), (1 - confidence) / 4, dtype=np.float32)
    p[np.arange(len(tags)), tags] = confidence
    return p


def test_decode_basic_spans():
    text = "Hi Ann a@b.c"
    tags = [O_, O_, O_, B_NAME, I_NAME, I_NAME, O_, B_EMAIL, I_EMAIL, I_EMAIL, I_EMAIL, I_EMAIL]
    spans = decode(text, probs_for(tags), LABELS)
    assert [(s["start"], s["end"], s["label"]) for s in spans] == [
        (3, 6, "first_name"),
        (7, 12, "email"),
    ]
    assert all(0 < s["score"] <= 1 for s in spans)


def test_threshold_controls_redaction():
    text = "Ann"
    probs = probs_for([B_NAME, I_NAME, I_NAME], confidence=0.4)  # P(O) = 0.15 -> p_entity 0.85
    assert decode(text, probs, LABELS, threshold=0.5)
    assert decode(text, probs, LABELS, threshold=0.9) == []


def test_whitespace_is_trimmed():
    text = " Ann "
    spans = decode(text, probs_for([I_NAME, B_NAME, I_NAME, I_NAME, I_NAME]), LABELS)
    assert [(s["start"], s["end"]) for s in spans] == [(1, 4)]


def test_b_tag_splits_adjacent_entities():
    spans = decode("AnnBo", probs_for([B_NAME, I_NAME, I_NAME, B_NAME, I_NAME]), LABELS)
    assert [(s["start"], s["end"]) for s in spans] == [(0, 3), (3, 5)]


def test_word_snap_redacts_whole_words():
    text = "Joseph Zarecki"
    tags = [O_, O_, B_NAME, I_NAME, O_, O_, O_, O_, O_, O_, O_, O_, O_, O_]
    assert decode(text, probs_for(tags), LABELS)[0]["end"] == 4
    snapped = decode(text, probs_for(tags), LABELS, word_snap=True)
    assert [(s["start"], s["end"]) for s in snapped] == [(0, 6)]


def test_empty_text_and_shape_check():
    assert decode("", np.zeros((0, 5), dtype=np.float32), LABELS) == []
    with pytest.raises(ValueError):
        decode("abc", np.zeros((2, 5), dtype=np.float32), LABELS)
