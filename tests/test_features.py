import numpy as np
import pytest

from src.data.schema import LABELS
from src.features.encode import (
    make_windows,
    n_tags,
    spans_to_tags,
    stitch,
    tags_to_spans,
    window_starts,
)
from src.features.vocab import PAD_ID, UNK_ID, CharVocab


def test_vocab_fit_is_deterministic_and_frequency_ordered():
    vocab = CharVocab.fit(["aab", "ba", "c"], min_freq=1)
    assert vocab.code_points == (ord("a"), ord("b"), ord("c"))  # a:3, b:2, c:1
    assert CharVocab.fit(["c", "ba", "aab"]) == vocab


def test_vocab_min_freq_and_unknown_chars():
    vocab = CharVocab.fit(["aaa", "b"], min_freq=2)
    ids = vocab.encode("ab?")
    assert ids.tolist() == [2, UNK_ID, UNK_ID]
    assert PAD_ID not in vocab.encode("aaaa")


def test_vocab_works_on_code_points_not_utf16():
    # An emoji is one code point in Python (and in JS via Array.from), so one id.
    vocab = CharVocab.fit(["👋 hi"])
    assert len(vocab.encode("👋 hi")) == 4


def test_vocab_roundtrip():
    vocab = CharVocab.fit(["héllo wörld 👋"])
    restored = CharVocab.from_dict(vocab.to_dict())
    assert restored == vocab and restored.sha256() == vocab.sha256()
    with pytest.raises(ValueError):
        CharVocab.from_dict({**vocab.to_dict(), "pad_id": 5})


SPANS = [
    {"start": 0, "end": 6, "label": "first_name"},
    {"start": 7, "end": 14, "label": "last_name"},
    {"start": 14, "end": 17, "label": "last_name"},  # adjacent span with the same label
]


def test_spans_tags_roundtrip():
    tags = spans_to_tags(20, SPANS, LABELS)
    assert tags_to_spans(tags, LABELS) == SPANS
    assert tags.max() < n_tags(LABELS)


def test_inside_tag_after_outside_starts_a_span():
    k = LABELS.index("email")
    assert tags_to_spans([0, 2 + 2 * k, 2 + 2 * k, 0], LABELS) == [
        {"start": 1, "end": 3, "label": "email"}
    ]


@pytest.mark.parametrize("n", [1, 10, 16, 17, 40, 100])
def test_windows_cover_document(n):
    window, stride = 16, 12
    starts = window_starts(n, window, stride)
    covered = set()
    for s in starts:
        covered.update(range(s, min(s + window, n)))
    assert covered == set(range(n))
    assert starts[-1] + window >= n
    assert all(s >= 0 for s in starts)


def test_make_windows_pads_short_documents():
    out = make_windows(np.arange(1, 6, dtype=np.int32), window=8, stride=4)
    assert out.tolist() == [[1, 2, 3, 4, 5, 0, 0, 0]]


@pytest.mark.parametrize("n", [5, 16, 37, 100])
def test_stitch_inverts_windowing(n):
    window, stride = 16, 12
    per_char = np.random.default_rng(0).random((n, 3)).astype(np.float32)
    windows = make_windows(np.arange(n), window, stride, pad_id=-1)
    probs = np.stack(
        [np.where((w >= 0)[:, None], per_char[np.clip(w, 0, None)], 0.0) for w in windows]
    )
    np.testing.assert_array_equal(stitch(probs, n, window, stride), per_char)


def test_stitch_prefers_the_most_central_window():
    window, stride, n = 8, 4, 12  # windows start at 0 and 4
    probs = np.stack([np.full((window, 1), 0.0), np.full((window, 1), 1.0)]).astype(np.float32)
    out = stitch(probs, n, window, stride)[:, 0]
    # Char 5 is 5 from the first window's start but 1 from the second's: second window wins
    # only where it is more central.
    assert out[:4].tolist() == [0.0] * 4
    assert out[8:].tolist() == [1.0] * 4
