from src.models.regex_baseline import RegexBaseline

MODEL = RegexBaseline()


def labels_of(text):
    return {(text[s["start"] : s["end"]], s["label"]) for s in MODEL.predict(text)}


def test_proposal_example():
    text = "Hi, I'm Joseph Zarecki. Text me at 727-360-0442 or email josephzarecki55@yahoo.com"
    found = labels_of(text)
    assert ("727-360-0442", "phone_number") in found
    assert ("josephzarecki55@yahoo.com", "email") in found


def test_structured_formats():
    text = (
        "SSN 245-47-7741, card 5375 4892 1378 9562, VPN 12.34.56.78, MAC 00:2C:6B:7D:4E:1A, "
        "on 2023-09-15 at 10:30 AM, see https://example.com/a?b=1."
    )
    found = labels_of(text)
    assert {
        ("245-47-7741", "ssn"),
        ("5375 4892 1378 9562", "credit_debit_card"),
        ("12.34.56.78", "ipv4"),
        ("00:2C:6B:7D:4E:1A", "mac_address"),
        ("2023-09-15", "date"),
        ("10:30 AM", "time"),
        ("https://example.com/a?b=1", "url"),
    } <= found


def test_no_overlapping_predictions():
    text = "Logged 2024-07-06T23:08:32 from user@example.com"
    spans = MODEL.predict(text)
    assert ("2024-07-06T23:08:32", "date_time") in labels_of(text)
    assert all(a["end"] <= b["start"] for a, b in zip(spans, spans[1:], strict=False))


def test_plain_text_has_no_predictions():
    assert MODEL.predict("The quick brown fox jumps over the lazy dog.") == []
