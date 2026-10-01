from scripts.evaluate_gsm8k import extract_gold, extract_prediction


def test_extract_gold():
    assert extract_gold("reasoning\n#### 1,234") == "1234"


def test_strict_prediction():
    value, strict = extract_prediction("answer\n#### 42")
    assert value == "42"
    assert strict


def test_fallback_prediction():
    value, strict = extract_prediction("The answer is 17.")
    assert value == "17"
    assert not strict