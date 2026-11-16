from src.preprocess import clean_text, normalize_arabic


def test_urls_and_numbers_are_replaced():
    out = clean_text("Win $500 now at http://bit.ly/abc")
    assert "http" not in out
    assert "urltoken" in out and "numtoken" in out


def test_lowercase_and_punctuation():
    assert clean_text("HELLO!!!   World") == "hello world"


def test_arabic_normalization():
    assert normalize_arabic("إأآ") == "ااا"
    assert normalize_arabic("مدرسة") == "مدرسه"
    assert normalize_arabic("مُحَمَّد") == "محمد"
