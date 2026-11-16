"""Text cleaning for English + Arabic messages."""
import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
NUM_RE = re.compile(r"\d+")
DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0640]")  # tashkeel + tatweel
PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_arabic(text: str) -> str:
    """Unify Arabic letter variants so the model sees one form."""
    text = DIACRITICS_RE.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    return text


def clean_text(text: str) -> str:
    """Lowercase, replace URLs/numbers with tokens, strip punctuation."""
    text = str(text).lower()
    text = URL_RE.sub(" urltoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = normalize_arabic(text)
    text = PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
