"""Answer scoring: normalised token-substring match against accepted aliases."""

import re
import string
import unicodedata

_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation/articles, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    toks = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(toks)


def is_correct(generation: str, answers: list[str]) -> bool:
    """True if any accepted answer appears as a normalised token-substring."""
    g = normalize(generation)
    if not g:
        return False
    for a in answers:
        na = normalize(a)
        if not na:
            continue
        # word-boundary-ish containment so "ohio" doesn't match "ohioan" spuriously
        if re.search(rf"(^|\s){re.escape(na)}($|\s)", g):
            return True
    return False
