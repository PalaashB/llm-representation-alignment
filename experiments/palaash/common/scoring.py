"""Answer scoring: normalised token-substring match against accepted aliases.

Two shapes of item are supported, and `score` dispatches between them:

  `answers`  a list of aliases, ANY of which counts (short-answer recall —
             "Wellington" for the capital of New Zealand).
  `requires` a list of alias *groups*, ALL of which must appear — one group per
             item the answer has to contain. This is what makes a list question
             checkable: "name the four DNA bases" is only right if adenine,
             thymine, guanine and cytosine are all there, in any order and with
             any wording around them.

Conjunctive scoring is deliberately harsher than it looks. A model that gets
three of four items scores zero, which is the point: it separates a model that
knows the whole set from one that recites the famous part of it, and that
separation is what a small model loses first.
"""

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


def _present(normalized_generation: str, alias: str) -> bool:
    """True if `alias` appears in an already-normalised generation.

    Whole-token containment, so "ohio" does not match "ohioan" and the symbol
    "k" does not match "kelvin".
    """
    na = normalize(alias)
    if not na:
        return False
    return re.search(rf"(^|\s){re.escape(na)}($|\s)", normalized_generation) is not None


def is_correct(generation: str, answers: list[str]) -> bool:
    """True if ANY accepted answer appears as a normalised token-substring."""
    g = normalize(generation)
    if not g:
        return False
    return any(_present(g, a) for a in answers)


def is_correct_all(generation: str, requires: list[list[str]]) -> bool:
    """True if EVERY required group is satisfied by at least one of its aliases.

    Order does not matter and surrounding text is ignored, so "The four bases
    are adenine (A), thymine (T), guanine (G) and cytosine (C)." scores the same
    as a bare comma-separated list.
    """
    g = normalize(generation)
    if not g or not requires:
        return False
    return all(any(_present(g, alias) for alias in group) for group in requires)


def n_required_present(generation: str, requires: list[list[str]]) -> int:
    """How many required groups were found. Partial credit is never used for
    accuracy — `is_correct_all` is all-or-nothing — but it is recorded per
    generation, because "3 of 4 items" and "0 of 4" are very different failures
    and only one of them means the stitch has broken."""
    g = normalize(generation)
    if not g:
        return 0
    return sum(any(_present(g, alias) for alias in group) for group in requires)


def score(generation: str, item: dict) -> bool:
    """Score a generation against a prompt-bank item of either shape."""
    if item.get("requires"):
        return is_correct_all(generation, item["requires"])
    return is_correct(generation, item["answers"])
