"""Bank `list_hard`: conjunctive list questions composed from audited single facts.

Why this bank exists
--------------------
Two separate problems with the existing banks, both fatal to the measurement
rather than to the method:

**The splits are too small to resolve anything.** `factual` gives 35 prompts per
split, so one prompt is 2.9 points and a 95% interval near 85% is ~23 points
wide. The published tables rank 50 cells whose true differences are far smaller
than that, and the clearest evidence is on record: `L8->L14` warm scored 65.7%
on dev and 85.7% on test — same adapter, same configuration, two 35-prompt
splits. Nothing about the method changed between those numbers.

**`factual` has no headroom to compete for.** Small 88.6%, large 97.1%: 8.6
points, and the small model is already there at 2.5x the speed. A stitch cannot
demonstrate anything in an 8.6-point window measured with a 23-point interval.

Both are fixed by the same construction. Conjunction amplifies a per-fact gap
multiplicatively: if the 3B is right on ~98% of these facts and the 1B on ~74%,
then on a three-fact question the ceiling is 0.98^3 = 94% and the floor is
0.74^3 = 41%. A 24-point single-fact gap becomes a ~53-point one, from exactly
the same underlying knowledge — no new facts had to be written or verified.
Longer answers come free with it, which is the other thing a latency claim
needs (see `Bank.latency_steps` and the decode-share argument in
stitching_small_to_large/README.md).

Where the facts come from
-------------------------
Every item is composed from `common/prompts_hard_factual.py`, whose gold
answers have already been through the both-models-wrong audit described in
stitching_large_to_small/README.md — that process caught three real bank bugs.
Composing rather than authoring means the expansion inherits that audit instead
of adding 600 new unverified answers, which is the failure mode this file is
most exposed to. The categories used are the ones that survive extraction with
an unambiguous subject; `science`, `misc` and `geography` are phrased too
heterogeneously to compose safely and are left out.

Split discipline
----------------
The three-way split happens over **facts, not over composed prompts**. If it
were over prompts, the same fact would appear in a `fit` prompt and a `dev`
prompt, and the adapter — fit on states captured from the fit prompts — would
have seen the answer it is later scored on. Partitioning the underlying facts
first makes that impossible, and each item carries its `split` so the loaders
use this partition rather than re-shuffling.
"""

from __future__ import annotations

import re

import numpy as np

from common.prompts_hard_factual import HARD_FACTUAL_PROMPTS

LIST_HARD_SYSTEM_PROMPT = (
    "You are a precise factual assistant. Answer with ONLY the items requested, "
    "as a short comma-separated list in the order asked. Do not number them, "
    "explain them, or add any other text."
)

# (subject regex, question template, joiner) per composable category. The
# template takes the joined subject list; `{n}` is the item count.
CATEGORIES = {
    "capital": (
        r"^What is the capital city of (.+)\?$",
        "What are the capital cities of {subjects}?"),
    "us_capital": (
        r"^What is the capital of the U\.S\. state of (.+)\?$",
        "What are the capital cities of the U.S. states of {subjects}?"),
    "atomic_number": (
        r"^What is the atomic number of (.+)\?$",
        "What are the atomic numbers of {subjects}?"),
    "element": (
        r"^What is the chemical symbol for (.+)\?$",
        "What are the chemical symbols for {subjects}?"),
    "currency": (
        r"^What is the official currency of (.+)\?$",
        "What are the official currencies of {subjects}?"),
    "literature": (
        r"^Who wrote (?:the novel )?'(.+)'\?$",
        "Who wrote {subjects}?"),
    "history": (
        r"^(.+)\. In what year did this happen\?$",
        "In what year did each of these happen: {subjects}?"),
}

# Subjects that need re-quoting when composed, so the question reads naturally
# and the items stay separable by a reader auditing the gold answers.
QUOTED = {"literature"}
JOINER = {"history": "; "}

GROUP_SIZE = 3
# ^ Three, not five. Conjunction cuts both ways: at five items the large model's
#   own ceiling falls to ~0.98^5 = 90% and keeps dropping, which shrinks the
#   headroom the bank exists to create. Three is where the gap is widest.

# Per-category reuse, chosen from measurement rather than intuition — the same
# procedure stitching_large_to_small/README.md used to build `hard_factual`.
#
# A first composition weighted all six categories equally and scored +11.0 pts
# of headroom on llama/dev (n=172). Breaking that down by category showed the
# gap was very unevenly distributed (that run is preserved as
# results/llama/list_hard/headroom_dev_v1_uniform_categories.json):
#
#   category      n   small   large     gap   divergent
#   us_capital   26   73.1%  100.0%   +26.9           7
#   history      12   66.7%   91.7%   +25.0           3
#   element      19   84.2%  100.0%   +15.8           3
#   capital      40   85.0%  100.0%   +15.0           6
#   atomic_number 42 100.0%  100.0%    +0.0           0
#   currency     33   81.8%   81.8%    +0.0           5
#
# Two categories contribute no gap at all and were 44% of the split, diluting
# the headroom the bank exists to provide. Atomic numbers turn out to be
# something a 1B knows cold once the element is common — the generations are
# correct and fully spelled out, so this is a real result about the model, not a
# scoring artefact. Currency splits the other way: both models sit at 81.8%, and
# the 5 divergent cases are offset by 5 the small model gets and the large one
# misses, so the *net* gap is zero even though the category is not trivial.
#
# Note this does not reproduce the single-fact category table in
# stitching_large_to_small/README.md, and should not be expected to. There,
# national capitals and element symbols showed 0% divergence; here their
# three-fact conjunctions give +15.0 and +15.8. Conjunction changes which
# categories are productive, because it converts a high per-fact accuracy into a
# much lower per-question one.
#
# So the unproductive pair is kept (they still add prompts, and a bank of only
# the hardest categories would overstate the effect) but down-weighted heavily.
PRODUCTIVE = ("us_capital", "history", "element", "capital")
UNPRODUCTIVE = ("atomic_number", "currency")
USES_PER_FACT = {"fit": 5, "dev": 11, "test": 11}
USES_UNPRODUCTIVE = {"fit": 2, "dev": 2, "test": 2}
# ^ How many composed prompts each fact may appear in. Facts are partitioned
#   across splits first, so reuse never crosses a split boundary; within a split
#   it only means two prompts share one of their three items, with different
#   partners. dev/test get a higher factor because they need >= 150 prompts from
#   a quarter of the facts each.
#
#   The honest caveat: prompts sharing a fact are not independent trials, so a
#   Wilson interval computed as if they were is somewhat narrower than the truth.
#   The effect is bounded — each prompt shares at most one of its three items
#   with any other, and a prompt is only correct if all three are — but it means
#   "n=170" here is worth a little less than 170 independent prompts. It is
#   still worth far more than the 35 it replaces, where one prompt moved the
#   score by 2.9 points.

SPLIT_FRACS = (0.50, 0.25, 0.25)


def _subjects() -> dict[str, list[dict]]:
    """Parsed (subject, aliases) per category, in bank order for determinism."""
    out: dict[str, list[dict]] = {}
    for p in HARD_FACTUAL_PROMPTS:
        spec = CATEGORIES.get(p["category"])
        if not spec:
            continue
        m = re.match(spec[0], p["question"])
        if not m:
            continue
        out.setdefault(p["category"], []).append(
            {"subject": m.group(1), "answers": list(p["answers"]), "src": p["id"]})
    return out


def _partition(items: list[dict], seed: int) -> dict[str, list[dict]]:
    """Deterministic fact-level three-way split."""
    idx = np.arange(len(items))
    np.random.default_rng(seed).shuffle(idx)
    n_fit = int(round(len(items) * SPLIT_FRACS[0]))
    n_dev = int(round(len(items) * SPLIT_FRACS[1]))
    return {"fit": [items[k] for k in idx[:n_fit]],
            "dev": [items[k] for k in idx[n_fit:n_fit + n_dev]],
            "test": [items[k] for k in idx[n_fit + n_dev:]]}


def _compose(facts: list[dict], category: str, split: str, uses: int,
             seed: int) -> list[dict]:
    """Group facts into conjunctive prompts, each fact used at most `uses` times.

    Round-robin over `uses` independent shuffles rather than sampling with
    replacement: it guarantees an even number of appearances per fact (so no
    fact dominates the split) and never repeats a group.
    """
    template = CATEGORIES[category][1]
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    for r in range(uses):
        order = list(rng.permutation(len(facts)))
        for s in range(0, len(facts) - GROUP_SIZE + 1, GROUP_SIZE):
            grp = tuple(sorted(order[s:s + GROUP_SIZE]))
            if grp in seen:
                continue
            seen.add(grp)
            chosen = [facts[k] for k in grp]
            subs = [f"'{c['subject']}'" if category in QUOTED else c["subject"]
                    for c in chosen]
            sep = JOINER.get(category, ", ")
            joined = sep.join(subs[:-1]) + f"{sep}and {subs[-1]}"
            out.append({
                "id": f"lh_{category}_{split}_{len(out):04d}",
                "category": f"list_{category}",
                "split": split,
                "question": template.format(subjects=joined),
                # Conjunctive scoring: every group must be satisfied, in any
                # order and with any wording around it.
                "requires": [list(c["answers"]) for c in chosen],
                "answers": [", ".join(c["answers"][0] for c in chosen)],
                "source_ids": [c["src"] for c in chosen],
            })
    return out


def build(seed: int = 0) -> list[dict]:
    out = []
    for ci, (category, facts) in enumerate(sorted(_subjects().items())):
        uses = (USES_UNPRODUCTIVE if category in UNPRODUCTIVE else USES_PER_FACT)
        parts = _partition(facts, seed + ci)
        for split, pool in parts.items():
            out += _compose(pool, category, split, uses[split],
                            seed + 100 * ci + len(split))
    return out


LIST_HARD_PROMPTS = build()

SPLIT_COUNTS = {s: sum(1 for p in LIST_HARD_PROMPTS if p["split"] == s)
                for s in ("fit", "dev", "test")}
