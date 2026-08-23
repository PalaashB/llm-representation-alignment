"""A generic instruction corpus for adapter fitting, disjoint from every eval bank.

The adapter's job at decode time is to map the small model's residual at
*generated* positions into the large model's geometry. The eval banks are a bad
source of those positions in bulk: `factual` answers are 3-5 tokens, so 70 fit
prompts yield 258 answer rows, and the only way to get more from a bank is to
make the bank bigger, which is slow, hand-verified work that also risks
contaminating the thing being measured.

This corpus exists to supply answer positions in volume instead. Its items are
open-ended instructions whose continuations run 40-80 tokens, so a few hundred
of them produce tens of thousands of answer rows — two orders of magnitude more
than the bank path, for a fraction of the authoring effort.

Three properties make it safe to fit on:

* **No eval overlap.** Nothing here is a short-answer factual question, and no
  item shares a topic with `prompts.py`, `prompts_list.py` or
  `prompts_hard_factual.py` in a way that could leak a gold answer. It is
  checked, not asserted: `overlap_with` compares against a bank's questions and
  the capture path refuses a corpus that trips it.
* **No gold answers, because none are needed.** These items are never scored.
  The teacher's own continuation *is* the target — that is what distillation
  fits against, and what the ridge path takes hidden states from.
* **Generated offline.** `datasets` is not installed in this environment and
  there is no cached wikitext, so a corpus that needs a download is a corpus
  that does not run. Items are composed from templates and topics here.

The `wikitext` option is wired for environments that do have it, and fails with
an explicit message rather than silently falling back — a fit set quietly
substituted for a different one is exactly the class of bug this folder keeps
finding.
"""

from __future__ import annotations

import numpy as np

# Instruction shapes that elicit multi-sentence continuations. Deliberately not
# questions with short factual answers — the point is answer *length*.
INSTRUCTION_TEMPLATES = (
    "Explain what {t} is, in two or three sentences.",
    "Describe {t} in a short paragraph.",
    "Write two sentences about why {t} matters.",
    "Summarise the main idea behind {t}.",
    "Give a brief overview of {t} for someone new to it.",
    "In a few sentences, explain how {t} works.",
    "What should someone know about {t}? Answer in a short paragraph.",
    "Outline the key points about {t}.",
)

# Topics chosen to span registers (technical, everyday, historical, abstract)
# without touching the eval banks' territory of single obscure facts.
TOPICS = (
    "photosynthesis", "the water cycle", "plate tectonics", "erosion",
    "the greenhouse effect", "ocean currents", "the nitrogen cycle", "weathering",
    "natural selection", "genetic inheritance", "cell division", "enzymes",
    "the immune system", "antibiotics", "vaccination", "the nervous system",
    "digestion", "blood circulation", "muscle contraction", "sleep",
    "gravity", "friction", "magnetism", "electric current", "sound waves",
    "light refraction", "thermodynamics", "nuclear fission", "radioactivity",
    "the periodic table", "chemical bonding", "acids and bases", "catalysts",
    "combustion", "distillation", "crystallisation", "polymers",
    "arithmetic", "geometry", "probability", "statistics", "algebra",
    "calculus", "prime numbers", "logarithms", "matrices", "graph theory",
    "computer memory", "operating systems", "compilers", "databases",
    "computer networks", "encryption", "machine learning", "version control",
    "algorithms", "data structures", "recursion", "caching", "concurrency",
    "the internet", "web browsers", "cloud computing", "open source software",
    "supply and demand", "inflation", "interest rates", "taxation", "trade",
    "insurance", "budgeting", "central banks", "unemployment", "productivity",
    "democracy", "the rule of law", "constitutions", "civil rights",
    "public health", "urban planning", "public transport", "recycling",
    "renewable energy", "solar power", "wind turbines", "nuclear power",
    "electric vehicles", "energy efficiency", "climate adaptation",
    "agriculture", "irrigation", "crop rotation", "food preservation",
    "nutrition", "exercise", "mental health", "first aid",
    "the printing press", "the industrial revolution", "the silk road",
    "cartography", "navigation", "timekeeping", "the postal system",
    "libraries", "museums", "archaeology", "written language", "translation",
    "architecture", "bridges", "tunnels", "concrete", "steel", "glassmaking",
    "textiles", "pottery", "woodworking", "metallurgy",
    "photography", "cinema", "music theory", "orchestras", "poetry",
    "storytelling", "typography", "graphic design", "colour theory",
    "chess strategy", "team sports", "athletics training", "sailing",
    "mountaineering", "cooking techniques", "baking", "fermentation",
    "coffee", "tea", "beekeeping", "gardening", "composting",
    "weather forecasting", "earthquakes", "volcanoes", "glaciers", "deserts",
    "rainforests", "coral reefs", "wetlands", "migration of birds",
    "pollination", "soil health", "biodiversity", "conservation",
    "space exploration", "satellites", "telescopes", "the solar system",
    "black holes", "the moon landing", "rockets", "orbital mechanics",
)


def generic_items(n: int | None = None, seed: int = 0) -> list[dict]:
    """Instruction items shaped like a prompt bank, minus the gold answers.

    Every (template, topic) pairing is enumerated and then shuffled, so asking
    for a prefix of any length gives a mix of both axes rather than all eight
    phrasings of the first topic. Ids are stable under `n`, which keeps a small
    smoke-test capture a strict subset of a large one.
    """
    items = [{"id": f"gen_{ti:03d}_{qi}", "category": "generic_corpus",
              "question": tmpl.format(t=topic), "answers": None}
             for ti, topic in enumerate(TOPICS)
             for qi, tmpl in enumerate(INSTRUCTION_TEMPLATES)]
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    return items[:n] if n else items


def wikitext_items(n: int | None = None, seed: int = 0) -> list[dict]:
    """Wikitext-103 paragraphs as continuation prompts, if `datasets` is present.

    Raises rather than falling back. A capture that silently fitted on a
    different corpus than the one named would be undetectable in the sidecar,
    and the whole point of scoping artefacts by their inputs is that the inputs
    are knowable from the filename.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "--fit-corpus wikitext needs the `datasets` package, which is not "
            "installed here (and no wikitext is cached). Use --fit-corpus generic, "
            "which is offline and purpose-built for this, or `pip install datasets`."
        ) from e
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    keep, rng = [], np.random.default_rng(seed)
    for row in ds:
        t = row["text"].strip()
        if len(t) > 400 and not t.startswith("="):
            keep.append(t)
        if n and len(keep) >= n * 4:
            break
    rng.shuffle(keep)
    return [{"id": f"wt_{k:05d}", "category": "wikitext",
             "question": f"Continue this passage:\n\n{t[:600]}", "answers": None}
            for k, t in enumerate(keep[:n] if n else keep)]


FIT_CORPORA = {"generic": generic_items, "wikitext": wikitext_items}


def load_corpus(name: str, n: int | None = None, seed: int = 0) -> list[dict]:
    if name not in FIT_CORPORA:
        raise SystemExit(f"--fit-corpus must be one of {sorted(FIT_CORPORA)}, got {name!r}")
    return FIT_CORPORA[name](n, seed)


def overlap_with(items: list[dict], bank_prompts) -> list[str]:
    """Corpus ids whose question text collides with a bank question.

    Cheap and exact rather than fuzzy: the corpora here are generated from
    disjoint templates, so any collision at all means something has been wired
    up wrong — a bank passed in as a corpus, most likely — and the caller should
    stop rather than sample around it.
    """
    bank = {p["question"].strip().lower() for p in bank_prompts}
    return [it["id"] for it in items if it["question"].strip().lower() in bank]
