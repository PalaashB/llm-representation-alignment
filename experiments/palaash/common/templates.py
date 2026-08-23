"""Prompt-template variety for state capture.

Both stitching packages captured every fit prompt under one chat template and
one system prompt, and it wrecked the fit distribution. On the `factual`
capture that is 258 answer rows against 5180 prompt rows: 95% of what the
adapter was fit on is the *same boilerplate repeated 70 times*. A ridge solve
handed that data does the sensible thing and memorises the template — hence
R2_all = 0.999 — while the positions that actually matter once decoding starts
are a rounding error in the objective.

Up-weighting answer rows (ANSWER_WEIGHT=4) does not fix it: 258x4 against 5180
is still only 17% of the total weight. Two changes are needed and this module
is the first. Varying the framing means identical prompt positions are no
longer the bulk of the rows, so the prompt positions that remain carry
information about *language* rather than about one fixed prefix. The second is
row selection, in each package's `data.py`.

What is safe to vary. The system prompt and the framing around the question
can be paraphrased freely; the question itself cannot, because the bank's gold
answer is attached to its exact wording. So the question text is always passed
through verbatim and only wrapped. Wrapping is also the variation that matters:
it moves the *positions* of the answer tokens relative to the sequence start
and changes what precedes them, which is exactly the invariance the adapter
needs and did not have.

Eval never uses these. `build_prompt_ids` with its default system prompt is
what every accuracy and latency pass runs, so the variety is training-time
augmentation only and cannot flatter a score.
"""

from __future__ import annotations

import numpy as np

# Paraphrases of common.model_utils.SYSTEM_PROMPT. Each asks for the same
# behaviour — a bare short fact — so the teacher's answers stay comparable
# across variants and the bank's scoring stays valid.
FACTUAL_SYSTEM_VARIANTS = (
    "You are a precise factual assistant. Answer with ONLY the specific fact "
    "requested — a name, number, or short phrase. Do not add explanation.",
    "Answer with just the fact asked for. No preamble, no explanation, no full "
    "sentences — a name, a number, or a short phrase.",
    "You are a factual lookup service. Reply with the requested value alone.",
    "Respond with the shortest correct answer: a single name, number, or phrase. "
    "Never explain your reasoning.",
    "Be terse. Give only the specific fact requested and nothing else.",
    "You answer factual questions. Output only the answer itself, with no "
    "surrounding text.",
)

# Paraphrases of common.prompts_list.LIST_SYSTEM_PROMPT.
LIST_SYSTEM_VARIANTS = (
    "You are a precise factual assistant. Answer with ONLY the items requested, "
    "as a short comma-separated list. Do not number them, explain them, or add "
    "any other text.",
    "Reply with just the requested items, comma-separated. No numbering, no "
    "explanation, no extra words.",
    "List only the items asked for, separated by commas. Nothing else.",
    "You produce short comma-separated lists of facts. Give the items and stop.",
    "Answer as a plain comma-separated list of the requested items, with no "
    "commentary.",
    "Output the items requested as a bare list separated by commas.",
)

# The question goes in verbatim; only the framing around it varies.
QUESTION_FRAMINGS = (
    "{q}",
    "Question: {q}",
    "{q}",                       # the bare form twice: it is the eval form, so
                                 # it should stay the plurality of the fit set
    "Please answer: {q}",
    "I need to know: {q}",
    "{q} Answer concisely.",
    "Quick question — {q}",
    "Answer this: {q}",
    "{q} (short answer only)",
    "Here is a question. {q}",
)


def system_variants(base_system: str | None) -> tuple[str, ...]:
    """The paraphrase set matching whichever system prompt a bank uses.

    Banks that supply their own system prompt (the list banks, whose default
    would otherwise suppress list answers) get the list paraphrases; everything
    else gets the factual ones. An unrecognised custom system prompt is used
    alone rather than guessed at — silently substituting a paraphrase that asks
    for different behaviour would change what the teacher generates.
    """
    if base_system is None:
        return FACTUAL_SYSTEM_VARIANTS
    if base_system == LIST_SYSTEM_VARIANTS[0]:
        return LIST_SYSTEM_VARIANTS
    if base_system == FACTUAL_SYSTEM_VARIANTS[0]:
        return FACTUAL_SYSTEM_VARIANTS
    return (base_system,)


def variant_for(index: int, base_system: str | None,
                seed: int = 0) -> tuple[str, str]:
    """(system prompt, question framing) for the capture item at `index`.

    Deterministic in `index` so a re-capture reproduces the same fit set, and
    decorrelated between the two axes so the framings are not locked to one
    system prompt. Returned as a format string for the caller to apply, since
    only the caller knows the question text.
    """
    systems = system_variants(base_system)
    rng = np.random.default_rng([seed, index])
    return (systems[int(rng.integers(len(systems)))],
            QUESTION_FRAMINGS[int(rng.integers(len(QUESTION_FRAMINGS)))])


def apply_framing(framing: str, question: str) -> str:
    return framing.format(q=question)
