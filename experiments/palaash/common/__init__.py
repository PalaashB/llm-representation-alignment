"""Leaf utilities shared by both projects in this folder.

    model_utils  model loading, chat formatting, greedy generation, hidden-state
                 extraction (the only module here that imports torch)
    prompts      the factual-question bank both projects are evaluated on
    scoring      normalised gold-answer matching

Nothing here imports from `diagnosis` or `stitching`; the dependency runs one
way only, which is what lets both projects share it without either reaching into
the other.
"""
