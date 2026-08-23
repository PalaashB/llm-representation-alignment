"""Question 1 — can alignment identify the root causes of hallucinations?

Pipeline package, parameterised by a small-vs-large model pair (see
diagnosis.config.PAIRS: "llama", "qwen", and the two cross-family pairs).
Steps (each also runnable as `python -m diagnosis.<module>`, which uses the
default pair):

    select_prompts  -> results/<pair>/generations.csv, selection.json
    extract_states  -> results/<pair>/states/*.npz
    train_dm        -> results/<pair>/dm/*.npz, dm_summary.json
    analyze         -> results/<pair>/figures/*.png (+ printed verdict)
    cka             -> results/<pair>/cka/*.npz, cka_summary.json
    fit_adapter     -> results/<pair>/adapters/adapter_i{i}_j{j}.npz + .json

Run a whole pair with `python -m diagnosis.run [--pair qwen]` from
experiments/palaash. Stitching now lives in the sibling `stitching/` package.
"""
