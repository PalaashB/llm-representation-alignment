"""Question 1 — can alignment identify the root causes of hallucinations?

Pipeline package, parameterised by a small-vs-large model pair (see
q1.config.PAIRS: currently "llama" and "qwen"). Steps (each also runnable as
`python -m q1.<module>`, which uses the default pair):

    select_prompts  -> results/<pair>/generations.csv, selection.json
    extract_states  -> results/<pair>/states/*.npz
    train_dm        -> results/<pair>/dm/*.npz, dm_summary.json
    analyze         -> results/<pair>/figures/*.png (+ printed verdict)

Run a whole pair with `python run_q1.py [--pair qwen]` from the repo root.
"""
