"""ACCURACY-oriented stitching, large -> small: the large model's early layers
feed an adapter which injects into the small model's late layers, so the small
model does the writing with the large model's reading.

Goal: beat the small model's accuracy without fine-tuning either LLM. This is
NOT a latency win — the path runs part of both models — and the sibling package
`stitching_small_to_large` is the one aimed at speed.

Run from `experiments/palaash`:

    python -m stitching_large_to_small.run headroom --pair llama
    python -m stitching_large_to_small.run sweep    --pair llama

See stitching_large_to_small/README.md for the method and the measured results.
"""
