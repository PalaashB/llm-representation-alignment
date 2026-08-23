"""LATENCY-oriented stitching, small -> large: the small model's early layers
feed an adapter which injects into the large model's late layers, so the skipped
large blocks never run during decode.

Run from `experiments/palaash`:

    python -m stitching_small_to_large.run headroom --pair llama
    python -m stitching_small_to_large.run sweep    --pair llama --modes exit warm

The opposite direction (large -> small, aimed at accuracy) is the sibling
package `stitching_large_to_small`.

See stitching_small_to_large/README.md for the method and the measured results.
"""
