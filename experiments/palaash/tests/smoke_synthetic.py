"""Synthetic smoke test: run extract -> train -> analyze on fake hidden states,
without downloading or loading any real model weights.

It patches diagnosis.extract_states.hidden_states_for_prompt with a deterministic fake
whose small- and large-"model" states share a low-dim latent per prompt (so the
DM ridge fit has real linear signal), and points RESULTS_ROOT at a temp dir.

Covers:
  * prompt-aligned pairs (llama2qwen, qwen2llama) — the two fake models see
    DIFFERENT sequence lengths, as real cross-family tokenizers would; the
    pipeline must still produce one final-token row per prompt per model.
  * a token-aligned pair (llama) — equal sequence lengths, LAST_K rows kept —
    to confirm the existing path still produces the same layout.

Run from experiments/palaash:
    python tests/smoke_synthetic.py
"""

import json
import shutil
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diagnosis.config as config
from diagnosis.config import PAIRS, LAST_K
from diagnosis import extract_states, train_dm, analyze, cka, fit_adapter
from common.prompts import PROMPTS

LATENT = 32          # shared latent dim linking small and large fake states
_PROJ_CACHE = {}


def _seed(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def _proj(tag: str, layer: int, dim: int) -> np.ndarray:
    key = (tag, layer, dim)
    if key not in _PROJ_CACHE:
        rng = np.random.default_rng(_seed(f"proj|{tag}|{layer}|{dim}"))
        _PROJ_CACHE[key] = rng.standard_normal((LATENT, dim)).astype(np.float32)
    return _PROJ_CACHE[key]


@dataclass
class FakeLM:
    tag: str
    n_layers: int
    hidden_size: int
    seq_len: int     # emulates the tokenizer: cross-family models get different lengths


def fake_hidden_states_for_prompt(lm: FakeLM, question: str, last_k: int | None = None):
    """Same contract as model_utils.hidden_states_for_prompt, no torch."""
    seq = lm.seq_len
    n_used = last_k if (last_k is not None and seq > last_k) else seq
    z = np.random.default_rng(_seed(f"latent|{question}")).standard_normal(LATENT)
    states = []
    for L in range(lm.n_layers + 1):
        noise = np.random.default_rng(
            _seed(f"noise|{lm.tag}|{L}|{question}")).standard_normal((n_used, LATENT))
        rows = (z[None, :] + 0.05 * noise) @ _proj(lm.tag, L, lm.hidden_size)
        states.append(rows.astype(np.float32))
    return states, n_used, n_used - 1


def run_pair(pair_name: str, n_div: int, n_ctrl: int) -> None:
    pair = PAIRS[pair_name]
    print(f"\n{'#' * 70}\n# smoke: pair={pair_name} align={pair.align} "
          f"({n_div} divergent / {n_ctrl} control prompts)\n{'#' * 70}")

    ids = [p["id"] for p in PROMPTS[: n_div + n_ctrl]]
    out_dir = config.results_dir(pair)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "selection.json", "w") as f:
        json.dump({"divergent_small_wrong_large_right": ids[:n_div],
                   "control_both_right": ids[n_div:]}, f)

    seq_s, seq_l = (40, 40) if pair.align == "token" else (37, 52)
    lm_s = FakeLM("small-" + pair_name, pair.n_layers_small, pair.dim_small, seq_s)
    lm_l = FakeLM("large-" + pair_name, pair.n_layers_large, pair.dim_large, seq_l)

    extract_states.run(pair, lm_s, lm_l)

    # ── layout checks ─────────────────────────────────────────────────────────
    sdir = config.states_dir(pair)
    meta = np.load(sdir / "meta.npz")
    x = np.load(sdir / "x_small.npz")
    y = np.load(sdir / "y_large.npz")
    n_prompts = n_div + n_ctrl
    rows_per_prompt = 1 if pair.align == "prompt" else min(seq_s, LAST_K)
    n_rows = n_prompts * rows_per_prompt
    assert len(x.files) == pair.n_layers_small + 1 and len(y.files) == pair.n_layers_large + 1
    assert x["layer_00"].shape == (n_rows, pair.dim_small), x["layer_00"].shape
    assert y["layer_00"].shape == (n_rows, pair.dim_large), y["layer_00"].shape
    assert meta["prompt_index"].shape == (n_rows,)
    assert meta["is_last"].sum() == n_prompts
    if pair.align == "prompt":
        assert (meta["is_last"] == 1).all(), "prompt-aligned rows must all be last tokens"
    assert (np.sort(np.unique(meta["prompt_index"])) == np.arange(n_prompts)).all()

    # ── train + analyze on the synthetic states ───────────────────────────────
    summary = train_dm.run(pair)
    assert "divergent" in summary and "control" in summary and "divergence" in summary
    for name in ("divergent", "control"):
        br = np.array(summary[name]["best_r2_all_per_small_layer"])
        assert br.shape == (pair.n_layers_small + 1,)
        assert np.isfinite(br).all(), f"{name}: non-finite best-R2 curve"
    if pair.align == "prompt":
        # every row is a last token, so the all-token and last-token grids coincide
        g = np.load(config.dm_dir(pair) / "grids_control.npz")
        assert np.allclose(g["r2_all"], g["r2_last"], equal_nan=True)

    analyze.run(pair)
    fig = config.figures_dir(pair) / "divergence_curve.png"
    assert fig.exists(), "analyze did not write divergence_curve.png"

    # ── cka on the same synthetic states ──────────────────────────────────────
    cs = cka.run(pair)
    assert "divergent" in cs and "control" in cs and "divergence" in cs
    for name in ("divergent", "control"):
        curve = np.array(cs[name]["best_cka_last_per_small_layer"])
        assert curve.shape == (pair.n_layers_small + 1,)
        assert np.isfinite(curve).all(), f"{name}: non-finite best-CKA curve"
        # the fake models share a strong low-dim latent, so CKA must find it
        assert (curve > 0.5).all(), f"{name}: CKA missed the shared latent"
    if pair.align == "prompt":
        g = np.load(config.cka_dir(pair) / "grids_control.npz")
        assert np.allclose(g["cka_all"], g["cka_last"], equal_nan=True)
    assert (config.figures_dir(pair) / "cka_curve.png").exists()
    assert (config.results_dir(pair) / "verdict_cka.txt").exists()

    # ── fit_adapter (numpy-only; stitch itself needs real models) ─────────────
    steps = "layout, train, analyze, cka"
    if pair.align == "token":
        i = summary["divergence_layer_small"]
        j = summary["divergent"]["best_match_large_layer_last"][i]
        assert 1 <= j < pair.n_layers_large, \
            f"selected target j={j} is not an injectable residual stream"
        meta = fit_adapter.run(pair, i, j)
        assert meta["small_layer_i"] == i and meta["large_layer_j"] == j

        arrays, loaded = fit_adapter.load_adapter(pair, i, j)
        assert arrays["W"].shape == (pair.dim_small, pair.dim_large)
        assert arrays["b"].shape == (pair.dim_large,)
        assert loaded["form"] == meta["form"]
        assert all(np.isfinite(a).all() for a in arrays.values())

        # the saved coefficients must reproduce the reported in-sample fit
        X, Y, _, set_label, _ = fit_adapter.load_layer_pair(pair, i, j)
        m = set_label == fit_adapter.SET_LABEL[meta["fit_set"]]
        q = fit_adapter.quality(
            fit_adapter.apply_adapter(X[m], **{k: arrays[k]
                                               for k in ("W", "b", "mu_x", "sd_x")}), Y[m])
        assert abs(q["r2"] - meta["in_sample"]["r2"]) < 1e-4, \
            f"saved adapter does not reproduce reported R2: {q['r2']} vs {meta['in_sample']['r2']}"

        # the synthetic large states are a fixed linear map of a shared latent,
        # so a linear adapter must recover them well
        assert q["r2"] > 0.5, f"adapter failed on linear synthetic data: R2={q['r2']}"

        # a post-norm / embedding target must be refused, not silently fitted
        for bad in (0, pair.n_layers_large):
            try:
                fit_adapter.run(pair, i, bad)
            except SystemExit:
                pass
            else:
                raise AssertionError(f"fit_adapter accepted invalid target j={bad}")
        steps += ", fit_adapter"
    print(f"[ok] {pair_name}: {steps} all pass")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="q1_smoke_"))
    config.RESULTS_ROOT = tmp          # redirect all results_dir/states_dir/... calls
    extract_states.hidden_states_for_prompt = fake_hidden_states_for_prompt
    try:
        run_pair("llama2qwen", n_div=8, n_ctrl=24)   # prompt-aligned, unequal seq lens
        run_pair("qwen2llama", n_div=8, n_ctrl=24)   # prompt-aligned, other direction
        run_pair("llama", n_div=5, n_ctrl=8)         # token-aligned regression check
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
