"""Step 6 — inject a small-model layer into the large model's residual stream.

This is the first step in the pipeline that *acts* on the diagnosis rather than
measuring it. Given the adapter saved by `fit_adapter`, it runs the small model,
maps its layer-i activations into the large model's geometry, overwrites the
large model's residual stream at the input to block j, and finishes the large
forward pass.

Injection convention (verified, not assumed — see `check_identity_injection`):
in transformers 5.x `output_hidden_states=True` is served by capture hooks that
record the first layer's *input* and then each layer's *output*, so

    hidden_states[j] == input to decoder block j          for j < n_layers
    hidden_states[n_layers] == model.norm(...)            (NOT a residual stream)

so injecting at `model.layers[j]` via a forward *pre*-hook writes exactly the
tensor the DM grid was fit to predict. Injecting at n_layers is impossible,
which is the same fact `DROP_POST_NORM_LAYER` encodes on the analysis side.

Scope (v1): no KV-cache surgery. Greedy decoding re-runs both models over the
full growing sequence each step ("lockstep, cache-free"). That is O(n^2) but
obviously correct — the large model's cache can never fall out of sync with the
injected stream because there is no cache. Token-aligned pairs only, so both
models consume the identical token sequence at every step.

    python -m q1.stitch --pair llama --check
    python -m q1.stitch --pair llama --prompt "What is the capital of New Zealand?"
    python -m q1.stitch --pair llama --run-selected

Outputs (in results/<pair>/stitch/): stitch_i{i}_j{j}.json, checks_i{i}_j{j}.json
"""

from __future__ import annotations

import argparse
import contextlib
import json

import numpy as np
import torch

from q1.config import (
    ModelPair, PAIRS, DEFAULT_PAIR, STITCH_MAX_NEW_TOKENS, PRESERVE_PREFIX,
    results_dir, stitch_dir,
)
from q1.fit_adapter import load_adapter, default_layers, validate_layers
from q1.model_utils import LM, load_pair, build_prompt_ids, generate_answer
from q1.prompts import PROMPTS

BY_ID = {p["id"]: p for p in PROMPTS}


# ── module plumbing ───────────────────────────────────────────────────────────
def decoder_stack(lm: LM) -> torch.nn.Module:
    """The module that owns the decoder blocks, whatever the wrapper nesting is.

    Returned rather than just the block list because the early-exit path in
    `stitch_fast` also needs its siblings (embed_tokens, rotary_emb, norm).
    """
    m = lm.model
    for path in (("model",), ("model", "language_model"), ("transformer",), ()):
        node = m
        try:
            for attr in path:
                node = getattr(node, attr)
        except AttributeError:
            continue
        for name in ("layers", "h"):
            if isinstance(getattr(node, name, None), torch.nn.ModuleList):
                return node
    raise RuntimeError(f"could not locate the decoder stack on {type(m).__name__}")


def decoder_layers(lm: LM) -> torch.nn.ModuleList:
    """The decoder block list, whatever the wrapper nesting is for this model."""
    stack = decoder_stack(lm)
    return stack.layers if hasattr(stack, "layers") else stack.h


@contextlib.contextmanager
def residual_override(lm: LM, j: int, make_tensor):
    """Overwrite the residual stream entering decoder block j.

    `make_tensor(current)` receives the incoming hidden states and returns the
    replacement (or None to leave them alone — used by the null-hook check).
    Registered with_kwargs so the layer's other arguments (position_embeddings,
    attention_mask, past_key_values, ...) pass through untouched.
    """
    layer = decoder_layers(lm)[j]

    def pre_hook(module, args, kwargs):
        if args:
            current = args[0]
        elif "hidden_states" in kwargs:
            current = kwargs["hidden_states"]
        else:
            raise RuntimeError("decoder layer called without hidden_states")
        new = make_tensor(current)
        if new is None:
            return None                      # leave the call exactly as it was
        new = new.to(device=current.device, dtype=current.dtype)
        if new.shape != current.shape:
            raise RuntimeError(f"injection shape {tuple(new.shape)} != "
                               f"stream shape {tuple(current.shape)}")
        if args:
            return (new,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = new
        return args, kwargs

    handle = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


# ── the adapter as a torch op ─────────────────────────────────────────────────
class Adapter:
    """Y_hat = ((X - mu_x) / sd_x) @ W + b, applied in float32 then cast back."""

    def __init__(self, arrays: dict, meta: dict, device: str):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device, torch.float32)
        self.W, self.b = t(arrays["W"]), t(arrays["b"])
        self.mu_x, self.sd_x = t(arrays["mu_x"]), t(arrays["sd_x"])
        self.meta = meta
        self.i, self.j = meta["small_layer_i"], meta["large_layer_j"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.to(torch.float32) - self.mu_x) / self.sd_x) @ self.W + self.b


# ── forward passes ────────────────────────────────────────────────────────────
@torch.no_grad()
def baseline_first_token(lm: LM, ids: torch.Tensor) -> int:
    """The token the model would emit next, unmodified."""
    return int(lm.model(ids).logits[0, -1].argmax())


@torch.no_grad()
def small_layer_states(lm: LM, ids: torch.Tensor, i: int) -> torch.Tensor:
    """(1, seq, dim_small) — the small model's layer-i residual stream."""
    out = lm.model(ids, output_hidden_states=True)
    return out.hidden_states[i]


@torch.no_grad()
def stitched_logits(lm_small: LM, lm_large: LM, ids: torch.Tensor, adapter: Adapter,
                    last_only: bool = False,
                    preserve_prefix: int = PRESERVE_PREFIX) -> torch.Tensor:
    """Run the stitched forward pass and return the large model's logits.

    `preserve_prefix` leaves the first N positions of the large model's own
    residual stream in place. N=1 is not a tuning knob — see PRESERVE_PREFIX:
    overwriting the attention-sink token at position 0 destroys the forward pass
    outright, and the adapter cannot reconstruct it.
    """
    xs = small_layer_states(lm_small, ids, adapter.i)
    mapped = adapter(xs)                                    # (1, seq, dim_large)

    def make(current):
        if last_only:
            new = current.clone()
            new[:, -1, :] = mapped[:, -1, :].to(current.dtype)
            return new
        new = mapped.to(current.dtype).clone()
        if preserve_prefix > 0:
            n = min(preserve_prefix, new.shape[1])
            new[:, :n, :] = current[:, :n, :]
            return new
        return new

    with residual_override(lm_large, adapter.j, make):
        return lm_large.model(ids).logits


@torch.no_grad()
def stitched_generate(lm_small: LM, lm_large: LM, question: str, adapter: Adapter,
                      max_new_tokens: int = STITCH_MAX_NEW_TOKENS,
                      last_only: bool = False,
                      preserve_prefix: int = PRESERVE_PREFIX) -> dict:
    """Cache-free lockstep greedy decode. Both models see the same growing
    sequence; the small model supplies layer i for every position each step."""
    tok = lm_large.tokenizer
    ids = build_prompt_ids(tok, question, lm_large.device)
    n_prompt = ids.shape[1]
    first_token_id = None
    for _ in range(max_new_tokens):
        logits = stitched_logits(lm_small, lm_large, ids, adapter, last_only, preserve_prefix)
        nxt = int(logits[0, -1].argmax())
        if first_token_id is None:
            first_token_id = nxt
        if nxt == tok.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    gen = ids[0, n_prompt:]
    return {
        "text": tok.decode(gen, skip_special_tokens=True).strip(),
        "first_token_id": first_token_id,
        "first_token": tok.decode([first_token_id]) if first_token_id is not None else "",
        "n_generated": int(gen.shape[0]),
    }


# ── correctness checks (plumbing, not accuracy) ───────────────────────────────
@torch.no_grad()
def check_null_hook(lm_large: LM, ids: torch.Tensor, j: int) -> dict:
    """A registered pre-hook that changes nothing must not change the output."""
    base = lm_large.model(ids).logits
    with residual_override(lm_large, j, lambda cur: None):
        hooked = lm_large.model(ids).logits
    return _compare(base, hooked, "null_hook")


@torch.no_grad()
def check_identity_injection(lm_large: LM, ids: torch.Tensor, j: int) -> dict:
    """Injecting the large model's OWN hidden_states[j] must reproduce the
    baseline exactly. This is what validates the injection index/convention: if
    hidden_states[j] were the *output* of block j rather than its input, this
    would be off by one layer and the logits would differ."""
    out = lm_large.model(ids, output_hidden_states=True)
    base, true_hs = out.logits, out.hidden_states[j]
    with residual_override(lm_large, j, lambda cur: true_hs):
        injected = lm_large.model(ids).logits
    return _compare(base, injected, "identity_injection")


@torch.no_grad()
def check_offbyone_injection(lm_large: LM, ids: torch.Tensor, j: int) -> dict:
    """Negative control: injecting hidden_states[j+1] at block j SHOULD differ.
    If this also matched, the identity check above would be vacuous."""
    out = lm_large.model(ids, output_hidden_states=True)
    base, wrong_hs = out.logits, out.hidden_states[j + 1]
    with residual_override(lm_large, j, lambda cur: wrong_hs):
        injected = lm_large.model(ids).logits
    r = _compare(base, injected, "offbyone_injection")
    r["expected"] = "differs"
    r["passed"] = not r["tokens_match"]
    return r


def _compare(a: torch.Tensor, b: torch.Tensor, name: str) -> dict:
    max_abs = float((a.float() - b.float()).abs().max())
    tokens_match = bool((a[0].argmax(-1) == b[0].argmax(-1)).all())
    return {
        "check": name, "max_abs_logit_diff": max_abs,
        "bit_identical": max_abs == 0.0, "tokens_match": tokens_match,
        "passed": tokens_match and max_abs == 0.0,
    }


@torch.no_grad()
def check_sink_position(lm_large: LM, ids: torch.Tensor, j: int) -> dict:
    """Quantify why PRESERVE_PREFIX exists: report the residual norm of position
    0 against the rest at the injection layer. This is diagnostic — it always
    'passes' — but it records the number that justifies the constant."""
    hs = lm_large.model(ids, output_hidden_states=True).hidden_states[j][0].float()
    norms = hs.norm(dim=-1)
    sink, rest = float(norms[0]), norms[1:]
    return {"check": "sink_position", "passed": True,
            "sink_norm": sink, "other_norm_mean": float(rest.mean()),
            "other_norm_max": float(rest.max()),
            "ratio": sink / float(rest.mean())}


def check_adapter_rows(pair: ModelPair, adapter: Adapter) -> dict:
    """Cosine / relative L2 of adapter(X_i) against true Y_j on cached rows —
    reported from the adapter's own sidecar, which measured it on a prompt-level
    holdout at fit time."""
    m = adapter.meta
    return {"check": "adapter_rows", "fit_set": m["fit_set"],
            "in_sample": m["in_sample"], "held_out": m["held_out"],
            "held_out_last_token": m["held_out_last_token"],
            "passed": m["held_out"] is not None and m["held_out"]["cosine_mean"] > 0.5}


def run_checks(pair: ModelPair, lm_small: LM, lm_large: LM, adapter: Adapter,
               probe: str = "What is the capital city of New Zealand?") -> list[dict]:
    ids = build_prompt_ids(lm_large.tokenizer, probe, lm_large.device)
    checks = [
        check_null_hook(lm_large, ids, adapter.j),
        check_identity_injection(lm_large, ids, adapter.j),
        check_offbyone_injection(lm_large, ids, adapter.j),
        check_sink_position(lm_large, ids, adapter.j),
        check_adapter_rows(pair, adapter),
    ]
    print("\n" + "=" * 64 + "\nPLUMBING CHECKS\n" + "=" * 64)
    for c in checks:
        flag = "PASS" if c["passed"] else "FAIL"
        extra = ""
        if "max_abs_logit_diff" in c:
            extra = (f"  max|dlogit|={c['max_abs_logit_diff']:.3e}  "
                     f"tokens_match={c['tokens_match']}")
        elif c["check"] == "sink_position":
            extra = (f"  |h[0]|={c['sink_norm']:.1f} vs mean {c['other_norm_mean']:.1f} "
                     f"({c['ratio']:.0f}x) -> preserve_prefix must be >= 1")
        elif c["check"] == "adapter_rows" and c["held_out"]:
            extra = (f"  held-out cos={c['held_out']['cosine_mean']:.4f} "
                     f"relL2={c['held_out']['rel_l2_mean']:.4f}")
        print(f"  [{flag}] {c['check']}{extra}")
    return checks


# ── driver ────────────────────────────────────────────────────────────────────
def run(pair: ModelPair, lm_small: LM | None = None, lm_large: LM | None = None,
        i: int | None = None, j: int | None = None,
        prompts: list[str] | None = None, run_selected: bool = False,
        checks_only: bool = False, last_only: bool = False,
        max_new_tokens: int = STITCH_MAX_NEW_TOKENS, n_control: int = 8,
        preserve_prefix: int = PRESERVE_PREFIX) -> dict:
    if pair.align != "token":
        raise SystemExit(f"pair '{pair.name}' is {pair.align}-aligned; stitching v1 is "
                         f"token-aligned pairs only.")
    if i is None or j is None:
        di, dj = default_layers(pair)
        i, j = (di if i is None else i), (dj if j is None else j)
    validate_layers(pair, i, j)

    arrays, meta = load_adapter(pair, i, j)
    if lm_small is None or lm_large is None:
        lm_small, lm_large = load_pair(pair)
    adapter = Adapter(arrays, meta, lm_large.device)
    span = ("final position only" if last_only
            else f"all positions (first {preserve_prefix} preserved)")
    print(f"[stitch] {pair.name}: {pair.small_tag} L{i} -> {pair.large_tag} L{j}  {span}")

    out_dir = stitch_dir(pair)
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = run_checks(pair, lm_small, lm_large, adapter)
    with open(out_dir / f"checks_i{i:02d}_j{j:02d}.json", "w") as f:
        json.dump({"pair": pair.name, "i": i, "j": j, "checks": checks}, f, indent=2)
    hard_fail = [c for c in checks
                 if not c["passed"] and c["check"] in ("null_hook", "identity_injection",
                                                       "offbyone_injection")]
    if hard_fail:
        raise SystemExit(f"injection plumbing check(s) failed: "
                         f"{[c['check'] for c in hard_fail]} — refusing to report stitched "
                         f"generations from a path that is not verified.")
    if checks_only:
        print(f"\nWrote {out_dir}/checks_i{i:02d}_j{j:02d}.json")
        return {"checks": checks}

    # ── pick prompts ──────────────────────────────────────────────────────────
    items = []
    if prompts:
        items += [{"prompt_id": None, "set": "custom", "question": q, "gold": None}
                  for q in prompts]
    if run_selected or not items:
        sel_path = results_dir(pair) / "selection.json"
        if not sel_path.exists():
            raise SystemExit(f"{sel_path} not found — run the select step, or pass --prompt.")
        with open(sel_path) as f:
            sel = json.load(f)
        # The lockstep decode is cache-free, so control prompts are capped even
        # under --run-selected; all 120 would dominate the runtime for no extra
        # information about whether the stitch path works.
        n_div, n_ctrl = (None, n_control) if run_selected else (2, 2)
        div = sel["divergent_small_wrong_large_right"][:n_div]
        ctrl = sel["control_both_right"][:n_ctrl]
        for pid in div:
            items.append({"prompt_id": pid, "set": "divergent",
                          "question": BY_ID[pid]["question"], "gold": BY_ID[pid].get("answers")})
        for pid in ctrl:
            items.append({"prompt_id": pid, "set": "control",
                          "question": BY_ID[pid]["question"], "gold": BY_ID[pid].get("answers")})

    # ── generate: small alone / large alone / stitched ────────────────────────
    print("\n" + "=" * 64 + f"\nSTITCHED GENERATIONS  ({len(items)} prompts)\n" + "=" * 64)
    rows = []
    for k, it in enumerate(items, 1):
        q = it["question"]
        small_txt = generate_answer(lm_small, q, max_new_tokens)
        large_txt = generate_answer(lm_large, q, max_new_tokens)
        st = stitched_generate(lm_small, lm_large, q, adapter, max_new_tokens,
                               last_only, preserve_prefix)
        # v1's scoped claim is prefill + first generated token, so compare that
        # explicitly against both baselines rather than only the full string.
        ids = build_prompt_ids(lm_large.tokenizer, q, lm_large.device)
        ft_large = baseline_first_token(lm_large, ids)
        ft_small = baseline_first_token(lm_small, ids)
        row = {**it, "small": small_txt, "large": large_txt,
               "stitched": st["text"],
               "stitched_first_token": st["first_token"],
               "large_first_token": lm_large.tokenizer.decode([ft_large]),
               "small_first_token": lm_small.tokenizer.decode([ft_small]),
               "first_token_eq_large": st["first_token_id"] == ft_large,
               "first_token_eq_small": st["first_token_id"] == ft_small,
               "stitched_eq_large": st["text"] == large_txt,
               "stitched_eq_small": st["text"] == small_txt}
        rows.append(row)
        tag = f"[{it['set']}]" + (f" {it['prompt_id']}" if it["prompt_id"] else "")
        print(f"\n({k}/{len(items)}) {tag}  {q}")
        if it["gold"]:
            print(f"      gold     : {it['gold']}")
        print(f"      {pair.small_tag:>8} : {small_txt!r}")
        print(f"      {pair.large_tag:>8} : {large_txt!r}")
        print(f"      stitched : {st['text']!r}")
        print(f"      first tok: {st['first_token']!r}  "
              f"(== {pair.large_tag}: {row['first_token_eq_large']}, "
              f"== {pair.small_tag}: {row['first_token_eq_small']})")

    payload = {
        "pair": pair.name, "small_layer_i": i, "large_layer_j": j,
        "last_only": last_only, "preserve_prefix": preserve_prefix,
        "max_new_tokens": max_new_tokens,
        "adapter": {k: meta[k] for k in ("fit_set", "n_fit_rows", "ridge_alpha",
                                         "in_sample", "held_out", "held_out_last_token")},
        "checks": checks,
        "n_prompts": len(rows),
        "n_stitched_eq_large": sum(r["stitched_eq_large"] for r in rows),
        "n_first_token_eq_large": sum(r["first_token_eq_large"] for r in rows),
        "n_first_token_eq_small": sum(r["first_token_eq_small"] for r in rows),
        "by_set": {
            s: {
                "n": sum(r["set"] == s for r in rows),
                "first_token_eq_large": sum(r["set"] == s and r["first_token_eq_large"]
                                            for r in rows),
                "first_token_eq_small": sum(r["set"] == s and r["first_token_eq_small"]
                                            for r in rows),
                "stitched_eq_large": sum(r["set"] == s and r["stitched_eq_large"]
                                         for r in rows),
            } for s in sorted({r["set"] for r in rows})
        },
        "generations": rows,
    }
    json_path = out_dir / f"stitch_i{i:02d}_j{j:02d}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    try:
        import csv
        csv_path = out_dir / f"stitch_i{i:02d}_j{j:02d}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["prompt_id", "set", "question", "gold",
                                              "small", "large", "stitched",
                                              "small_first_token", "large_first_token",
                                              "stitched_first_token",
                                              "first_token_eq_large", "first_token_eq_small",
                                              "stitched_eq_large", "stitched_eq_small"])
            w.writeheader()
            for r in rows:
                w.writerow({**r, "gold": "|".join(r["gold"]) if r["gold"] else ""})
        print(f"\nWrote {json_path}\n      {csv_path}")
    except Exception as e:
        print(f"\nWrote {json_path}  [csv skipped: {e}]")

    n = len(rows)
    print("\n" + "=" * 64)
    print(f"v1 scope is prefill + first generated token:")
    print(f"  first token == {pair.large_tag}-alone : "
          f"{payload['n_first_token_eq_large']}/{n}")
    print(f"  first token == {pair.small_tag}-alone : "
          f"{payload['n_first_token_eq_small']}/{n}")
    print(f"  full text  == {pair.large_tag}-alone : "
          f"{payload['n_stitched_eq_large']}/{n}  "
          f"(drift past the first token is expected at v1)")
    for s, d in payload["by_set"].items():
        print(f"    [{s}] n={d['n']}  first_tok=={pair.large_tag}: "
              f"{d['first_token_eq_large']}  first_tok=={pair.small_tag}: "
              f"{d['first_token_eq_small']}")
    print("=" * 64)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", choices=sorted(PAIRS), default=DEFAULT_PAIR)
    ap.add_argument("--i", type=int, default=None)
    ap.add_argument("--j", type=int, default=None)
    ap.add_argument("--prompt", action="append", default=None,
                    help="question to stitch (repeatable)")
    ap.add_argument("--run-selected", action="store_true",
                    help="run every divergent prompt from selection.json (+ --n-control controls)")
    ap.add_argument("--n-control", type=int, default=8,
                    help="control prompts to include under --run-selected (default: 8)")
    ap.add_argument("--check", action="store_true", help="run plumbing checks only")
    ap.add_argument("--last-only", action="store_true",
                    help="overwrite only the final token position instead of the whole stream")
    ap.add_argument("--preserve-prefix", type=int, default=PRESERVE_PREFIX,
                    help=f"leading positions to leave as the large model's own stream "
                         f"(default {PRESERVE_PREFIX}; 0 overwrites the attention sink and "
                         f"produces noise)")
    ap.add_argument("--max-new-tokens", type=int, default=STITCH_MAX_NEW_TOKENS)
    a = ap.parse_args()
    run(PAIRS[a.pair], i=a.i, j=a.j, prompts=a.prompt, run_selected=a.run_selected,
        checks_only=a.check, last_only=a.last_only, max_new_tokens=a.max_new_tokens,
        n_control=a.n_control, preserve_prefix=a.preserve_prefix)


if __name__ == "__main__":
    main()
