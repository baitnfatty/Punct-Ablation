"""During-generation punctuation knockout — the pre-registered exploratory complement
to C-A (PREREG_punct_mechanism §2; addresses the frozen-CoT caveat the red-team flagged).

C-A's knockout was applied to a FROZEN, answer-bearing CoT. Here the knockout is active
THROUGHOUT generation: the model reasons with attention into prompt punctuation masked
the whole way. Metrics (no confirmatory weight, exploratory):
  - answer flip rate: baseline answer vs knockout answer differ?
  - CoT edit distance: normalized token-level Levenshtein of the generated text.
Both reported for punctuation KO AND an attention-matched non-punct control (so a flip
is interpretable as punct-specific vs generic-high-attention), mirroring C-A's control.

Qwen3-4B-Instruct-2507, cuda/fp16, greedy (deterministic). Subset of N items. Slow
(3 generations/item). Run twice + byte-compare.

  python -u studies/run_during_gen.py --n 30
"""

import argparse
import json
import os
import re
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gentrace.intervene import Intervenor
from run_punct_ablation import attn_received, classify_positions, matched_control

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def final_answer(text):
    m = re.findall(r"\b(True|False|Unknown)\b", text)
    return m[-1] if m else None


def norm_levenshtein(a, b):
    """token-level normalized edit distance in [0,1]."""
    if not a and not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb] / max(la, lb)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--out", default="runs/studies/during_gen.json")
    args = ap.parse_args()

    intv = Intervenor(MODEL, dtype="float16", device="cuda")
    layers = list(range(intv.A.n_layers))
    stim = json.load(open("studies/ruletaker_punct.json"))["prompts"][:args.n]

    def gen(enc, cm=None):
        import contextlib
        with (cm or contextlib.nullcontext()), torch.no_grad():
            out = intv.model.generate(**enc, max_new_tokens=args.max_new,
                                      do_sample=False, pad_token_id=intv.tok.eos_token_id)
        g = out[0][enc["input_ids"].shape[1]:]
        return intv.tok.decode(g, skip_special_tokens=True), g.tolist()

    rows = []
    for it in stim:
        enc = intv.encode(it["prompt"])
        plen = enc["input_ids"].shape[1]
        seq = enc["input_ids"][0].tolist()
        lo, hi = 3, plen                              # user-content span (chat template)
        # find <|im_end|> end of user content
        im_end = intv.tok.convert_tokens_to_ids("<|im_end|>")
        hi = seq.index(im_end) if im_end in seq else plen
        punct = classify_positions(seq, intv.tok, lo, hi)["punct_all"]
        # attention-matched control from a baseline prompt forward
        with torch.no_grad():
            o = intv.model(enc["input_ids"], output_attentions=True)
        ar = attn_received(o.attentions, plen)
        del o
        nonpunct = [i for i in range(lo, hi) if i not in set(punct)]
        attnM, nv, nt = matched_control(ar, punct, nonpunct)

        base_txt, base_ids = gen(enc)
        p_txt, p_ids = gen(enc, intv.knockout_attn(layers, punct))
        a_txt, a_ids = gen(enc, intv.knockout_attn(layers, attnM))
        ba, pa, aa = final_answer(base_txt), final_answer(p_txt), final_answer(a_txt)
        rows.append({
            "id": it["id"], "gold": it["label"], "base_ans": ba,
            "punct_ko_ans": pa, "attnM_ko_ans": aa,
            "punct_flip": bool(ba is not None and pa != ba),
            "attnM_flip": bool(ba is not None and aa != ba),
            "punct_edit": round(norm_levenshtein(base_ids, p_ids), 4),
            "attnM_edit": round(norm_levenshtein(base_ids, a_ids), 4),
            "match_valid": f"{nv}/{nt}"})
        print(f"  id={it['id']:>3} base={ba} punctKO={pa} attnMKO={aa} "
              f"edit p={rows[-1]['punct_edit']:.2f} aM={rows[-1]['attnM_edit']:.2f}", flush=True)

    usable = [r for r in rows if r["base_ans"] is not None]
    n = len(usable)
    agg = {
        "n": n, "n_total": len(rows),
        "punct_flip_rate": round(sum(r["punct_flip"] for r in usable) / n, 4),
        "attnM_flip_rate": round(sum(r["attnM_flip"] for r in usable) / n, 4),
        "punct_edit_mean": round(statistics.mean(r["punct_edit"] for r in usable), 4),
        "attnM_edit_mean": round(statistics.mean(r["attnM_edit"] for r in usable), 4),
    }
    print(f"\n=== during-generation KO (n={n}, exploratory — no confirmatory weight) ===")
    print(f"  flip rate:  punct {agg['punct_flip_rate']}   attn-matched {agg['attnM_flip_rate']}")
    print(f"  CoT edit:   punct {agg['punct_edit_mean']}   attn-matched {agg['attnM_edit_mean']}")
    print("  read: punct ≈ attn-matched → during-generation agrees with the frozen-CoT "
          "null (no punct-specific disruption); punct ≫ attn-matched → frozen-CoT masked "
          "a real effect.")
    artifact = {"study": "during_gen_ko", "model": MODEL, "dtype": "float16",
                "gpu_name": torch.cuda.get_device_name(0), "max_new": args.max_new,
                "aggregate": agg, "shortcuts_taken": [], "per_item": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
