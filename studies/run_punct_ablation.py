"""Punctuation ablation harness — FIXED per PREREG_punct_mechanism.md §4 (teacher-forced,
patch-style). Consumes the cached baselines from gen_baseline.py.

Fixes applied (PREREG §4.1–4.6, from REDTEAM_punct_prereg.md D1/D4):
  1. punct AND every control candidate pool restricted to the USER-CONTENT token span
     (chat-template wrapper excluded — the old pools included the position-0
     <|im_start|> sink and template tokens: control contamination).
  2. per-item ASSERTION: punct token count within the span == n_periods + n_commas +
     n_question from the stimulus record; mismatch HALTS the run (never silent).
  3. attn_received is a per-ELIGIBLE-query mean (causal-mask-aware; the old all-query
     mean inflated early positions).
  4. knockout hook raises on a missing mask (see gentrace/intervene.py).
  5. dead space-less label ids removed (each script builds " "+label ids).
  6. aggregate stats use SAMPLE sd.
Plus: position classes (P_theory / P_last / P_qmark / P_comma / P_instr per PREREG §4),
C_attnM caliper [0.5x, 2x] with a matching-quality report and the <80%-valid item flag,
co-primary Δ margin alongside Δ correct-logit, seeded bootstrap CIs (B=10k, seed 0).

Modes:
  --gate   G1 (PREREG §10): whole-class KO(all-punct) vs KO(C_attnM) only, ~3 forwards/
           item. Decision: bootstrap 95% CI of the paired contrast covers 0 -> KILL.
  --full   the 2-interventions x 4-position-sets grid on CLEAN pools (continuity with
           the pre-fix pilot; confirmatory grid per PREREG §5–7 lands separately).

Ungated / uncertified (mutates the forward pass). Deterministic.
"""

import argparse
import json
import math
import os
import random
import statistics

import torch

from gentrace.intervene import Intervenor, LABELS

PUNCT_MARKS = (".", ",", "?", ";", ":")
FUNCTION_WORDS = {"is", "are", "then", "if", "and", "or", "a", "the", "they", "someone"}


def user_content_span(seq, tok):
    """[lo, hi) token span of the user message content. Template renders
    <|im_start|>user\\n{content}<|im_end|>... -> content starts at index 3 and ends
    before the first <|im_end|>. Both assumptions guarded."""
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if seq[0] != im_start:
        raise SystemExit(f"template drift: seq[0]={seq[0]} is not <|im_start|>")
    if tok.decode(seq[1:3]) != "user\n":            # guard the lo=3 assumption (verifier N4)
        raise SystemExit(f"template drift: seq[1:3] decodes to {tok.decode(seq[1:3])!r}, not 'user\\n'")
    hi = seq.index(im_end)
    lo = 3
    if hi <= lo:
        raise SystemExit("template drift: empty user-content span")
    return lo, hi


def classify_positions(seq, tok, lo, hi):
    """PREREG §4 position classes, within the user-content span only."""
    def mark(i):
        return tok.decode([seq[i]]).strip()
    punct = [i for i in range(lo, hi) if mark(i) in PUNCT_MARKS]
    qmarks = [i for i in punct if mark(i) == "?"]
    if len(qmarks) != 1:
        raise SystemExit(f"expected exactly one '?' in span, got {len(qmarks)}")
    q = qmarks[0]
    p_instr = [i for i in punct if i > q]                       # instruction tail
    before = [i for i in punct if i < q]
    p_comma = [i for i in before if mark(i) == ","]             # if-then boundary commas
    periods = [i for i in before if mark(i) == "."]
    return {
        "punct_all": punct, "P_qmark": [q], "P_instr": p_instr, "P_comma": p_comma,
        "P_theory": periods[:-1], "P_last": periods[-1:],
    }


def attn_received(attentions, T):
    """Per key-position attention received, as a per-ELIGIBLE-query mean (PREREG §4.3):
    key k is attendable by queries k..T-1 (causal), so divide by T-k, not T."""
    acc = torch.zeros(T)
    for A in attentions:
        acc += A[0].mean(dim=0).float().cpu().sum(dim=0)        # [heads,q,kv] -> [kv]
    acc /= len(attentions)
    eligible = torch.arange(T, 0, -1, dtype=torch.float32)
    return acc / eligible


def matched_control(scores, targets, candidates, lo=0.5, hi=2.0):
    """Greedy attention-matching without replacement + PREREG §4 caliper: a match is
    valid iff its score is within [lo*, hi*] of the target's. Returns (positions,
    n_valid, n_total)."""
    used, out, n_valid = set(), [], 0
    for p in targets:
        best, bd = None, None
        for c in candidates:
            if c in used:
                continue
            dd = abs(scores[c].item() - scores[p].item())
            if bd is None or dd < bd:
                best, bd = c, dd
        if best is None:
            continue
        used.add(best)
        out.append(best)
        t, m = scores[p].item(), scores[best].item()
        if t > 0 and lo * t <= m <= hi * t:
            n_valid += 1
    if len(out) != len(targets):                    # verifier N5: never under-match silently
        raise SystemExit(f"matched_control under-matched: {len(out)}/{len(targets)} "
                         "(candidate pool exhausted)")
    return out, n_valid, len(targets)


def label_ids(tok):
    return {w: tok(" " + w, add_special_tokens=False).input_ids[0] for w in LABELS}


def readout(model, ids, rpos, lids, correct):
    """(correct-label logit, margin) at the readout position; margin is the
    shift-invariant co-primary (PREREG §2)."""
    with torch.no_grad():
        lg = model(ids).logits[0, rpos, :]
    vals = {w: float(lg[i]) for w, i in lids.items()}
    c = vals[correct]
    other = max(v for k, v in vals.items() if k != correct)
    return c, c - other


def boot_ci(diffs, B=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(B):
        means.append(statistics.mean(diffs[rng.randrange(n)] for _ in range(n)))
    means.sort()
    # conventional nearest-rank percentile: p-th value at ceil(p*B)-1, 0-indexed
    # (2.5/97.5% of B=10000 -> 249/9749, not the old 250/9750). Clamped to [0, B-1].
    lo_i = min(B - 1, max(0, math.ceil(0.025 * B) - 1))
    hi_i = min(B - 1, max(0, math.ceil(0.975 * B) - 1))
    return means[lo_i], means[hi_i]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="runs/studies/punct_baseline.json")
    ap.add_argument("--stimuli", default="studies/ruletaker_punct.json")
    ap.add_argument("--out", default=None,
                    help="default derives from mode: punct_ablation_{gate|full}.json")
    ap.add_argument("--gate", action="store_true", help="G1 only (PREREG §10)")
    ap.add_argument("--full", action="store_true", help="2x4 grid on clean pools")
    ap.add_argument("--dtype", default="float16", choices=("float16", "float32"),
                    help="model dtype (float32 = the Modal A100 fp16-question re-check)")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    if not (args.gate or args.full):
        raise SystemExit("pick a mode: --gate (G1, PREREG §10) or --full")
    if not os.path.exists(args.baseline):
        raise SystemExit(f"no baseline cache at {args.baseline} — run gen_baseline.py first")

    B = json.load(open(args.baseline))
    stim = {s["id"]: s for s in json.load(open(args.stimuli))["prompts"]}
    items = B["per_item"]
    if args.n:
        items = items[:args.n]
    intv = Intervenor(B["model"], dtype=args.dtype, device="cuda")
    lids = label_ids(intv.tok)
    layers = list(range(intv.A.n_layers))
    rng = random.Random(0)

    results, skipped, flagged, forwards = [], 0, 0, 0
    for it in items:
        if not it.get("correct"):
            skipped += 1
            continue
        seq, rpos, correct = it["seq"], it["readout_pos"], it["label"]
        lo, hi = user_content_span(seq, intv.tok)
        cls = classify_positions(seq, intv.tok, lo, hi)
        punct = cls["punct_all"]
        # PREREG §4.2 assertion — halt on mismatch, never skip silently
        s = stim[it["id"]]
        want = s["n_periods"] + s["n_commas"] + s["n_question"]
        if len(punct) != want:
            raise SystemExit(
                f"ASSERTION FAILED id={it['id']}: {len(punct)} punct tokens in span "
                f"vs {want} punctuation chars in the stimulus — tokenizer merge or "
                f"span bug; halting (PREREG §4.2)")
        ids = torch.tensor([seq], device=intv.device)
        with torch.no_grad():
            o = intv.model(ids, output_attentions=True)
        forwards += 1
        base_c = float(o.logits[0, rpos, lids[correct]])
        vals = {w: float(o.logits[0, rpos, i]) for w, i in lids.items()}
        base_m = base_c - max(v for k, v in vals.items() if k != correct)
        ar = attn_received(o.attentions, len(seq))
        del o
        nonpunct = [i for i in range(lo, hi) if i not in set(punct)]
        attnM, n_valid, n_tot = matched_control(ar, punct, nonpunct)
        well_matched = (n_valid / n_tot) >= 0.8 if n_tot else False
        if not well_matched:
            flagged += 1

        def run(cm):
            nonlocal forwards
            with cm:
                c, m = readout(intv.model, ids, rpos, lids, correct)
            forwards += 1
            return round(c - base_c, 4), round(m - base_m, 4)

        row = {"id": it["id"], "label": correct, "hops": it["hops"],
               "n_punct": len(punct), "match_valid": f"{n_valid}/{n_tot}",
               "well_matched": well_matched, "base_logit": round(base_c, 4),
               "base_margin": round(base_m, 4), "deltas": {}, "margins": {}}
        if args.gate:
            sets = {"punct": punct, "attn_matched": attnM}
            for name, pos in sets.items():
                dc, dm = run(intv.knockout_attn(layers, pos))
                row["deltas"][f"knockout:{name}"] = dc
                row["margins"][f"knockout:{name}"] = dm
        else:
            # per-item seeded rng so sampled draws do not depend on --n (verifier note);
            # 'random' drawn from NON-PUNCT positions to match the "clean pools" contract
            # (verifier note: the old range(lo,hi) pool included punctuation).
            irng = random.Random(it["id"])
            content_pool = [i for i in nonpunct
                            if intv.tok.decode([seq[i]]).strip().lower() not in FUNCTION_WORDS]
            k = len(punct)
            sets = {"punct": punct,
                    "content": irng.sample(content_pool, min(k, len(content_pool))),
                    "random": irng.sample(nonpunct, min(k, len(nonpunct))),
                    "attn_matched": attnM}
            for iv in ("write", "knockout"):
                for name, pos in sets.items():
                    cm = (intv.ablate_write(layers, pos, "both") if iv == "write"
                          else intv.knockout_attn(layers, pos))
                    dc, dm = run(cm)
                    row["deltas"][f"{iv}:{name}"] = dc
                    row["margins"][f"{iv}:{name}"] = dm
        results.append(row)
        print(f"  id={it['id']:>3} k={len(punct):>2} match={n_valid}/{n_tot} "
              + " ".join(f"{c.split(':')[0][:2]}[{c.split(':')[1][:2]}]={v:+.2f}"
                         for c, v in row["deltas"].items()), flush=True)

    # ---- aggregate: paired punct − attn_matched, KO (the C1-shaped contrast) ----
    def contrast(metric_key, rows):
        d = [r[metric_key]["knockout:punct"] - r[metric_key]["knockout:attn_matched"]
             for r in rows]
        out = {"mean": round(statistics.mean(d), 4) if d else None,
               "sd_sample": round(statistics.stdev(d), 4) if len(d) > 1 else None,
               "n": len(d)}
        if len(d) > 1:
            lo_, hi_ = boot_ci(d)
            out["boot95"] = [round(lo_, 4), round(hi_, 4)]
            out["ci_covers_0"] = bool(lo_ <= 0 <= hi_)
        return out

    logit_c = contrast("deltas", results)
    margin_c = contrast("margins", results)
    wm = [r for r in results if r["well_matched"]]
    # co-primary margin reported on the well-matched subset too (verifier N1 / PREREG §2)
    wm_c = ({"logit": contrast("deltas", wm), "margin": contrast("margins", wm)}
            if len(wm) > 1 else None)

    print(f"\n=== paired KO contrast punct − attn_matched (n={len(results)}, "
          f"skipped {skipped}, match-flagged {flagged}, forwards {forwards}) ===")
    print(f"  Δcorrect-logit: {logit_c}")
    print(f"  Δmargin (co-primary): {margin_c}")
    if wm_c:
        print(f"  well-matched subset: logit {wm_c['logit']}")
    if wm_c:
        print(f"                       margin {wm_c['margin']}")

    verdict = None
    if args.gate:
        kill = logit_c.get("ci_covers_0", True)
        sign_ok = (logit_c["mean"] or 0) < 0 and (margin_c["mean"] or 0) < 0
        verdict = ("KILL — CI covers 0 (no C1-shaped effect under clean controls)" if kill
                   else ("PROCEED" if sign_ok
                         else "ANOMALOUS — CI excludes 0 but sign/co-primary disagree; "
                              "report, do not proceed silently"))
        print(f"\n=== G1 GATE (PREREG §10): {verdict} ===")

    gpu_name = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    mode = "gate" if args.gate else "full"
    out_path = args.out or f"runs/studies/punct_ablation_{mode}.json"   # per-mode default (N3)
    artifact = {"study": f"punct_ablation_{mode}", "model": B["model"],
                "dtype": args.dtype, "gpu_name": gpu_name,
                "harness": "fixed per PREREG §4 (clean pools, assertion, per-eligible-query "
                           "attn, caliper, sample sd, co-primary margin)",
                "n_items": len(results), "n_skipped": skipped, "n_match_flagged": flagged,
                "n_forwards": forwards,
                "ko_punct_minus_attnM_logit": logit_c,
                "ko_punct_minus_attnM_margin": margin_c,
                "well_matched_subset": wm_c,
                "gate_verdict": verdict,                                 # persisted (N2)
                "shortcuts_taken": [], "per_item": results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
