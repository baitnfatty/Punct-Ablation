"""Pre-registered positive control runner (studies/PREREG_gpt2_control.md).

On the FINE-TUNED GPT-2 (models/gpt2-ruletaker), over the SAME 180 study items as the
Qwen G1 run: contrast A (P&P-shaped necessity: KO all-punct vs baseline — raw damage +
accuracy drop) and contrast B (the controlled question: KO all-punct − KO
attention-matched, paired, co-primary Δlogit/Δmargin) using the SAME fixed machinery
(clean pools, per-eligible-query attention matching, caliper, seeded bootstrap).

No chat template (span = the whole prompt); readout = label logits at the last position
of "{prompt} Answer:" — exactly the training format. VERIFICATION GATE runs first
(pre-registered): write-zeroing exact + attention-into-key exactly zero ON GPT-2, and
the per-item punct-count assertion — any failure halts before results exist.

Fully deterministic (no RNG in the A/B path) -> run twice + byte-compare for §9-style
determinism. Ungated; Rule 9 applies to any claim.
"""

import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gentrace.intervene import Intervenor, LABELS
from run_punct_ablation import (attn_received, boot_ci, classify_positions,
                                matched_control)

MODEL_DIR = "models/gpt2-ruletaker"
SCAFFOLD = " Answer:"


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stimuli", default="studies/ruletaker_punct.json")
    ap.add_argument("--out", default="runs/studies/gpt2_control.json")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    man = json.load(open("runs/studies/gpt2_finetune_manifest.json"))
    if not man.get("gate_passed"):
        raise SystemExit(f"proceed gate FAILED at fine-tune (val acc "
                         f"{man.get('final_val_acc')}) — per prereg: report, stop")

    # Amendment A2: GPT-2 ROCm kernels showed 1e-4-digit fp nondeterminism; force
    # deterministic implementations so the byte-compare standard holds.
    torch.use_deterministic_algorithms(True)
    intv = Intervenor(MODEL_DIR, dtype="float32", device="cuda")
    tok = intv.tok
    lids = {w: tok(" " + w).input_ids[0] for w in LABELS}
    for w, i in lids.items():
        assert i == man["label_ids"][w], "label-id drift vs training manifest — halt"
    layers = list(range(intv.A.n_layers))
    stim = json.load(open(args.stimuli))["prompts"]
    if args.n:
        stim = stim[:args.n]

    # ---------- pre-registered VERIFICATION GATE (before any result) ----------
    it0 = stim[0]
    ids0 = torch.tensor([tok(it0["prompt"] + SCAFFOLD).input_ids], device="cuda")
    plen0 = len(tok(it0["prompt"]).input_ids)
    cls0 = classify_positions(ids0[0].tolist(), tok, 0, plen0)
    P = cls0["punct_all"][len(cls0["punct_all"]) // 2]
    L = intv.A.n_layers // 2
    rec = {}
    def vh(mod, a, k, out):
        hs = a[0] if a else k["hidden_states"]
        t = out[0] if isinstance(out, (tuple, list)) else out
        rec["in"], rec["out"] = hs[0, P].float().clone(), t[0, P].float().clone()
    with intv.ablate_write([L], [P], "both"):
        h = intv.A.layers[L].register_forward_hook(vh, with_kwargs=True)
        with torch.no_grad():
            intv.model(ids0)
        h.remove()
    wz = (rec["out"] - rec["in"]).abs().max().item()
    with intv.knockout_attn(layers, [P]):
        with torch.no_grad():
            o = intv.model(ids0, output_attentions=True)
        ko_in = max(float(A[0][:, :, P].max()) for A in o.attentions)
    print(f"[verify] GPT-2 write-zero max|Δ|={wz:.3e}  knockout max-attn-into-P={ko_in:.3e}",
          flush=True)
    if wz > 0 or ko_in > 0:
        raise SystemExit("VERIFICATION GATE FAILED on GPT-2 — halting (prereg)")

    # ---------- baseline + contrasts ----------
    rows, skipped, flagged, n_correct_base = [], 0, 0, 0
    for it in stim:
        seq = tok(it["prompt"] + SCAFFOLD).input_ids
        plen = len(tok(it["prompt"]).input_ids)
        ids = torch.tensor([seq], device="cuda")
        cls = classify_positions(seq, tok, 0, plen)
        punct = cls["punct_all"]
        want = it["n_periods"] + it["n_commas"] + it["n_question"]
        if len(punct) != want:
            raise SystemExit(f"ASSERTION FAILED id={it['id']}: {len(punct)} punct tokens "
                             f"vs {want} chars (BPE merge?) — halting (prereg)")
        with torch.no_grad():
            o = intv.model(ids, output_attentions=True)
        lg = o.logits[0, -1, :]
        vals = {w: float(lg[i]) for w, i in lids.items()}
        pred = max(vals, key=vals.get)
        if pred != it["label"]:
            skipped += 1
            continue
        n_correct_base += 1
        base_c = vals[it["label"]]
        base_m = base_c - max(v for k, v in vals.items() if k != it["label"])
        ar = attn_received(o.attentions, len(seq))
        del o
        nonpunct = [i for i in range(0, plen) if i not in set(punct)]
        attnM, n_valid, n_tot = matched_control(ar, punct, nonpunct)
        if (n_valid / n_tot) < 0.8:
            flagged += 1

        def probe(cm):
            with cm, torch.no_grad():
                l2 = intv.model(ids).logits[0, -1, :]
            v = {w: float(l2[i]) for w, i in lids.items()}
            c = v[it["label"]]
            m = c - max(x for k, x in v.items() if k != it["label"])
            return c - base_c, m - base_m, max(v, key=v.get) == it["label"]

        dp_c, dp_m, ko_ok = probe(intv.knockout_attn(layers, punct))
        da_c, da_m, _ = probe(intv.knockout_attn(layers, attnM))
        # Amendment A1: the faithful P&P necessity analog — WA-both, all layers
        wp_c, wp_m, wa_ok = probe(intv.ablate_write(layers, punct, "both"))
        wa_c, wa_m, _ = probe(intv.ablate_write(layers, attnM, "both"))
        rows.append({"id": it["id"], "label": it["label"], "n_punct": len(punct),
                     "match_valid": f"{n_valid}/{n_tot}",
                     "well_matched": (n_valid / n_tot) >= 0.8,
                     "attnM_includes_pos0": bool(0 in attnM),      # amendment A3
                     "base_logit": round(base_c, 4), "base_margin": round(base_m, 4),
                     "ko_punct": [round(dp_c, 4), round(dp_m, 4)],
                     "ko_attnM": [round(da_c, 4), round(da_m, 4)],
                     "wa_punct": [round(wp_c, 4), round(wp_m, 4)],
                     "wa_attnM": [round(wa_c, 4), round(wa_m, 4)],
                     "correct_after_punct_ko": bool(ko_ok),
                     "correct_after_punct_wa": bool(wa_ok)})
        print(f"  id={it['id']:>3} KO[p]={dp_c:+.2f} KO[aM]={da_c:+.2f} "
              f"WA[p]={wp_c:+.2f} WA[aM]={wa_c:+.2f}", flush=True)

    n = len(rows)
    def agg(key_idx, a_key, b_key, subset=None):
        rs = subset if subset is not None else rows
        if len(rs) < 2:
            return {"n": len(rs)}
        d = [r[a_key][key_idx] - r[b_key][key_idx] for r in rs]
        lo, hi = boot_ci(d)
        return {"mean": round(statistics.mean(d), 4),
                "sd_sample": round(statistics.stdev(d), 4), "n": len(rs),
                "boot95": [round(lo, 4), round(hi, 4)],
                "ci_covers_0": bool(lo <= 0 <= hi)}

    def necessity(prefix, ok_key):
        raw = [r[f"{prefix}_punct"][0] for r in rows]
        acc = sum(r[ok_key] for r in rows) / n
        return {"punct_vs_baseline_mean": round(statistics.mean(raw), 4),
                "accuracy_after": round(acc, 4),
                "baseline_accuracy": round(n_correct_base / len(stim), 4)}

    contrast_A_ko = necessity("ko", "correct_after_punct_ko")
    contrast_A_wa = necessity("wa", "correct_after_punct_wa")     # A′: faithful P&P analog
    no_pos0 = [r for r in rows if not r["attnM_includes_pos0"]]   # amendment A3 split
    out_c = {
        "A_ko_readside": contrast_A_ko,
        "A_wa_faithful_PnP": contrast_A_wa,
        "B_ko_logit": agg(0, "ko_punct", "ko_attnM"),
        "B_ko_margin": agg(1, "ko_punct", "ko_attnM"),
        "B_wa_logit": agg(0, "wa_punct", "wa_attnM"),
        "B_wa_margin": agg(1, "wa_punct", "wa_attnM"),
        "B_wa_logit_no_pos0_matched": agg(0, "wa_punct", "wa_attnM", no_pos0),
        "B_ko_logit_no_pos0_matched": agg(0, "ko_punct", "ko_attnM", no_pos0),
        "n_attnM_includes_pos0": sum(r["attnM_includes_pos0"] for r in rows),
    }
    print(f"\n=== A' WA (faithful P&P necessity): {contrast_A_wa}")
    print(f"=== A  KO (read-side variant):       {contrast_A_ko}")
    for k in ("B_wa_logit", "B_wa_margin", "B_ko_logit", "B_ko_margin",
              "B_wa_logit_no_pos0_matched", "B_ko_logit_no_pos0_matched"):
        print(f"=== {k}: {out_c[k]}")

    artifact = {"study": "gpt2_positive_control", "model_dir": MODEL_DIR,
                "weights_sha256": man["weights_sha256"], "dtype": "float32",
                "gpu_name": torch.cuda.get_device_name(0),
                "deterministic_algorithms": True,
                "n_items": n, "n_skipped": skipped, "n_match_flagged": flagged,
                "contrasts": out_c,
                "shortcuts_taken": [], "per_item": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
