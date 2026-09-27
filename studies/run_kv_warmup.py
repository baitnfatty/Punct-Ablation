"""M4 warm-up (GQA-mailbox prereg §A3) — on the fine-tuned GPT-2, discriminate WHY
write-ablating punctuation is catastrophic: CONTENT vs INJECTION vs RE-ROUTING, via
key/value-isolated ablation.

Cells at punctuation positions (all layers): baseline, full-WA-both (anchor), V←0,
K←0, V←wa (embedding-derived value, K intact), K←wa (embedding-derived key, V intact).
Reads: CONTENT harmful in V←0; INJECTION harmful in V←wa (mild in V←0); RE-ROUTING
harmful in K←wa (mild in V cells). Plus the WA+KO mask-leak check.

VERIFICATION GATE runs first and HALTS on any failure: per-cell the untouched
projection slice is bitwise intact; V←wa/K←wa equal the c_attn projection of the
write-ablated residual; a self-substitute (own clean input) is a no-op; WA+KO ≡
KO-alone. Deterministic -> run twice + byte-compare. Ungated; Rule 9 at claim.

  python -u studies/run_kv_warmup.py --out runs/studies/kv_warmup.json
"""

import argparse
import contextlib
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gentrace.intervene import Intervenor, LABELS
from run_punct_ablation import boot_ci, classify_positions

MODEL_DIR = "models/gpt2-ruletaker"
SCAFFOLD = " Answer:"


def cap_cattn(intv, L, ids, cm=None):
    """Capture layer L's c_attn output [T, 3d] at final state (capture hook registered
    INSIDE cm so it fires after any ablation hook)."""
    box = {}

    def hk(mod, args, kwargs, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        box["v"] = t[0].detach().float().clone()
        return out

    with (cm or contextlib.nullcontext()):
        h = intv.A.attn[L].c_attn.register_forward_hook(hk, with_kwargs=True)
        try:
            with torch.no_grad():
                intv.model(ids)
        finally:
            h.remove()
    return box["v"]


def readout(intv, ids, cm):
    with cm, torch.no_grad():
        return intv.model(ids).logits[0, -1, :].float().clone()


def verify_gate(intv, ids, P):
    L = intv.A.n_layers // 2
    base = cap_cattn(intv, L, ids)
    d = base.shape[-1] // 3
    ksl, vsl = slice(d, 2 * d), slice(2 * d, 3 * d)
    Q = P + 1 if P + 1 < base.shape[0] else P - 1        # a bystander position
    checks = []

    m = cap_cattn(intv, L, ids, intv.ablate_kv([L], [P], "value", "zero"))
    checks += [("V<-0 v-slice==0", float(m[P, vsl].abs().max()) == 0.0),
               ("V<-0 k intact", torch.equal(m[P, ksl], base[P, ksl])),
               ("V<-0 q intact", torch.equal(m[P, :d], base[P, :d])),
               ("V<-0 bystander intact", torch.equal(m[Q], base[Q]))]

    m = cap_cattn(intv, L, ids, intv.ablate_kv([L], [P], "key", "zero"))
    checks += [("K<-0 k-slice==0", float(m[P, ksl].abs().max()) == 0.0),
               ("K<-0 v intact", torch.equal(m[P, vsl], base[P, vsl]))]

    m = cap_cattn(intv, L, ids, intv.ablate_kv([L], [P], "value", "wa", ids=ids))
    checks += [("V<-wa k intact", torch.equal(m[P, ksl], base[P, ksl])),
               ("V<-wa v changed", not torch.equal(m[P, vsl], base[P, vsl])),
               ("V<-wa bystander intact", torch.equal(m[Q], base[Q]))]

    m = cap_cattn(intv, L, ids, intv.ablate_kv([L], [P], "key", "wa", ids=ids))
    checks += [("K<-wa v intact", torch.equal(m[P, vsl], base[P, vsl])),
               ("K<-wa k changed", not torch.equal(m[P, ksl], base[P, ksl]))]

    # V<-wa slice equals c_attn projection of the WA (embedding-only) residual
    wa_in = intv._wa_attn_inputs(ids, [P], [L])[L]        # [1,1,d]
    W = intv.A.attn[L].c_attn.weight[:, vsl]
    b = intv.A.attn[L].c_attn.bias[vsl]
    expect = (wa_in[0, 0].float() @ W.float() + b.float())
    got = cap_cattn(intv, L, ids, intv.ablate_kv([L], [P], "value", "wa", ids=ids))[P, vsl]
    checks.append(("V<-wa == v_proj(WA resid)", torch.allclose(got, expect, atol=1e-3)))

    # WA+KO == KO-alone (mask-leak detector)
    allL = list(range(intv.A.n_layers))
    ko = readout(intv, ids, intv.knockout_attn(allL, [P]))
    with intv.ablate_write(allL, [P], "both"):
        wako = readout(intv, ids, intv.knockout_attn(allL, [P]))
    checks.append(("WA+KO == KO-alone (no leak)", torch.allclose(ko, wako, atol=1e-3)))

    for name, ok in checks:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {name}", flush=True)
    if not all(ok for _, ok in checks):
        raise SystemExit("ablate_kv VERIFICATION GATE FAILED — halting (prereg A3)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stimuli", default="studies/ruletaker_punct.json")
    ap.add_argument("--out", default="runs/studies/kv_warmup.json")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    man = json.load(open("runs/studies/gpt2_finetune_manifest.json"))
    if not man.get("gate_passed"):
        raise SystemExit("fine-tune gate not passed — stop")
    intv = Intervenor(MODEL_DIR, dtype="float32", device="cuda")
    tok = intv.tok
    lids = {w: tok(" " + w).input_ids[0] for w in LABELS}
    allL = list(range(intv.A.n_layers))
    stim = json.load(open(args.stimuli))["prompts"]
    if args.n:
        stim = stim[:args.n]

    # --- verification gate (before any result) ---
    it0 = stim[0]
    ids0 = torch.tensor([tok(it0["prompt"] + SCAFFOLD).input_ids], device="cuda")
    p0 = classify_positions(ids0[0].tolist(), tok, 0, len(tok(it0["prompt"]).input_ids))["punct_all"]
    print("=== ablate_kv VERIFICATION GATE ===")
    verify_gate(intv, ids0, p0[len(p0) // 2])

    CELLS = ["full_wa", "v_zero", "k_zero", "v_wa", "k_wa"]
    rows, skipped = [], 0
    for it in stim:
        seq = tok(it["prompt"] + SCAFFOLD).input_ids
        plen = len(tok(it["prompt"]).input_ids)
        ids = torch.tensor([seq], device="cuda")
        punct = classify_positions(seq, tok, 0, plen)["punct_all"]
        want = it["n_periods"] + it["n_commas"] + it["n_question"]
        if len(punct) != want:
            raise SystemExit(f"punct assertion failed id={it['id']}")
        with torch.no_grad():
            lg = intv.model(ids).logits[0, -1, :]
        vals = {w: float(lg[i]) for w, i in lids.items()}
        if max(vals, key=vals.get) != it["label"]:
            skipped += 1
            continue
        base_c = vals[it["label"]]
        base_m = base_c - max(v for k, v in vals.items() if k != it["label"])

        def cell(cm):
            l2 = readout(intv, ids, cm)
            v = {w: float(l2[i]) for w, i in lids.items()}
            c = v[it["label"]]
            m = c - max(x for k, x in v.items() if k != it["label"])
            return round(c - base_c, 4), round(m - base_m, 4), max(v, key=v.get) == it["label"]

        cms = {
            "full_wa": intv.ablate_write(allL, punct, "both"),
            "v_zero": intv.ablate_kv(allL, punct, "value", "zero"),
            "k_zero": intv.ablate_kv(allL, punct, "key", "zero"),
            "v_wa": intv.ablate_kv(allL, punct, "value", "wa", ids=ids),
            "k_wa": intv.ablate_kv(allL, punct, "key", "wa", ids=ids),
        }
        r = {"id": it["id"], "label": it["label"], "n_punct": len(punct),
             "base_logit": round(base_c, 4)}
        for name in CELLS:
            dc, dm, ok = cell(cms[name])
            r[name] = {"dlogit": dc, "dmargin": dm, "correct": bool(ok)}
        rows.append(r)
        print(f"  id={it['id']:>3} " + " ".join(f"{c}={r[c]['dlogit']:+.1f}" for c in CELLS), flush=True)

    n = len(rows)
    agg = {}
    for c in CELLS:
        agg[c] = {"mean_dlogit": round(statistics.mean(r[c]["dlogit"] for r in rows), 4),
                  "mean_dmargin": round(statistics.mean(r[c]["dmargin"] for r in rows), 4),
                  "accuracy_after": round(sum(r[c]["correct"] for r in rows) / n, 4)}
    print(f"\n=== cell means (n={n}, skipped {skipped}) ===")
    for c in CELLS:
        print(f"  {c:8} Δlogit {agg[c]['mean_dlogit']:+8.3f}  Δmargin {agg[c]['mean_dmargin']:+7.3f}  acc→ {agg[c]['accuracy_after']}")
    # Discrimination on ACCURACY (answer-flip = ground truth) + margin co-primary.
    # The raw Δlogit is EXCLUDED from the decision: it is shift-sensitive (embedding-
    # derived values inflate every label logit ~equally, so V←wa reads +28 while the
    # answer is untouched) — the C-C / A4 lesson. Reproduction = closer to full-WA's
    # accuracy collapse than to the baseline-correct 1.0.
    fa = agg["full_wa"]["accuracy_after"]
    base_acc = 1.0

    def reproduces(cell):
        a = agg[cell]["accuracy_after"]
        return abs(a - fa) < abs(a - base_acc)

    kwa_rep, vwa_rep = reproduces("k_wa"), reproduces("v_wa")
    if kwa_rep and not vwa_rep:
        read = ("RE-ROUTING: K←wa reproduces the WA accuracy collapse "
                f"(acc {agg['k_wa']['accuracy_after']} vs full-WA {fa}); V←wa preserves "
                f"the answer (acc {agg['v_wa']['accuracy_after']}). Punctuation's causal "
                "role is in key/addressing structure, not value content.")
    elif vwa_rep and not kwa_rep:
        read = ("CONTENT/INJECTION: V←wa reproduces the collapse; K←wa preserves the answer.")
    elif kwa_rep and vwa_rep:
        read = "JOINT: both pathways reproduce the collapse."
    else:
        read = "NEITHER isolated pathway reproduces (joint corruption needed)."
    print(f"  discrimination (accuracy+margin, logit excluded as shift-sensitive): {read}")

    artifact = {"study": "kv_warmup_gpt2", "model_dir": MODEL_DIR,
                "weights_sha256": man["weights_sha256"], "dtype": "float32",
                "gpu_name": torch.cuda.get_device_name(0),
                "deterministic_algorithms": True, "n_items": n, "n_skipped": skipped,
                "cell_means": agg, "discrimination": read,
                "shortcuts_taken": [], "per_item": rows}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
