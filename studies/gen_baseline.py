"""Baseline CoT + forced-answer readout for the RuleTaker items; caches each full
sequence for the teacher-forced ablation harness.

Per item: generate the chain-of-thought (greedy), append a fixed `Answer:` scaffold,
and read the label logits at that ONE fixed readout position — no fragile hunting for
the answer token inside the CoT. Records the full input_ids (prompt + CoT + scaffold),
the readout position, the 3 label token ids, the baseline label logits, the argmax,
and correctness. The ablation harness later teacher-forces this exact sequence with an
intervention on the PROMPT punctuation positions and re-reads the same position → Δ.

Slow: ~40 s/item on the 6900XT (generation), ~2 h for the full 180. Deterministic
(greedy). Run a small --n first to validate the readout, then the full set.

  python -u studies/gen_baseline.py --n 5          # validate
  python -u studies/gen_baseline.py                 # full 180 (~2h)
"""

import argparse
import json
import os

import torch

from gentrace.intervene import Intervenor, LABELS

SCAFFOLD = "\n\nAnswer:"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=0, help="0 = all")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--stimuli", default="studies/ruletaker_punct.json")
    ap.add_argument("--out", default="runs/studies/punct_baseline.json")
    args = ap.parse_args()

    intv = Intervenor(args.model, dtype="float16", device="cuda")
    # after "Answer:" the label token carries a leading space
    lab_ids = {w: intv.tok(" " + w, add_special_tokens=False).input_ids[0] for w in LABELS}
    scaffold_ids = intv.tok(SCAFFOLD, add_special_tokens=False).input_ids

    stim = json.load(open(args.stimuli))["prompts"]
    if args.n:
        stim = stim[:args.n]

    rows, ok = [], 0
    for item in stim:
        enc = intv.encode(item["prompt"])
        plen = int(enc["input_ids"].shape[1])
        with torch.no_grad():
            gen = intv.model.generate(**enc, max_new_tokens=args.max_new,
                                      do_sample=False, pad_token_id=intv.tok.eos_token_id)
        seq = gen[0].tolist() + scaffold_ids            # prompt + CoT + scaffold
        with torch.no_grad():
            logits = intv.model(torch.tensor([seq], device=intv.device)).logits[0, -1, :]
        labl = {w: float(logits[i]) for w, i in lab_ids.items()}
        pred = max(labl, key=labl.get)
        hit = (pred == item["label"])
        ok += hit
        rows.append({
            "id": item["id"], "label": item["label"], "hops": item["hops"],
            "prompt_len": plen, "readout_pos": len(seq) - 1, "seq_len": len(seq),
            "baseline_logits": {k: round(v, 4) for k, v in labl.items()},
            "pred": pred, "correct": bool(hit), "seq": seq,
        })
        print(f"  id={item['id']:>3} {item['label']:7}->{pred:7} {'ok' if hit else 'X '} "
              f"gen={len(seq) - plen - len(scaffold_ids):>3} "
              f"logits={ {k: round(v, 1) for k, v in labl.items()} }", flush=True)

    acc = ok / len(rows) if rows else 0.0
    print(f"\nforced-scaffold readout accuracy: {ok}/{len(rows)} = {acc:.2f}", flush=True)
    artifact = {
        "study": "punct_baseline", "model": args.model, "scaffold": SCAFFOLD,
        "label_ids": lab_ids, "n": len(rows),
        "readout_accuracy": round(acc, 4), "max_new": args.max_new,
        "shortcuts_taken": [], "per_item": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(artifact, f)
    print(f"  -> {args.out}  ({len(rows)} items cached)", flush=True)


if __name__ == "__main__":
    main()
