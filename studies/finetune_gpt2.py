"""Fine-tune GPT-2 (124M) on RuleTaker-style classification — the pre-registered
positive control (studies/PREREG_gpt2_control.md). Replicates P&P's regime: small
model fine-tuned to emit the label directly (no CoT).

Format: "{prompt} Answer:" -> single label token (" True"/" False"/" Unknown");
loss on the label token only. AdamW lr 5e-5, batch 16, <=5 epochs, early stop when
val accuracy >= 0.90 (the pre-registered proceed gate). fp32, cuda, seed 0.

Writes: models/gpt2-ruletaker/ (weights — NOT committed) and
runs/studies/gpt2_finetune_manifest.json (config, seeds, per-epoch val accuracy,
weight sha256 — committed). Wall-clock to stderr only.
"""

import hashlib
import json
import os
import random
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openai-community/gpt2"
OUT_DIR = "models/gpt2-ruletaker"
MANIFEST = "runs/studies/gpt2_finetune_manifest.json"
LABELS = ("True", "False", "Unknown")
LR, BATCH, MAX_EPOCHS, GATE = 5e-5, 16, 5, 0.90
SEED = 0


def encode_all(tok, items, lab_ids):
    """Tokenize '{prompt} Answer:' + label token; loss on the label only."""
    out = []
    for it in items:
        ids = tok(it["prompt"] + " Answer:").input_ids
        lab = lab_ids[it["label"]]
        out.append((ids + [lab], len(ids)))          # (sequence, label position)
    return out


def batches(data, bs, rng=None):
    idx = list(range(len(data)))
    if rng:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [data[j] for j in idx[i:i + bs]]


def collate(batch, pad_id, device):
    L = max(len(s) for s, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    for r, (seq, lpos) in enumerate(batch):
        ids[r, :len(seq)] = torch.tensor(seq)
        att[r, :len(seq)] = 1
        lab[r, lpos] = seq[lpos]                     # loss on the label token only
    return ids.to(device), att.to(device), lab.to(device)


@torch.no_grad()
def val_accuracy(model, tok, val, lab_ids, device):
    inv = {v: k for k, v in lab_ids.items()}
    ok = 0
    model.eval()
    for it in val:
        ids = tok(it["prompt"] + " Answer:", return_tensors="pt").input_ids.to(device)
        logits = model(ids).logits[0, -1, :]
        pred = max(lab_ids, key=lambda w: float(logits[lab_ids[w]]))
        ok += (pred == it["label"])
    return ok / len(val)


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    lab_ids = {}
    for w in LABELS:
        enc = tok(" " + w).input_ids
        if len(enc) != 1:                            # pre-registered assert
            raise SystemExit(f"label ' {w}' is {len(enc)} GPT-2 tokens; expected 1 — halt")
        lab_ids[w] = enc[0]
    print(f"[ft] label ids: { {w: i for w, i in lab_ids.items()} }", flush=True)

    train = json.load(open("studies/ruletaker_train.json"))["prompts"]
    val = json.load(open("studies/ruletaker_val.json"))["prompts"]
    study = {x["prompt"] for x in json.load(open("studies/ruletaker_punct.json"))["prompts"]}
    assert not ({x["prompt"] for x in train} & study), "train overlaps study set — halt"
    assert not ({x["prompt"] for x in val} & study), "val overlaps study set — halt"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    data = encode_all(tok, train, lab_ids)
    rng = random.Random(SEED)

    base_acc = val_accuracy(model, tok, val, lab_ids, device)
    print(f"[ft] pre-finetune val acc = {base_acc:.3f} (chance ~0.33)", flush=True)
    history = [{"epoch": 0, "val_acc": round(base_acc, 4)}]
    t0 = time.time()
    best = base_acc
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        tot, nb = 0.0, 0
        for b in batches(data, BATCH, rng):
            ids, att, lab = collate(b, tok.pad_token_id, device)
            loss = model(ids, attention_mask=att, labels=lab).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        acc = val_accuracy(model, tok, val, lab_ids, device)
        best = max(best, acc)
        print(f"[ft] epoch {ep}: mean loss {tot/nb:.4f}  val acc {acc:.3f}  "
              f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)
        history.append({"epoch": ep, "mean_loss": round(tot / nb, 4), "val_acc": round(acc, 4)})
        if acc >= GATE:
            print(f"[ft] proceed gate {GATE} reached — stopping early", flush=True)
            break

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    wfile = os.path.join(OUT_DIR, "model.safetensors")
    sha = hashlib.sha256(open(wfile, "rb").read()).hexdigest() if os.path.exists(wfile) else None
    manifest = {
        "purpose": "pre-registered positive control (PREREG_gpt2_control.md)",
        "base_model": MODEL_ID, "dtype": "float32", "seed": SEED,
        "train": {"file": "studies/ruletaker_train.json", "seed": 1, "n": len(train)},
        "val": {"file": "studies/ruletaker_val.json", "seed": 2, "n": len(val)},
        "label_ids": lab_ids, "lr": LR, "batch": BATCH, "max_epochs": MAX_EPOCHS,
        "proceed_gate": GATE, "history": history,
        "final_val_acc": history[-1]["val_acc"],
        "gate_passed": bool(history[-1]["val_acc"] >= GATE),
        "weights_sha256": sha, "out_dir": OUT_DIR,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "shortcuts_taken": [],
    }
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[ft] saved {OUT_DIR} (sha {str(sha)[:16]}…) -> {MANIFEST}", flush=True)
    print(f"[ft] GATE {'PASSED' if manifest['gate_passed'] else 'FAILED'} "
          f"(final val acc {manifest['final_val_acc']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
