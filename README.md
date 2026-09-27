# punct-ablation

Pre-registered punctuation-ablation study under attention-matched controls, with
its two adversarial reviews and every artifact the numbers come from.

- **`PAPER.md`** — short-paper draft (start here).
- **`studies/NOTE_punct_null.md`** — the research note (v3.2, citable). Claims C-A
  through C-E, the trust chain, the numbers table, limits, and regeneration commands.
- **`reviews/`** — the two fresh-context adversarial reviews (Rule 9) that the note
  passed, verbatim. Both returned SUPPORTED-WITH-CAVEATS; every required edit is
  applied in the note.
- **`prereg/`** — the pre-registrations (zero-shot mechanism study, frozen and
  git-anchored before its result; GPT-2 positive control).
- **`runs/studies/`** — committed artifacts: the zero-shot gate, its baselines, the
  A100 fp32 re-check report, the GPT-2 fine-tune manifest, control run, key/value
  decomposition, and the exploratory during-generation run.
- **`studies/*.py`** — the harness: stimulus generator, baseline generation, ablation
  gate, GPT-2 fine-tune, control run, key/value warm-up, during-generation knockout,
  nonparametric re-analysis.
- **`gentrace/`** — the minimal subset of the GenTrace instrument these scripts need
  (`intervene.py` and the read-only `capture_core.load_model` / `Adapter` it uses).
  The GenTrace selfcheck gate is **not** part of this package; nothing here is
  gate-certified, as the note states.

## Headline

In Qwen3-4B-Instruct-2507 reasoning zero-shot with CoT, knocking out all attention
into prompt punctuation shows no punctuation-specific excess damage detectable at
MDE ≈ 1.4 logits versus attention-matched non-punctuation positions (−0.58, 95% CI
[−1.20, +0.04], n=167; reproduces on A100 fp32). The same harness detects the large
effect the literature reports in fine-tuned GPT-2 (accuracy 1.000 → 0.273), but
~97% of it is reproduced by attention-matched controls, and a key/value
decomposition localizes the collapse to the key projection (keys ← WA: 0.364;
values ← WA: 0.988). Scopes and caveats are in `PAPER.md` §4 and the note's §4.

## Regenerating

Layout mirrors the source repository so the note's §6 commands run unchanged from
this directory (scripts resolve `runs/…` and `studies/…` relative to the repo root):

```bash
python studies/analyze_gate_nonparam.py            # recompute C-A p-values from the committed gate artifact (no GPU)
python studies/gen_baseline.py                     # regenerate zero-shot CoT baselines (Qwen3-4B-Instruct-2507)
python studies/run_punct_ablation.py --gate        # C-A gate
python studies/make_ruletaker_punct.py --seed 1 --per-label 2667 --out studies/ruletaker_train.json
python studies/finetune_gpt2.py                    # GPT-2 positive-control checkpoint (models/, gitignored)
python studies/run_gpt2_control.py                 # C-B / C-C / C-D
python studies/run_kv_warmup.py                    # C-E
python studies/run_during_gen.py                   # exploratory during-generation KO
```

Not included: the fine-tuned GPT-2 checkpoint (regenerable, ~480 MB) and the Modal
pipeline that executed the A100 fp32 re-check (its report is committed under
`runs/studies/modal/`). Requires `torch` and `transformers`; the original runs used
torch 2.9.1 with eager attention.

## License

Apache-2.0 (see `LICENSE`).
