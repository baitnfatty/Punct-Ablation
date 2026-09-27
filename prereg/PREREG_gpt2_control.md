# PRE-REGISTERED POSITIVE CONTROL — fine-tuned GPT-2 punctuation sensitivity

**Status: declared 2026-07-08, BEFORE any training or ablation run.** Supports the G1
null note (PREREG_punct_mechanism.md, frozen; its §10 stop is honored — this is not a
redesign of that study but the instrument-sensitivity demonstration its null needs).
Scope: replicate the REGIME of Punctuation & Predicates (2508.14067) — a small model
FINE-TUNED on RuleTaker-style classification, where they found punctuation necessary &
sufficient (GPT-2-small) — and run OUR fixed harness (attention-matched control
included, which P&P lacked) on it.

## Question

Does the G1 instrument detect a punctuation effect in the regime where the literature
says one exists? Secondary: does P&P's punctuation effect survive an
attention-matched control?

## Training protocol (fixed before data)

- Base: `openai-community/gpt2` (124M), fp32, seed 0, deliberate HF pull.
- Data: our generator (`make_ruletaker_punct.py` machinery), **train seed 1, n=8000;
  val seed 2, n=500**; balanced T/F/U; **asserted disjoint** from the 180 study items
  (seed 0). Format: `"{prompt} Answer: {label}"`; loss on the label token only
  (classification-style, P&P-like; no CoT).
- Labels ` True` / ` False` / ` Unknown` must each be a single GPT-2 token (assert;
  halt if not).
- AdamW lr 5e-5, batch 16, ≤5 epochs, early stop on val accuracy.
- **Proceed gate: val accuracy ≥ 0.90** (P&P reached 93–96%). Below 0.90 after the
  budget → report and stop (ablating a model that cannot do the task is
  uninterpretable — the G1 lesson).
- Model artifact is NOT committed (gitignored); the training manifest (config, seeds,
  val accuracy, weight sha256) is.

## Ablation protocol (fixed before data)

- Stimuli: the SAME 180 study items (never trained on; train/val seeds differ) —
  apples-to-apples with the Qwen G1 run. Analysis set = items FT-GPT-2 answers
  correctly at the readout (same rule as G1).
- Readout: teacher-force `"{prompt} Answer:"`, label logits at the final position.
  Co-primary Δ correct-logit and Δ margin (same as G1).
- Contrasts, same fixed machinery (clean pools, per-eligible-query attention matching,
  caliper, seeded bootstrap B=10k):
  - **A (P&P-shaped necessity):** KO(all-punct) vs baseline — raw damage + accuracy
    drop. This is the replication target.
  - **B (the controlled question):** KO(all-punct) − KO(attention-matched), paired —
    the G1 contrast, on the fine-tuned regime.
- GPT-2 adapter notes: `capture_core.Adapter` already supports the GPT-2 architecture
  (`core.h` branch, read-only use). The knockout mask path MUST be re-verified on
  GPT-2 before any result is read (verify_intervene checks: write-zeroing exact;
  attention-into-key exactly zero; the raise-on-missing-mask guard stays — if GPT-2's
  eager path does not materialize a float mask, the run halts rather than silently
  no-ops).

## Pre-committed interpretation cells

| A (vs baseline) | B (vs attn-matched) | Reading |
|---|---|---|
| large | CI < 0 | Instrument detects a punctuation-SPECIFIC effect in the FT regime → positive control PASSES; the Qwen G1 null is a regime difference, and the null note says so with teeth. |
| large | CI covers 0 | P&P-shaped effect reproduces but is fully explained by attention profile → their punctuation claim reduces to sink structure under the proper control (secondary finding). Positive control passes only for GROSS sensitivity; the null note reports this honestly. |
| small | — | FT-GPT-2 insensitive to punctuation ablation at this N → replication failure; report both halves, investigate before any null note ships. |

- Determinism: ablation artifact byte-compare, two runs (same §9 style).
- shortcuts_taken in every artifact; no exclusions beyond baseline-correct.
- The words "positive control passes" appear in the null note ONLY for cell 1 (or,
  qualified as gross-sensitivity-only, cell 2).

## AMENDMENTS (dated; logged before the data they govern)

**2026-07-08 A1 (declared after the first KO run was read, before ANY WA number
exists):** The original §"Ablation protocol" mis-specified contrast A as
KO-vs-baseline and labeled it "P&P-shaped". P&P's actual necessity intervention is
**activation ZEROING at punctuation positions** (their §4.1), whose analog in this
harness is **WA-both (residual-write ablation, all layers)** — KO (attention
knockout) removes READING of the position, a different mechanism they never tested.
Amendment: contrast **A′ = WA-both(all-punct) vs baseline** (+ accuracy drop) is
added as the faithful P&P replication target; the interpretation cells apply to A′.
The already-observed KO results are reported alongside, honestly labeled as the
read-side variant (first KO pass: KO-punct did NOT damage the model — Δ +3.6,
accuracy 0.917→0.964 — and B_KO was strongly positive, i.e. KO of attention-matched
controls hurt far more; note the GPT-2 caveat below).

**2026-07-08 A2 (determinism):** first run pair differed at the 1e-4 rounding digit
on some per-item values (headlines and all matched positions identical) —
nondeterministic fp reductions in GPT-2 ROCm kernels. Fix:
`torch.use_deterministic_algorithms(True)` in the runner; byte-compare re-run
required. No result from the nondeterministic pair carries weight beyond the A1
labeling above.

**2026-07-08 A4 (trivial n deviation, logged per REDTEAM_punct_null.md D5):** the
protocol said train n=8000 / val n=500; the generator's per-label balancing produced
8001 (3×2667) / 501 (3×167). No other deviation.

**2026-07-08 A3 (GPT-2 sink note, descriptive):** GPT-2 prepends no BOS; its
attention sink is the FIRST CONTENT TOKEN (position 0), which sits inside the
non-punct candidate pool by construction. The attention-matched control can
therefore select the position-0 sink, making B partially a "punct vs sink"
contrast. The runner now records whether position 0 is in each item's matched set;
B is reported with and without pos-0-matched items.
