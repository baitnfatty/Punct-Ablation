# NOTE — Punctuation ablation across two regimes: a sensitivity-scoped zero-shot
# null, a gross-effect replication, and what survives an attention-matched control

**Status: v3.2 — C-E's fresh Rule 9 (`REDTEAM_punct_null_v3.md`) returned
SUPPORTED-WITH-CAVEATS; its three required wording fixes (and its optional §4 framing
note) are applied here. C-A/C-B/C-C carry `REDTEAM_punct_null.md` (v2); all ten of its
required wording changes and six defects are resolved. v3.2 adds the pre-registered
exploratory during-generation KO (Limit 1, §3) — exploratory only, outside both reviews'
scope. CITABLE — Matt read both reviews and accepted their caveats as scoped,
2026-09-26.** All numbers
regenerable from committed files (§6).

## 1. Claims (each scoped; nothing stronger is asserted)

**C-A (zero-shot, sensitivity-scoped).** In Qwen3-4B-Instruct-2507 answering
RuleTaker-style theories zero-shot with CoT, knocking out all attention into prompt
punctuation positions (all layers) shows **no punctuation-specific excess damage
detectable at this design's sensitivity** (MDE ≈1.4 logits): paired contrast vs
attention-received-matched non-punctuation positions (n=167) −0.58, 95% CI
[−1.20, +0.04] — covering 0 by 0.04, logit leaning negative (sign p=.02, Wilcoxon
p=.08); co-primary margin flat (+0.22 [−0.34, +0.80]; sign p=.88). Pre-registered
gate verdict KILL. This is a failure to detect at stated sensitivity, **not** an
equivalence finding. The paired means reproduce to ~0.03 on an independent
hardware/software stack at cuda/float32 (A100: −0.61 [−1.23, +0.02]) — same harness,
same frozen fp16-generated CoT baselines re-executed; per-item contrasts differ by up
to ~2.9 logits fp16↔fp32.

**C-B (instrument sensitivity — positive control).** The same harness detects a large
punctuation effect where one exists: in GPT-2-124M fine-tuned on the same task family
(val acc 0.976), write-ablating punctuation positions (all layers) moves the
correct-label logit −22.69 on average and collapses analysis-set accuracy
**1.000 → 0.273** (below 3-way chance; baseline over all 180 items is 0.917). The
KO arm also detects large effects where they exist: KO of attention-matched controls
costs −11.2 logits and drops accuracy to **0.739** — so the KO-based C-A null is not
KO-arm blindness either.

**C-C (what survives the control P&P lacked).** The fine-tuned-GPT-2 gross necessity
effect is **~97% reproduced by attention-matched non-punctuation controls** (implied
accuracy after control-WA 0.279 vs punctuation-WA 0.273; gross margin damage −9.83 vs
−10.10). Punctuation & Predicates (arXiv 2508.14067) reported punctuation necessity
in fine-tuned GPT-2 with no control conditions of any kind; under an
attention-matched control, the bulk of "punctuation is necessary" is reproduced by
equally-attended non-punctuation positions. **Direction:** a small
punctuation-specific residual on the co-primary margin is **indicated, not
excluded** (−0.27 [−0.55, +0.01]; punctuation more damaging in 102/165 items, sign
p=.003, Wilcoxon p≈.045). The +4.76 logit-scale figure is shift-sensitive (sd 16.6)
and carries no directional weight. Note the control set is **content-bearing**
(includes subject and query-attribute tokens), so this contrast bounds
punctuation-specialness against equally-attended content positions; it does not
isolate attention profile per se.

**C-D (the KO/WA asymmetry — descriptive).** In the fine-tuned model, blocking all
READS of punctuation K/V costs little (Δ +3.60 logit; analysis-set accuracy
1.000 → 0.964, margin −1.14) while write-ablating those positions is catastrophic
(−22.69; 0.273). C-D states the asymmetry; **C-E resolves which pathway carries it**
(v2 flagged this as an open gap — the discriminating experiment has now been run).

**C-E (key vs value decomposition — the pathway, pre-registered §A3).** The WA
catastrophe runs through the **KEY/addressing pathway, not value content**. At
punctuation positions (all layers, FT-GPT-2, n=165), substituting each projection's
input with the write-ablated embedding-only residual — the exact WA counterfactual,
one projection at a time — gives: **K←wa** (keys embedding-derived, values intact)
accuracy **0.364**, margin **−9.70** — recovering most of the collapse (full-WA is
0.273 / −10.10; K←wa's 0.364 sits at chance/majority-class, a ~9% gap above full-WA's
below-chance floor, so keys recover most but not all of it); **V←wa** (values
embedding-derived, keys intact) accuracy **0.988** — the answer is preserved (its
margin −5.21 means values are near-inert, not provably zero). The cruder zero-drain
cells corroborate, and K←0 (0.527) is *less* damaging than K←wa (0.364) — a
mis-addressed key hurts more than an absent one. Accuracy and the co-primary margin
**agree** (K←wa more damaging than V←wa in 156/165 items). Note that the automated
`reproduces` rule keys on which cell's accuracy sits nearest full-WA vs the 1.000
baseline; since K←wa lands at chance/majority (0.364) rather than at full-WA's 0.273,
the discrimination verdict rests on the per-item margin agreement (156/165) more than
on accuracy nearness — 0.364 should not be read as "reproduces 0.273" on accuracy
alone. The Δlogit is **excluded**
(V←wa reads +28.06 because embedding-derived values inflate every label logit
~equally — shift-sensitive, answer-irrelevant; the C-C lesson). Reconciles C-D: KO
removes punctuation from attention entirely (its softmax mass redistributes → fine);
K←wa leaves it *in* the attention with corrupted keys — **the damage flows through
the key/addressing pathway** (no attention-pattern readout was taken, so this is a
pathway localization, not a specific-re-routing claim). Punctuation's values are
~contract-empty for this task; its causal weight is in the key pathway. Scope:
FT-GPT-2 only — a GPT-2 mechanism, explicitly not asserted of Qwen (the cross-model
non-transfer this whole note documents). A *causal* localization of the
attention-sink/no-op account to keys.

## 2. What produced these numbers (trust chain, stated exactly)

- Instrument: `gentrace/intervene.py` (UNGATED forward-mutation tool; both
  interventions verified mechanically exact on both architectures — write-zeroing
  residual delta 0.0; knockout attention-into-key 0.0). No GenTrace gate or
  calibration certifies anything here.
- Stimuli: 180 RuleTaker-style theories, computed labels (seed 0); GPT-2 fine-tune
  used seeds 1/2 (n=8001/501 — the prereg said 8000/500; deviation logged as
  amendment A4), asserted disjoint.
- **Pre-registration status, per claim (the honest chain):** C-A's protocol
  (`PREREG_punct_mechanism.md` v2.1) is **git-anchored frozen before its result**
  (commit d45306f 09:40 vs result 0bfb35d 09:45); the kill criterion was mechanical.
  For C-B/C-C (`PREREG_gpt2_control.md`): the interpretation cells are git-anchored
  **5 minutes before** the ablation artifacts (20:42 vs 20:47);
  prereg-before-training and amendment-A1-before-any-WA-number are **self-attested**,
  structurally corroborated (the 20:42 runner is KO-only — no WA code path existed —
  and A1's rationale was already in the mechanism prereg frozen at 09:40). The B_wa
  contrast itself was constructed at A1 time as the mechanical WA mirror of the
  pre-committed KO contrast (same positions, pairing, metrics, bootstrap) and was
  never separately pre-registered.
- **A1's consequence, stated plainly:** under the original (mis-specified) prereg,
  the KO result lands in cell 3 ("replication failure — investigate before any null
  note ships"); A1's correction to the P&P-faithful intervention moved the arc to
  cell 2. The correction is right on the merits (verified against the paper), but it
  was declared after the original target contrast had failed.
- Determinism: byte-identical pairs for (a) the pre-registered **5-item subset** of
  the C-A gate on the local fp16 stack (run after the gate result, before this
  note), (b) the **full-N** fp32 reproduction (both passes in one A100 container),
  (c) the full GPT-2 control run (after amendment A2's deterministic-algorithms fix;
  the first, nondeterministic pair differed at the 1e-4 rounding digit and is
  described from the run log — it was not preserved).

## 3. Numbers table

| Quantity | Value | Artifact |
|---|---|---|
| C-A paired KO contrast, fp16 local | −0.5757 [−1.2005, +0.0394]; margin +0.2173 [−0.3391, +0.7965]; n=167 | `runs/studies/punct_ablation_gate.json` |
| C-A nonparametric re-analysis | logit sign p=.020, Wilcoxon p=.082; margin sign p=.877, Wilcoxon p=.420 | `python studies/analyze_gate_nonparam.py` over the same artifact |
| C-A fp32 A100 reproduction (means) | −0.6053 [−1.2328, +0.0152]; margin +0.2262 | `runs/studies/modal/punct_gate_fp32_report.json` |
| C-A raw KO vs baseline | punct −2.39; attn-matched −1.82 | gate artifact |
| C-A during-generation KO (exploratory) | answer flip rate: punct 0.048 vs attn-matched 0.714; CoT edit dist 0.48 vs 0.84 (n=21) | `runs/studies/during_gen.json` |
| C-B fine-tune | val 0.329 → 0.976 (1 epoch); gate ≥0.90 passed | `runs/studies/gpt2_finetune_manifest.json` |
| C-B WA necessity | −22.69; analysis-set accuracy 1.000 → 0.273 | `runs/studies/gpt2_control.json` |
| C-B KO-arm sensitivity | control-KO −11.2 logit; accuracy → 0.739 (derived from per-item margins) | same |
| C-C paired WA contrast | logit +4.76 [2.27, 7.29] (shift-sensitive, no directional weight); margin −0.27 [−0.55, +0.01], sign p=.003 | same |
| C-C implied control-WA accuracy | 0.279 (vs punct 0.273) | same (derived) |
| C-D KO read-side | +3.60; accuracy 1.000 → 0.964, margin −1.14 | same |
| C-E full-WA anchor | acc 1.000 → 0.273, margin −10.10 | `runs/studies/kv_warmup.json` |
| C-E K←wa (routing) | acc 0.364 (≈ chance / majority-class), margin −9.70 (recovers most of full-WA's collapse; ~9% accuracy gap above its below-chance floor) | same |
| C-E V←wa (content) | acc 0.988, margin −5.21 (answer preserved) | same |
| C-E zero-drain corroboration | V←0 acc 0.915; K←0 acc 0.527 | same |

## 4. Limits

1. **The C-A readout is near a copy task — but the during-generation variant now
   addresses it (exploratory).** The frozen, unablated CoT **explicitly states the
   answer** (e.g. `**Answer: True**`) immediately before the readout scaffold, so C-A
   proper measures prompt-side routing robustness at a readout that can copy the
   stated answer. The pre-registered exploratory complement (knockout active
   THROUGHOUT generation, no frozen CoT) has now been run (n=21): knocking out
   punctuation flips the generated answer **4.8%** of the time and perturbs the CoT
   by edit-distance 0.48, versus **71.4%** flips / 0.84 for the attention-matched
   control. So the null is **not** a frozen-CoT artifact — punctuation is
   non-disruptive even during live reasoning, while equally-attended content
   positions are highly disruptive, consistent with C-A/C-C. Exploratory (no
   confirmatory weight; not in the v3 Rule 9's scope — its own review if it ever
   carries a standalone claim).
2. **Two models, one task family, one stimulus generator.** No claim generalizes
   beyond these. P&P's exact dataset/checkpoints were not used (none released).
3. **Intervention fidelity:** WA-both is the **closest analog in this harness** to
   P&P's zeroing, not a faithful replication — P&P zero the full activation at
   single layers (layer-resolved); WA-both removes all-layer writes while preserving
   embedding+RoPE. Layer-resolved claims are not addressed.
4. **Whole-class resolution.** All contrasts ablate every punctuation position at
   once. Single-edge / per-layer / per-group effects are untested.
5. **Newlines were not in the punctuation class** (tokenizer-strip artifact).
6. **KO and WA each conflate** identity/position/content; C-E separates keys from
   values via the embedding-derived (WA-counterfactual) substitution, but only in
   FT-GPT-2, whole-class, all-layers. Its `ablate_kv` hook is verified (13/13 gate:
   untouched projection bitwise intact per cell; the WA-derived K/V provably *differs*
   from the clean baseline — the check that establishes the capture is the intended
   embedding-only counterfactual, and the one that caught a full-depth capture bug
   before any C-E result; WA+KO ≡ KO-alone as a mask-leak check). Byte-deterministic.
   The C-E Rule 9 (`REDTEAM_punct_null_v3.md`) additionally reproduced `_wa_attn_inputs`
   against an independent full-WA forward to max|Δ|=0.0.
7. **C-E is a single-model mechanism** (FT-GPT-2). "Keys not values" is not asserted
   of Qwen or any pretrained model; the mailbox study tests it on Qwen with both
   ports measured.
8. **Uncertified instrument by design;** trust basis is the pre-registration chain
   as stated in §2, determinism byte-compares, mechanical intervention verification,
   and the fresh Rule 9 of this note.
9. Baseline-correct-only analysis sets condition on model success.

## 5. Relation to prior work (scoped)

- **P&P (2508.14067):** gross necessity replicated via the closest-analog
  intervention (C-B); the bulk of the effect is reproduced by attention-matched
  content-bearing controls (C-C), a control condition their design lacked (they used
  none); a small punctuation-specific margin residual remains indicated. They pose no
  zero-shot transfer question; our C-A speaks only to the KO (read-side) form in one
  zero-shot model.
- **Attention-sink / massive-activation line:** C-E gives a *causal* localization of
  the parked-attention account — in FT-GPT-2 the delimiter's causal weight is in its
  key/addressing, not its (near-inert) value; consistent with near-zero-value sinks.
- **LLM-Microscope (2502.15007):** the correlational "punctuation as context memory"
  claim finds no support at the frozen-CoT readout in the Qwen regime and only a
  small margin residual in the fine-tuned regime; causal testing at other readouts
  remains open.

## 6. Regeneration

- C-A gate: `python studies/run_punct_ablation.py --gate` (baselines:
  `studies/gen_baseline.py`; note: the committed gate artifact predates the
  `--dtype`/`gpu_name` keys — numbers reproduce, bytes differ); fp32:
  `modal run modal/app.py::fp32_recheck`; nonparametrics:
  `python studies/analyze_gate_nonparam.py`.
- C-B/C-C: `python studies/make_ruletaker_punct.py --seed 1 --per-label 2667 --out
  studies/ruletaker_train.json` (val: seed 2 / 167) → `python
  studies/finetune_gpt2.py` → `python studies/run_gpt2_control.py`.
- C-E: `python studies/run_kv_warmup.py` (on the fine-tuned GPT-2 from the C-B step;
  verification gate + cell grid; deterministic).
- shortcuts_taken: [] in every artifact above.
