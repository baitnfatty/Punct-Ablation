# RED-TEAM REVIEW v3 — C-E (key/value decomposition), studies/NOTE_punct_null.md (Rule 9)

Fresh-session adversarial review, no access to the implementing conversation. Primary
target: **C-E** (the new key-vs-value claim). C-A/C-B/C-C were settled in
`REDTEAM_punct_null.md` and are not re-litigated. Also audited: the v3 edits to C-D,
limits 6–9, and §5.

**Execution profile.** Recomputations ran on host python (my own code over the committed
per_item, not the study's aggregation path). Four live checks executed in rig:2.9.1
(ROCm, `models/gpt2-ruletaker`, cuda/float32, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
deterministic algorithms on): a WA-capture fidelity check, a bug-vs-fix check, a gate
under buggy-vs-fixed code, and a K/V perturbation-norm check. Byte-compares with `cmp`.
No servers started (viewer owns 5002). Container files chowned back; scratch scripts removed.

---

## VERDICT

**C-E: SUPPORTED-WITH-CAVEATS.** The dissociation is real and survives every artifact
attack I ran. In FT-GPT-2, substituting the write-ablated (embedding-only) input into the
**key** projection at punctuation positions collapses the answer (acc 1.000→0.364, margin
−9.70) while the same substitution into the **value** projection preserves it (acc 0.988,
margin −5.21). Both co-primary metrics (accuracy AND margin) agree in direction — this is
NOT the C-C metric reversal. The Δlogit exclusion is principled here, not the C-C pattern
in reverse. The pre-registration of the *direction* is genuinely git-anchored 74 minutes
before the result and before any `ablate_kv` code existed. The WA-derived capture does
what C-E depends on (verified live to max|Δ|=0.0 against an independent full-WA forward),
and the note's claim that a full-depth capture bug was caught by the gate is TRUE (I
reproduced the gate FAIL under the buggy code). Caveats are about overstated wording
("reproducing full-WA"), one gate check that is vacuous w.r.t. the specific bug, and the
single-model scope — none overturn the claim.

**v3 edits to C-D / limits 6–9 / §5: HONEST.** C-D correctly demoted to "descriptive";
the interpretive scaffolding that the prior review flagged was moved into C-E where it is
now backed by the discriminating experiment. No contamination of C-D. Limits 6–9 and the
§5 sink-line are accurate, with two small wording fixes required below.

---

## 1. Recomputed numbers (my code vs the note)

Byte-compare: `runs/studies/kv_warmup.json` == `_b` — **byte-identical**
(sha256 `90bd086e…`, cmp clean).

All five cell accuracies and the margins reproduce EXACTLY from my own aggregation over
`per_item` (n=165, n_skipped=15):

| cell | note acc | my acc | note margin | my margin |
|---|---|---|---|---|
| full_wa (anchor) | 0.273 | **0.2727** | −10.10 | **−10.0982** |
| v_zero | 0.915 | **0.9152** | — | −2.7665 |
| k_zero | 0.527 | **0.5273** | — | −7.5594 |
| v_wa | 0.988 | **0.9879** | −5.21 | **−5.2097** |
| k_wa | 0.364 | **0.3636** | −9.70 | **−9.6996** |

Live spot-check (cuda/float32, one container): item id=0 reproduces the stored per_item
to 3 decimals exactly — full_wa Δlogit +20.861 / Δmargin −10.960 / correct=False; k_wa
+7.274 / −7.834 / correct=True; v_wa +31.078 / −6.050 / correct=True. The instrument
reproduces bit-for-bit on the live stack.

## 2. The load-bearing WA-capture check (the thing C-E depends on)

**Does `_wa_attn_inputs` produce the embedding-only counterfactual?** I ran a full-depth
WA-both forward and captured the attn-module input at a punctuation position with my own
hook, then compared to `intv._wa_attn_inputs`. **max|Δ| = 0.0** at the deepest layer.
The captured input is exactly what a full-WA forward feeds `c_attn`. Separately I confirmed
the residual-stream channel at that position is write-free under WA-both: the *block* input
at P is constant across all 12 layers (norm 4.6555, Δ=0 everywhere) — position P carries
only its embedding through the residual. (The attn-module input rises in norm across depth
only because it is the post-`ln_1` image of that fixed residual; each layer's LayerNorm has
different scale. This is correct, and `_slice_wa` recomputes the K/V slice from that same
post-LN input, so the substitution is in the right space.)

**Was the bug real and was it caught by the gate?** The result commit (3c4f5d9) changed
`_wa_attn_inputs` from `ablate_write(layers,…)` to `ablate_write(range(n_layers),…)`.
- The bug is real: I reimplemented the buggy path. At a deep layer it produces the CLEAN
  residual input (max|Δ| vs clean = 0.0) — i.e. the substitution would be a near no-op,
  NOT the embedding-only counterfactual. The fixed path differs from clean by ‖Δ‖≈7.2 and
  matches the true full-WA input.
- The gate catches it: I monkeypatched the buggy `_wa_attn_inputs` and ran the actual
  `verify_gate`. It **FAILED** on `V<-wa v changed` and `K<-wa k changed` (under the bug
  the "substituted" slice equals the clean slice, so `not torch.equal(...)` is False).
  The note's limit-6 claim "a full-depth WA-capture bug was caught by that gate before any
  C-E result existed" is verified.
- Caveat: the gate check the note foregrounds — `V<-wa == v_proj(WA resid)` — is **vacuous
  w.r.t. this bug** (both sides call `_wa_attn_inputs`, so they agree even when both are
  wrong). It passed under the buggy code. The checks that actually catch the bug are the
  `changed`-vs-independent-clean-baseline checks. The gate is non-vacuous as a whole, but
  limit 6 should not imply the `== v_proj` check is what establishes capture correctness.

## 3. Standing-order attacks — results

**(b) Is the Δlogit exclusion discarding the metric that disagrees (C-C in reverse)?**
No. On K←wa vs V←wa, accuracy AND margin AGREE in direction: K←wa margin −9.70 (near
full-WA −10.10), V←wa −5.21 (about half); K←wa acc 0.364 (near full-WA), V←wa 0.988.
Per-item paired: K←wa is more damaging on margin than V←wa in **156/165 items (94.5%)**;
K←wa wrong while V←wa right in **105 items**, the reverse in **2**, both-wrong in **0**.
The Δlogit genuinely is a shift artifact (V←wa reads +28 while the answer is untouched).
This is the opposite of C-C, where the co-primary margin REVERSED the headline logit. A4
(the shift-invariant co-primary rule, git-anchored pre-result) is satisfied here.

**(c) Key-side control analogous to V←0?** The grid HAS it: K←0 (key zero-drain). K←0 acc
0.527 already hurts more than V←0 0.915. So the key side dominates the value side on BOTH
the zero-drain and the WA-derived source — a coherent 2×2, not a one-cell effect.

**(a)/(d) Is K←wa's damage a norm/magnitude artifact — "any big key perturbation destroys
the forward pass"?** Falsified two ways.
1. Relative perturbation magnitude ‖Δ‖/‖clean‖ (measured live over all layer×punct on
   item 0): KEY mean 1.30, VALUE mean 1.59 — the VALUE perturbation is if anything LARGER,
   yet it preserves the answer. The asymmetry runs opposite to a magnitude confound.
2. K←0 zeroes the key entirely (maximal key perturbation) yet is LESS damaging (acc 0.527)
   than K←wa (0.364), where the key is replaced with a comparable-magnitude but
   mis-addressed embedding-derived key. So it is not "big key perturbation = damage"; a
   specific WRONG address is worse than NO address. This actively supports "mis-addressing,"
   not merely "any perturbation."

## 4. Pre-registration reality for C-E (git-anchored, not merely self-attested)

Timeline (all 2026-07-08): mailbox addenda with the A3 directional prediction **6c9b5ad
21:27** → `ablate_kv` code + runner + the {K,V}×{clean,zero,wa} grid in the prereg
**1e39276 22:35** → **result 3c4f5d9 22:41** → A5 K-port reframe **541cbc5 22:47**.

- At **21:27, 74 minutes before the result and before any `ablate_kv` code existed**, A3
  committed the exact directional mapping: *"key-only ≈ WA-both ≫ value-only → key-re-routing
  wins"* vs *"value-only ≈ WA-both ≫ key-only → corruption-via-reads."* The direction was
  declared, not read off the result.
- The direction is not fragile to the zero→wa refinement: the originally-committed zero
  cells (K←0 0.527 vs V←0 0.915) already show the key side dominating; the WA-derived cells
  (added 22:35, still pre-result) sharpen it (0.364 vs 0.988) but do not invent it.
- The full {K,V}×{clean,zero,wa} grid and the three-way CONTENT/INJECTION/RE-ROUTING → cell
  mapping were git-anchored at 22:35, 6 minutes before the result, in the same commit as
  the code. Thin margin, same working session — but genuinely pre-result and, unusually,
  the *direction* is anchored 74 min out.
- The runner's automated discrimination logic was changed at the result commit from a
  Δlogit rule (which, on the actual numbers, would have printed "value-pathway/injection"
  — the OPPOSITE conclusion) to an accuracy rule. This LOOKS like the p-hack pattern, but
  two pre-result commitments make it legitimate: (i) A3's prediction was always stated on
  the answer-catastrophe axis ("harmful in K←wa" = answer damaged), so the accuracy rule
  matches the prereg and the old logit auto-print did not; (ii) A4 (shift-invariant
  co-primary, pre-result) mandates dropping a raw logit that shifts. And the margin — the
  independent shift-invariant metric — agrees with accuracy. So the metric switch is
  prereg-consistent, not result-chasing. I flag it because on its face it is exactly the
  banned move; it clears only because the direction and the metric-discipline rule both
  predate the data and the margin corroborates.
- A5 ("promote K to primary") is a POST-result reframe of the FORWARD Qwen study; correctly
  labeled "reframe from the warm-up," it does not retroactively touch C-E. No contamination.

## 5. Strongest case against C-E (≥3 vectors)

1. **K←wa lands at chance, not at full-WA's below-chance corruption — "reproducing full-WA"
   overstates.** K←wa acc 0.3636 is exactly the majority-class baseline (60/165) and ~3-way
   chance (0.333); full-WA 0.2727 is BELOW chance (systematically wrong). On margin, K←wa is
   0.40 LESS damaging than full-WA per item. So K←wa reproduces "the model can no longer
   answer" (collapse to chance) but NOT full-WA's below-chance systematic corruption — there
   is a residual ~9% gap and a nonzero V←wa margin hit (−5.21), i.e. a value/joint component
   the "keys, not values" headline glosses. Supported form: keys carry the *bulk* of the
   collapse; values are near-inert but not provably zero. The word "reproducing" should be
   "recovers most of" and the residual named.

2. **Single model, single 1-epoch fine-tune, whole-class, all-layers, one stimulus
   generator.** This is a claim about `models/gpt2-ruletaker`, not about GPT-2, not about
   fine-tuned transformers, and explicitly not about Qwen (the note says so). The
   addressing-vs-content dichotomy is itself a construct of THIS harness's two counterfactuals
   (K/V isolation via the fused-c_attn slice); "keys" here means the c_attn key slice at
   punctuation positions across all 12 layers at once, not a localized circuit. A different
   checkpoint, seed, or per-layer resolution could split differently. C-E does not establish
   a whole-class or cross-model "punctuation routes by keys" fact.

3. **The "mis-addressed / routed through wrong" mechanistic gloss is a step beyond the
   data.** What is measured: replacing the punct key with its embedding-derived key breaks
   the answer while replacing the value does not, and a wrong key is worse than no key.
   That licenses "the causal weight is in the key/addressing pathway." It does NOT directly
   show attention is re-routed to specific wrong positions (no attention-pattern readout is
   in the artifact); "routed through wrong" is an inference. The claim is safe as
   "damage flows through the key pathway," slightly ahead of itself as "mis-addressed."

## 6. What C-E does and does NOT establish

DOES: in this FT-GPT-2 checkpoint, at punctuation positions (all layers, whole class), the
write-ablation catastrophe flows predominantly through the KEY projection, not the VALUE
projection; punctuation values are near-inert for this task; a specifically-wrong key is
more damaging than a zeroed key. Both co-primary metrics agree; the intervention is verified
mechanically exact and byte-deterministic; the direction was pre-registered before the code.

DOES NOT: (a) generalize beyond this checkpoint/task/generator — not a GPT-2, fine-tuned-
transformer, or pretrained-model fact, and explicitly not Qwen; (b) prove values are
*exactly* zero-contract (V←wa margin −5.21 ≠ 0; K←wa reaches chance, not full-WA's
below-chance); (c) demonstrate the attention pattern is re-routed to specific positions
(no pattern readout); (d) resolve per-layer or single-edge structure (all-layers, whole-
class). A verified intervention certifies the arithmetic of each cell, never the mechanistic
interpretation — the note states this.

## 7. Required wording changes

1. **C-E**: "reproducing full-WA (0.273 / −10.10)" → "recovering most of the full-WA
   collapse (K←wa acc 0.364 ≈ chance / majority-class 0.364; full-WA 0.273 is below chance)
   — a residual ~9% accuracy gap and the V←wa margin (−5.21) leave a small value/joint
   component not excluded." Keep the headline (keys carry the bulk); name the residual.
2. **C-E**: "routed through wrong → catastrophic" → "damage flows through the key/addressing
   pathway" (drop the specific "re-routed to wrong positions" implication — no attention-
   pattern readout supports it; the K←0-vs-K←wa contrast supports "mis-addressing" only as
   an inference).
3. **Limit 6**: the gate line implies `V←wa == c_attn projection of the WA residual` is the
   correctness check; it is vacuous w.r.t. the capture bug (both sides use the same
   `_wa_attn_inputs`). State that the bug is caught by the `changed`-vs-clean-baseline
   checks. ("13/13 gate" count is correct.)
4. **Optional (§4 discrimination framing)**: note that the automated `reproduces` rule keys
   on accuracy nearest full-WA vs baseline 1.0; since K←wa sits at chance/majority (0.364),
   the rule's verdict rests on the margin agreement (156/165) more than on accuracy nearness.
   Not a defect — the margin carries it — but worth one sentence so a reader does not read
   0.364≈0.273 as "reproduces" on accuracy alone.

None of these overturn C-E. With edits 1–3 applied, the claim as scoped (FT-GPT-2, key
pathway carries the bulk of the WA catastrophe, values near-inert, pre-registered) is
faithful to the artifacts.

---
*Review executed 2026-07-08. Recomputations: host python (own code) over committed
per_item. Live checks: cuda/float32 GPT-2 in rig:2.9.1 (four container runs, stdout only) —
WA-capture fidelity (max|Δ|=0.0), buggy-vs-fixed capture, gate under buggy-vs-fixed code
(FAIL confirmed under bug), K/V perturbation norms. Qwen not loaded. No servers. This file
is the only repo write.*
