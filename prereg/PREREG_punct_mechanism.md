# PRE-REGISTRATION — Delimiter mechanism study: parsing boundaries or working memory?

**Status: FROZEN v2.1 (2026-07-08), binding at the commit that lands this version.**
Amendments after this commit require a dated entry in the amendment log. All three
freeze conditions met before any clean-harness (class-resolved) result was seen:
(a) power paragraph from the full-N old-harness pilot (§6), (b) Matt's read of the
red-team + his donor-patching decision (C5-only confirmatory; patching pre-committed
as the §12 escalation — relayed cross-session, Matt-approved), (c) harness fixes
committed (dbb3098) + statically verified 6/6 (runs/studies/VERIFY_harness_fixes.md)
+ verifier findings applied (8690c24). v1 was written while the full-N aggregate
pilot (old harness) was still running; v2 applied the adversarial review's REVISE
list, still before those results were read; v2.1 adds §12.

---

## 0. One-sentence contribution

Punctuation & Predicates (arXiv 2508.14067) showed — by **representation-level
zeroing** of punctuation-token activations (necessity AND sufficiency, layer-resolved,
plus interchange interventions at first-period positions) — that punctuation is
causally load-bearing for RuleTaker classification in models **fine-tuned on the
task** (GPT-2-small full-FT; DeepSeek-1.3B, Gemma-2B LoRA), model-specifically.
Zeroing a token's activations removes its **identity and its content together**, and
P&P never compares attention-knockout (is the position *read*?) against write-ablation
(does its *formed content* matter?), never controls for generic sink structure, and
never resolves the formation pathway (attention-aggregation vs MLP). This study
dissociates exactly that, at fixed tokenization, in a **non-fine-tuned instruct model
with CoT**, under attention-matched and position-matched controls: **either**
delimiter K/V carry write-formed, theory-specific content read downstream (working
memory) **or** their static identity/position suffices (parsing boundary) **or** their
formed content is generic bias (sink account). Each cell is pre-committed in §8.

## 1. Prior work and what each piece owns (rewritten per red-team scoop check)

- **P&P (2508.14067)** — owns the phenomenon AND the representation-level method
  (zeroing, necessity+sufficiency, interchange at first-period, layer-resolved), in
  fine-tuned small models with direct classification readout. Open after P&P: the
  identity-vs-content dissociation (their zeroing conflates both), route-vs-formation
  (no attention analysis), sink-matched controls, and whether any of it holds
  zero-shot with CoT.
- **LLM-Microscope (2502.15007)** — the closest *memory* claim: punctuation/stopword
  hidden states are highly "contextualized" (a trained decoder reconstructs prefixes
  from them) and text-level removal hurts. **Correlational + text-level only** — no
  causal intervention on representations. This study is the causal test its claim
  needs; must be cited.
- **Sink / massive-activation line (2402.17762; ICLR'25 attention-sink emergence;
  2402.09221 dark-signals — periods behave BOS-like; 2510.06477; survey 2604.10098)**
  — owns the null: delimiter positions attract structural attention with near-zero,
  largely **input-invariant** value content. Consequence (red-team attack 1): "content
  formation matters" (WA≈KO) does NOT by itself imply *theory* content — it can be
  generic bias formation. §7's C5 exists to arbitrate this.
- **Filler/pause line (2404.15758 "dot by dot"; 2510.01032 meaningless-token gains)**
  — added content-free tokens help via extra compute / activation redistribution.
  Adjacent-sounding, different object (inserted filler vs the prompt's own
  delimiters); named here to preempt conflation.
- **Deductive-circuit work (2605.27824)** — head-level causal mediation for logical
  reasoning on Qwen3-family models; red-team verified full-text: **zero** analysis of
  punctuation/delimiter/boundary positions. Orthogonal; cite. (Crowding risk is
  time-shaped: that group works causal mediation on this task family.)
- **Delimiters-as-engineering (2604.10135 think-in-sentences; 2511.08128 gist tokens
  anchored at sentence-final punctuation)** — inserting/training delimiter anchors
  helps; neither shows a vanilla model's native delimiters carry aggregated state.
- **This repo's own history** (syllogism arc): every structural-position signal so far
  resolved to a surface feature under controls. The deflationary outcome is the
  base-rate expectation and is treated as first-class — but see §8's honesty note on
  its asymmetric value.

## 2. Instrument, certification status, and WHAT IS ACTUALLY MEASURED

- `gentrace/intervene.py` (UNGATED, uncertified by design — mutating the forward
  breaks Claim B). Uses gated `capture_core.load_model`/`Adapter` read-only
  (`attn_implementation="eager"` is pinned by `load_model` — load-bearing for the
  knockout mask path). Any claim from this study is Rule-9'd from scratch.
- Model: `Qwen/Qwen3-4B-Instruct-2507`, cuda/fp16, greedy, deterministic. **Scope
  limit, pre-declared:** single model ⇒ claims scoped "in Qwen3-4B-Instruct";
  cross-model confirmation is the Modal phase.
- **Measured quantity (frozen-CoT scope — red-team D3, named honestly):** the CoT is
  generated once, UNABLATED, and frozen; interventions on prompt positions are applied
  while teacher-forcing that same clean CoT; Δ is read at the fixed `"\n\nAnswer:"`
  readout. Every confirmatory number is therefore **prompt-side routing robustness at
  a readout downstream of an intact, answer-bearing CoT** — NOT "reasoning is
  disrupted." All §8 sentences carry this scope. (Exploratory complement,
  pre-registered: during-generation re-run of KO on a 30-item subset; metric = answer
  flip rate + CoT edit distance; no confirmatory weight.)
- Readout metric: Δ correct-label logit (` True`/` False`/` Unknown`), **co-primary
  with Δ margin** (correct − max other) — margin is shift-invariant; pre-registered
  check: sign agreement of the two on every confirmatory contrast (disagreement =
  reported, contrast counted NOT passed).

## 3. Stimuli

`studies/ruletaker_punct.json` (committed generator, seed 0): 180 items, T/F/U
60/60/60, hops 1–3 (79/63/38 — unbalanced; any hops claim is exploratory and must
name this), computed labels (forward-chaining, self-checking), distractors.
**Correction (red-team D6):** every rule carries exactly one **if-then boundary
comma** ("If someone is X**,** then they are Y."); conjunction is expressed via
" and " and contributes no comma (`n_commas == n_rules + 2` for 180/180 — the +2 is
the instruction tail). The generator docstring will be corrected; stimuli themselves
are unchanged and correct.

**Analysis set:** items the model answers correctly at baseline (pre-registered
exclusion; conditions on the easy-item subpopulation — named as a limitation). The
realized n is reported; expected ≈145.

## 4. Position classes and controls

All classes are restricted to the **user-content token span** (chat-template wrapper
excluded). **Required harness fixes before any confirmatory run (red-team D1 — the
punct set itself was verified clean on the pilot; the CONTROL POOLS were not):**
1. Restrict punct AND every control candidate pool to the user-content span (the
   current `nonpunct`/`random` pools include template positions, incl. the position-0
   `<|im_start|>` sink — control contamination, biases contrasts unpredictably).
2. Pre-registered per-item assertion: punct token count `== n_periods + n_commas +
   n_question` from the stimulus record (halt on mismatch, never skip silently).
3. `attn_received` becomes a **per-eligible-query mean** (causal-mask-aware; the
   current all-query mean inflates early positions).
4. `_mask_keys` must **raise** on a missing/non-float attention mask (currently a
   silent no-op — the banned silent-failure mode).
5. `Intervenor._label_ids` (space-less, dead code) removed or fixed.
6. Aggregate stats use **sample** sd (harness currently uses population sd).

Punctuation classes (per item):
- **P_theory** — sentence-final periods of facts/rules, excluding the final one.
- **P_last** — the final theory period (theory→query boundary).
- **P_qmark** — the `?` ending the query.
- **P_comma** — the if-then boundary comma of each rule (ALL items; antecedent/
  consequent boundary — redefined per D6).
- **P_instr** — punctuation of the fixed tail `Answer True, False, or Unknown.`
  (2 commas + final period). Same bias-formation opportunity, **no theory content** —
  the in-design arbiter for the bias confound; carries confirmatory weight via C5.

Control sets (seeded RNG, per item):
- **C_attnM** — attention-received-matched non-punct positions (per-eligible-query
  metric), WITH a pre-registered **matching-quality report and caliper**: a match is
  valid if its attention is within [0.5×, 2×] of the target's; items where <80% of
  punct positions have valid matches are flagged; C1/C2 are reported on all items AND
  on the well-matched subset (declared fallback — divergence is reported, not hidden).
- **C_posM** — position-matched **function-word** tokens (nearest token whose decoded
  form is in a declared closed list: is/are/then/if/and/or/a/the/they/someone).
  Role-restricted per red-team D5.3 — the nearest *content* word would subtract
  semantic damage, not recency. If no function word exists within ±3 positions, the
  position is dropped from C3 and counted in the matching report.
- **C_content** — random non-punct, non-function-word content positions,
  count-matched (secondary/continuity).

## 5. Interventions (the mechanism triplet)

For position set S, layers L:
1. **KO** — `knockout_attn(L, S)`: no query reads S's K/V. Removes identity +
   position + content. NOTE (red-team D2): masking also **redistributes** S's softmax
   mass; C_attnM is the control for this, valid only where matching succeeds (§4).
2. **WA-both** — `ablate_write(L, S, 'both')`: S's residual stops accumulating
   writes. Verified by code chain: after all-layer WA-both, downstream attention
   reads exactly **token embedding + RoPE position** — identity survives, content
   does not.
3. **WA-attn / WA-mlp** — formation pathway. Pre-declared caveat: the two channels
   interact (ablating attn-writes changes what the MLP sees); the 0.7 share heuristic
   in C4 is a labeled heuristic over interacting quantities, not a clean decomposition.

Confirmatory WA/KO comparisons are made at the **all-layers** setting (both total);
layer-banded versions (4 bands × 9 layers) are exploratory.

## 6. Statistics (repaired per red-team D4)

- **Confirmatory family (Holm-corrected, α=0.05, two-sided): {C1, C2, C3a, C3b, C5}.**
  Wilcoxon signed-rank on paired per-item contrasts + seeded bootstrap 95% CI
  (B=10,000, seed 0).
- **C4 is an ESTIMATE, not a test** (no p-value, outside Holm): group-level ratio
  **R = mean(Δ WA-both) / mean(Δ KO)** (ratio of means, both on all-punct, all-layers),
  seeded-bootstrap CI. Regime read off the CI: CI ⊂ (−∞, 0.25] ⇒ identity regime;
  CI ⊂ [0.5, 1] with WA-attn/WA-both ≥ 0.7 (same estimator form) ⇒ aggregation
  regime; CI straddling ⇒ mixed. **R < 0 or R > 1 (reachable: pilot has
  positive deltas): no regime is assigned; reported as anomalous with the raw
  distribution shown.** Thresholds 0.25/0.5/0.7 are conventions fixed before data,
  not principled constants — stated as such.
- **Single-position KO sub-study** for per-position class comparability (P_qmark has
  |S|=1; P_theory ≈ 8–15): KO of one position at a time. Redundancy caveat
  pre-declared (single-position under-counts if delimiters are mutually redundant;
  whole-class carries joint necessity; both reported).
- **Power (written 2026-07-08 from the full-N old-harness pilot, per the declared
  dependency; that run's contrasts remain uninterpreted per §10):** n=167 baseline-
  correct items; paired sample sd of the pooled punct−attnM contrast = **4.57** (KO)
  / **4.51** (WA-both) in correct-logit units. At the Holm family floor (α=0.01
  two-sided, 5 tests) and power 0.90, the minimum detectable paired |mean| is
  **≈1.37 logits** (KO) / **≈1.35** (WA). Conservativeness: this sd comes from the
  CONTAMINATED old harness (template/sink positions in control pools produced
  outlier control deltas up to +7), so it is plausibly an over-estimate of the
  clean-harness sd; the realized clean sd is reported by G1 and the confirmatory
  run. Scope: sd was measured on the Δcorrect-logit metric only (the old harness
  did not record margins); Δmargin sd is assumed comparable and is checked at G1.
  Per-position sub-studies (C3/C5, |S|=1 per item) have smaller effective variance
  ratios and no pilot sd — they are powered by the same n but declared
  exploratory-strength if their realized CIs are wide.
- **Runtime budget (N-is-sacred):** ≈75–90 forwards/item × ~145 items ≈ 11–13k
  forwards ≈ 4–7 h on the 6900 XT (fp16, eager). A shortfall halts and reports;
  never silently subsampled.

## 7. Confirmatory hypotheses

- **C1 (route-through):** KO(all-punct) − KO(C_attnM) < 0.
- **C2 (content formation):** WA-both(all-punct) − WA-both(C_attnM) < 0.
- **C3 (aggregation-not-boundary):** per-position KO at {P_qmark, P_last} exceeds
  per-position KO at P_theory — **C3a**; AND the {P_qmark, P_last} per-position KO
  exceeds its C_posM counterpart — **C3b**. Both must individually pass at the
  Holm-adjusted level (declared conjunction rule).
- **C5 (theory-content, the bias arbiter — promoted per red-team fix 2):**
  per-position WA-both at P_theory ∪ P_last exceeds per-position WA-both at P_instr.
  Working-memory predicts ≫ (theory delimiters carry item-specific content;
  instruction punctuation forms the same generic bias without theory content); the
  sink/bias account predicts ≈. Without C5, WA≈KO is UNINTERPRETABLE as memory.
- **C4 (identity-vs-content ratio):** estimate per §6; feeds the §8 table but never
  a significance claim.

## 8. Pre-committed interpretation table (every sentence scoped to §2's frozen-CoT quantity)

| C1 | C3 | C5 | C4 regime | Licensed sentence |
|----|----|----|-----------|-------------------|
| sig | sig | sig | aggregation | "Delimiter K/V carry write-formed, theory-specific content, formed chiefly by attention, preferentially at aggregating positions, and read downstream (at a readout downstream of an intact CoT)" — (label per §12 claim ladder: "working memory" PENDING donor-patching; until then this row is claimed only as C5-grade formation-dependence) |
| sig | any | **ns** | aggregation | "Delimiter content matters but is not theory-specific — consistent with write-formed generic bias (sink account)" — deflation of the memory reading despite WA≈KO |
| sig | sig | — | identity | "Late delimiters are preferentially read as identity/position anchors; their formed content is causally inert" — routing-not-memory |
| sig | ns | — | identity | "Delimiters are uniformly-read boundary markers (parsing); P&P's effect is segmentation" — clean deflation |
| ns | — | — | — | "No delimiter-specific routing beyond generic sink structure in this model at this sensitivity" — null note, scoped |

Honesty notes (per red-team): (a) v1's "sentence-gist per period" row asserted what
the content encodes; nothing in this design decodes content — removed. Content
identification (probing / donor-patching / interchange à la P&P) is follow-up work.
(b) The outcome space is **asymmetric**: rows 2–5 largely confirm the sink-literature
prior at mid-sequence delimiters (value: first causal formation-side test of that
prior on a reasoning task in a non-fine-tuned model — a negative note, not a
symmetric co-equal result). The study is worth running for the instrument and the
row-1/row-2 discrimination, not for outcome symmetry.

## 9. Validity, determinism, attrition

- Baseline-correct-only inclusion (§3); no other exclusions permitted.
- Determinism (Tier 2): 5-item subset run twice, artifacts byte-identical, before the
  confirmatory run is trusted.
- fp16 robustness: same subset, KO + WA-both on cpu/fp32; check = sign agreement of
  paired contrasts.
- Per-item punct-count assertion (§4.2) active in every run; failure halts.
- `shortcuts_taken` in every artifact; wall-clock to stdout only.

## 10. Gating (repaired per red-team fix 6)

The currently-running full-N aggregate pilot (old harness: contaminated control
pools, pooled punct incl. P_instr/P_qmark) is **demoted to power estimation only** —
its contrast is NOT the proceed/kill gate (a gate must not run on an instrument this
prereg itself indicts).

**Gate G1 (cheap, on the FIXED harness):** re-run C1's whole-class contrast only
(KO all-punct vs C_attnM, all-layers, ~300 forwards, ≈15 min) on the full analysis
set. If the paired 95% CI covers 0 → the C1-shaped effect does not exist under clean
controls; report the null, stop; any redesign is a new pre-registration.
If G1 passes → run the full confirmatory grid.

## 11. Out of scope (named so they aren't smuggled in later)

Steering / project_out; text-level punctuation edits; content probing/decoding;
cross-model runs; certified-trace integration; CoT-position interventions beyond
the declared exploratory flip-rate subset. (**Donor-patching is NOT out of scope**
— it is the pre-committed conditional escalation, §12; it is simply not part of
the confirmatory grid.)

## 12. Conditional escalation — donor-patching protocol (pre-committed, not yet built)

**Trigger:** C1 AND C2 pass (Holm-corrected) and the C4/C5 pattern lands in row-1
or row-2 territory of §8. If the trigger fires, this protocol is the pre-registered
follow-up. It is NOT part of the confirmatory grid; its harness does not yet exist
(new hook + verify_intervene.py extension required before execution).

**Why pre-committed:** C5's position confound (instruction-tail punctuation has a
fixed extreme position and no within-prompt downstream consumers; "theory WA >
instr WA" can reflect position/consumer asymmetry rather than content specificity).
C5 is an arbiter of record, not a decisive one; this section fixes the escalation
and its interpretation before any grid data is seen.

**Design (fixed now; implementation details amendable with dated entries):**
- **Donor:** the same punctuation token type (".", ",", "?") embedded at a
  depth-matched position in a length-matched DISTRACTOR-ONLY theory (same
  generator, no derivable chain to any query) — NOT a lone token (a solitary
  period at position 0 forms BOS-like sink structure and is not "context-free" in
  the relevant sense).
- **Operation:** patch the donor token's residual trajectory into the target
  position layer-by-layer (resid at every layer boundary), reusing the existing
  layer-hook style. Target-position K/V then form through the model's own
  projections with the TARGET's RoPE position — sidestepping the pre/post-RoPE
  re-rotation trap entirely.
- **Verification precondition (before any result is read):** extend
  verify_intervene.py — (a) patched position's resid equals donor values at every
  layer boundary; (b) a self-patch (donor = the position's own trajectory) is a
  numerical no-op on the readout.
- **Hypotheses (pre-committed):**
  - PATCH ≈ baseline while WA-both < 0 → WA's damage was generic/bias structure
    the donor also supplies → sink account wins; the row-1 label is DENIED.
  - PATCH ≈ WA-both < 0 → the removed component is context-specific →
    working-memory label EARNED.
  - Intermediate → partial specificity; report ratio + seeded bootstrap CI (same
    estimator form as C4); no label upgrade.
- **Scope:** inherits §2's frozen-CoT scope, same analysis set, same paired
  statistics.

**Claim ladder (binding):** until this protocol runs and lands in its second cell,
row 1's licensed sentence remains at C5-grade — "write-formed, theory-specific
formation-dependence at delimiter positions" — and the phrase "working memory" is
NOT used in any artifact or summary.

## 13. shortcuts_taken (at pre-registration time)

- Single model, single dtype (fp16) confirmatory run — declared scope limit.
- Layer bands, not per-layer sweep, for the exploratory depth map — runtime budget.
- The bias arbiter in the confirmatory grid is C5 (contrastive), not donor-patching
  (surgical) — patching is the pre-committed §12 escalation, not built now:
  the donor definition is a real degrees-of-freedom surface (a lone "." forms its
  own sink; a donor trajectory carries positional signature) and rushing it into a
  frozen design risks freezing a subtle bug; G1 may kill everything in 15 min; and
  if effects land, the grid's class/layer structure tells us where to aim the
  patch, so the escalation is better-designed after the grid.
- Harness fixes §4.1–4.6 are specified here but not yet applied (the old harness is
  mid-run on the GPU); they land as one commit BEFORE G1, with the fixed harness
  diffed against this section.

---
*Revision log: v2 2026-07-08 — applied REDTEAM_punct_prereg.md REVISE list (P&P
recharacterized from primary source; LLM-Microscope + filler line + circuits cited;
C5 promoted as bias arbiter; C4 made an estimate with defined R<0/R>1 behavior; Holm
family = actual tests; C3 conjunction rule declared; frozen-CoT scope named
everywhere; control pools/matching/caliper/assertions specified; pilot demoted to
power-estimation, G1 gate added; P_comma redefined; margin co-primary; runtime
budget declared). v2.1 2026-07-08 — donor-patching escalation protocol added as §12
(pre-committed trigger, donor design, verification precondition, hypotheses, and the
binding claim ladder: "working memory" is unusable until patching's second cell),
per Matt's approved decision relayed from the drafting session; §11 pointer and §8
row-1 qualifier updated; shortcuts note records why the hook is not built now.
FROZEN at this commit — all three freeze conditions met (see status header).*
*Amendment log (post-freeze): (none).*
