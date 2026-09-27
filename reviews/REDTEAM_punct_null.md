# RED-TEAM REVIEW — studies/NOTE_punct_null.md (Rule 9)

Fresh-session adversarial review with no access to the implementing conversation.
Scoped inputs: the note, the two pre-registrations, the named artifacts, the named
code, and arXiv 2508.14067 (fetched). STATUS.md and prior REDTEAM_*.md were not read.

**Execution profile of this review:** all recomputations ran on host CPU (own code,
not the study's aggregation path); one live spot-check executed on cuda/float32
(ROCm, rig:2.9.1 container, models/gpt2-ruletaker); the Qwen model was never loaded
(Qwen recomputes done from stored artifacts). Byte-compares run with `cmp`.

---

## Verdicts

| Claim | Verdict |
|---|---|
| **C-A** (Qwen zero-shot KO null) | **SUPPORTED-WITH-CAVEATS** — numbers exact, prereg genuinely frozen pre-result; wording must be sensitivity-scoped and the frozen-CoT limit understates that the CoT text literally states the answer |
| **C-B** (GPT-2 positive control) | **SUPPORTED-WITH-CAVEATS** — numbers exact, P&P regime characterization verified; prereg has no git-anchored precedence over training, and the accuracy pair mixes denominators |
| **C-C** (specificity fails attention-matched control) | **SUPPORTED-WITH-CAVEATS, required rewording** — the deflationary core holds (gross effect ~97% reproduced by controls); the headline direction clause "damages the model MORE" is **NOT-SUPPORTED**: the co-primary margin points the opposite way (punct more damaging in 102/165 items, sign p=.003, Wilcoxon p≈.045) |
| **C-D** (corruption-injection reading) | **SUPPORTED-WITH-CAVEATS** — honestly labeled interpretive; the discriminating experiment (WA+KO rescue, or key-only vs value-only ablation) was never run; "harmless" and the KO accuracy presentation overstate |

---

## 1. Recomputed numbers (my code vs the note's)

All from committed artifacts; every headline reproduces exactly.

| Quantity | Note | My recompute | Source |
|---|---|---|---|
| C-A paired KO logit contrast | −0.5757 [−1.2005, +0.0394], n=167 | −0.5757, sd 4.0502, [−1.2005, +0.0394] (seed-0 boot); indep. seed 12345: [−1.1861, +0.0333] | `runs/studies/punct_ablation_gate.json` per_item |
| C-A margin contrast | +0.2173 [−0.3391, +0.7965] | +0.2173 [−0.3391, +0.7965] | same |
| C-A raw KO means | punct −2.39 / attnM −1.82 | −2.3930 / −1.8172 | same |
| C-A "robust re-analysis" row | logit sign p=.02, Wilcoxon p=.08; margin sign p=.88, W p=.42 | sign p=.0200, W p=.0817; margin sign p=.8771, W p=.4201 | recomputed from the gate artifact — the out-of-repo citation is unnecessary |
| C-A fp32 vs fp16 | "agreement to ~0.03 logits" | means agree to 0.0296; **per-item contrasts differ up to 2.875 logits (mean |Δ| 0.117)** | gate + `modal/punct_gate_fp32_a.json` |
| C-B A′ WA necessity | −22.69; acc 0.917→0.273 | −22.6857; acc-after 0.2727 **on the 165-item analysis set (baseline = 1.000 there; 0.9167 is over all 180)** | `runs/studies/gpt2_control.json` |
| C-B fine-tune | 0.329→0.976, gate ≥0.90 | matches manifest; weights sha matches artifact | `gpt2_finetune_manifest.json` |
| C-C B_wa logit | +4.76 [2.27, 7.29] | +4.7644, sd 16.5572, [2.2739, 7.2942] | gpt2_control per_item |
| C-C B_wa margin | −0.27 [−0.55, +0.01] | −0.2691 [−0.5516, +0.0142]; **sign test 102/165 punct-worse, p=.003; Wilcoxon p≈.045** | same |
| C-D KO read-side | +3.60; acc 0.964 | +3.6014; 0.9636 on the analysis set (i.e. 1.000→0.964, a drop, not the 0.917→0.964 rise A1's phrasing implies) | same |

New numbers the artifact does not report but per-item margins imply (base_margin + Δmargin > 0):
- **accuracy after attnM-WA ≈ 0.279** (vs punct-WA 0.273 — the controls collapse the model equally);
- **accuracy after attnM-KO ≈ 0.739** (vs punct-KO 0.964 — the KO arm demonstrably detects large effects, which is the missing piece of the "not instrument blindness" argument for the KO-based C-A).
- Gross WA margin damage: punct −10.098 vs attnM −9.829 (punct-specific residual −0.269 ≈ 2.7% of gross). Gross WA logit: punct −22.686 vs attnM −27.450.

Live spot-check (cuda/float32, rig:2.9.1, one container run, stdout only): item id=0
reproduces to 4 decimals — base_logit −73.1902, wa_punct [20.8610, −10.9595],
wa_attnM [−12.8685, −9.5497], ko_punct [−0.9270, −2.0839]. The matched-control
positions for this item decode to ` If, they, are, big, is, big, Mon, Mon, Is, or,
sad, sad` — i.e. **the subject token ('Mon'×2) and the query attribute ('sad'×2) are
in the control set**. See attack 1.

Stimuli: `make_ruletaker_punct.py --seed 0` regenerates `studies/ruletaker_punct.json`
byte-identically (verified). Train/val/study exact-prompt disjointness independently
verified (0/0/0 overlaps, n=8001/501/180). n_commas == n_rules+2 holds 180/180.

## 2. Determinism audit

- `gpt2_control.json` vs `_b`: **byte-identical** (sha256 a424fcb6…, cmp clean).
- `modal/punct_gate_fp32_a.json` vs `_b`: **byte-identical** (f9407c25…), n=167 full-N,
  both passes inside one container as documented.
- `punct_det_a.json` vs `_b`: **byte-identical** (f2170d8d…) — but this pair is the
  pre-registered **5-item subset (n=4, 12 forwards)**, not the full gate; it was run
  at 20:09, ~10 h AFTER the gate artifact (09:44) and its result commit (09:45).
  Prereg §9's letter ("before the confirmatory run is trusted") is satisfied since the
  note postdates it, but the note's phrase "byte-identical artifact pairs for the C-A
  gate (local fp16)" reads as a full-gate byte-compare that does not exist on disk.
  Full-N byte-compare exists only on the CUDA stack.
- A2 account check: the note's description (first GPT-2 pair differing at the 1e-4
  digit, ROCm nondeterminism, `use_deterministic_algorithms` fix) matches amendment
  A2 verbatim. The nondeterministic first pair is not preserved on disk, so the
  "1e-4, headlines identical" characterization is narrative-only (low stakes: the
  final deterministic pair reproduces every number I checked).
- Regeneration gap: the committed gate artifact predates the `--dtype`/`gpu_name`
  runner extension (commit 759cab1); `run_punct_ablation.py --gate` today writes two
  extra keys, so the committed artifact is not byte-regenerable by the current code
  (numbers would match; bytes would not). `runs/studies/punct_baseline.json` — a
  required input — is untracked.

## 3. Amendment-timeline audit (PREREG_gpt2_control.md)

Git facts (all 2026-07-08): mechanism prereg frozen d45306f **09:40** → gate result
0bfb35d **09:45**. GPT-2 prereg + fine-tune manifest + KO-only runner committed
1870243 **20:42** (fine-tune result already in the commit message) → amendments
A1–A3 + WA runner code + both result artifacts committed d5b06b7 **20:47** → note
9dd5f33 **20:48**.

- **C-A's prereg is genuinely frozen before its result** (09:40 vs 09:45), and the
  G1 KILL criterion was mechanical (CI covers 0). This freeze is the strongest link
  in the note's trust chain.
- **C-B/C-C's prereg has no git-anchored precedence over training**: "declared
  BEFORE any training" is attested only by the file's own narrative — the prereg
  first entered git in the same commit as the training result. What IS git-anchored:
  the interpretation cells (including cell 2's exact wording) precede the ablation
  artifacts by 5 minutes (20:42 vs 20:47).
- **A1 is structurally corroborated but consequential.** Corroborated: the 20:42
  runner is KO-only (no WA code path existed — verified by diff), so no WA number
  could exist when A1 was declared; and A1's rationale (P&P necessity = activation
  zeroing, whose analog is WA, not KO) was already in the mechanism prereg **frozen
  at 09:40**, eleven hours before any GPT-2 work — A1 realigned the GPT-2 prereg
  with an earlier frozen characterization, and the paper confirms it (§1 below).
  Consequential: under the original prereg (A = KO), the observed KO null lands in
  cell 3 — "replication failure; investigate before any null note ships." A1 moved
  the arc to cell 2 — "positive control passes (gross)." The amendment that decided
  whether the note could ship at all was declared after the original target contrast
  had already failed. Legitimate on the merits; the note must not describe C-B/C-C
  as simply "frozen before data."
- **B_wa (C-C's contrast) was never pre-registered.** The prereg's B is KO-based;
  A1 re-targets the cells to A′ but never names a WA-paired B. B_wa first exists in
  the 20:47 commit, alongside its results. It is the mechanical WA mirror of the
  pre-committed KO contrast (same positions, pairing, metrics, bootstrap — thin DoF
  surface), but its pre-registration status is "constructed at A1 time," and the
  note's §2 ("Protocols frozen before data … for C-B/C-C") overstates it. Limit 5
  concedes only the direction cell, not the contrast itself.
- **A3** is honest and moot as reported: `n_attnM_includes_pos0 = 0`; the with/without
  split is degenerate (identical numbers).

## 4. P&P characterization check (arXiv 2508.14067, fetched)

Verified TRUE: necessity intervention is token-activation **zeroing** ("a token
activation z is zeroed out, and all other activations are left untouched"),
sufficiency is the complement; models are **fine-tuned** (GPT-2 full FT; DeepSeek +
Gemma LoRA) on RuleTaker to 93–96%; **no control conditions of any kind** (no random
non-punct, no attention-matched — the note if anything understates this); **no
attention-knockout experiments**. GPT-2 necessity+sufficiency holds in 5/12 layers
(periods early 0–4, "?" late 7–11).

Discrepancies:
1. The paper poses **no explicit zero-shot transfer question** (limitations mention
   only "a wider variety of models"). The note's §5 "their zero-shot transfer
   question answered in the negative" attributes to P&P a question they did not ask —
   and the C-A KO null cannot answer the zeroing/WA version of that question anyway
   (WA was never run on Qwen; the gate killed first).
2. "Faithful P&P replication target" overstates: P&P zero the **full activation at
   single layers** (layer-resolved); WA-both zeroes **all-layer writes** while
   preserving embedding+RoPE. Closest-analog-in-this-harness, not faithful.

## 5. Strongest case against this note

1. **(C-C) The headline direction is contradicted by the note's own co-primary
   metric, and the control is not content-neutral.** On margin — the shift-invariant
   metric limit 5 itself calls "the honest headline" — punct WA is MORE damaging
   than control WA in 102/165 items (sign p=.003, Wilcoxon p≈.045, mean −0.27,
   bootstrap CI [−0.55, +0.01] covering 0 by 0.014). The +4.76 logit direction the
   note leads with is shift-sensitive (WA moves the whole logit vector; sd 16.6 vs
   margin sd 1.9). Meanwhile the matched controls include the subject and
   query-attribute tokens (' Mon', ' sad' — verified live), so "controls hurt more
   on the logit" is overdetermined by their semantic content, not evidence about
   punctuation. A P&P author could read this same artifact as: *a punctuation-specific
   margin effect survives an aggressive content-bearing control at p≈.003–.045* —
   the opposite of the note's headline. What actually survives attack: the GROSS
   effect is ~97% reproduced by attention-matched controls (accuracy 0.279 vs 0.273;
   margin −9.83 vs −10.10), which does deflate P&P's uncontrolled necessity claim.
2. **(C-A) The readout is close to a copy task.** The frozen CoT ends with the
   answer written out — item 0's context ends `**Answer: True**. ✅<|im_end|>` two
   tokens before the readout scaffold (verified by decoding the committed baseline).
   Prompt-side interventions are being tested against a readout that can copy the
   stated answer from the immediately preceding text. The null may be a property of
   the readout design, not of punctuation routing; the during-generation variant
   that could detect reasoning disruption was pre-registered exploratory and never
   run. Limit 1 names the frozen-CoT scope but not that the answer string itself is
   in context — a materially stronger hollowing.
3. **(C-B/C-C) The pre-registration for the arc's most novel claim is narrative plus
   a five-minute anchor.** The entire GPT-2 chain — prereg, training, KO run, A1,
   WA code, two determinism pairs, results, amendments — entered git in two commits
   five minutes apart, after all data existed. The only git-anchored pre-commitments
   are the interpretation cells (20:42) and the frozen mechanism prereg's P&P
   characterization (09:40). Everything else, including "declared before training"
   and "A1 before any WA number," is self-attestation.
4. **(C-D) Routing and content were never separated.** WA-both changes punct
   positions' keys AND values; KO removes both reads and mass. "Reading corrupted
   values injects damage" is one consistent story; "embedding-like keys re-route
   attention globally" is another. A value-only ablation, a WA+KO rescue (if
   injection-via-reads is the mechanism, blocking reads on top of WA should
   restore accuracy toward 0.96), or the pre-committed donor-patching would
   discriminate. None was run. The note correctly labels C-D interpretive; the
   labels "harmless" (actually 1.000→0.964, margin −1.14) and "corrupting"
   (the operation removes writes; "corruption" presupposes the mechanism) lean
   past the data.

## 6. Defects found (beyond wording)

- D1 — **Mixed accuracy denominators**: 0.917 is over all 180 items; 0.273 and 0.964
  are over the 165-item analysis set (where baseline = 1.000 by construction).
  "0.917 → 0.273" is not a like-for-like pair, and A1's "0.917→0.964" makes KO look
  beneficial when it is a 1.000→0.964 drop.
- D2 — **Accuracy after control-WA/KO absent from the artifact aggregates** (the
  numbers that make C-C's magnitude case decisive — 0.279 / 0.739 — must be derived
  from per-item margins; matched-control positions themselves are not recorded).
- D3 — Note §3 row 8 cites "MEMprobe-side session record" — an out-of-repo,
  non-regenerable pointer, violating the every-number-links-to-an-artifact rule.
  (The numbers are correct; I reproduced all four exactly from the gate artifact.)
- D4 — `punct_baseline.json` (required pipeline input) untracked; committed gate
  artifact not byte-regenerable by the current runner (post-hoc `--dtype`/`gpu_name`
  keys). Verification-gate outputs (write-zero/knockout exactness) live in stdout
  only, not persisted in any artifact.
- D5 — Prereg said train n=8000 / val n=500; actuals are 8001/501 (3×2667, 3×167).
  Trivial, but an undocumented deviation from a document whose trust basis is
  self-attestation.
- D6 — "Robust re-analysis" (note §3): technical usage, but adjacent to the banned
  celebration sense; rename "nonparametric re-analysis." No other banned-vocabulary
  hits.

## 7. Required wording changes for the note to be honest

1. **C-A**: replace "does not damage the answer readout more than" with a
   sensitivity-scoped statement: "shows no punctuation-specific excess damage
   detectable at this design's sensitivity (MDE ≈1.4 logits; paired mean −0.58,
   95% CI [−1.20, +0.04], covering 0 by 0.04; logit leans negative, sign p=.02;
   co-primary margin flat)." A CI-covers-0 KILL is not an equivalence finding.
2. **C-A**: "agreement to ~0.03 logits" → "agreement of the paired means to ~0.03
   (per-item contrasts differ by up to ~2.9 logits fp16↔fp32)"; "independent stack"
   → "same harness and same frozen fp16-generated CoT baselines, re-executed on an
   independent hardware/software stack."
3. **Limit 1** must state that the frozen CoT text explicitly contains the stated
   answer immediately before the readout scaffold.
4. **C-C**: delete the "damages the model MORE" clause. Supported form: "the gross
   necessity effect is ~97% reproduced by attention-matched controls (implied
   accuracy 0.279 vs 0.273; margin −9.83 vs −9.83−0.27); a small punctuation-specific
   residual on the co-primary margin (−0.27 [−0.55, +0.01]; punct more damaging in
   102/165 items, sign p=.003) is indicated, not excluded. The +4.76 logit figure is
   shift-sensitive and carries no directional weight." Also state the matched
   controls are content-bearing tokens, so the contrast bounds punctuation-specialness
   against equally-attended content positions rather than isolating attention profile.
5. **§2**: replace "Protocols frozen before data … PREREG_gpt2_control.md for
   C-B/C-C" with the accurate chain: cells git-anchored 5 min before ablation
   results; prereg-before-training and A1-before-WA are self-attested (structurally
   corroborated by the KO-only runner at 20:42); B_wa constructed at A1 time, never
   pre-registered.
6. **C-B/C-D**: put all accuracy pairs on one denominator (1.000→0.273 WA,
   1.000→0.964 KO on the analysis set; 0.917 baseline over 180) and add the derived
   control accuracies (0.279, 0.739). Cite KO-attnM damage (−11.2 logit, acc 0.739)
   as the KO-arm sensitivity evidence — C-B's WA-based sensitivity does not by
   itself cover the KO-based C-A null.
7. "Faithful P&P analog" → "closest analog in this harness" with the two named
   differences (all-layer writes vs per-layer full-state zeroing; embedding+RoPE
   preserved).
8. **§5**: P&P line — do not attribute a zero-shot transfer question to them; and
   the negative answer covers only the KO (read-side) form, since WA was never run
   on Qwen. LLM-Microscope line — scope "does not survive causal testing" to the
   frozen-CoT readout in the Qwen regime.
9. **§2 determinism**: name the local fp16 pair as the pre-registered 5-item subset
   (n=4), run after the gate result but before the note; full-N byte-compare is
   CUDA-only.
10. **§3 row 8**: replace the MEMprobe-side pointer with the in-repo recompute
    (one script over `punct_ablation_gate.json` reproduces all four p-values).

## 8. Instrument limits (what this evidence can and cannot establish)

`gentrace/intervene.py` is an ungated forward-mutation tool; no GenTrace gate or
calibration certifies anything here, and the note says so. The mechanical checks
(write-zero Δ=0, attention-into-key=0) certify the interventions do what they claim
at the tensor level — they do not certify the causal reading of any contrast. KO
removes identity+position+content AND redistributes softmax mass; WA-both removes
formed content but leaves embedding+RoPE readable; neither isolates keys from
values, so route-vs-content is only partially dissociated (C-D's gap). All
interventions are whole-class, all-layers; P&P's layer-resolved claims are not
addressed at that resolution. Every Qwen number is measured at one teacher-forced
readout position downstream of an intact CoT that states the answer; every GPT-2
number is on a 1-epoch fine-tune of one 124M model on one stimulus generator.
Pre-registration here is a trust mechanism, not a validity mechanism: for C-A it is
git-anchored; for C-B/C-C it is substantially self-attested. A valid intervention
run certifies the arithmetic of the contrast, never the interpretation.

---
*Review executed 2026-07-08. Recomputations: host python (own code). Live check:
cuda/float32 GPT-2 in rig:2.9.1 (one container, stdout only). Qwen model not loaded.
No servers started. No repo artifacts modified; this file is the only write.*
