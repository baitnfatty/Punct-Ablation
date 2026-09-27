#!/usr/bin/env python
"""RuleTaker-style stimuli for the punctuation-ablation study — deterministic, byte-reproducible.

Why this set (see the plan): Punctuation & Predicates (2508.14067) used RuleTaker; we
need stimuli whose *prompt* punctuation is compute-bearing — periods delimit each fact
and rule, every rule carries one if-then boundary comma ("If someone is X, then they
are Y."; conjunction is expressed via " and " and adds NO comma — n_commas ==
n_rules + 2, the +2 being the instruction tail), and a question mark closes the query —
so that ablating a punctuation position could plausibly disrupt parsing the theory.
Labels are COMPUTED by forward-chaining (not asserted), the construction is multi-hop
(1–3) so reasoning is required, and True/False/Unknown are balanced.

Theory language:
  fact  : "{Subject} is {attr}."
  rule  : "If someone is {a}, then they are {b}."   (+ conjunctive "…{a} and {b}…" via " and ", no comma)
  neg   : "If someone is {a}, then they are not {b}."   (used only to force a False label)
  query : "Is {Subject} {attr}? Answer True, False, or Unknown."

Label semantics (open-world): forward-chain positive premises; query is True if the
positive literal is derivable, False if the negative literal is, Unknown otherwise.
Distractor facts/rules pad the theory (more punctuation, harder parse); every item's
label is re-derived by the closure and must equal the intended target or it is dropped
(self-checking). Seeded RNG (seed 0) → byte-reproducible; sorted; shortcuts_taken: [].
"""

import argparse
import json
import os
import random

NAMES = ["Harry", "Anne", "Bob", "Charlie", "Dave", "Erin", "Fiona", "Gary",
         "Ivy", "Jack", "Kara", "Liam", "Mona", "Nate", "Opal", "Pete"]
ATTRS = ["red", "big", "cold", "round", "kind", "young", "smart", "quiet",
         "rough", "nice", "green", "soft", "tall", "heavy", "fast", "clean",
         "sad", "warm", "strong", "loud"]


def closure(subjects, facts, rules):
    """Forward-chain to the fixpoint. facts: set of (subj, attr, bool). rules:
    list of (premise_attrs, concl_attr, concl_bool) — all premises positive."""
    known = set(facts)
    changed = True
    while changed:
        changed = False
        for s in subjects:
            for prem, c_attr, c_pol in rules:
                if all((s, a, True) in known for a in prem):
                    t = (s, c_attr, c_pol)
                    if t not in known:
                        known.add(t)
                        changed = True
    return known


def label_of(s, q, known):
    if (s, q, True) in known:
        return "True"
    if (s, q, False) in known:
        return "False"
    return "Unknown"


def build_item(rng, target, item_id):
    subj = rng.choice(NAMES)
    attrs = rng.sample(ATTRS, 10)
    hops = rng.choice([1, 2, 3])
    chain = attrs[:hops + 1]                        # a0 -> a1 -> ... -> a_hops
    facts, rules, tfacts, trules = set(), [], [], []
    facts.add((subj, chain[0], True))
    tfacts.append(f"{subj} is {chain[0]}.")
    for i in range(hops):
        if i > 0 and rng.random() < 0.35:           # conjunctive rule (" and ", no comma)
            extra = chain[i - 1]
            rules.append(([chain[i], extra], chain[i + 1], True))
            trules.append(f"If someone is {chain[i]} and {extra}, then they are {chain[i + 1]}.")
        else:
            rules.append(([chain[i]], chain[i + 1], True))
            trules.append(f"If someone is {chain[i]}, then they are {chain[i + 1]}.")

    if target == "True":
        q = chain[hops]
    elif target == "False":
        neg = attrs[hops + 1]
        rules.append(([chain[hops]], neg, False))
        trules.append(f"If someone is {chain[hops]}, then they are not {neg}.")
        q = neg
    else:                                           # Unknown
        q = attrs[hops + 1]                          # never entailed

    for _ in range(rng.choice([1, 2, 3])):          # distractor facts
        ds = rng.choice([n for n in NAMES if n != subj])
        da = rng.choice(ATTRS)
        facts.add((ds, da, True))
        tfacts.append(f"{ds} is {da}.")
    for _ in range(rng.choice([0, 1, 2])):          # distractor rules
        a, b = rng.sample(ATTRS, 2)
        rules.append(([a], b, True))
        trules.append(f"If someone is {a}, then they are {b}.")

    subjects = {s for s, _, _ in facts} | {subj}
    lab = label_of(subj, q, closure(subjects, facts, rules))
    if lab != target:                               # a distractor changed the answer
        return None
    body = tfacts + trules
    rng.shuffle(body)
    prompt = " ".join(body) + f" Is {subj} {q}? Answer True, False, or Unknown."
    return {"id": item_id, "prompt": prompt, "label": lab, "subject": subj,
            "query_attr": q, "hops": hops, "n_facts": len(tfacts),
            "n_rules": len(trules),
            "n_periods": prompt.count("."), "n_commas": prompt.count(","),
            "n_question": prompt.count("?")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="studies/ruletaker_punct.json")
    ap.add_argument("--per-label", type=int, default=60)   # 60 each -> 180
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed (study set = 0; GPT-2 control train = 1, val = 2)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    labels = ["True", "False", "Unknown"]
    per = {l: 0 for l in labels}
    items, seen = [], set()
    guard = 0
    while any(per[l] < args.per_label for l in labels) and guard < 500000:
        guard += 1
        target = min(labels, key=lambda l: (per[l], labels.index(l)))
        it = build_item(rng, target, len(items))
        if it is None or it["prompt"] in seen:
            continue
        seen.add(it["prompt"])
        it["id"] = len(items)
        items.append(it)
        per[it["label"]] += 1

    from collections import Counter
    payload = {
        "dataset": "ruletaker_punct",
        "purpose": ("RuleTaker-style theories with compute-bearing prompt punctuation "
                    "(periods delimit facts/rules, one if-then boundary comma per rule "
                    "+ two in the instruction tail, one question mark), computed "
                    "True/False/Unknown labels, hops 1-3. Stimuli for the punctuation "
                    "write-ablation vs attention-knockout study; ablation targets "
                    "PROMPT punctuation positions only."),
        "n": len(items),
        "labels": dict(sorted(Counter(x["label"] for x in items).items())),
        "hops": dict(sorted(Counter(x["hops"] for x in items).items())),
        "median_periods": sorted(x["n_periods"] for x in items)[len(items) // 2],
        "n_with_commas": sum(1 for x in items if x["n_commas"] > 0),
        "seed": args.seed, "shortcuts_taken": [],
        "prompts": items,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[ruletaker] wrote {len(items)} items -> {args.out}")
    print(f"  labels={payload['labels']}  hops={payload['hops']}  "
          f"median_periods={payload['median_periods']}  with_commas={payload['n_with_commas']}")
    for it in items[:3]:
        print(f"  [{it['label']:7}] ({it['n_periods']}. {it['n_commas']},) {it['prompt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
