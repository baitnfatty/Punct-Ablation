"""Nonparametric re-analysis of the G1 gate contrast — in-repo replacement for the
out-of-repo pointer flagged as D3 in REDTEAM_punct_null.md. Pure python (sign test,
Wilcoxon signed-rank with normal approximation) over the committed gate artifact.

  python studies/analyze_gate_nonparam.py [--in runs/studies/punct_ablation_gate.json]
"""

import argparse
import json
import math


def sign_test(d):
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    n = pos + neg
    # two-sided exact binomial (normal approx for n>50 is fine; do exact)
    from math import comb
    k = min(pos, neg)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return pos, neg, min(1.0, p)


def wilcoxon(d):
    d = [x for x in d if x != 0]
    n = len(d)
    ranked = sorted((abs(x), x) for x in d)
    # average ranks for ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranked[j + 1][0] == ranked[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    w_pos = sum(r for r, (_, x) in zip(ranks, ranked) if x > 0)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_pos - mu) / sd
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return w_pos, z, p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="runs/studies/punct_ablation_gate.json")
    a = ap.parse_args()
    r = json.load(open(a.inp))
    for name, key in (("logit", "deltas"), ("margin", "margins")):
        d = [it[key]["knockout:punct"] - it[key]["knockout:attn_matched"]
             for it in r["per_item"]]
        pos, neg, sp = sign_test(d)
        w, z, wp = wilcoxon(d)
        print(f"{name:6}  n={len(d)}  neg(punct-worse)={neg} pos={pos}  "
              f"sign p={sp:.4f}  Wilcoxon z={z:+.2f} p={wp:.4f}")


if __name__ == "__main__":
    main()
