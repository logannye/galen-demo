"""Phase 3.4: Onc/immuno performance with AE cluster collapse + crediting.

Re-evaluates Sprint K saved Hybrid output with:
  1. Output-time collapse: dedup top-K by cluster membership
  2. Eval-time crediting: GT cluster member match counts as hit

No new LLM calls. Pure post-processing.

Pre-registered in docs/PHASE_3_PRE_REGISTRATION.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..baselines.ae_cluster_postprocess import (
    collapse_top_k, hit_at_k_clustered, rank_clustered, load_clusters,
)


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def hit_at_k_literal(top_k, gt_set, k):
    return bool(set(top_k[:k]) & set(gt_set))


def mcnemar_one_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    p = 0.0
    for x in range(b, n + 1):
        p += math.comb(n, x) * (0.5 ** n)
    return p


def bootstrap_ci(hits, n, n_boot=1000, seed=42):
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    arr = np.array([1] * hits + [0] * (n - hits))
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot.append(arr[idx].mean())
    boot.sort()
    return (boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)])


def evaluate(records, label="Onc+Immuno"):
    """Compute literal hit@K, collapsed-only hit@K, full cluster-aware hit@K."""
    n = len(records)

    results = {
        "label": label, "n": n,
        "literal": {}, "collapsed": {}, "cluster_aware": {},
    }

    for k in (1, 3, 5, 10):
        hits_lit = sum(1 for r in records
                        if hit_at_k_literal(r["hybrid_top10"],
                                              r["causal_side_effects_umls"], k))
        # Collapsed: same top-10 but cluster-dedup, then literal match
        hits_col = sum(1 for r in records
                        if hit_at_k_literal(collapse_top_k(r["hybrid_top10"]),
                                              r["causal_side_effects_umls"], k))
        # Cluster-aware on the collapsed list
        hits_clust = sum(1 for r in records
                          if hit_at_k_clustered(set(r["causal_side_effects_umls"]),
                                                  collapse_top_k(r["hybrid_top10"]), k))
        results["literal"][f"hit@{k}"] = hits_lit
        results["literal"][f"hit@{k}_rate"] = hits_lit / max(n, 1)
        results["collapsed"][f"hit@{k}"] = hits_col
        results["collapsed"][f"hit@{k}_rate"] = hits_col / max(n, 1)
        results["cluster_aware"][f"hit@{k}"] = hits_clust
        results["cluster_aware"][f"hit@{k}_rate"] = hits_clust / max(n, 1)
        # Bootstrap CIs
        results["cluster_aware"][f"hit@{k}_ci"] = bootstrap_ci(hits_clust, n)

    return results


def paired_mcnemar(records, k=3, mode="cluster_aware"):
    """Paired McNemar new (cluster-aware on collapsed) vs old (literal).

    b = new wins (new hit, old miss)
    c = new loses (new miss, old hit)
    """
    b = c = 0
    for r in records:
        gt = set(r["causal_side_effects_umls"])
        old = hit_at_k_literal(r["hybrid_top10"], gt, k)
        if mode == "cluster_aware":
            new = hit_at_k_clustered(gt, collapse_top_k(r["hybrid_top10"]), k)
        elif mode == "collapsed":
            new = hit_at_k_literal(collapse_top_k(r["hybrid_top10"]), gt, k)
        if new and not old:
            b += 1
        elif old and not new:
            c += 1
    return b, c, mcnemar_one_sided(b, c)


def main():
    print("=" * 78)
    print("Phase 3.4: AE-cluster collapse + cluster-aware eval (n=140 onc/immuno)")
    print("=" * 78)

    # Load Sprint K saved results
    main_records = json.load(open(RESULTS / "sprint_k_safety_sonnet.json"))["per_drug"]
    ood_records = json.load(open(RESULTS / "sprint_k_ood_safety_sonnet.json"))["per_drug"]
    all_records = [dict(r, dataset="main") for r in main_records] + \
                   [dict(r, dataset="ood") for r in ood_records]
    oi = [r for r in all_records
           if r.get("therapeutic_area") in ("Oncology", "Immunology")
           and not r.get("skipped")]
    print(f"\nLoaded {len(oi)} onc/immuno cases\n")

    # Print cluster summary
    _, meta = load_clusters()
    print("AE clusters:")
    for cid, m in meta.items():
        print(f"  {cid:<35s} {m['n_members']:>2d} members  → "
              f"{m['representative_name']}")

    # ---------- Evaluate ----------
    print("\n" + "=" * 78)
    print("Onc/Immuno (n={})".format(len(oi)))
    print("=" * 78)
    oi_eval = evaluate(oi, "Onc+Immuno")

    print(f"\n{'metric':<8s} {'literal':<22s} {'collapsed':<22s} {'cluster-aware':<22s}")
    for k in (1, 3, 5, 10):
        lit = oi_eval["literal"][f"hit@{k}"]
        col = oi_eval["collapsed"][f"hit@{k}"]
        clu = oi_eval["cluster_aware"][f"hit@{k}"]
        ci = oi_eval["cluster_aware"][f"hit@{k}_ci"]
        print(f"  hit@{k:<3d}  {lit:>3d}/{oi_eval['n']} ({lit/oi_eval['n']:>5.1%})       "
              f"{col:>3d}/{oi_eval['n']} ({col/oi_eval['n']:>5.1%})       "
              f"{clu:>3d}/{oi_eval['n']} ({clu/oi_eval['n']:>5.1%}) "
              f"[{ci[0]:.0%}-{ci[1]:.0%}]")

    # Paired McNemar
    print("\nPaired McNemar (new vs literal Sprint K, one-sided):")
    for k in (1, 3, 5, 10):
        b, c, p = paired_mcnemar(oi, k=k, mode="cluster_aware")
        print(f"  hit@{k}: b={b} (cluster-aware wins), c={c} (literal wins), p={p:.4f}")

    # By TA
    print("\n" + "=" * 78)
    print("Per-TA breakdown")
    print("=" * 78)
    for ta in ("Oncology", "Immunology"):
        sub = [r for r in oi if r.get("therapeutic_area") == ta]
        sub_eval = evaluate(sub, ta)
        print(f"\n{ta} (n={sub_eval['n']})")
        for k in (1, 3, 5, 10):
            lit = sub_eval["literal"][f"hit@{k}"]
            clu = sub_eval["cluster_aware"][f"hit@{k}"]
            print(f"  hit@{k}: literal {lit/sub_eval['n']:>5.1%} → "
                  f"cluster-aware {clu/sub_eval['n']:>5.1%}  "
                  f"(+{(clu-lit)/sub_eval['n']:>4.1%})")

    # Full Sprint K regression check
    full = [r for r in all_records if not r.get("skipped")]
    full_eval = evaluate(full, "FULL Sprint K")
    print("\n" + "=" * 78)
    print(f"FULL Sprint K regression check (n={full_eval['n']})")
    print("=" * 78)
    for k in (1, 3, 5, 10):
        lit = full_eval["literal"][f"hit@{k}"]
        clu = full_eval["cluster_aware"][f"hit@{k}"]
        print(f"  hit@{k}: literal {lit/full_eval['n']:>5.1%} → "
              f"cluster-aware {clu/full_eval['n']:>5.1%}  "
              f"(+{(clu-lit)/full_eval['n']:>4.1%})")

    # PPV at confidence with cluster-aware
    print("\n" + "=" * 78)
    print("Cluster-aware PPV at confidence thresholds (onc/immuno)")
    print("=" * 78)
    for t in (0.30, 0.50, 0.70, 0.85):
        confident = [r for r in oi
                     if (r.get("confidence_top10") or [{}])[0].get("confidence", 0) >= t]
        n_c = len(confident)
        if n_c == 0:
            continue
        h3_lit = sum(1 for r in confident
                     if hit_at_k_literal(r["hybrid_top10"],
                                           r["causal_side_effects_umls"], 3))
        h3_clu = sum(1 for r in confident
                     if hit_at_k_clustered(set(r["causal_side_effects_umls"]),
                                              collapse_top_k(r["hybrid_top10"]), 3))
        print(f"  conf ≥ {t:.2f}  n={n_c:>3d}  "
              f"PPV@3 literal {h3_lit/n_c:>5.1%} → cluster {h3_clu/n_c:>5.1%}")

    # Decision per pre-reg
    print("\n" + "=" * 78)
    print("Decision per pre-reg")
    print("=" * 78)
    h1 = oi_eval["cluster_aware"]["hit@1_rate"]
    h3 = oi_eval["cluster_aware"]["hit@3_rate"]
    h10 = oi_eval["cluster_aware"]["hit@10_rate"]
    full_h10 = full_eval["cluster_aware"]["hit@10_rate"]
    no_reg = full_h10 >= 0.86

    if h1 >= 0.75 and h3 >= 0.92 and h10 >= 0.96 and no_reg:
        decision = "STRONG WIN"
    elif h1 >= 0.70 and h3 >= 0.88 and no_reg:
        decision = "MODERATE WIN"
    elif h1 < 0.65 or h3 < 0.85 or not no_reg:
        decision = "LOSS"
    else:
        decision = "NULL"

    print(f"\nOnc/Immuno hit@1  {h1:.1%}  (target: STRONG ≥75%, MODERATE ≥70%)")
    print(f"Onc/Immuno hit@3  {h3:.1%}  (target: STRONG ≥92%, MODERATE ≥88%)")
    print(f"Onc/Immuno hit@10 {h10:.1%} (target: STRONG ≥96%)")
    print(f"Full hit@10       {full_h10:.1%} (regression check: ≥86%)")
    print(f"\nDECISION: {decision}")

    out = {
        "n_onc_immuno": len(oi),
        "oi_eval": oi_eval,
        "full_eval": full_eval,
        "decision": decision,
    }
    out_path = RESULTS / "phase_3_clusters.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
