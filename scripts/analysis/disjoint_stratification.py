"""Sprint 2.1a: stratify Sprint 1 results by train-test target overlap.

The Sprint 1 result showed SCM beats LLM-drug-blind by 0.062 MAP. The
critical caveat: training and test drugs share many binding targets, so
the SCM's learned α(S|T) edges may be benefiting from straightforward
interpolation rather than true mechanism composition.

This analysis stratifies the existing per-drug Sprint 1 results by
target-overlap metrics (Jaccard with training set; training depth per
target) and checks whether the SCM advantage holds in the LOW-overlap
strata where interpolation is least available.

No new LLM calls. No re-training. Pure statistical decomposition of
the existing n=200 evaluation.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / max(len(s1 | s2), 1)


def _wilcoxon_one_sided(diffs: list[float]) -> tuple[int, float, float]:
    """Wilcoxon signed-rank, one-sided H1: mean diff > 0 (i.e. A > B).
    Returns (n_nonzero, z, p_one_sided)."""
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 0, 0.0, 0.5
    indexed = sorted(enumerate(nonzero), key=lambda x: abs(x[1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(indexed[j + 1][1]) == abs(indexed[i][1]):
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    W_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    mean_W = n * (n + 1) / 4
    var_W = n * (n + 1) * (2 * n + 1) / 24
    z = (W_plus - mean_W) / math.sqrt(var_W) if var_W > 0 else 0.0
    p_one = 0.5 * (1 - math.erf(z / math.sqrt(2)))
    return n, z, p_one


def main() -> int:
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    with open(RESULTS / "sprint1_per_drug.json") as f:
        pd_data = json.load(f)
    per_drug = pd_data["per_drug"]

    # Build target sets per drug + training/test split
    target_sets = {
        d["cid"]: {t["uniprot"] for t in d["binding_profile"]}
        for d in cat["drugs"]
    }
    splits = {d["cid"]: d["split"] for d in cat["drugs"]}
    train_cids = {c for c, s in splits.items() if s == "train"}

    # Per-target training depth
    target_depth: dict[str, int] = defaultdict(int)
    for cid in train_cids:
        for t in target_sets[cid]:
            target_depth[t] += 1

    # Annotate each test drug with disjointness metrics
    train_target_sets = [target_sets[c] for c in train_cids]
    for r in per_drug:
        cid = r["cid"]
        ts = target_sets[cid]
        r["max_train_jaccard"] = max(
            jaccard(ts, tts) for tts in train_target_sets
        )
        r["mean_train_jaccard"] = statistics.mean(
            jaccard(ts, tts) for tts in train_target_sets
        )
        depths = [target_depth.get(t, 0) for t in ts]
        r["median_target_depth"] = statistics.median(depths)
        r["mean_target_depth"] = statistics.mean(depths)
        r["min_target_depth"] = min(depths)

    print("=" * 78)
    print("Sprint 2.1a: disjointness stratification on Sprint 1 results")
    print("=" * 78)
    print(f"n test drugs: {len(per_drug)}")
    print(f"n train drugs: {len(train_cids)}")

    # Distribution overview
    jaccards = [r["max_train_jaccard"] for r in per_drug]
    jaccards.sort()
    print(f"\nmax_train_jaccard distribution:")
    print(f"  min={jaccards[0]:.3f} p25={jaccards[len(jaccards)//4]:.3f} "
          f"median={jaccards[len(jaccards)//2]:.3f} "
          f"p75={jaccards[3*len(jaccards)//4]:.3f} max={jaccards[-1]:.3f}")

    depths = [r["median_target_depth"] for r in per_drug]
    depths.sort()
    print(f"\nmedian_target_depth distribution (# training drugs per target):")
    print(f"  min={depths[0]:.0f} p25={depths[len(depths)//4]:.0f} "
          f"median={depths[len(depths)//2]:.0f} "
          f"p75={depths[3*len(depths)//4]:.0f} max={depths[-1]:.0f}")

    # Stratified comparisons
    print("\n" + "=" * 78)
    print("STRATIFIED ANALYSES — does the SCM advantage hold at low overlap?")
    print("=" * 78)

    def report_stratum(name: str, subset: list[dict]) -> None:
        n = len(subset)
        if n < 10:
            print(f"\n{name}: n={n} (too few for stable stats)")
            return
        scm_aps = [r["scm_metrics"]["ap"] for r in subset]
        llm_aps = [r["llm_drug_blind_metrics"]["ap"] for r in subset]
        name_aps = [r["llm_with_name_metrics"]["ap"] for r in subset]
        rnd_aps = [r["random_metrics"]["ap"] for r in subset]
        diffs = [s - l for s, l in zip(scm_aps, llm_aps)]
        n_wilc, z, p = _wilcoxon_one_sided(diffs)
        d_mean = statistics.mean(diffs)
        d_sd = statistics.stdev(diffs) if len(diffs) > 1 else 1.0
        cohens_d = d_mean / d_sd if d_sd > 0 else 0.0
        scm_wins = sum(1 for d in diffs if d > 0)
        print(f"\n{name} (n={n}):")
        print(f"  SCM MAP:            {statistics.mean(scm_aps):.4f}")
        print(f"  LLM-drug-blind MAP: {statistics.mean(llm_aps):.4f}")
        print(f"  LLM-with-name MAP:  {statistics.mean(name_aps):.4f}")
        print(f"  Random MAP:         {statistics.mean(rnd_aps):.4f}")
        print(f"  SCM − LLM-blind:    {d_mean:+.4f}  (Cohen d={cohens_d:+.3f})")
        print(f"  Wilcoxon SCM>LLM-blind: z={z:.3f}  p(one-sided)={p:.4g}")
        print(f"  SCM wins / n: {scm_wins}/{n} = {scm_wins/n:.1%}")

    # Strata by max_train_jaccard
    print("\n--- by max_train_jaccard (lower = more target-disjoint) ---")
    sorted_by_jacc = sorted(per_drug, key=lambda r: r["max_train_jaccard"])
    for pct, thresh_idx in [(100, len(sorted_by_jacc)),
                              (50, len(sorted_by_jacc) // 2),
                              (25, len(sorted_by_jacc) // 4),
                              (10, len(sorted_by_jacc) // 10)]:
        subset = sorted_by_jacc[:thresh_idx]
        max_j = max(r["max_train_jaccard"] for r in subset) if subset else 0
        report_stratum(
            f"Bottom {pct}% by max_train_jaccard (max_jaccard ≤ {max_j:.3f})",
            subset,
        )

    # Hard threshold: drugs with NO near-duplicate in training (max Jaccard ≤ 0.3)
    strict = [r for r in per_drug if r["max_train_jaccard"] <= 0.3]
    report_stratum("Strict disjoint (max_jaccard ≤ 0.30)", strict)
    strict_05 = [r for r in per_drug if r["max_train_jaccard"] <= 0.5]
    report_stratum("Moderate disjoint (max_jaccard ≤ 0.50)", strict_05)

    # Strata by median target depth
    print("\n--- by median_target_depth (lower = sparser training coverage) ---")
    sorted_by_depth = sorted(per_drug, key=lambda r: r["median_target_depth"])
    for pct, thresh_idx in [(50, len(sorted_by_depth) // 2),
                              (25, len(sorted_by_depth) // 4),
                              (10, len(sorted_by_depth) // 10)]:
        subset = sorted_by_depth[:thresh_idx]
        max_d = max(r["median_target_depth"] for r in subset) if subset else 0
        report_stratum(
            f"Bottom {pct}% by median_target_depth (depth ≤ {max_d:.0f})",
            subset,
        )

    # Save annotated per-drug results
    out_path = RESULTS / "sprint2_1a_disjoint_stratification.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_test": len(per_drug),
            "n_train": len(train_cids),
            "per_drug_with_disjoint_metrics": per_drug,
        }, f, indent=2)
    print(f"\nSaved annotated results: {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
