"""Phase 1 diagnostic: classify each L.1 miss into a failure mode.

For each ground-truth SE missed by Hybrid at top-10 in L.1, label it:
  - OOV-GT: SE not in 681-term vocabulary (unhittable)
  - WEAK-TARGET: all binding targets have <3 training drugs
  - MISSING-EDGE: target+SE pair has α<0.05 in current substrate
  - CLOSE-MISS: SE was ranked 11-50 (close to top-10 cutoff)
  - PURE-MISS: SE was ranked >50 with strong target coverage (architecturally hard)

Drug-level classification = most common SE-level mode across the drug's missed SEs.

Reports per-mode counts + upper bound on what each intervention can recover.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
DOCS = WORKSPACE / "docs"

# Tunable thresholds
WEAK_TARGET_DRUG_COUNT = 3   # Sprint 2.1 floor
MISSING_EDGE_ALPHA = 0.05    # α below this = essentially no signal
CLOSE_MISS_RANK_MAX = 50     # rank 11-50 = close miss


def main():
    print("=" * 78)
    print("Phase 1 — L.1 miss-mode diagnostic")
    print("=" * 78)

    # Load data
    catalog = json.load(open(RESULTS / "catalog.json"))
    vocab = json.load(open(RESULTS / "side_effect_vocab.json"))
    vocab_umls = set(vocab["umls_ids"])
    edges = json.load(open(RESULTS / "scm_edges_blended_j.json"))
    l1 = json.load(open(RESULTS / "sprint_l_external_safety.json"))

    # Build target-coverage map (how many training drugs bind each target?)
    target_train_count = Counter()
    for d in catalog["drugs"]:
        if d.get("split") != "train":
            continue
        for b in d.get("binding_profile", []):
            target_train_count[b["uniprot"]] += 1

    # Build name → binding_profile map (for L.1 drugs we need to look up targets)
    name_to_profile: dict[str, list[dict]] = {}
    for d in catalog["drugs"]:
        nm = d["drug_name"].lower()
        # Prefer test-split entry if duplicate
        if nm not in name_to_profile or d.get("split") == "test":
            name_to_profile[nm] = d.get("binding_profile", [])

    # For each L.1 drug, classify
    records = l1["per_drug"]
    print(f"\nL.1 total drugs: {len(records)}")
    hits = [r for r in records if r.get("hybrid_rank") is not None and r["hybrid_rank"] <= 10]
    misses = [r for r in records if r not in hits]
    print(f"  hits @ top-10: {len(hits)} ({len(hits)/len(records):.1%})")
    print(f"  misses       : {len(misses)} ({len(misses)/len(records):.1%})")

    # Per-SE-level analysis across all misses
    se_mode_counts = Counter()
    se_details: list[dict] = []  # per-SE detail rows
    drug_mode_counts = Counter()
    drug_details = []

    for r in misses:
        drug = r["drug_search_name"].lower()
        gt_ses = r.get("causal_se_umls") or []
        profile = name_to_profile.get(drug, [])
        profile_uniprots = [b["uniprot"] for b in profile]

        # Per-target training coverage for this drug
        coverage = [target_train_count.get(t, 0) for t in profile_uniprots]
        any_strong_target = any(c >= WEAK_TARGET_DRUG_COUNT for c in coverage)
        max_target_cov = max(coverage) if coverage else 0

        # Per-SE classification
        per_se_modes = []
        for se in gt_ses:
            if se not in vocab_umls:
                mode = "OOV-GT"
            elif not any_strong_target:
                mode = "WEAK-TARGET"
            else:
                # Check if any of this drug's targets has an α-edge to this SE
                max_alpha = 0.0
                for t in profile_uniprots:
                    a = edges.get(t, {}).get(se, 0.0)
                    if a > max_alpha:
                        max_alpha = a
                if max_alpha < MISSING_EDGE_ALPHA:
                    mode = "MISSING-EDGE"
                else:
                    # Substrate has signal — check if reranker just put it deep
                    # We don't have per-SE rank in the L.1 records, only the
                    # best-ranked GT-SE. So we use the drug-level hybrid_rank
                    # as a proxy for "where the best GT-SE landed".
                    rk = r.get("hybrid_rank")
                    if rk is not None and rk <= CLOSE_MISS_RANK_MAX:
                        mode = "CLOSE-MISS"
                    else:
                        mode = "PURE-MISS"
            per_se_modes.append(mode)
            se_mode_counts[mode] += 1
            se_details.append({
                "drug": drug,
                "se_umls": se,
                "mode": mode,
                "in_vocab": se in vocab_umls,
                "max_target_train_drugs": max_target_cov,
                "n_targets": len(profile_uniprots),
            })

        # Drug-level: dominant mode (with tie-break by mode priority)
        priority = ["OOV-GT", "WEAK-TARGET", "MISSING-EDGE", "CLOSE-MISS", "PURE-MISS"]
        cnt = Counter(per_se_modes)
        if cnt:
            best = max(priority, key=lambda m: (cnt.get(m, 0), -priority.index(m)))
            if cnt.get(best, 0) > 0:
                drug_mode = best
            else:
                drug_mode = "PURE-MISS"
        else:
            drug_mode = "PURE-MISS"
        drug_mode_counts[drug_mode] += 1
        drug_details.append({
            "drug": drug,
            "hybrid_rank": r.get("hybrid_rank"),
            "n_gt_ses": len(gt_ses),
            "n_targets": len(profile_uniprots),
            "max_target_cov": max_target_cov,
            "per_se_modes": per_se_modes,
            "drug_mode": drug_mode,
        })

    # SE-level distribution
    print("\n" + "=" * 78)
    print("SE-level miss-mode distribution (across all missed GT-SEs)")
    print("=" * 78)
    total_se = sum(se_mode_counts.values())
    print(f"\n{'Mode':<14s} {'n':>6s} {'pct':>7s}   Intervention")
    print("-" * 78)
    interventions = {
        "OOV-GT":       "Vocabulary expansion (add UMLS codes from OnSIDES tail)",
        "WEAK-TARGET":  "Training-drug expansion (OnSIDES) for under-covered targets",
        "MISSING-EDGE": "Training-drug expansion adds new (target, SE) signal",
        "CLOSE-MISS":   "Reranker tuning / calibration v3",
        "PURE-MISS":    "Architecturally hard — needs richer mechanism (Reactome, paths)",
    }
    for mode in ["OOV-GT", "WEAK-TARGET", "MISSING-EDGE", "CLOSE-MISS", "PURE-MISS"]:
        n = se_mode_counts.get(mode, 0)
        pct = n / max(total_se, 1)
        print(f"{mode:<14s} {n:>6d}  {pct:>6.1%}   {interventions[mode]}")
    print(f"{'TOTAL':<14s} {total_se:>6d}")

    # Drug-level distribution
    print("\n" + "=" * 78)
    print("Drug-level miss-mode distribution (dominant SE-mode per missed drug)")
    print("=" * 78)
    total_d = sum(drug_mode_counts.values())
    print(f"\n{'Mode':<14s} {'n':>6s} {'pct':>7s}")
    print("-" * 30)
    for mode in ["OOV-GT", "WEAK-TARGET", "MISSING-EDGE", "CLOSE-MISS", "PURE-MISS"]:
        n = drug_mode_counts.get(mode, 0)
        pct = n / max(total_d, 1)
        print(f"{mode:<14s} {n:>6d}  {pct:>6.1%}")
    print(f"{'TOTAL':<14s} {total_d:>6d}")

    # Upper-bound analysis: if intervention X recovers all of mode Y, what's the hit@10 gain?
    # Assume each drug only needs ONE GT-SE in top-10 to count as a hit.
    print("\n" + "=" * 78)
    print("Upper-bound recovery per intervention (drug-level hit@10 gain)")
    print("=" * 78)
    base_hits = len(hits)
    n_total = len(records)
    base_rate = base_hits / n_total
    print(f"\nBaseline L.1 hit@10: {base_hits}/{n_total} = {base_rate:.1%}\n")
    print(f"{'Intervention':<35s} {'Mode recovered':<14s} {'+drugs':>8s} {'New hit@10':>12s}")
    print("-" * 78)
    for mode, label in [
        ("OOV-GT", "Vocabulary expansion"),
        ("WEAK-TARGET", "Training-drug expansion"),
        ("MISSING-EDGE", "Training-drug expansion"),
        ("CLOSE-MISS", "Reranker / calibration v3"),
        ("PURE-MISS", "Architecturally hard"),
    ]:
        n_recovered = drug_mode_counts.get(mode, 0)
        new_hit = (base_hits + n_recovered) / n_total
        print(f"{label:<35s} {mode:<14s} {n_recovered:>+8d}  {new_hit:>11.1%}")

    # Combined intervention bounds
    print("\nCombined bounds:")
    onsides = drug_mode_counts.get("WEAK-TARGET", 0) + drug_mode_counts.get("MISSING-EDGE", 0)
    onsides_plus_vocab = onsides + drug_mode_counts.get("OOV-GT", 0)
    rerank = drug_mode_counts.get("CLOSE-MISS", 0)
    onsides_vocab_rerank = onsides_plus_vocab + rerank
    print(f"  OnSIDES (recovers WEAK + MISSING-EDGE):           {(base_hits+onsides)/n_total:.1%}  (+{onsides} drugs)")
    print(f"  OnSIDES + vocab (+ OOV-GT):                       {(base_hits+onsides_plus_vocab)/n_total:.1%}  (+{onsides_plus_vocab} drugs)")
    print(f"  OnSIDES + vocab + reranker (+ CLOSE-MISS):        {(base_hits+onsides_vocab_rerank)/n_total:.1%}  (+{onsides_vocab_rerank} drugs)")
    print(f"  Architectural ceiling (only PURE-MISS unfixable): {(n_total - drug_mode_counts.get('PURE-MISS', 0))/n_total:.1%}")

    # Save detailed JSON
    out = {
        "se_level_counts": dict(se_mode_counts),
        "drug_level_counts": dict(drug_mode_counts),
        "n_total_drugs": n_total,
        "n_hits_baseline": base_hits,
        "drug_details": drug_details,
        "intervention_bounds": {
            "onsides_only": (base_hits + onsides) / n_total,
            "onsides_plus_vocab": (base_hits + onsides_plus_vocab) / n_total,
            "onsides_plus_vocab_plus_rerank": (base_hits + onsides_vocab_rerank) / n_total,
            "architectural_ceiling": (n_total - drug_mode_counts.get("PURE-MISS", 0)) / n_total,
        },
        "thresholds": {
            "weak_target_drug_count": WEAK_TARGET_DRUG_COUNT,
            "missing_edge_alpha": MISSING_EDGE_ALPHA,
            "close_miss_rank_max": CLOSE_MISS_RANK_MAX,
        },
    }
    out_path = RESULTS / "phase_1_l1_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
