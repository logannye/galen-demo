"""Sprint G Track 1: miss analysis on Sprint F Hybrid misses.

For each case in sprint_f_safety.json where Hybrid hit@10 missed (rank
> 10 or None), output a structured diagnostic:
  - drug_id, drug_name, therapeutic_area, causal_off_target
  - Ground truth AEs (UMLS + display names)
  - Hybrid top-10 predictions (UMLS + display names)
  - Whether Hybrid ranked the ground truth at all (rank if so)

Then classify each miss into one of:
  - VOCAB_MISMATCH: Hybrid predicted a semantically-correct concept
    using a different UMLS code than benchmark
  - WRONG_MECHANISM: Hybrid predicted off-target AEs not in ground truth
  - TRUE_GAP: Hybrid predicted plausible alternatives, ground truth
    requires deeper mechanism knowledge
  - BENCHMARK_ERROR: ground truth AE is itself questionable or wrong UMLS

Output: results/sprint_f_miss_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def main() -> int:
    with open(RESULTS / "sprint_f_safety.json") as f:
        safety = json.load(f)
    with open(RESULTS / "side_effect_vocab.json") as f:
        vocab = json.load(f)
    se_names = vocab["display_names"]
    with open(RESULTS / "hpo_umls_rollup.json") as f:
        rollup_payload = json.load(f)
    rollup_map = rollup_payload["rollup"]

    misses = []
    for r in safety["per_drug"]:
        if r.get("skipped"):
            continue
        hybrid_rank = r.get("hybrid_rank")
        if hybrid_rank is not None and hybrid_rank <= 10:
            continue
        misses.append(r)

    print(f"Total cases: {len([r for r in safety['per_drug'] if not r.get('skipped')])}")
    print(f"Hybrid misses (rank > 10 or None): {len(misses)}")

    detailed = []
    for r in misses:
        drug_id = r["drug_id"]
        drug_name = r["drug_search_name"]
        ta = r["therapeutic_area"]
        gene = r["causal_off_target"]
        causal_se_display = r["causal_se"]
        hybrid_top10 = r.get("hybrid_top10", [])
        hybrid_rank = r.get("hybrid_rank")

        # Map hybrid_top10 (UMLS list) to display names
        top10_display = [
            (u, se_names.get(u, u)) for u in hybrid_top10
        ]

        # Causal SE UMLS — pull from benchmark (have to look up by drug_id)
        # The safety record doesn't include causal_side_effects_umls directly;
        # we rely on the display string + ranks.
        detailed.append({
            "drug_id": drug_id,
            "drug_search_name": drug_name,
            "therapeutic_area": ta,
            "causal_off_target": gene,
            "causal_se_display": causal_se_display,
            "hybrid_rank_strict": hybrid_rank,
            "hybrid_top10": top10_display,
            "n_binding_targets": r.get("n_binding_targets"),
            "biologic_recovery": r.get("biologic_recovery", False),
            "scm_rank": r.get("scm_rank"),
            "llm_with_name_rank": r.get("llm_with_name_rank"),
        })

    out_path = RESULTS / "sprint_f_miss_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_misses": len(detailed),
            "misses": detailed,
        }, f, indent=2)
    print(f"\n[save] {out_path}")

    # Print a readable summary
    print("\n" + "=" * 78)
    print("HYBRID MISSES — diagnostic table")
    print("=" * 78)
    for d in detailed:
        print(f"\n{d['drug_id']:<35s} TA={d['therapeutic_area']:<22s} "
              f"target={d['causal_off_target']}")
        print(f"  CAUSAL: {d['causal_se_display']}")
        print(f"  Hybrid rank: {d['hybrid_rank_strict']}  "
              f"SCM rank: {d['scm_rank']}  "
              f"LLM-with-name rank: {d['llm_with_name_rank']}")
        print(f"  Hybrid top-10:")
        for i, (u, name) in enumerate(d['hybrid_top10'], 1):
            print(f"    {i:>2d}. {u:<10s} {name}")

    # Per-TA breakdown of misses
    from collections import Counter
    ta_counts = Counter(d['therapeutic_area'] for d in detailed)
    print(f"\n\n[summary] misses by TA: {dict(ta_counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
