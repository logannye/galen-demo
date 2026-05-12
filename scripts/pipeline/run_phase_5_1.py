"""Phase 5.1: Combined onc/immuno held-out validation (n=38).

Combines Phase 4.3 cases (n=30) with Phase 5.1 immuno expansion (n=8).
Re-runs Phase 4.1 production stack with Phase 5.2 expanded priors.

Definitive validation of the production stack on a larger held-out
cohort of FDA-approved onc/immuno drugs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"

# Swap the case file used by Phase 4.3 runner
import scripts.pipeline.run_phase_4_3 as runner
runner.__dict__["__phase5_combined__"] = True


def main():
    # Just import the main and run with substituted case file
    from scripts.pipeline.run_phase_4_3 import (
        _process_case, bootstrap_ci, hit_at_k_clustered, collapse_top_k,
        apply_curated_prior_override_v2, load_curated_priors_for_override,
        ThreadPoolExecutor, as_completed, time, hybrid_rerank,
        explain_predictions, load_action_types, score_drug_side_effects_signed,
        SonnetClient,
    )
    from scripts.pipeline.run_phase_4_3 import N_WORKERS

    print("=" * 78)
    print("Phase 5.1: Combined onc/immuno held-out (n=38) with Phase 5.2 priors")
    print("=" * 78)

    client = SonnetClient()
    edges = json.load(open(RESULTS / "scm_edges_blended_j.json"))
    signed = json.load(open(RESULTS / "scm_edges_signed.json"))
    signed_edges = signed.get("edges", {})
    target_action_n = signed.get("target_action_n_drugs", {})
    vocab_payload = json.load(open(RESULTS / "side_effect_vocab.json"))
    se_vocab = vocab_payload["umls_ids"]
    se_names = vocab_payload["display_names"]
    tv = json.load(open(RESULTS / "target_vocab.json"))
    target_info = {t["uniprot"]: t for t in tv["targets"]}
    curated_priors = load_curated_priors_for_override()

    cases = json.load(open(RESULTS / "phase_5_1_combined_cases.json"))["cases"]
    print(f"\nLoaded {len(cases)} combined held-out cases")

    results = [None] * len(cases)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(_process_case, c, edges, signed_edges,
                       target_action_n, target_info, vocab_payload, se_vocab,
                       se_names, client, curated_priors): i
            for i, c in enumerate(cases)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"  case {idx} ({cases[idx]['name']}) FAILED: {e}", flush=True)
                results[idx] = {"name": cases[idx]["name"], "skipped": True}
            done += 1
            if done % 5 == 0:
                el = time.monotonic() - t0
                print(f"  {done}/{len(cases)} ({el/60:.1f}m)", flush=True)

    results = [r for r in results if r is not None]
    print(f"\nDone in {(time.monotonic() - t0)/60:.1f}m")

    out_path = RESULTS / "phase_5_1_results.json"
    with open(out_path, "w") as f:
        json.dump({"n": sum(1 for r in results if not r.get("skipped")),
                   "per_drug": results}, f, indent=2)

    valid = [r for r in results if not r.get("skipped")]
    n = len(valid)

    def hit_lit(rec, k):
        return bool(set(rec.get("hybrid_top10", [])[:k]) & set(rec.get("gt_umls", [])))

    def hit_clust(rec, k):
        return hit_at_k_clustered(
            set(rec.get("gt_umls", [])),
            collapse_top_k(rec.get("hybrid_top10", [])), k,
        )

    print("\n" + "=" * 78)
    print(f"Phase 5.1 RESULTS — n={n} combined held-out onc/immuno")
    print("=" * 78)
    print(f"\n{'metric':<8s} {'LITERAL':<25s} {'CLUSTER-AWARE':<25s}")
    for k in (1, 3, 5, 10):
        h_lit = sum(1 for r in valid if hit_lit(r, k))
        h_clu = sum(1 for r in valid if hit_clust(r, k))
        ci = bootstrap_ci(h_clu, n)
        print(f"  hit@{k:<3d}  {h_lit:>2d}/{n} ({h_lit/n:.1%})            "
              f"{h_clu:>2d}/{n} ({h_clu/n:.1%}) [CI {ci[0]:.0%}-{ci[1]:.0%}]")

    # Per-TA
    print("\nPer-TA breakdown (cluster-aware):")
    for ta in ("Oncology", "Immunology", "Other"):
        sub = [r for r in valid if r.get("therapeutic_area") == ta]
        if not sub:
            continue
        print(f"\n  {ta} (n={len(sub)})")
        for k in (1, 3, 5, 10):
            h = sum(1 for r in sub if hit_clust(r, k))
            ci = bootstrap_ci(h, len(sub))
            print(f"    hit@{k}: {h}/{len(sub)} ({h/len(sub):.1%}) [CI {ci[0]:.0%}-{ci[1]:.0%}]")

    summary = {"n_combined": n}
    out2 = RESULTS / "phase_5_1_summary.json"
    with open(out2, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
