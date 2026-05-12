"""End-to-end Week 1 gate: re-run Phase 6.1 with SMILES->TargetNet input.

Strategy: monkey-patch `reconstruct_binding` in `run_phase_4_3` to use
TargetNet, then call the existing pipeline unchanged. This ensures
identical scoring/Hybrid/TA logic, isolating the change to ONLY the
binding-profile-resolution stage.

Output: results/phase_6_1_smiles_results.json

Pre-reg gate: Onc cluster-aware hit@10 drop <=5pp.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import scripts.pipeline.run_phase_4_3 as p43
from scripts.pipeline.run_phase_4_3 import (
    _process_case, bootstrap_ci, hit_at_k_clustered, collapse_top_k,
    load_curated_priors_for_override,
    SonnetClient, N_WORKERS,
)
from scripts.data.biologic_binding_profiles import get_biologic_binding
from scripts.targetnet.predict import predict_binding_profile

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"

TARGETNET_TOP_K = 20
TARGETNET_PROB_THRESHOLD = 0.5

# Track which source each drug's binding profile came from
_bp_sources: dict[str, str] = {}


def _get_smiles_by_molregno(molregno):
    if molregno is None:
        return None
    conn = sqlite3.connect(CHEMBL_DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT canonical_smiles FROM compound_structures WHERE molregno = ?",
        (int(molregno),),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def reconstruct_binding_smiles_first(name, molregno=None):
    """Replacement for p43.reconstruct_binding.

    1. SMILES->TargetNet (if compound has structure in ChEMBL)
    2. Biologic curated binding fallback
    """
    smi = _get_smiles_by_molregno(molregno)
    if smi:
        bp = predict_binding_profile(
            smi, top_k=TARGETNET_TOP_K,
            prob_threshold=TARGETNET_PROB_THRESHOLD,
        )
        if bp:
            _bp_sources[name] = "targetnet"
            return bp, molregno
    bp = get_biologic_binding(name) or []
    _bp_sources[name] = "biologic_fallback" if bp else "empty"
    return bp, None


# Monkey-patch
p43.reconstruct_binding = reconstruct_binding_smiles_first


def main():
    print("=" * 78)
    print("End-to-end Week 1 gate: Phase 6.1 with SMILES->TargetNet input")
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

    cases = json.load(open(RESULTS / "phase_6_1_combined_cases.json"))["cases"]
    print(f"\nLoaded {len(cases)} cases")
    print(f"TargetNet config: top_k={TARGETNET_TOP_K}, "
          f"prob_threshold={TARGETNET_PROB_THRESHOLD}\n")

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
                r = fut.result()
                r["bp_source"] = _bp_sources.get(r.get("name", ""), "unknown")
                results[idx] = r
            except Exception as e:
                print(f"  case {idx} ({cases[idx]['name']}) FAILED: {e}",
                      flush=True)
                results[idx] = {"name": cases[idx]["name"], "skipped": True,
                                 "error": str(e)}
            done += 1
            if done % 5 == 0:
                el = time.monotonic() - t0
                print(f"  {done}/{len(cases)} ({el/60:.1f}m)", flush=True)

    results = [r for r in results if r is not None]
    print(f"\nDone in {(time.monotonic() - t0)/60:.1f}m")

    out_path = RESULTS / "phase_6_1_smiles_results.json"
    with open(out_path, "w") as f:
        json.dump({"n": sum(1 for r in results if not r.get("skipped")),
                   "per_drug": results}, f, indent=2)

    valid = [r for r in results if not r.get("skipped")]
    n = len(valid)

    def hit_clust(rec, k):
        return hit_at_k_clustered(
            set(rec.get("gt_umls", [])),
            collapse_top_k(rec.get("hybrid_top10", [])), k,
        )

    print("\n" + "=" * 78)
    print(f"Phase 6.1 SMILES->TargetNet RESULTS — n={n}")
    print("=" * 78)

    from collections import Counter
    bp_sources = Counter(r.get("bp_source", "unknown") for r in valid)
    print(f"\nBinding-profile source:")
    for s, c in bp_sources.most_common():
        print(f"  {s}: {c}")

    print("\nAggregate (cluster-aware):")
    for k in (1, 3, 5, 10):
        h = sum(1 for r in valid if hit_clust(r, k))
        ci = bootstrap_ci(h, n)
        print(f"  hit@{k}: {h}/{n} ({h/n:.1%}) [CI {ci[0]:.0%}-{ci[1]:.0%}]")

    print("\nPer-TA (cluster-aware):")
    BASELINE_ONC_HIT10 = 1.00  # Phase 6.1 baseline
    for ta in ("Oncology", "Immunology", "Ophthalmology", "Neurology", "Other"):
        sub = [r for r in valid if r.get("therapeutic_area") == ta]
        if not sub:
            continue
        print(f"\n  {ta} (n={len(sub)})")
        for k in (1, 3, 5, 10):
            h = sum(1 for r in sub if hit_clust(r, k))
            ci = bootstrap_ci(h, len(sub))
            print(f"    hit@{k}: {h}/{len(sub)} ({h/len(sub):.1%}) "
                  f"[CI {ci[0]:.0%}-{ci[1]:.0%}]")
        if ta == "Oncology":
            onc_h10 = sum(1 for r in sub if hit_clust(r, 10)) / len(sub)
            drop_pp = (BASELINE_ONC_HIT10 - onc_h10) * 100
            gate = ("PASS" if drop_pp <= 5 else
                    "MARGINAL" if drop_pp <= 10 else "FAIL")
            print(f"\n  E2E GATE (Onc hit@10 vs Phase 6.1 baseline):")
            print(f"    Baseline: {BASELINE_ONC_HIT10:.1%}")
            print(f"    SMILES:   {onc_h10:.1%}")
            print(f"    Drop:     {drop_pp:.1f}pp")
            print(f"    Gate:     {gate} (PASS <=5pp, MARGINAL <=10pp, FAIL >10pp)")


if __name__ == "__main__":
    main()
