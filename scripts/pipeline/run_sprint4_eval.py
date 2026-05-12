"""Sprint 4G+H: re-evaluate SCM with multi-source-blended α(S|T).

We test:
  1. Sprint 1+2 ablation: SCM (blended) on the n=200 held-out test set,
     comparing to RF-ECFP, LLM-drug-blind, LLM-with-name (cached from
     Sprint 1).
  2. Sprint 3 clinical-failure benchmark: SCM (blended) on the 15 FDA-
     withdrawn drugs. Did the blended SCM finally catch hERG → Torsade,
     PTGS2 → MI, etc.?

This runner uses the BLENDED α(S|T) (from scripts.scm.multi_source_edges)
which incorporates SIDER + CTD + AOP-Wiki + OpenTargets + PharmGKB.
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
import sys
from pathlib import Path

import numpy as np

from ..baselines.rf_ecfp_baseline import (
    build_ecfp_matrix, compute_ecfp, fetch_smiles_for_drugs,
    rank_test_drug_rf, train_rf_models,
)
from ..baselines.logreg_baseline import build_se_label_matrix
from ..data.build_catalog import query_binding_profile
from ..data.clinical_failures import CLINICAL_FAILURES
from ..evaluation.metrics import per_drug_metrics
from ..pipeline.run_sprint3_clinical_failures import (
    best_causal_rank, fetch_smiles, lookup_chembl_molregno,
)
from ..scm.scoring import top_k_predictions


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
TOP_K_RANK = 50


def load_blended_edges() -> dict[str, dict[str, float]]:
    """Load multi-source blended α(S|T)."""
    with open(RESULTS / "scm_edges_blended.json") as f:
        return json.load(f)


def main() -> int:
    print("=" * 78)
    print("Sprint 4 evaluation: blended SCM on Sprint 1+2 + Sprint 3 benchmarks")
    print("=" * 78)

    edges_blended = load_blended_edges()
    print(f"[load] blended α: {len(edges_blended)} targets")

    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    with open(RESULTS / "side_effect_vocab.json") as f:
        vocab_payload = json.load(f)
    se_vocab = vocab_payload["umls_ids"]

    # ============= Sprint 1+2 ablation re-run =============
    test_drugs = [d for d in cat["drugs"] if d["split"] == "test"]
    print(f"\n--- Sprint 1+2 ablation re-run (n={len(test_drugs)}) ---")
    with open(RESULTS / "sprint1_per_drug.json") as f:
        sprint1 = json.load(f)
    s1_by_cid = {r["cid"]: r for r in sprint1["per_drug"]}

    blended_metrics: list[dict] = []
    for d in test_drugs:
        cid = d["cid"]
        gold = set(d["side_effects_in_vocab"])
        bp = d["binding_profile"]
        ranked = top_k_predictions(bp, edges_blended, se_vocab, k=len(se_vocab))
        m = per_drug_metrics(ranked, gold)
        blended_metrics.append({"cid": cid, "metrics": m})

    # Compare to Sprint 1 SCM (SIDER-only)
    pairs = []
    for r in blended_metrics:
        s1 = s1_by_cid.get(r["cid"])
        if s1:
            pairs.append((s1["scm_metrics"], r["metrics"], s1["llm_drug_blind_metrics"]))
    n = len(pairs)
    map_sider = statistics.mean(p[0]["ap"] for p in pairs)
    map_blend = statistics.mean(p[1]["ap"] for p in pairs)
    map_llm_b = statistics.mean(p[2]["ap"] for p in pairs)
    print(f"  SIDER-only α       MAP: {map_sider:.4f}")
    print(f"  Blended α          MAP: {map_blend:.4f}  (diff +{map_blend-map_sider:+.4f})")
    print(f"  LLM-drug-blind     MAP: {map_llm_b:.4f}")
    p10_sider = statistics.mean(p[0]["p@10"] for p in pairs)
    p10_blend = statistics.mean(p[1]["p@10"] for p in pairs)
    p10_llm_b = statistics.mean(p[2]["p@10"] for p in pairs)
    print(f"  SIDER-only α      P@10: {p10_sider:.4f}")
    print(f"  Blended α         P@10: {p10_blend:.4f}  (diff +{p10_blend-p10_sider:+.4f})")
    print(f"  LLM-drug-blind    P@10: {p10_llm_b:.4f}")

    # ============= Sprint 3 clinical-failure benchmark =============
    print(f"\n--- Sprint 3 clinical-failure benchmark re-run (n={len(CLINICAL_FAILURES)}) ---")

    # Pre-train RF-ECFP for comparison (same setup as Sprint 3)
    train_drugs = [d for d in cat["drugs"] if d["split"] == "train"]
    smiles_map = fetch_smiles_for_drugs(cat["drugs"])
    X_train_ecfp, train_ecfp_idx = build_ecfp_matrix(train_drugs, smiles_map)
    Y_train = build_se_label_matrix(train_drugs, se_vocab)
    Y_train_ecfp = Y_train[train_ecfp_idx]
    print("[setup] Training RF-ECFP on training set...")
    rf_models = train_rf_models(X_train_ecfp, Y_train_ecfp, n_estimators=50, max_depth=8)
    base_rate_array = np.clip(Y_train.mean(axis=0), 1e-6, 1.0 - 1e-6)

    drugs_by_name = {(d["drug_name"] or "").lower(): d for d in cat["drugs"]}
    results: list[dict] = []
    for i, cf in enumerate(CLINICAL_FAILURES, start=1):
        # Resolve binding profile
        binding_profile = []
        molregno_resolved = None
        if cf.in_sider_catalog:
            d = drugs_by_name.get(cf.drug_id.lower()) or drugs_by_name.get(cf.drug_search_name.lower())
            if d:
                binding_profile = d["binding_profile"]
                molregno_resolved = d["molregno"]
        else:
            molregno = lookup_chembl_molregno(cf.drug_search_name)
            if molregno:
                molregno_resolved = molregno
                conn = sqlite3.connect("/Volumes/Databank/databases/chembl_36.db")
                binding_profile = query_binding_profile(conn, molregno)
                conn.close()
        if not binding_profile:
            print(f"  [{i:2d}/{len(CLINICAL_FAILURES)}] {cf.drug_id}: SKIP (no binding)")
            continue

        # Blended SCM scoring
        ranked_blend = top_k_predictions(
            binding_profile, edges_blended, se_vocab, k=len(se_vocab),
        )
        rank_blend = best_causal_rank(ranked_blend, cf.causal_side_effects_umls)

        # RF-ECFP
        rf_rank = None
        if molregno_resolved is not None:
            smi = smiles_map.get(str(molregno_resolved)) or fetch_smiles(molregno_resolved)
            if smi:
                fp = compute_ecfp(smi)
                if fp is not None:
                    rf_ranked = rank_test_drug_rf(fp, rf_models, se_vocab, base_rate_array)
                    rf_rank = best_causal_rank(rf_ranked, cf.causal_side_effects_umls)

        results.append({
            "drug_id": cf.drug_id,
            "causal_off_target": cf.causal_off_target_gene,
            "causal_se": cf.causal_side_effects_display,
            "scm_blended_rank": rank_blend,
            "rf_ecfp_rank": rf_rank,
        })
        print(f"  [{i:2d}/{len(CLINICAL_FAILURES)}] {cf.drug_id:<24s} "
              f"causal={cf.causal_off_target_gene:<8s} "
              f"SCM-blend rank={rank_blend}  RF-ECFP rank={rf_rank}")

    # Summary
    n = len(results)
    def hits_at(rank_key: str, k: int) -> int:
        return sum(1 for r in results if r[rank_key] is not None and r[rank_key] <= k)

    print(f"\n=== Sprint 3 hit rate (n={n}) ===")
    print(f"  {'method':<22s} {'hit@3':<8s} {'hit@10':<8s} {'hit@20':<8s} {'hit@50':<8s}")
    for label, key in [("SCM (blended)", "scm_blended_rank"),
                         ("RF-ECFP", "rf_ecfp_rank")]:
        print(f"  {label:<22s} {hits_at(key,3)}/{n:<7d} {hits_at(key,10)}/{n:<7d} "
              f"{hits_at(key,20)}/{n:<7d} {hits_at(key,50)}/{n:<7d}")

    # Load Sprint 3 SIDER-only baseline for comparison
    sprint3_path = RESULTS / "sprint3_clinical_failures.json"
    if sprint3_path.exists():
        with open(sprint3_path) as f:
            s3 = json.load(f)
        sider_results = s3["per_case"]
        print(f"\n=== Comparison to SIDER-only SCM (Sprint 3) ===")
        print(f"  {'method':<22s} {'hit@3':<8s} {'hit@10':<8s} {'hit@20':<8s} {'hit@50':<8s}")
        def hits_at_s3(rank_key, k):
            return sum(1 for r in sider_results if r[rank_key] is not None and r[rank_key] <= k)
        n3 = len(sider_results)
        print(f"  {'SCM (SIDER-only)':<22s} {hits_at_s3('scm_rank',3)}/{n3:<7d} "
              f"{hits_at_s3('scm_rank',10)}/{n3:<7d} {hits_at_s3('scm_rank',20)}/{n3:<7d} "
              f"{hits_at_s3('scm_rank',50)}/{n3:<7d}")
        print(f"  {'LLM-drug-blind':<22s} {hits_at_s3('llm_drug_blind_rank',3)}/{n3:<7d} "
              f"{hits_at_s3('llm_drug_blind_rank',10)}/{n3:<7d} "
              f"{hits_at_s3('llm_drug_blind_rank',20)}/{n3:<7d} "
              f"{hits_at_s3('llm_drug_blind_rank',50)}/{n3:<7d}")
        print(f"  {'LLM-with-name':<22s} {hits_at_s3('llm_with_name_rank',3)}/{n3:<7d} "
              f"{hits_at_s3('llm_with_name_rank',10)}/{n3:<7d} "
              f"{hits_at_s3('llm_with_name_rank',20)}/{n3:<7d} "
              f"{hits_at_s3('llm_with_name_rank',50)}/{n3:<7d}")

    # Save
    out = {
        "n_test_drugs_sprint12": len(blended_metrics),
        "n_clinical_failures": n,
        "blended_sprint12_summary": {
            "blended_map": map_blend,
            "sider_only_map": map_sider,
            "llm_drug_blind_map": map_llm_b,
            "blended_p10": p10_blend,
            "sider_only_p10": p10_sider,
            "llm_drug_blind_p10": p10_llm_b,
        },
        "clinical_failure_results": results,
        "hits_summary": {
            "scm_blended": {f"at_{k}": hits_at("scm_blended_rank", k)
                              for k in (3, 5, 10, 20, 50)},
            "rf_ecfp": {f"at_{k}": hits_at("rf_ecfp_rank", k)
                          for k in (3, 5, 10, 20, 50)},
        },
    }
    out_path = RESULTS / "sprint4_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
