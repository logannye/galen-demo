"""Sprint 4 v2: blended SCM + lift-normalized scoring.

Combines:
  - Multi-source α (Sprint 4F: SIDER + CTD + AOP + OT + PharmGKB)
  - Lift normalization: score(S|drug) = NoisyOR(S|drug) / base_rate(S)

This should fix the rare-tox prediction failure: common side effects
(Nausea, Headache) get penalized by their high base rate; rare-but-
causally-specific predictions (Torsade given hERG-binding) get rewarded.

Runs both:
  - Sprint 1+2 ablation on n=200 held-out
  - Sprint 3 clinical-failure benchmark
"""
from __future__ import annotations

import json
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
from ..scm.lift_scoring import lift_top_k_predictions, load_base_rates
from ..scm.scoring import top_k_predictions


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def main() -> int:
    print("=" * 78)
    print("Sprint 4 v2: blended α + LIFT-normalized scoring")
    print("=" * 78)

    with open(RESULTS / "scm_edges_blended.json") as f:
        edges_blended = json.load(f)
    with open(RESULTS / "scm_edges.json") as f:
        edges_sider = json.load(f)
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    se_vocab = v["umls_ids"]
    base_rates = load_base_rates()
    print(f"[setup] blended α targets: {len(edges_blended)}; "
          f"base_rates: {len(base_rates)}")

    # ============ Sprint 1+2 ablation ============
    test_drugs = [d for d in cat["drugs"] if d["split"] == "test"]
    with open(RESULTS / "sprint1_per_drug.json") as f:
        s1 = json.load(f)
    s1_by_cid = {r["cid"]: r for r in s1["per_drug"]}

    print(f"\n--- Sprint 1+2 ablation (n={len(test_drugs)}) ---")
    rows = []
    for d in test_drugs:
        cid = d["cid"]
        gold = set(d["side_effects_in_vocab"])
        bp = d["binding_profile"]
        ranked_sider_noisy = top_k_predictions(bp, edges_sider, se_vocab, k=len(se_vocab))
        ranked_blended_noisy = top_k_predictions(bp, edges_blended, se_vocab, k=len(se_vocab))
        ranked_blended_lift = lift_top_k_predictions(
            bp, edges_blended, se_vocab, base_rates, k=len(se_vocab),
        )
        s1_rec = s1_by_cid.get(cid, {})
        rows.append({
            "cid": cid,
            "sider_noisyor": per_drug_metrics(ranked_sider_noisy, gold),
            "blended_noisyor": per_drug_metrics(ranked_blended_noisy, gold),
            "blended_lift": per_drug_metrics(ranked_blended_lift, gold),
            "llm_blind": s1_rec.get("llm_drug_blind_metrics", {}),
            "llm_name": s1_rec.get("llm_with_name_metrics", {}),
        })

    print(f"  {'method':<28s} {'MAP':<8s} {'P@10':<8s} {'P@20':<8s} {'R@10':<8s} {'MRR':<8s}")
    for arm_label, arm in [("SIDER-only noisy-OR", "sider_noisyor"),
                            ("Blended noisy-OR", "blended_noisyor"),
                            ("Blended + LIFT", "blended_lift"),
                            ("LLM-drug-blind", "llm_blind"),
                            ("LLM-with-name", "llm_name")]:
        if not rows[0].get(arm):
            continue
        map_ = statistics.mean(r[arm].get("ap", 0) for r in rows)
        p10 = statistics.mean(r[arm].get("p@10", 0) for r in rows)
        p20 = statistics.mean(r[arm].get("p@20", 0) for r in rows)
        r10 = statistics.mean(r[arm].get("r@10", 0) for r in rows)
        mrr = statistics.mean(r[arm].get("rr", 0) for r in rows)
        print(f"  {arm_label:<28s} {map_:<8.4f} {p10:<8.4f} {p20:<8.4f} {r10:<8.4f} {mrr:<8.4f}")

    # ============ Sprint 3 clinical failures ============
    print(f"\n--- Sprint 3 clinical-failure benchmark (n={len(CLINICAL_FAILURES)}) ---")

    # Prepare RF-ECFP (for comparison)
    train_drugs = [d for d in cat["drugs"] if d["split"] == "train"]
    smiles_map = fetch_smiles_for_drugs(cat["drugs"])
    X_train_ecfp, train_ecfp_idx = build_ecfp_matrix(train_drugs, smiles_map)
    Y_train = build_se_label_matrix(train_drugs, se_vocab)
    Y_train_ecfp = Y_train[train_ecfp_idx]
    print("[setup] Training RF-ECFP...")
    rf_models = train_rf_models(X_train_ecfp, Y_train_ecfp, n_estimators=50, max_depth=8)
    base_rate_array = np.clip(Y_train.mean(axis=0), 1e-6, 1.0 - 1e-6)

    drugs_by_name = {(d["drug_name"] or "").lower(): d for d in cat["drugs"]}
    cf_results: list[dict] = []
    for i, cf in enumerate(CLINICAL_FAILURES, start=1):
        bp = []
        molregno_resolved = None
        if cf.in_sider_catalog:
            d = drugs_by_name.get(cf.drug_id.lower()) or drugs_by_name.get(cf.drug_search_name.lower())
            if d:
                bp = d["binding_profile"]
                molregno_resolved = d["molregno"]
        else:
            molregno = lookup_chembl_molregno(cf.drug_search_name)
            if molregno:
                molregno_resolved = molregno
                conn = sqlite3.connect("/Volumes/Databank/databases/chembl_36.db")
                bp = query_binding_profile(conn, molregno)
                conn.close()
        if not bp:
            continue

        # 3 SCM modes
        sider_noisy = top_k_predictions(bp, edges_sider, se_vocab, k=len(se_vocab))
        blended_noisy = top_k_predictions(bp, edges_blended, se_vocab, k=len(se_vocab))
        blended_lift = lift_top_k_predictions(bp, edges_blended, se_vocab, base_rates, k=len(se_vocab))

        rec = {
            "drug_id": cf.drug_id,
            "causal_off_target": cf.causal_off_target_gene,
            "sider_noisyor_rank": best_causal_rank(sider_noisy, cf.causal_side_effects_umls),
            "blended_noisyor_rank": best_causal_rank(blended_noisy, cf.causal_side_effects_umls),
            "blended_lift_rank": best_causal_rank(blended_lift, cf.causal_side_effects_umls),
        }
        # RF-ECFP
        rf_r = None
        if molregno_resolved:
            smi = smiles_map.get(str(molregno_resolved)) or fetch_smiles(molregno_resolved)
            if smi:
                fp = compute_ecfp(smi)
                if fp is not None:
                    rf_ranked = rank_test_drug_rf(fp, rf_models, se_vocab, base_rate_array)
                    rf_r = best_causal_rank(rf_ranked, cf.causal_side_effects_umls)
        rec["rf_ecfp_rank"] = rf_r
        cf_results.append(rec)
        print(f"  [{i:2d}/{len(CLINICAL_FAILURES)}] {cf.drug_id:<24s} "
              f"sider={rec['sider_noisyor_rank']} "
              f"blend-noisy={rec['blended_noisyor_rank']} "
              f"blend-LIFT={rec['blended_lift_rank']} "
              f"RF={rec['rf_ecfp_rank']}")

    n = len(cf_results)
    def hits_at(key: str, k: int) -> int:
        return sum(1 for r in cf_results if r[key] is not None and r[key] <= k)

    print(f"\n=== Sprint 3 hit rate (n={n}) ===")
    print(f"  {'method':<26s} {'hit@3':<8s} {'hit@10':<8s} {'hit@20':<8s} {'hit@50':<8s}")
    for label, key in [("SCM: SIDER + noisy-OR", "sider_noisyor_rank"),
                         ("SCM: Blended + noisy-OR", "blended_noisyor_rank"),
                         ("SCM: Blended + LIFT", "blended_lift_rank"),
                         ("RF-ECFP", "rf_ecfp_rank")]:
        print(f"  {label:<26s} {hits_at(key, 3)}/{n:<7d} {hits_at(key, 10)}/{n:<7d} "
              f"{hits_at(key, 20)}/{n:<7d} {hits_at(key, 50)}/{n:<7d}")

    # Reference: Sprint 3 LLM numbers
    sprint3_path = RESULTS / "sprint3_clinical_failures.json"
    if sprint3_path.exists():
        with open(sprint3_path) as f:
            s3 = json.load(f)
        s3_results = s3["per_case"]
        n3 = len(s3_results)
        def hits_s3(key: str, k: int) -> int:
            return sum(1 for r in s3_results if r[key] is not None and r[key] <= k)
        print(f"  {'LLM-drug-blind (Sprint3)':<26s} {hits_s3('llm_drug_blind_rank', 3)}/{n3:<7d} "
              f"{hits_s3('llm_drug_blind_rank', 10)}/{n3:<7d} "
              f"{hits_s3('llm_drug_blind_rank', 20)}/{n3:<7d} "
              f"{hits_s3('llm_drug_blind_rank', 50)}/{n3:<7d}")
        print(f"  {'LLM-with-name (Sprint3)':<26s} {hits_s3('llm_with_name_rank', 3)}/{n3:<7d} "
              f"{hits_s3('llm_with_name_rank', 10)}/{n3:<7d} "
              f"{hits_s3('llm_with_name_rank', 20)}/{n3:<7d} "
              f"{hits_s3('llm_with_name_rank', 50)}/{n3:<7d}")

    # Save
    out = {
        "n_test_sprint12": len(rows),
        "n_cf": n,
        "ablation_sprint12": rows,
        "clinical_failures": cf_results,
        "hits": {
            label: {f"at_{k}": hits_at(key, k) for k in (3, 5, 10, 20, 50)}
            for label, key in [
                ("sider_noisyor", "sider_noisyor_rank"),
                ("blended_noisyor", "blended_noisyor_rank"),
                ("blended_lift", "blended_lift_rank"),
                ("rf_ecfp", "rf_ecfp_rank"),
            ]
        },
    }
    with open(RESULTS / "sprint4_v2_lift_evaluation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {RESULTS / 'sprint4_v2_lift_evaluation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
