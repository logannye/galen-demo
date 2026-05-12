"""Sprint 7E: stratified re-run of clinical-safety benchmark with Sprint 7
enhancements.

Sprint 7 enhancements vs Sprint 5:
  - Expanded vocab: 500 → 605 UMLS terms (105 new onc/immuno AEs)
  - Multi-source α: re-ingested on expanded vocab (504k → 505k edges)
  - 6th source: curated class-effect priors (124 high-confidence pairs)
  - TA-conditional LLM hybrid prompt (Onc / Immuno / CV-Metabolic / CNS)

Stratifies cases by therapeutic area and compares Sprint 7 vs Sprint 5
performance (Sprint 5 result: 47% hit@3, 57% hit@10).
"""
from __future__ import annotations

import json
import math
import sqlite3
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..baselines.llm_baselines import (
    rank_side_effects_llm_drug_blind, rank_side_effects_llm_with_name,
)
from ..baselines.llm_hybrid_reranker import hybrid_rerank
from ..baselines.rf_ecfp_baseline import (
    build_ecfp_matrix, compute_ecfp, fetch_smiles_for_drugs,
    rank_test_drug_rf, train_rf_models,
)
from ..baselines.logreg_baseline import build_se_label_matrix
from ..data.build_catalog import query_binding_profile
from ..data.clinical_safety_benchmark import SAFETY_CASES, passes_eligibility
from ..demo.scm_explainer import explain_predictions
from ..llm import SonnetClient
from ..pipeline.run_sprint3_clinical_failures import (
    best_causal_rank, fetch_smiles, lookup_chembl_molregno,
)
from ..scm.scoring import score_drug_side_effects


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
N_WORKERS = 16
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"


# Onc/Immuno classification by drug_id (manual triage of Sprint 5 cases)
TA_MAP: dict[str, str] = {
    # Oncology
    "troglitazone": "Other",   # actually metabolic
    "rosiglitazone": "Other",
    "rofecoxib": "Other",
    "valdecoxib": "Other",
    "doxorubicin": "Oncology",
    "daunorubicin": "Oncology",
    "trastuzumab_card": "Oncology",
    "sunitinib_card": "Oncology",
    "sorafenib_htn": "Oncology",
    "pazopanib_card": "Oncology",
    "cisplatin_renal": "Oncology",
    "lenalidomide": "Oncology",
    "thalidomide": "Oncology",
    "bortezomib": "Oncology",
    # Immunology
    "adalimumab_methotrexate_ra": "Immunology",
    "etanercept_methotrexate_ra": "Immunology",
    "infliximab_methotrexate_ra": "Immunology",
    "tocilizumab_methotrexate_ra": "Immunology",
    "tofacitinib_uc": "Immunology",
    "baricitinib_ra": "Immunology",
    "upadacitinib_ra": "Immunology",
    "ustekinumab_crohns": "Immunology",
    "risankizumab_crohns": "Immunology",
    "vedolizumab_uc": "Immunology",
    "secukinumab_psoriatic": "Immunology",
    "ixekizumab_psoriatic": "Immunology",
    "rituximab_pml": "Immunology",
    "rituximab_aav": "Immunology",
    "methotrexate_hep": "Immunology",
    "anifrolumab_sle": "Immunology",
    "dupilumab_eczema": "Immunology",
    # CV / metabolic / CNS / etc. default below
}


def classify_ta(drug_id: str, severity: str, causal_target_gene: str) -> str:
    """Classify a safety case by therapeutic area for TA-conditional prompt."""
    explicit = TA_MAP.get(drug_id)
    if explicit:
        return explicit
    name_lower = drug_id.lower()
    # Heuristics
    onc_genes = {"ERBB2", "TOP2A", "TOP2B", "KDR", "PDGFRB", "ALK",
                 "EGFR", "BRAF", "MAP2K1", "CDK4", "CDK6", "PARP1",
                 "BCL2", "MTOR", "PIK3CA", "PIK3CG", "ABL1", "KIT",
                 "MET", "FLT1", "PDCD1", "CD274", "CTLA4", "LAG3"}
    immuno_genes = {"TNF", "IL6R", "IL17A", "IL12B", "IL23A", "IL5",
                     "IL5RA", "IL4R", "IL1B", "JAK1", "JAK2", "JAK3",
                     "S1PR1", "ITGA4", "ITGB7", "C5", "MS4A1", "CD22",
                     "CD19", "TNFRSF17"}
    if causal_target_gene.upper() in onc_genes:
        return "Oncology"
    if causal_target_gene.upper() in immuno_genes:
        return "Immunology"
    # CV
    cv_keys = ("hf", "cad", "mcrpc", "ahus", "afib", "obesity", "diabetes")
    if any(k in name_lower for k in cv_keys):
        return "Cardiovascular & metabolic"
    return "Other"


def _process_safety_case(
    cf, edges: dict, target_info: dict, vocab_payload: dict,
    se_vocab: list[str], se_names: dict,
    drugs_by_name: dict, smiles_map: dict,
    rf_models: dict, base_rate_array: np.ndarray,
    client: SonnetClient,
) -> dict:
    # Resolve binding profile
    bp = []
    molregno_resolved = None
    cat_drug = drugs_by_name.get(cf.drug_search_name.lower())
    if cat_drug is not None:
        bp = cat_drug["binding_profile"]
        molregno_resolved = cat_drug["molregno"]
    else:
        molregno = lookup_chembl_molregno(cf.drug_search_name)
        if molregno:
            molregno_resolved = molregno
            conn = sqlite3.connect(CHEMBL_DB)
            bp = query_binding_profile(conn, molregno)
            conn.close()
    if not bp:
        return {
            "drug_id": cf.drug_id, "drug_search_name": cf.drug_search_name,
            "skipped": True, "reason": "no binding profile",
        }

    ta = classify_ta(cf.drug_id, cf.severity, cf.causal_off_target_gene)

    # SCM
    scored = score_drug_side_effects(bp, edges, se_vocab)
    explanations = explain_predictions(
        scored[:100], bp, edges, target_info, se_names,
        top_k_se=100, top_k_targets=5,
    )
    scm_ranked = [s for s, _ in scored]
    scm_rank = best_causal_rank(scm_ranked, cf.causal_side_effects_umls)

    # Sprint 7 hybrid (TA-conditional)
    hybrid = hybrid_rerank(
        bp, scored, explanations, vocab_payload, client=client,
        top_k_scm_candidates=100, therapeutic_area=ta,
    )
    hybrid_rank = best_causal_rank(hybrid.ranked_side_effects, cf.causal_side_effects_umls)

    # LLM-drug-blind for comparison
    llm_blind = rank_side_effects_llm_drug_blind(
        bp, vocab_payload, client=client, top_k=50,
    )
    llm_blind_rank = best_causal_rank(
        llm_blind.ranked_side_effects, cf.causal_side_effects_umls,
    )

    # LLM-with-name baseline
    llm_name = rank_side_effects_llm_with_name(
        cf.drug_search_name, bp, vocab_payload, client=client, top_k=50,
    )
    llm_name_rank = best_causal_rank(
        llm_name.ranked_side_effects, cf.causal_side_effects_umls,
    )

    # RF-ECFP
    rf_rank = None
    if molregno_resolved is not None:
        smi = smiles_map.get(str(molregno_resolved)) or fetch_smiles(molregno_resolved)
        if smi:
            fp = compute_ecfp(smi)
            if fp is not None:
                rf_ranked = rank_test_drug_rf(fp, rf_models, se_vocab, base_rate_array)
                rf_rank = best_causal_rank(rf_ranked, cf.causal_side_effects_umls)

    return {
        "drug_id": cf.drug_id,
        "drug_search_name": cf.drug_search_name,
        "severity": cf.severity,
        "therapeutic_area": ta,
        "causal_off_target": cf.causal_off_target_gene,
        "causal_se": cf.causal_side_effects_display,
        "n_binding_targets": len(bp),
        "scm_rank": scm_rank,
        "hybrid_rank": hybrid_rank,
        "llm_drug_blind_rank": llm_blind_rank,
        "llm_with_name_rank": llm_name_rank,
        "rf_ecfp_rank": rf_rank,
        "hybrid_top10": hybrid.ranked_side_effects[:10],
    }


def _mcnemar_one_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    p = 0.0
    for x in range(b, n + 1):
        p += math.comb(n, x) * (0.5 ** n)
    return p


def main() -> int:
    print("=" * 78)
    print("Sprint 7E: stratified clinical-safety benchmark with onc/immuno enhancements")
    print("=" * 78)

    with open(RESULTS / "scm_edges_blended.json") as f:
        edges = json.load(f)
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    with open(RESULTS / "side_effect_vocab.json") as f:
        vocab_payload = json.load(f)
    se_vocab = vocab_payload["umls_ids"]
    se_names = vocab_payload["display_names"]
    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    target_info = {t["uniprot"]: t for t in tv["targets"]}

    target_set = set(target_info.keys())
    vocab_set = set(se_vocab)
    eligible = [c for c in SAFETY_CASES if passes_eligibility(c, vocab_set, target_set)]
    print(f"[setup] vocab={len(se_vocab)}; edges={len(edges)} targets; "
          f"eligible cases={len(eligible)}")

    # RF-ECFP
    train_drugs = [d for d in cat["drugs"] if d["split"] == "train"]
    smiles_map = fetch_smiles_for_drugs(cat["drugs"])
    X_train_ecfp, train_ecfp_idx = build_ecfp_matrix(train_drugs, smiles_map)
    Y_train = build_se_label_matrix(train_drugs, se_vocab)
    Y_train_ecfp = Y_train[train_ecfp_idx]
    print("[setup] Training RF-ECFP on expanded vocab...")
    rf_models = train_rf_models(X_train_ecfp, Y_train_ecfp,
                                  n_estimators=50, max_depth=8)
    base_rate_array = np.clip(Y_train.mean(axis=0), 1e-6, 1.0 - 1e-6)

    client = SonnetClient()
    drugs_by_name = {(d["drug_name"] or "").lower(): d for d in cat["drugs"]}

    results: list[dict] = [None] * len(eligible)
    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(_process_safety_case, cf, edges, target_info,
                       vocab_payload, se_vocab, se_names,
                       drugs_by_name, smiles_map, rf_models,
                       base_rate_array, client): i
            for i, cf in enumerate(eligible)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"[safety] case {idx} FAILED: {e}", flush=True)
            done += 1
            if done % 10 == 0:
                elapsed = time.monotonic() - t_start
                print(f"  [safety] {done}/{len(eligible)} ({elapsed/60:.1f}m)",
                      flush=True)
    results = [r for r in results if r is not None and not r.get("skipped")]
    print(f"[safety] done in {(time.monotonic() - t_start)/60:.1f}m, n={len(results)}")

    with open(RESULTS / "sprint7_safety.json", "w") as f:
        json.dump({"n_cases": len(results), "per_drug": results}, f, indent=2)

    # ============= Aggregate + stratified analysis =============
    print("\n" + "=" * 78)
    print(f"Sprint 7 RESULTS (n={len(results)})")
    print("=" * 78)

    def hit(records: list[dict], key: str, k: int) -> int:
        return sum(1 for r in records if r[key] is not None and r[key] <= k)

    n = len(results)
    print(f"\n--- Overall hit rates ---")
    print(f"  {'method':<22s} {'hit@3':<10s} {'hit@10':<10s} {'hit@20':<10s} {'hit@50':<10s}")
    for label, key in [("SCM (blended Sprint 7)", "scm_rank"),
                         ("RF-ECFP", "rf_ecfp_rank"),
                         ("LLM-drug-blind", "llm_drug_blind_rank"),
                         ("LLM-with-name", "llm_with_name_rank"),
                         ("Hybrid SCM+LLM (S7)", "hybrid_rank")]:
        h3, h10, h20, h50 = hit(results, key, 3), hit(results, key, 10), hit(results, key, 20), hit(results, key, 50)
        print(f"  {label:<22s} {h3}/{n} ({h3/n:.0%})   "
              f"{h10}/{n} ({h10/n:.0%})  {h20}/{n} ({h20/n:.0%})  {h50}/{n} ({h50/n:.0%})")

    # McNemar test: Sprint 7 Hybrid > LLM-drug-blind on hit@10
    b = sum(1 for r in results
            if (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
            and not (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    c = sum(1 for r in results
            if not (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
            and (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    p_one = _mcnemar_one_sided(b, c)
    print(f"\nMcNemar Hybrid > LLM-drug-blind hit@10: b={b} c={c} p={p_one:.4g}")

    # Stratified by TA
    print(f"\n--- Stratified by therapeutic area ---")
    for ta in ("Oncology", "Immunology", "Cardiovascular & metabolic", "CNS & psychiatry", "Other"):
        sub = [r for r in results if r["therapeutic_area"] == ta]
        if not sub:
            continue
        print(f"\n  [{ta}]  n={len(sub)}")
        print(f"    {'method':<22s} {'hit@3':<10s} {'hit@10':<10s} {'hit@20':<10s}")
        for label, key in [("SCM (S7)", "scm_rank"),
                             ("LLM-drug-blind", "llm_drug_blind_rank"),
                             ("LLM-with-name", "llm_with_name_rank"),
                             ("Hybrid (S7)", "hybrid_rank")]:
            h3 = hit(sub, key, 3); h10 = hit(sub, key, 10); h20 = hit(sub, key, 20)
            print(f"    {label:<22s} {h3}/{len(sub)} ({h3/len(sub):.0%})   "
                  f"{h10}/{len(sub)} ({h10/len(sub):.0%})  "
                  f"{h20}/{len(sub)} ({h20/len(sub):.0%})")

    # Save summary
    summary = {
        "n_cases": n,
        "overall_hits": {
            label: {f"at_{k}": hit(results, key, k) for k in (3, 5, 10, 20, 50)}
            for label, key in [
                ("scm", "scm_rank"),
                ("hybrid", "hybrid_rank"),
                ("llm_drug_blind", "llm_drug_blind_rank"),
                ("llm_with_name", "llm_with_name_rank"),
                ("rf_ecfp", "rf_ecfp_rank"),
            ]
        },
        "mcnemar_hybrid_vs_llm_blind_hit10_p": p_one,
        "mcnemar_b_hybrid_hit_llm_miss": b,
        "mcnemar_c_hybrid_miss_llm_hit": c,
    }
    with open(RESULTS / "sprint7_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {RESULTS / 'sprint7_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
