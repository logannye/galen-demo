"""Sprint 5: hybrid SCM+LLM re-ranking evaluation (full).

Runs the LLM-augmented hybrid re-ranker on:
  - Sprint 1+2 ablation (n=200, common ADR prediction)
  - Sprint 5 expanded clinical-safety benchmark (n≈57, rare/causal tox)

For each test case:
  1. SCM (blended) scoring → top-100 candidates with per-target attribution
  2. LLM-drug-blind re-ranker (the new hybrid arm)
  3. Comparison vs SCM-Blended, LLM-drug-blind, LLM-with-name baselines

Parallelized via ThreadPoolExecutor(16 workers).
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
from ..evaluation.metrics import per_drug_metrics
from ..llm import SonnetClient
from ..pipeline.run_sprint3_clinical_failures import (
    best_causal_rank, fetch_smiles, lookup_chembl_molregno,
)
from ..scm.scoring import score_drug_side_effects, top_k_predictions


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
N_WORKERS = 16
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"


def _mcnemar_one_sided(b: int, c: int) -> float:
    """Exact binomial McNemar one-sided p(A > B) given (a-hit-b-miss=b, a-miss-b-hit=c)."""
    n = b + c
    if n == 0:
        return 1.0
    p = 0.0
    for x in range(b, n + 1):
        p += math.comb(n, x) * (0.5 ** n)
    return p


def _wilcoxon_one_sided(diffs: list[float]) -> tuple[int, float, float]:
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


def _process_sprint12_case(
    drug: dict, edges: dict, target_info: dict,
    vocab_payload: dict, se_vocab: list[str], se_names: dict,
    client: SonnetClient,
) -> dict:
    cid = drug["cid"]
    gold = set(drug["side_effects_in_vocab"])
    bp = drug["binding_profile"]
    scored = score_drug_side_effects(bp, edges, se_vocab)
    explanations = explain_predictions(
        scored[:100], bp, edges, target_info, se_names,
        gold_set=gold, top_k_se=100, top_k_targets=5,
    )
    hybrid = hybrid_rerank(bp, scored, explanations, vocab_payload, client=client)
    scm_blended = [s for s, _ in scored]
    return {
        "cid": cid,
        "drug_name": drug["drug_name"],
        "gold_size": len(gold),
        "scm_blended_metrics": per_drug_metrics(scm_blended, gold),
        "hybrid_metrics": per_drug_metrics(hybrid.ranked_side_effects, gold),
        "hybrid_top10": hybrid.ranked_side_effects[:10],
        "hybrid_confidence": hybrid.confidence,
    }


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

    # SCM (blended) scoring
    scored = score_drug_side_effects(bp, edges, se_vocab)
    explanations = explain_predictions(
        scored[:100], bp, edges, target_info, se_names,
        top_k_se=100, top_k_targets=5,
    )
    scm_ranked = [s for s, _ in scored]
    scm_rank = best_causal_rank(scm_ranked, cf.causal_side_effects_umls)

    # Hybrid
    hybrid = hybrid_rerank(bp, scored, explanations, vocab_payload, client=client)
    hybrid_rank = best_causal_rank(hybrid.ranked_side_effects, cf.causal_side_effects_umls)

    # LLM-drug-blind (fresh call)
    llm_blind = rank_side_effects_llm_drug_blind(
        bp, vocab_payload, client=client, top_k=50,
    )
    llm_blind_rank = best_causal_rank(
        llm_blind.ranked_side_effects, cf.causal_side_effects_umls,
    )

    # LLM-with-name (fresh call)
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
        "causal_off_target": cf.causal_off_target_gene,
        "causal_se": cf.causal_side_effects_display,
        "n_binding_targets": len(bp),
        "scm_blended_rank": scm_rank,
        "hybrid_rank": hybrid_rank,
        "llm_drug_blind_rank": llm_blind_rank,
        "llm_with_name_rank": llm_name_rank,
        "rf_ecfp_rank": rf_rank,
        "hybrid_top10": hybrid.ranked_side_effects[:10],
    }


def main() -> int:
    print("=" * 78)
    print("Sprint 5: hybrid SCM+LLM re-ranking evaluation")
    print("=" * 78)

    # Load inputs
    with open(RESULTS / "scm_edges_blended.json") as f:
        edges = json.load(f)
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    with open(RESULTS / "side_effect_vocab.json") as f:
        vocab_payload = json.load(f)
    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    target_info = {t["uniprot"]: t for t in tv["targets"]}
    se_vocab = vocab_payload["umls_ids"]
    se_names = vocab_payload["display_names"]
    print(f"[setup] blended α: {len(edges)} targets; vocab: {len(se_vocab)} SEs")

    # Sprint 1 LLM baselines (cached)
    with open(RESULTS / "sprint1_per_drug.json") as f:
        sprint1 = json.load(f)
    s1_by_cid = {r["cid"]: r for r in sprint1["per_drug"]}

    client = SonnetClient()

    # ============= Sprint 1+2 hybrid =============
    test_drugs = [d for d in cat["drugs"] if d["split"] == "test"]
    print(f"\n--- Sprint 1+2 hybrid (n={len(test_drugs)}, parallel={N_WORKERS}) ---")
    t_start = time.monotonic()
    s12_results: list[dict] = [None] * len(test_drugs)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(_process_sprint12_case, d, edges, target_info,
                      vocab_payload, se_vocab, se_names, client): i
            for i, d in enumerate(test_drugs)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                s12_results[idx] = fut.result()
            except Exception as e:
                print(f"[s12] case {idx} FAILED: {e}", flush=True)
            done += 1
            if done % 20 == 0:
                elapsed = time.monotonic() - t_start
                print(f"  [s12] {done}/{len(test_drugs)} ({elapsed/60:.1f}m)", flush=True)
    s12_results = [r for r in s12_results if r is not None]
    print(f"[s12] done in {(time.monotonic() - t_start)/60:.1f}m")

    # ============= Sprint 5 clinical safety =============
    target_set = set(target_info.keys())
    vocab_set = set(se_vocab)
    eligible = [c for c in SAFETY_CASES if passes_eligibility(c, vocab_set, target_set)]
    print(f"\n--- Sprint 5 clinical-safety (n={len(eligible)} eligible, parallel={N_WORKERS}) ---")

    # Prep RF-ECFP
    train_drugs = [d for d in cat["drugs"] if d["split"] == "train"]
    smiles_map = fetch_smiles_for_drugs(cat["drugs"])
    X_train_ecfp, train_ecfp_idx = build_ecfp_matrix(train_drugs, smiles_map)
    Y_train = build_se_label_matrix(train_drugs, se_vocab)
    Y_train_ecfp = Y_train[train_ecfp_idx]
    print("[setup] Training RF-ECFP...")
    rf_models = train_rf_models(X_train_ecfp, Y_train_ecfp, n_estimators=50, max_depth=8)
    base_rate_array = np.clip(Y_train.mean(axis=0), 1e-6, 1.0 - 1e-6)

    drugs_by_name = {(d["drug_name"] or "").lower(): d for d in cat["drugs"]}
    t_start = time.monotonic()
    safety_results: list[dict] = [None] * len(eligible)
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
                safety_results[idx] = fut.result()
            except Exception as e:
                print(f"[safety] case {idx} FAILED: {e}", flush=True)
            done += 1
            if done % 10 == 0:
                elapsed = time.monotonic() - t_start
                print(f"  [safety] {done}/{len(eligible)} ({elapsed/60:.1f}m)", flush=True)
    safety_results = [r for r in safety_results if r is not None and not r.get("skipped")]
    print(f"[safety] done in {(time.monotonic() - t_start)/60:.1f}m")

    # Save results
    with open(RESULTS / "sprint5_sprint12.json", "w") as f:
        json.dump({"n_cases": len(s12_results), "per_drug": s12_results}, f, indent=2)
    with open(RESULTS / "sprint5_safety.json", "w") as f:
        json.dump({"n_cases": len(safety_results), "per_drug": safety_results}, f, indent=2)

    # ============= Aggregate analysis =============
    print("\n" + "=" * 78)
    print(f"Sprint 5 RESULTS")
    print("=" * 78)

    # Sprint 1+2 MAP comparison
    print(f"\n--- Sprint 1+2 ablation (n={len(s12_results)}) ---")
    scm_maps = [r["scm_blended_metrics"]["ap"] for r in s12_results]
    hyb_maps = [r["hybrid_metrics"]["ap"] for r in s12_results]
    llm_blind_maps = []
    llm_name_maps = []
    for r in s12_results:
        s1 = s1_by_cid.get(r["cid"], {})
        llm_blind_maps.append(s1.get("llm_drug_blind_metrics", {}).get("ap", 0))
        llm_name_maps.append(s1.get("llm_with_name_metrics", {}).get("ap", 0))
    print(f"  {'method':<22s} {'MAP':<8s} {'P@10':<8s} {'P@20':<8s} {'R@10':<8s}")
    print(f"  {'SCM-Blended':<22s} {statistics.mean(scm_maps):<8.4f} "
          f"{statistics.mean(r['scm_blended_metrics']['p@10'] for r in s12_results):<8.4f} "
          f"{statistics.mean(r['scm_blended_metrics']['p@20'] for r in s12_results):<8.4f} "
          f"{statistics.mean(r['scm_blended_metrics']['r@10'] for r in s12_results):<8.4f}")
    print(f"  {'Hybrid SCM+LLM':<22s} {statistics.mean(hyb_maps):<8.4f} "
          f"{statistics.mean(r['hybrid_metrics']['p@10'] for r in s12_results):<8.4f} "
          f"{statistics.mean(r['hybrid_metrics']['p@20'] for r in s12_results):<8.4f} "
          f"{statistics.mean(r['hybrid_metrics']['r@10'] for r in s12_results):<8.4f}")
    print(f"  {'LLM-drug-blind':<22s} {statistics.mean(llm_blind_maps):<8.4f}")
    print(f"  {'LLM-with-name':<22s} {statistics.mean(llm_name_maps):<8.4f}")

    diffs = [h - s for h, s in zip(hyb_maps, scm_maps)]
    n, z, p = _wilcoxon_one_sided(diffs)
    print(f"\n  Wilcoxon Hybrid > SCM-Blended: z={z:.2f} p={p:.4g}")
    diffs_llm = [h - lb for h, lb in zip(hyb_maps, llm_blind_maps)]
    n2, z2, p2 = _wilcoxon_one_sided(diffs_llm)
    print(f"  Wilcoxon Hybrid > LLM-drug-blind: z={z2:.2f} p={p2:.4g}")

    # Sprint 5 clinical safety hit-rate
    print(f"\n--- Sprint 5 clinical-safety benchmark (n={len(safety_results)}) ---")

    def hit(key: str, k: int) -> int:
        return sum(1 for r in safety_results if r[key] is not None and r[key] <= k)

    n = len(safety_results)
    print(f"  {'method':<22s} {'hit@3':<8s} {'hit@10':<8s} {'hit@20':<8s} {'hit@50':<8s}")
    for label, key in [("SCM-Blended", "scm_blended_rank"),
                         ("RF-ECFP", "rf_ecfp_rank"),
                         ("LLM-drug-blind", "llm_drug_blind_rank"),
                         ("LLM-with-name", "llm_with_name_rank"),
                         ("Hybrid SCM+LLM", "hybrid_rank")]:
        h3, h10, h20, h50 = hit(key, 3), hit(key, 10), hit(key, 20), hit(key, 50)
        print(f"  {label:<22s} {h3}/{n} ({h3/n:.0%})   "
              f"{h10}/{n} ({h10/n:.0%})  {h20}/{n} ({h20/n:.0%})  {h50}/{n} ({h50/n:.0%})")

    # Primary H1 McNemar: hybrid vs LLM-drug-blind, hit@10
    b = sum(1 for r in safety_results
            if (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
            and not (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    c = sum(1 for r in safety_results
            if not (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
            and (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    p_one = _mcnemar_one_sided(b, c)
    print(f"\n  McNemar Hybrid > LLM-drug-blind on hit@10:")
    print(f"    b={b} (hybrid hit, LLM-blind miss); c={c} (hybrid miss, LLM-blind hit)")
    print(f"    p_one_sided = {p_one:.4g}")

    # Save aggregate summary
    summary = {
        "sprint12_n": len(s12_results),
        "sprint12_scm_blended_map": statistics.mean(scm_maps),
        "sprint12_hybrid_map": statistics.mean(hyb_maps),
        "sprint12_llm_blind_map": statistics.mean(llm_blind_maps),
        "sprint12_llm_name_map": statistics.mean(llm_name_maps),
        "sprint12_wilcoxon_hybrid_vs_scm_p": p,
        "sprint12_wilcoxon_hybrid_vs_llm_blind_p": p2,
        "safety_n": n,
        "safety_hits": {
            "scm_blended": {f"at_{k}": hit("scm_blended_rank", k) for k in (3, 5, 10, 20, 50)},
            "hybrid": {f"at_{k}": hit("hybrid_rank", k) for k in (3, 5, 10, 20, 50)},
            "llm_drug_blind": {f"at_{k}": hit("llm_drug_blind_rank", k) for k in (3, 5, 10, 20, 50)},
            "llm_with_name": {f"at_{k}": hit("llm_with_name_rank", k) for k in (3, 5, 10, 20, 50)},
            "rf_ecfp": {f"at_{k}": hit("rf_ecfp_rank", k) for k in (3, 5, 10, 20, 50)},
        },
        "mcnemar_hybrid_vs_llm_blind_hit10_p": p_one,
        "mcnemar_b_hybrid_hit_llm_miss": b,
        "mcnemar_c_hybrid_miss_llm_hit": c,
    }
    with open(RESULTS / "sprint5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {RESULTS / 'sprint5_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
