"""Sprint 8A: expanded clinical-safety benchmark with new sources + action types.

Sprint 8A enhancements vs Sprint 7:
  - 7th α source: OnSIDES v3.1.1 (transformer-NLP FDA label AEs,
    matches 201/247 training drugs)
  - DGIdb 5.0 action types (909 drug-target action pairs)
  - DrugCentral action types (5,736 drug-target action pairs)
  - LLM Hybrid prompt surfaces per-target action class
    (inhibit/activate/modulator/binder)
  - Expanded benchmark: n=51 → n=149 (62 onc + 30 immuno + 22 CV + 13 CNS + ...)

Eval design: run once on n=149 expanded benchmark, then identify the
n=51 historical subset within those results for apples-to-apples vs
Sprint 7. McNemar Hybrid vs LLM-drug-blind reported on both.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..baselines.llm_baselines import (
    rank_side_effects_llm_drug_blind, rank_side_effects_llm_with_name,
)
from ..baselines.llm_hybrid_reranker import hybrid_rerank, load_action_types
from ..baselines.rf_ecfp_baseline import (
    build_ecfp_matrix, compute_ecfp, fetch_smiles_for_drugs,
    rank_test_drug_rf, train_rf_models,
)
from ..baselines.logreg_baseline import build_se_label_matrix
from ..data.build_catalog import query_binding_profile
from ..data.clinical_safety_benchmark import passes_eligibility
from ..data.clinical_safety_benchmark_v2 import EXPANDED_SAFETY_CASES
from ..demo.scm_explainer import explain_predictions
from ..llm import SonnetClient
from ..pipeline.run_sprint3_clinical_failures import (
    best_causal_rank, fetch_smiles, lookup_chembl_molregno,
)
from ..pipeline.run_sprint7_eval import classify_ta, TA_MAP
from ..scm.scoring import score_drug_side_effects


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
N_WORKERS = 16
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"


# Sprint 5/7 historical benchmark drug_ids (the n=51 we've been
# benchmarking against). Used to extract the n=51 subset from n=149
# results post-hoc for apples-to-apples comparison.
HISTORICAL_N51_IDS: set[str] = {
    # From clinical_safety_benchmark.py v1
    "terfenadine", "astemizole", "cisapride", "troglitazone", "cerivastatin",
    "rofecoxib", "valdecoxib", "pergolide", "sibutramine", "rosiglitazone",
    "mibefradil", "nefazodone", "bromfenac", "flecainide", "tegaserod",
    "pemoline", "phenformin", "propoxyphene",
    "amiodarone_pulm", "amiodarone_qt", "ondansetron", "haloperidol",
    "thioridazine", "ziprasidone", "droperidol", "citalopram", "moxifloxacin",
    "erythromycin", "dofetilide", "sotalol", "quinidine",
    "diclofenac", "naltrexone_hep", "methotrexate_hep", "valproate_hep",
    "tolcapone", "atorvastatin_hep", "nimesulide",
    "doxorubicin", "daunorubicin", "trastuzumab_card",
    "sunitinib_card", "sorafenib_htn", "pazopanib_card",
    "cisplatin_renal", "amphotericin_renal", "gentamicin_renal", "vancomycin_renal",
    "carbamazepine_sjs", "clozapine_agra", "rituximab_pml",
    "simvastatin_rhabdo", "pravastatin_rhabdo", "rosuvastatin_rhabdo", "lovastatin_rhabdo",
    "amitriptyline_arr", "imipramine_arr", "nortriptyline_arr",
    "captopril_hyperK", "lisinopril_hyperK",
    "bromocriptine_valve", "cabergoline_valve", "ibuprofen_renal", "fluoxetine_sui",
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

    ta = classify_ta(cf.drug_id, cf.severity, cf.causal_off_target_gene)

    # Sprint 8A: load merged action types for this drug (DGIdb + DrugCentral)
    action_types = {}
    if molregno_resolved is not None:
        try:
            action_types = load_action_types(molregno_resolved)
        except Exception:
            action_types = {}

    # SCM scoring
    scored = score_drug_side_effects(bp, edges, se_vocab)
    explanations = explain_predictions(
        scored[:100], bp, edges, target_info, se_names,
        top_k_se=100, top_k_targets=5,
    )
    scm_ranked = [s for s, _ in scored]
    scm_rank = best_causal_rank(scm_ranked, cf.causal_side_effects_umls)

    # Sprint 8A Hybrid (TA-conditional + action-types context)
    hybrid = hybrid_rerank(
        bp, scored, explanations, vocab_payload, client=client,
        top_k_scm_candidates=100, therapeutic_area=ta,
        action_types=action_types,
    )
    hybrid_rank = best_causal_rank(hybrid.ranked_side_effects, cf.causal_side_effects_umls)

    # LLM-drug-blind baseline
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
        "n_action_types": sum(1 for v in action_types.values() if v != "unknown"),
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


def hit(records: list[dict], key: str, k: int) -> int:
    return sum(1 for r in records if r[key] is not None and r[key] <= k)


def _bootstrap_ci(records: list[dict], key: str, k: int,
                    n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% bootstrap CI on hit@k proportion."""
    rng = np.random.RandomState(seed)
    n = len(records)
    if n == 0:
        return (0.0, 0.0)
    hits = np.array([1 if (r[key] is not None and r[key] <= k) else 0 for r in records])
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(hits[idx].mean())
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    return (lo, hi)


def _report_method_table(records: list[dict], label: str):
    n = len(records)
    print(f"\n--- {label} (n={n}) ---")
    print(f"  {'method':<22s} {'hit@3':<14s} {'hit@10':<14s} "
          f"{'hit@20':<14s} {'hit@50':<14s}")
    for lbl, key in [("SCM (Sprint 8A α)", "scm_rank"),
                       ("RF-ECFP", "rf_ecfp_rank"),
                       ("LLM-drug-blind", "llm_drug_blind_rank"),
                       ("LLM-with-name", "llm_with_name_rank"),
                       ("Hybrid (Sprint 8A)", "hybrid_rank")]:
        h3, h10, h20, h50 = (hit(records, key, k) for k in (3, 10, 20, 50))
        print(f"  {lbl:<22s} {h3}/{n} ({h3/n:.0%})    "
              f"{h10}/{n} ({h10/n:.0%})    "
              f"{h20}/{n} ({h20/n:.0%})    "
              f"{h50}/{n} ({h50/n:.0%})")


def main() -> int:
    print("=" * 78)
    print("Sprint 8A: n=149 expanded clinical-safety benchmark")
    print("  + OnSIDES (7th α source)")
    print("  + DGIdb + DrugCentral action types")
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
    eligible = [c for c in EXPANDED_SAFETY_CASES
                if passes_eligibility(c, vocab_set, target_set)]
    print(f"[setup] vocab={len(se_vocab)}; edges={len(edges)} targets; "
          f"expanded eligible cases={len(eligible)}")

    # RF-ECFP setup
    train_drugs = [d for d in cat["drugs"] if d["split"] == "train"]
    smiles_map = fetch_smiles_for_drugs(cat["drugs"])
    X_train_ecfp, train_ecfp_idx = build_ecfp_matrix(train_drugs, smiles_map)
    Y_train = build_se_label_matrix(train_drugs, se_vocab)
    Y_train_ecfp = Y_train[train_ecfp_idx]
    print("[setup] Training RF-ECFP...")
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
    print(f"\n[safety] done in {(time.monotonic() - t_start)/60:.1f}m, "
          f"n={len(results)}")

    with open(RESULTS / "sprint8a_safety.json", "w") as f:
        json.dump({"n_cases": len(results), "per_drug": results}, f, indent=2)
    print(f"[save] {RESULTS / 'sprint8a_safety.json'}")

    # ============= Primary signal: n=149 expanded =============
    print("\n" + "=" * 78)
    print("Sprint 8A PRIMARY (expanded n=149)")
    print("=" * 78)
    _report_method_table(results, "Overall (expanded)")

    # McNemar Hybrid > LLM-drug-blind on hit@10
    b = sum(1 for r in results
            if (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
            and not (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    c = sum(1 for r in results
            if not (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
            and (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    p_one = _mcnemar_one_sided(b, c)
    print(f"\n[expanded] McNemar Hybrid > LLM-drug-blind hit@10: b={b} c={c} p={p_one:.4g}")

    # Bootstrap CI on Hybrid hit@10
    lo, hi = _bootstrap_ci(results, "hybrid_rank", 10)
    h10 = hit(results, "hybrid_rank", 10)
    print(f"[expanded] Hybrid hit@10: {h10}/{len(results)} "
          f"({h10/len(results):.1%}) [95%CI: {lo:.1%}-{hi:.1%}]")

    # ============= Secondary signal: n=51 historical subset =============
    subset = [r for r in results if r["drug_id"] in HISTORICAL_N51_IDS]
    print("\n" + "=" * 78)
    print(f"Sprint 8A SECONDARY (n=51 historical subset; actual n={len(subset)})")
    print("=" * 78)
    _report_method_table(subset, "Historical subset (apples-to-apples vs Sprint 7)")

    b51 = sum(1 for r in subset
              if (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
              and not (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    c51 = sum(1 for r in subset
              if not (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
              and (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    p_one_51 = _mcnemar_one_sided(b51, c51)
    print(f"\n[n=51 subset] McNemar Hybrid > LLM-drug-blind: b={b51} c={c51} p={p_one_51:.4g}")

    # ============= Per-TA stratified =============
    print(f"\n--- Per-therapeutic-area (expanded n=149) ---")
    for ta in ("Oncology", "Immunology", "Cardiovascular & metabolic",
               "CNS & psychiatry", "Other"):
        sub = [r for r in results if r["therapeutic_area"] == ta]
        if not sub:
            continue
        _report_method_table(sub, f"[{ta}]")

    # ============= Save summary =============
    summary = {
        "n_expanded": len(results),
        "n_historical_subset": len(subset),
        "expanded": {
            "hits": {
                label: {f"at_{k}": hit(results, key, k) for k in (3, 5, 10, 20, 50)}
                for label, key in [
                    ("scm", "scm_rank"),
                    ("hybrid", "hybrid_rank"),
                    ("llm_drug_blind", "llm_drug_blind_rank"),
                    ("llm_with_name", "llm_with_name_rank"),
                    ("rf_ecfp", "rf_ecfp_rank"),
                ]
            },
            "mcnemar_hybrid_vs_llm_blind_hit10": {
                "b": b, "c": c, "p": p_one,
            },
            "hybrid_hit10_bootstrap_95ci": [lo, hi],
        },
        "historical_n51_subset": {
            "hits": {
                label: {f"at_{k}": hit(subset, key, k) for k in (3, 5, 10, 20, 50)}
                for label, key in [
                    ("scm", "scm_rank"),
                    ("hybrid", "hybrid_rank"),
                    ("llm_drug_blind", "llm_drug_blind_rank"),
                    ("llm_with_name", "llm_with_name_rank"),
                    ("rf_ecfp", "rf_ecfp_rank"),
                ]
            },
            "mcnemar_hybrid_vs_llm_blind_hit10": {
                "b": b51, "c": c51, "p": p_one_51,
            },
        },
    }
    with open(RESULTS / "sprint8a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {RESULTS / 'sprint8a_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
