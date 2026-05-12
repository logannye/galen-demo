"""Sprint E: clinical-safety benchmark with modern biologics deep-coverage.

Sprint E enhancements vs Sprint 8B:
  - Target vocab expansion 983 → 1,084 (+101 biologic targets including
    long-missing PDCD1/CTLA4/CD20/CD19/BCMA/etc.)
  - Biologic binding profiles: 81 → 173 (CAR-Ts, bispecifics, ADCs)
  - Curated class-effect priors: 124 → 307 pairs (CRS, ICANS, ADC payloads,
    newer IO, B-cell aplasia, modern biologic patterns)
  - Benchmark expansion: 149 eligible → 187 (+38 new biologic cases)
  - **Critical bug-fix**: Sprint 7's curated priors for PDCD1/CTLA4/CD20
    were silently filtered out because target_vocab didn't include those
    targets. Now corrected.

The substrate architecture (signed-α + PPI backoff) is INHERITED from
Sprint 8B (which was null). The Sprint E intervention is COVERAGE
expansion, which Sprint 8B's biologic-recovery demonstrated does work.

Eval design: single Sprint E-full arm. Compared against Sprint 8B
saved results on the paired intersection set.
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
from ..data.biologic_binding_profiles import get_biologic_binding
from ..data.clinical_safety_benchmark import passes_eligibility
from ..data.clinical_safety_benchmark_v3 import EXPANDED_SAFETY_CASES_V3
from ..demo.scm_explainer import explain_predictions
from ..llm import SonnetClient
from ..pipeline.run_sprint3_clinical_failures import (
    best_causal_rank, fetch_smiles, lookup_chembl_molregno,
)
from ..pipeline.run_sprint7_eval import classify_ta
from ..pipeline.run_sprint8a_eval import HISTORICAL_N51_IDS
from ..scm.scoring import score_drug_side_effects, score_drug_side_effects_signed


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
N_WORKERS = 16
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"


def _process_safety_case(
    cf, edges: dict, signed_edges: dict, target_action_n: dict,
    target_info: dict, vocab_payload: dict,
    se_vocab: list[str], se_names: dict,
    drugs_by_name: dict, smiles_map: dict,
    rf_models: dict, base_rate_array: np.ndarray,
    client: SonnetClient,
) -> dict:
    bp = []
    molregno_resolved = None
    biologic_recovery = False

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
        biologic_bp = get_biologic_binding(cf.drug_search_name)
        if biologic_bp:
            bp = biologic_bp
            biologic_recovery = True

    if not bp:
        return {
            "drug_id": cf.drug_id, "drug_search_name": cf.drug_search_name,
            "skipped": True, "reason": "no binding profile",
        }

    ta = classify_ta(cf.drug_id, cf.severity, cf.causal_off_target_gene)

    action_types = {}
    if molregno_resolved is not None:
        try:
            action_types = load_action_types(molregno_resolved)
        except Exception:
            action_types = {}

    scored = score_drug_side_effects_signed(
        bp, edges, signed_edges, action_types, target_action_n, se_vocab,
        min_drugs_for_signed=3,
    )
    explanations = explain_predictions(
        scored[:100], bp, edges, target_info, se_names,
        top_k_se=100, top_k_targets=5,
    )
    scm_ranked = [s for s, _ in scored]
    scm_rank = best_causal_rank(scm_ranked, cf.causal_side_effects_umls)

    hybrid = hybrid_rerank(
        bp, scored, explanations, vocab_payload, client=client,
        top_k_scm_candidates=100, therapeutic_area=ta,
        action_types=action_types,
    )
    hybrid_rank = best_causal_rank(hybrid.ranked_side_effects, cf.causal_side_effects_umls)

    llm_blind = rank_side_effects_llm_drug_blind(
        bp, vocab_payload, client=client, top_k=50,
    )
    llm_blind_rank = best_causal_rank(
        llm_blind.ranked_side_effects, cf.causal_side_effects_umls,
    )

    llm_name = rank_side_effects_llm_with_name(
        cf.drug_search_name, bp, vocab_payload, client=client, top_k=50,
    )
    llm_name_rank = best_causal_rank(
        llm_name.ranked_side_effects, cf.causal_side_effects_umls,
    )

    rf_rank = None
    if molregno_resolved is not None and not biologic_recovery:
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
        "biologic_recovery": biologic_recovery,
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
    return (boot_means[int(0.025 * n_boot)], boot_means[int(0.975 * n_boot)])


def _report_table(records: list[dict], label: str):
    n = len(records)
    print(f"\n--- {label} (n={n}) ---")
    print(f"  {'method':<22s} {'hit@3':<14s} {'hit@10':<14s} "
          f"{'hit@20':<14s} {'hit@50':<14s}")
    for lbl, key in [("SCM (Sprint E α)", "scm_rank"),
                       ("RF-ECFP", "rf_ecfp_rank"),
                       ("LLM-drug-blind", "llm_drug_blind_rank"),
                       ("LLM-with-name", "llm_with_name_rank"),
                       ("Hybrid (Sprint E)", "hybrid_rank")]:
        h3, h10, h20, h50 = (hit(records, key, k) for k in (3, 10, 20, 50))
        print(f"  {lbl:<22s} {h3}/{n} ({h3/n:.0%})    "
              f"{h10}/{n} ({h10/n:.0%})    "
              f"{h20}/{n} ({h20/n:.0%})    "
              f"{h50}/{n} ({h50/n:.0%})")


def _mcnemar_vs_external(records: list[dict], external_path: Path, k: int = 10) -> dict:
    if not external_path.exists():
        return {"available": False}
    with open(external_path) as f:
        ext = json.load(f)
    ext_by_id = {r["drug_id"]: r for r in ext.get("per_drug", [])}
    pair_rows = [(r, ext_by_id[r["drug_id"]]) for r in records
                 if r["drug_id"] in ext_by_id]
    b = sum(1 for cur, ext in pair_rows
            if (cur["hybrid_rank"] is not None and cur["hybrid_rank"] <= k)
            and not (ext["hybrid_rank"] is not None and ext["hybrid_rank"] <= k))
    c = sum(1 for cur, ext in pair_rows
            if not (cur["hybrid_rank"] is not None and cur["hybrid_rank"] <= k)
            and (ext["hybrid_rank"] is not None and ext["hybrid_rank"] <= k))
    p = _mcnemar_one_sided(b, c)
    return {
        "available": True,
        "n_paired": len(pair_rows),
        "b_cur_hit_ext_miss": b,
        "c_cur_miss_ext_hit": c,
        "p_one_sided": p,
    }


def main() -> int:
    print("=" * 78)
    print("Sprint E: Modern Biologics Deep-Coverage benchmark")
    print("  + Vocab expansion 983→1,084 (long-missing PDCD1/CTLA4/CD20/CD19/BCMA)")
    print("  + Biologic profiles 81→173 (CAR-Ts, bispecifics, ADCs)")
    print("  + Curated priors 124→307 (CRS, ICANS, ADC payloads, newer IO)")
    print("  + Benchmark v3: 149→187 eligible (~38 modern biologic cases)")
    print("=" * 78)

    with open(RESULTS / "scm_edges_blended_e.json") as f:
        edges = json.load(f)
    with open(RESULTS / "scm_edges_signed.json") as f:
        signed_payload = json.load(f)
    signed_edges = signed_payload.get("edges", {})
    target_action_n = signed_payload.get("target_action_n_drugs", {})
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
    eligible = [c for c in EXPANDED_SAFETY_CASES_V3
                if passes_eligibility(c, vocab_set, target_set)]
    print(f"[setup] vocab={len(se_vocab)}; edges={len(edges)} targets; "
          f"eligible cases={len(eligible)}")

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
            ex.submit(_process_safety_case, cf, edges, signed_edges,
                       target_action_n, target_info,
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
    n_biologic_recovered = sum(1 for r in results if r.get("biologic_recovery"))
    print(f"\n[safety] done in {(time.monotonic() - t_start)/60:.1f}m, "
          f"n={len(results)} ({n_biologic_recovered} via biologic recovery)")

    with open(RESULTS / "sprint_e_safety.json", "w") as f:
        json.dump({"n_cases": len(results), "per_drug": results}, f, indent=2)
    print(f"[save] {RESULTS / 'sprint_e_safety.json'}")

    # PRIMARY signal
    print("\n" + "=" * 78)
    print("Sprint E PRIMARY (full expanded benchmark)")
    print("=" * 78)
    _report_table(results, "Overall (full)")

    b = sum(1 for r in results
            if (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
            and not (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    c = sum(1 for r in results
            if not (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
            and (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    p_one = _mcnemar_one_sided(b, c)
    print(f"\n[full] McNemar Hybrid > LLM-drug-blind hit@10: b={b} c={c} p={p_one:.4g}")

    lo, hi = _bootstrap_ci(results, "hybrid_rank", 10)
    h10 = hit(results, "hybrid_rank", 10)
    print(f"[full] Hybrid hit@10: {h10}/{len(results)} ({h10/len(results):.1%}) "
          f"[95%CI: {lo:.1%}-{hi:.1%}]")

    # Sprint 8B paired comparison
    sprint8b_path = RESULTS / "sprint8b_safety.json"
    if sprint8b_path.exists():
        with open(sprint8b_path) as f:
            s8b = json.load(f)
        s8b_ids = {r["drug_id"] for r in s8b.get("per_drug", [])}
        s8b_intersect = [r for r in results if r["drug_id"] in s8b_ids]
        print("\n" + "=" * 78)
        print(f"Sprint E vs 8B: paired intersection set (n={len(s8b_intersect)})")
        print("=" * 78)
        _report_table(s8b_intersect, "Sprint 8B intersection")
        cmp_8b = _mcnemar_vs_external(s8b_intersect, sprint8b_path, k=10)
        print(f"\n[E vs 8B pair] McNemar E Hybrid > 8B Hybrid hit@10: "
              f"b={cmp_8b['b_cur_hit_ext_miss']} c={cmp_8b['c_cur_miss_ext_hit']} "
              f"p={cmp_8b['p_one_sided']:.4g}")
    else:
        cmp_8b = {"available": False}

    # Sprint 8A paired comparison
    sprint8a_path = RESULTS / "sprint8a_safety.json"
    cmp_8a = {"available": False}
    if sprint8a_path.exists():
        with open(sprint8a_path) as f:
            s8a = json.load(f)
        s8a_ids = {r["drug_id"] for r in s8a.get("per_drug", [])}
        s8a_intersect = [r for r in results if r["drug_id"] in s8a_ids]
        print("\n" + "=" * 78)
        print(f"Sprint E vs 8A: paired intersection set (n={len(s8a_intersect)})")
        print("=" * 78)
        _report_table(s8a_intersect, "Sprint 8A intersection")
        cmp_8a = _mcnemar_vs_external(s8a_intersect, sprint8a_path, k=10)
        print(f"\n[E vs 8A pair] McNemar E Hybrid > 8A Hybrid hit@10: "
              f"b={cmp_8a['b_cur_hit_ext_miss']} c={cmp_8a['c_cur_miss_ext_hit']} "
              f"p={cmp_8a['p_one_sided']:.4g}")

    # Historical n=51 subset
    subset = [r for r in results if r["drug_id"] in HISTORICAL_N51_IDS]
    print("\n" + "=" * 78)
    print(f"Sprint E n=51 historical (actual n={len(subset)})")
    print("=" * 78)
    _report_table(subset, "Historical subset")

    # Per-TA
    print(f"\n--- Per-therapeutic-area ---")
    for ta in ("Oncology", "Immunology", "Cardiovascular & metabolic",
               "CNS & psychiatry", "Other"):
        sub = [r for r in results if r["therapeutic_area"] == ta]
        if not sub:
            continue
        _report_table(sub, f"[{ta}]")

    summary = {
        "n_full": len(results),
        "n_biologic_recovered": n_biologic_recovered,
        "n_historical_subset": len(subset),
        "full": {
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
        "mcnemar_e_vs_8b_paired": cmp_8b,
        "mcnemar_e_vs_8a_paired": cmp_8a,
    }
    with open(RESULTS / "sprint_e_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {RESULTS / 'sprint_e_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
