"""Sprint I: 3-prong calibration sprint.

  - Prong I.1: refined override (apply_curated_prior_override_v2)
    with tighter criteria (SCM rank<=3, Hybrid rank>5, cap=1)
  - Prong I.2: expanded OOD benchmark (n=22 → n=60)
  - Prong I.3: dual-LLM arms (Sonnet 4.6 + Opus 4.7) — runs ONE arm
    per invocation, controlled by --llm flag (or env var)

Usage:
  python -m scripts.pipeline.run_sprint_i --llm sonnet
  python -m scripts.pipeline.run_sprint_i --llm opus
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..baselines.curated_prior_vote import (
    apply_curated_prior_override_v2, load_curated_priors_for_override,
)
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
from ..data.clinical_safety_benchmark_v5 import EXPANDED_SAFETY_CASES_V5
from ..data.clinical_safety_benchmark_ood_v2 import EXPANDED_OOD_CASES
from ..demo.scm_explainer import explain_predictions
from ..llm import SonnetClient, OpusClient
from ..pipeline.run_sprint3_clinical_failures import (
    fetch_smiles, lookup_chembl_molregno,
)
from ..pipeline.run_sprint7_eval import classify_ta
from ..pipeline.run_sprint_f_eval import (
    best_causal_rank, best_causal_rank_with_rollup,
)
from ..scm.scoring import score_drug_side_effects_signed


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
    client,
    rollup_map: dict, curated_priors: dict,
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
    hybrid_ranked = hybrid.ranked_side_effects

    # Sprint I refined override (v2)
    hybrid_i_ranked, promotions = apply_curated_prior_override_v2(
        hybrid_ranked, scm_ranked, bp, curated_priors,
    )

    hybrid_rank_pre = best_causal_rank(hybrid_ranked, cf.causal_side_effects_umls)
    hybrid_rank = best_causal_rank(hybrid_i_ranked, cf.causal_side_effects_umls)
    hybrid_rank_rollup = best_causal_rank_with_rollup(
        hybrid_i_ranked, cf.causal_side_effects_umls, rollup_map,
    )

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
        "biologic_recovery": biologic_recovery,
        "scm_rank": scm_rank,
        "hybrid_rank": hybrid_rank,
        "hybrid_rank_pre_override": hybrid_rank_pre,
        "hybrid_rank_rollup": hybrid_rank_rollup,
        "llm_drug_blind_rank": llm_blind_rank,
        "llm_with_name_rank": llm_name_rank,
        "rf_ecfp_rank": rf_rank,
        "n_promotions": len(promotions),
        "promotions": [{"umls": u, "alpha": a} for u, a in promotions],
        "hybrid_top10": hybrid_i_ranked[:10],
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
    print(f"  {'method':<25s} {'hit@3':<14s} {'hit@10':<14s} "
          f"{'hit@20':<14s} {'hit@50':<14s}")
    for lbl, key in [("SCM (Sprint I alpha)", "scm_rank"),
                       ("RF-ECFP", "rf_ecfp_rank"),
                       ("LLM-drug-blind", "llm_drug_blind_rank"),
                       ("LLM-with-name", "llm_with_name_rank"),
                       ("Hybrid pre-override", "hybrid_rank_pre_override"),
                       ("Hybrid Sprint I", "hybrid_rank")]:
        h3, h10, h20, h50 = (hit(records, key, k) for k in (3, 10, 20, 50))
        print(f"  {lbl:<25s} {h3}/{n} ({h3/n:.0%})    "
              f"{h10}/{n} ({h10/n:.0%})    "
              f"{h20}/{n} ({h20/n:.0%})    "
              f"{h50}/{n} ({h50/n:.0%})")


def run_stage(cases, edges, signed_edges, target_action_n, target_info,
               vocab_payload, se_vocab, se_names, drugs_by_name, smiles_map,
               rf_models, base_rate_array, client, rollup_map, curated_priors,
               label="MAIN"):
    target_set = set(target_info.keys())
    vocab_set = set(se_vocab)
    eligible = [c for c in cases
                if passes_eligibility(c, vocab_set, target_set)]
    print(f"\n[{label}] eligible: {len(eligible)}")

    results: list[dict] = [None] * len(eligible)
    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(_process_safety_case, cf, edges, signed_edges,
                       target_action_n, target_info, vocab_payload, se_vocab,
                       se_names, drugs_by_name, smiles_map, rf_models,
                       base_rate_array, client, rollup_map, curated_priors): i
            for i, cf in enumerate(eligible)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"[{label}] case {idx} FAILED: {e}", flush=True)
            done += 1
            if done % 10 == 0:
                elapsed = time.monotonic() - t_start
                print(f"  [{label}] {done}/{len(eligible)} ({elapsed/60:.1f}m)",
                      flush=True)
    results = [r for r in results if r is not None and not r.get("skipped")]
    print(f"\n[{label}] done in {(time.monotonic() - t_start)/60:.1f}m, "
          f"n={len(results)}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", choices=("sonnet", "opus"), default="sonnet",
                        help="Which LLM to use as Hybrid + baselines")
    args = parser.parse_args()
    llm_name = args.llm

    print("=" * 78)
    print(f"Sprint I: 3-prong calibration ({llm_name.upper()} arm)")
    print("  + I.1: refined override (SCM<=3, Hybrid>5, cap=1)")
    print("  + I.2: expanded OOD benchmark (n=60)")
    print(f"  + I.3: {llm_name.upper()} LLM arm")
    print("=" * 78)

    if llm_name == "sonnet":
        client = SonnetClient()
    else:
        client = OpusClient()

    with open(RESULTS / "scm_edges_blended_i.json") as f:
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

    with open(RESULTS / "hpo_umls_rollup.json") as f:
        rollup_payload = json.load(f)
    rollup_map = rollup_payload["rollup"]

    curated_priors = load_curated_priors_for_override()
    print(f"[setup] curated priors: {len(curated_priors)} targets")

    train_drugs = [d for d in cat["drugs"] if d["split"] == "train"]
    smiles_map = fetch_smiles_for_drugs(cat["drugs"])
    X_train_ecfp, train_ecfp_idx = build_ecfp_matrix(train_drugs, smiles_map)
    Y_train = build_se_label_matrix(train_drugs, se_vocab)
    Y_train_ecfp = Y_train[train_ecfp_idx]
    print("[setup] Training RF-ECFP...")
    rf_models = train_rf_models(X_train_ecfp, Y_train_ecfp,
                                  n_estimators=50, max_depth=8)
    base_rate_array = np.clip(Y_train.mean(axis=0), 1e-6, 1.0 - 1e-6)

    drugs_by_name = {(d["drug_name"] or "").lower(): d for d in cat["drugs"]}

    # MAIN
    print("\n" + "=" * 78)
    print(f"STAGE 1: MAIN ({llm_name})")
    print("=" * 78)
    main_results = run_stage(EXPANDED_SAFETY_CASES_V5, edges, signed_edges,
                              target_action_n, target_info, vocab_payload,
                              se_vocab, se_names, drugs_by_name, smiles_map,
                              rf_models, base_rate_array, client, rollup_map,
                              curated_priors, label="MAIN")
    main_path = RESULTS / f"sprint_i_safety_{llm_name}.json"
    with open(main_path, "w") as f:
        json.dump({"n_cases": len(main_results), "per_drug": main_results,
                    "llm": llm_name}, f, indent=2)
    print(f"[save] {main_path}")

    _report_table(main_results, f"Main ({llm_name})")

    b = sum(1 for r in main_results
            if (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
            and not (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    c = sum(1 for r in main_results
            if not (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
            and (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
    p_one = _mcnemar_one_sided(b, c)
    print(f"\n[main {llm_name}] McNemar Hybrid > LLM-drug-blind: b={b} c={c} p={p_one:.4g}")

    lo, hi = _bootstrap_ci(main_results, "hybrid_rank", 10)
    h10 = hit(main_results, "hybrid_rank", 10)
    print(f"[main {llm_name} strict] Hybrid hit@10: {h10}/{len(main_results)} "
          f"({h10/len(main_results):.1%}) [95%CI: {lo:.1%}-{hi:.1%}]")

    # OOD expanded
    print("\n\n" + "=" * 78)
    print(f"STAGE 2: OOD-v2 EXPANDED ({llm_name}, n=60 target)")
    print("=" * 78)
    ood_results = run_stage(EXPANDED_OOD_CASES, edges, signed_edges,
                             target_action_n, target_info, vocab_payload,
                             se_vocab, se_names, drugs_by_name, smiles_map,
                             rf_models, base_rate_array, client, rollup_map,
                             curated_priors, label="OOD")
    ood_path = RESULTS / f"sprint_i_ood_safety_{llm_name}.json"
    with open(ood_path, "w") as f:
        json.dump({"n_cases": len(ood_results), "per_drug": ood_results,
                    "llm": llm_name}, f, indent=2)
    print(f"[save] {ood_path}")

    _report_table(ood_results, f"OOD expanded ({llm_name})")

    if ood_results:
        lo_ood, hi_ood = _bootstrap_ci(ood_results, "hybrid_rank", 10)
        h10_ood = hit(ood_results, "hybrid_rank", 10)
        print(f"[OOD {llm_name}] Hybrid hit@10: {h10_ood}/{len(ood_results)} "
              f"({h10_ood/len(ood_results):.1%}) [95%CI: {lo_ood:.1%}-{hi_ood:.1%}]")

        h10_name_ood = hit(ood_results, "llm_with_name_rank", 10)
        diff_pp = (h10_ood - h10_name_ood) / len(ood_results) * 100
        print(f"[OOD {llm_name} diff vs LLM-with-name] "
              f"Hybrid {h10_ood/len(ood_results):.1%} vs "
              f"LLM-with-name {h10_name_ood/len(ood_results):.1%} ({diff_pp:+.1f}pp)")

        b_ood = sum(1 for r in ood_results
                     if (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
                     and not (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
        c_ood = sum(1 for r in ood_results
                     if not (r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
                     and (r["llm_drug_blind_rank"] is not None and r["llm_drug_blind_rank"] <= 10))
        p_ood = _mcnemar_one_sided(b_ood, c_ood)
        print(f"[OOD {llm_name}] McNemar Hybrid > LLM-drug-blind: "
              f"b={b_ood} c={c_ood} p={p_ood:.4g}")

    summary = {
        "llm": llm_name,
        "main": {
            "n": len(main_results),
            "hits": {
                label: {f"at_{k}": hit(main_results, key, k) for k in (3, 5, 10, 20, 50)}
                for label, key in [
                    ("scm", "scm_rank"),
                    ("hybrid", "hybrid_rank"),
                    ("hybrid_pre_override", "hybrid_rank_pre_override"),
                    ("llm_drug_blind", "llm_drug_blind_rank"),
                    ("llm_with_name", "llm_with_name_rank"),
                    ("rf_ecfp", "rf_ecfp_rank"),
                ]
            },
            "mcnemar_hybrid_vs_llm_blind_hit10": {"b": b, "c": c, "p": p_one},
            "hybrid_hit10_strict_95ci": [lo, hi],
        },
        "ood": {
            "n": len(ood_results),
            "hits": {
                label: {f"at_{k}": hit(ood_results, key, k) for k in (3, 5, 10, 20, 50)}
                for label, key in [
                    ("scm", "scm_rank"),
                    ("hybrid", "hybrid_rank"),
                    ("hybrid_pre_override", "hybrid_rank_pre_override"),
                    ("llm_drug_blind", "llm_drug_blind_rank"),
                    ("llm_with_name", "llm_with_name_rank"),
                ]
            } if ood_results else {},
        },
    }
    summary_path = RESULTS / f"sprint_i_summary_{llm_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
