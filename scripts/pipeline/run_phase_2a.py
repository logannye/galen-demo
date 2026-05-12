"""Phase 2.A: Hybrid candidate-pool expansion (top-100 → top-200).

Single intervention: change `top_k_scm_candidates=100` → `200` in the
hybrid_rerank() call. All other arms (SCM, LLM-blind, LLM-with-name)
are NOT re-evaluated — they are reused from Sprint K JSONs for paired
McNemar comparison.

Eval sets:
  - L.1 external held-out (n=80 SIDER test drugs)
  - Main n=224 (Sprint K curated benchmark, EXPANDED_SAFETY_CASES_V5)
  - OOD n=97 (Sprint K curated OOD benchmark, EXPANDED_OOD_CASES_V3)

Outputs:
  - results/phase_2a_l1_safety.json
  - results/phase_2a_main_safety.json
  - results/phase_2a_ood_safety.json
  - results/phase_2a_summary.json (paired McNemar + decision per pre-reg)
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

from ..baselines.curated_prior_vote import (
    apply_curated_prior_override_v2, load_curated_priors_for_override,
)
from ..baselines.llm_hybrid_reranker import hybrid_rerank, load_action_types
from ..pipeline.run_sprint3_clinical_failures import (
    fetch_smiles, lookup_chembl_molregno,
)
from ..data.biologic_binding_profiles import get_biologic_binding
from ..data.build_catalog import query_binding_profile
from ..data.clinical_safety_benchmark import passes_eligibility
from ..data.clinical_safety_benchmark_v5 import EXPANDED_SAFETY_CASES_V5
from ..data.clinical_safety_benchmark_ood_v3 import EXPANDED_OOD_CASES_V3
from ..demo.scm_explainer import explain_predictions
from ..llm import SonnetClient
from ..pipeline.run_sprint7_eval import classify_ta
from ..pipeline.run_sprint_f_eval import best_causal_rank
from ..pipeline.run_sprint_l import _build_l1_cases
from ..scm.scoring import score_drug_side_effects_signed

CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
N_WORKERS = 16

# Phase 2.A single intervention
TOP_K_POOL = 200
EXPLAIN_TOP_K = 200  # match the pool size for attribution coverage
AFFINITY_MODE = "log_sigmoid"  # same as Sprint K (no change)


def _process_curated_case(
    cf, edges, signed_edges, target_action_n,
    target_info, vocab_payload, se_vocab, se_names,
    drugs_by_name, client, curated_priors,
):
    """Process a single curated case (main or OOD) with top-200 pool.

    Reuses binding profile from catalog; if not present, falls back to
    ChEMBL lookup (rare for the curated benchmarks).
    """
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
        biologic_bp = get_biologic_binding(cf.drug_search_name)
        if biologic_bp:
            bp = biologic_bp

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
        min_drugs_for_signed=3, affinity_mode=AFFINITY_MODE,
    )
    explanations = explain_predictions(
        scored[:EXPLAIN_TOP_K], bp, edges, target_info, se_names,
        top_k_se=EXPLAIN_TOP_K, top_k_targets=5,
    )
    scm_ranked = [s for s, _ in scored]

    hybrid = hybrid_rerank(
        bp, scored, explanations, vocab_payload, client=client,
        top_k_scm_candidates=TOP_K_POOL,
        therapeutic_area=ta,
        action_types=action_types,
    )
    hybrid_ranked, _ = apply_curated_prior_override_v2(
        hybrid.ranked_side_effects, scm_ranked, bp, curated_priors,
    )

    hybrid_rank = best_causal_rank(
        hybrid_ranked, cf.causal_side_effects_umls,
    )

    return {
        "drug_id": cf.drug_id,
        "drug_search_name": cf.drug_search_name,
        "causal_side_effects_umls": list(cf.causal_side_effects_umls),
        "hybrid_rank": hybrid_rank,
        "hybrid_top10": hybrid_ranked[:10],
        "skipped": False,
    }


def _process_l1_case(
    cf, edges, signed_edges, target_action_n,
    target_info, vocab_payload, se_vocab, se_names,
    client, curated_priors,
):
    """Process a single L.1 external held-out case at top-200."""
    bp = cf["binding_profile"]
    if not bp:
        return {"drug_id": cf["drug_id"], "skipped": True}

    causal = tuple(cf["causal_se_umls"])
    molregno = cf.get("molregno")

    action_types = {}
    if molregno is not None:
        try:
            action_types = load_action_types(molregno)
        except Exception:
            action_types = {}

    scored = score_drug_side_effects_signed(
        bp, edges, signed_edges, action_types, target_action_n, se_vocab,
        min_drugs_for_signed=3, affinity_mode=AFFINITY_MODE,
    )
    explanations = explain_predictions(
        scored[:EXPLAIN_TOP_K], bp, edges, target_info, se_names,
        top_k_se=EXPLAIN_TOP_K, top_k_targets=5,
    )
    scm_ranked = [s for s, _ in scored]

    hybrid = hybrid_rerank(
        bp, scored, explanations, vocab_payload, client=client,
        top_k_scm_candidates=TOP_K_POOL,
        therapeutic_area="Other",
        action_types=action_types,
    )
    hybrid_ranked, _ = apply_curated_prior_override_v2(
        hybrid.ranked_side_effects, scm_ranked, bp, curated_priors,
    )
    hybrid_rank = best_causal_rank(hybrid_ranked, causal)

    return {
        "drug_id": cf["drug_id"],
        "drug_search_name": cf["drug_search_name"],
        "causal_se_umls": list(causal),
        "hybrid_rank": hybrid_rank,
        "hybrid_top10": hybrid_ranked[:10],
        "skipped": False,
    }


def _bootstrap_ci(records, key, k, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = sum(1 for r in records if not r.get("skipped"))
    if n == 0:
        return (0.0, 0.0)
    hits = np.array([
        1 if (not r.get("skipped")
              and r.get(key) is not None and r[key] <= k) else 0
        for r in records if not r.get("skipped")
    ])
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot.append(hits[idx].mean())
    boot.sort()
    return (boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)])


def _hit_at_k(records, key, k):
    return sum(
        1 for r in records
        if (not r.get("skipped")
            and r.get(key) is not None and r[key] <= k)
    )


def _mcnemar_one_sided(b: int, c: int) -> float:
    """One-sided exact binomial McNemar (H: new better than old)."""
    n = b + c
    if n == 0:
        return 1.0
    p = 0.0
    for x in range(b, n + 1):
        p += math.comb(n, x) * (0.5 ** n)
    return p


def _paired_mcnemar(new_records, sprint_k_records, label="hybrid_rank"):
    """Pair new (top-200) vs sprint-K (top-100) hybrid hits.

    Returns (b, c, p_one_sided, p_one_sided_old_better).
    b = new wins (new hit, old miss)
    c = new loses (new miss, old hit)
    """
    new_by_drug = {r["drug_search_name"].lower(): r for r in new_records
                    if not r.get("skipped")}
    k_by_drug = {(r.get("drug_search_name") or r.get("drug_id", "")).lower(): r
                  for r in sprint_k_records if not r.get("skipped")}
    paired = set(new_by_drug) & set(k_by_drug)
    b = c = 0
    for drug in paired:
        new_hit = (new_by_drug[drug].get(label) is not None
                   and new_by_drug[drug][label] <= 10)
        old_hit = (k_by_drug[drug].get(label) is not None
                   and k_by_drug[drug][label] <= 10)
        if new_hit and not old_hit:
            b += 1
        elif old_hit and not new_hit:
            c += 1
    p_new_better = _mcnemar_one_sided(b, c)
    p_old_better = _mcnemar_one_sided(c, b)
    return b, c, p_new_better, p_old_better, len(paired)


def _decide_l1(hit_rate: float, p: float, main_regressed: bool, ood_regressed: bool) -> str:
    """Decision per Phase 2.A pre-registered falsifiability matrix."""
    if main_regressed or ood_regressed:
        return "LOSS (curated regression — revert)"
    if hit_rate >= 0.50 and p <= 0.01:
        return "STRONG WIN"
    if hit_rate >= 0.45 and p <= 0.05:
        return "MODERATE WIN"
    if hit_rate >= 0.40 and hit_rate < 0.45:
        return "NULL"
    if hit_rate <= 0.40:
        return "LOSS (no L.1 lift)"
    return "NULL"


def run_l1(client, edges, signed_edges, target_action_n, target_info,
           vocab_payload, se_vocab, se_names, curated_priors):
    print("\n" + "=" * 78)
    print("Phase 2.A — L.1 external held-out (n=80) at top-200")
    print("=" * 78)
    l1_cases = _build_l1_cases()
    print(f"[L.1] running n={len(l1_cases)}")
    results = [None] * len(l1_cases)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(_process_l1_case, cf, edges, signed_edges,
                       target_action_n, target_info, vocab_payload, se_vocab,
                       se_names, client, curated_priors): i
            for i, cf in enumerate(l1_cases)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"[L.1] case {idx} FAILED: {e}", flush=True)
                results[idx] = {"drug_id": l1_cases[idx]["drug_id"],
                                "skipped": True, "reason": str(e)}
            done += 1
            if done % 10 == 0:
                el = time.monotonic() - t0
                print(f"  [L.1] {done}/{len(l1_cases)} ({el/60:.1f}m)",
                      flush=True)
    results = [r for r in results if r is not None]
    print(f"[L.1] done in {(time.monotonic() - t0)/60:.1f}m")
    out = {"n_cases": sum(1 for r in results if not r.get("skipped")),
           "top_k": TOP_K_POOL,
           "per_drug": results}
    with open(RESULTS / "phase_2a_l1_safety.json", "w") as f:
        json.dump(out, f, indent=2)
    return results


def run_curated_set(cases, label, fn_out, client, edges, signed_edges,
                     target_action_n, target_info, vocab_payload, se_vocab,
                     se_names, curated_priors, vocab_set, target_set,
                     drugs_by_name):
    print("\n" + "=" * 78)
    print(f"Phase 2.A — {label} (n={len(cases)}) at top-200")
    print("=" * 78)
    eligible = [c for c in cases if passes_eligibility(c, vocab_set, target_set)]
    print(f"[{label}] eligible: {len(eligible)}/{len(cases)}")

    results = [None] * len(eligible)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(_process_curated_case, cf, edges, signed_edges,
                       target_action_n, target_info, vocab_payload, se_vocab,
                       se_names, drugs_by_name, client, curated_priors): i
            for i, cf in enumerate(eligible)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"[{label}] case {idx} FAILED: {e}", flush=True)
                results[idx] = {"drug_id": eligible[idx].drug_id,
                                "skipped": True, "reason": str(e)}
            done += 1
            if done % 20 == 0:
                el = time.monotonic() - t0
                print(f"  [{label}] {done}/{len(eligible)} ({el/60:.1f}m)",
                      flush=True)
    results = [r for r in results if r is not None]
    print(f"[{label}] done in {(time.monotonic() - t0)/60:.1f}m")
    out = {"n_cases": sum(1 for r in results if not r.get("skipped")),
           "top_k": TOP_K_POOL,
           "per_drug": results}
    with open(RESULTS / fn_out, "w") as f:
        json.dump(out, f, indent=2)
    return results


def main():
    print("=" * 78)
    print("Phase 2.A: Hybrid candidate-pool expansion (top-100 → top-200)")
    print("=" * 78)

    client = SonnetClient()

    # Load substrate
    with open(RESULTS / "scm_edges_blended_j.json") as f:
        edges = json.load(f)
    with open(RESULTS / "scm_edges_signed.json") as f:
        signed_payload = json.load(f)
    signed_edges = signed_payload.get("edges", {})
    target_action_n = signed_payload.get("target_action_n_drugs", {})
    with open(RESULTS / "side_effect_vocab.json") as f:
        vocab_payload = json.load(f)
    se_vocab = vocab_payload["umls_ids"]
    se_names = vocab_payload["display_names"]
    vocab_set = set(se_vocab)
    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    target_info = {t["uniprot"]: t for t in tv["targets"]}
    target_set = set(target_info.keys())
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    drugs_by_name = {(d["drug_name"] or "").lower(): d for d in cat["drugs"]}

    curated_priors = load_curated_priors_for_override()

    # Run L.1 (or load existing result if already saved at top-200)
    l1_path = RESULTS / "phase_2a_l1_safety.json"
    if l1_path.exists():
        print(f"[L.1] using existing result at {l1_path}")
        with open(l1_path) as f:
            l1_results = json.load(f)["per_drug"]
    else:
        l1_results = run_l1(client, edges, signed_edges, target_action_n,
                             target_info, vocab_payload, se_vocab, se_names,
                             curated_priors)

    # Run main + OOD
    main_results = run_curated_set(
        EXPANDED_SAFETY_CASES_V5, "MAIN", "phase_2a_main_safety.json",
        client, edges, signed_edges, target_action_n, target_info,
        vocab_payload, se_vocab, se_names, curated_priors,
        vocab_set, target_set, drugs_by_name,
    )
    ood_results = run_curated_set(
        EXPANDED_OOD_CASES_V3, "OOD", "phase_2a_ood_safety.json",
        client, edges, signed_edges, target_action_n, target_info,
        vocab_payload, se_vocab, se_names, curated_priors,
        vocab_set, target_set, drugs_by_name,
    )

    # ---------- Report ----------
    print("\n\n" + "=" * 78)
    print("Phase 2.A REPORT")
    print("=" * 78)

    # L.1 paired vs Sprint L
    with open(RESULTS / "sprint_l_external_safety.json") as f:
        l1_baseline = json.load(f)["per_drug"]
    n_l1 = sum(1 for r in l1_results if not r.get("skipped"))
    h_l1 = _hit_at_k(l1_results, "hybrid_rank", 10)
    lo_l1, hi_l1 = _bootstrap_ci(l1_results, "hybrid_rank", 10)
    b, c, p_new, p_old, paired = _paired_mcnemar(l1_results, l1_baseline)
    print(f"\n[L.1] n={n_l1}")
    print(f"  Sprint L L.1 (top-100): 29/80 = 36.2%")
    print(f"  Phase 2.A (top-200):    {h_l1}/{n_l1} = {h_l1/max(n_l1,1):.1%} "
          f"[95%CI {lo_l1:.1%}-{hi_l1:.1%}]")
    print(f"  Paired n={paired}, b={b} (2A wins), c={c} (L wins), "
          f"McNemar p(2A>L)={p_new:.4f}, p(L>2A)={p_old:.4f}")

    # Main paired vs Sprint K
    with open(RESULTS / "sprint_k_safety_sonnet.json") as f:
        main_baseline = json.load(f)["per_drug"]
    n_main = sum(1 for r in main_results if not r.get("skipped"))
    h_main = _hit_at_k(main_results, "hybrid_rank", 10)
    lo_m, hi_m = _bootstrap_ci(main_results, "hybrid_rank", 10)
    b_m, c_m, p_m_new, p_m_old, paired_m = _paired_mcnemar(main_results, main_baseline)
    print(f"\n[MAIN] n={n_main}")
    print(f"  Sprint K main (top-100): 198/224 = 88.4%")
    print(f"  Phase 2.A (top-200):     {h_main}/{n_main} = {h_main/max(n_main,1):.1%} "
          f"[95%CI {lo_m:.1%}-{hi_m:.1%}]")
    print(f"  Paired n={paired_m}, b={b_m} (2A wins), c={c_m} (K wins), "
          f"McNemar p(2A>K)={p_m_new:.4f}, p(K>2A)={p_m_old:.4f}")

    # OOD paired vs Sprint K
    with open(RESULTS / "sprint_k_ood_safety_sonnet.json") as f:
        ood_baseline = json.load(f)["per_drug"]
    n_ood = sum(1 for r in ood_results if not r.get("skipped"))
    h_ood = _hit_at_k(ood_results, "hybrid_rank", 10)
    lo_o, hi_o = _bootstrap_ci(ood_results, "hybrid_rank", 10)
    b_o, c_o, p_o_new, p_o_old, paired_o = _paired_mcnemar(ood_results, ood_baseline)
    print(f"\n[OOD] n={n_ood}")
    print(f"  Sprint K OOD (top-100): 79/97 = 81.4%")
    print(f"  Phase 2.A (top-200):    {h_ood}/{n_ood} = {h_ood/max(n_ood,1):.1%} "
          f"[95%CI {lo_o:.1%}-{hi_o:.1%}]")
    print(f"  Paired n={paired_o}, b={b_o} (2A wins), c={c_o} (K wins), "
          f"McNemar p(2A>K)={p_o_new:.4f}, p(K>2A)={p_o_old:.4f}")

    # Regression evaluation (pre-reg lower bounds)
    main_rate = h_main / max(n_main, 1)
    ood_rate = h_ood / max(n_ood, 1)
    main_regressed_soft = main_rate < 0.84
    main_regressed_hard = main_rate < 0.80
    ood_regressed_soft = ood_rate < 0.75
    ood_regressed_hard = ood_rate < 0.70

    print("\n[Regression check]")
    print(f"  Main rate {main_rate:.1%}: {'HARD REGRESSION' if main_regressed_hard else ('SOFT REG' if main_regressed_soft else 'no regression')}")
    print(f"  OOD rate  {ood_rate:.1%}: {'HARD REGRESSION' if ood_regressed_hard else ('SOFT REG' if ood_regressed_soft else 'no regression')}")

    # Decision
    l1_rate = h_l1 / max(n_l1, 1)
    decision = _decide_l1(l1_rate, p_new,
                          main_regressed_hard, ood_regressed_hard)
    print(f"\n[DECISION per Phase 2.A pre-reg]: {decision}")

    # Save summary
    summary = {
        "phase": "2A",
        "intervention": "top_k_scm_candidates: 100 → 200",
        "l1": {
            "n": n_l1, "hit10": h_l1, "rate": l1_rate,
            "ci95": [lo_l1, hi_l1],
            "paired_vs_sprint_l": {"b": b, "c": c, "p_new_better": p_new,
                                    "p_old_better": p_old, "paired_n": paired},
            "baseline_rate": 0.362,
        },
        "main": {
            "n": n_main, "hit10": h_main, "rate": main_rate,
            "ci95": [lo_m, hi_m],
            "paired_vs_sprint_k": {"b": b_m, "c": c_m, "p_new_better": p_m_new,
                                    "p_old_better": p_m_old, "paired_n": paired_m},
            "baseline_rate": 0.884,
            "regression_soft": main_regressed_soft,
            "regression_hard": main_regressed_hard,
        },
        "ood": {
            "n": n_ood, "hit10": h_ood, "rate": ood_rate,
            "ci95": [lo_o, hi_o],
            "paired_vs_sprint_k": {"b": b_o, "c": c_o, "p_new_better": p_o_new,
                                    "p_old_better": p_o_old, "paired_n": paired_o},
            "baseline_rate": 0.814,
            "regression_soft": ood_regressed_soft,
            "regression_hard": ood_regressed_hard,
        },
        "decision": decision,
    }
    with open(RESULTS / "phase_2a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {RESULTS / 'phase_2a_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
