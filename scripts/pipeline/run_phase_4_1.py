"""Phase 4.1: Re-run Hybrid with new distinct-concept ranking prompt.

Only Hybrid is re-evaluated (other arms reused from Sprint K).
Applies Phase 3 cluster collapse + crediting (production default).

Pre-registered in docs/PHASE_4_1_PRE_REGISTRATION.md.
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

from ..baselines.ae_cluster_postprocess import (
    collapse_top_k, hit_at_k_clustered,
)
from ..baselines.curated_prior_vote import (
    apply_curated_prior_override_v2, load_curated_priors_for_override,
)
from ..baselines.llm_hybrid_reranker import hybrid_rerank, load_action_types
from ..data.biologic_binding_profiles import get_biologic_binding
from ..data.build_catalog import query_binding_profile
from ..data.clinical_safety_benchmark import passes_eligibility
from ..data.clinical_safety_benchmark_v5 import EXPANDED_SAFETY_CASES_V5
from ..data.clinical_safety_benchmark_ood_v3 import EXPANDED_OOD_CASES_V3
from ..demo.scm_explainer import explain_predictions
from ..llm import SonnetClient
from ..pipeline.run_sprint3_clinical_failures import lookup_chembl_molregno
from ..pipeline.run_sprint7_eval import classify_ta
from ..pipeline.run_sprint_f_eval import best_causal_rank
from ..scm.scoring import score_drug_side_effects_signed

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"
N_WORKERS = 16

AFFINITY_MODE = "log_sigmoid"
TOP_K_POOL = 100


def reconstruct_binding(name, drugs_by_name):
    cat_drug = drugs_by_name.get((name or "").lower())
    if cat_drug is not None:
        return cat_drug["binding_profile"], cat_drug["molregno"]
    molregno = lookup_chembl_molregno(name)
    if molregno:
        conn = sqlite3.connect(CHEMBL_DB)
        bp = query_binding_profile(conn, molregno)
        conn.close()
        if bp:
            return bp, molregno
    bp = get_biologic_binding(name)
    if bp:
        return bp, None
    return [], None


def _process_case(
    cf, edges, signed_edges, target_action_n, target_info,
    vocab_payload, se_vocab, se_names, drugs_by_name,
    client, curated_priors,
):
    bp, molregno = reconstruct_binding(cf.drug_search_name, drugs_by_name)
    if not bp:
        return {
            "drug_id": cf.drug_id, "drug_search_name": cf.drug_search_name,
            "skipped": True, "reason": "no binding profile",
        }

    ta = classify_ta(cf.drug_id, cf.severity, cf.causal_off_target_gene)
    action_types = {}
    if molregno is not None:
        try:
            action_types = load_action_types(molregno)
        except Exception:
            pass

    scored = score_drug_side_effects_signed(
        bp, edges, signed_edges, action_types, target_action_n, se_vocab,
        min_drugs_for_signed=3, affinity_mode=AFFINITY_MODE,
    )
    explanations = explain_predictions(
        scored[:TOP_K_POOL], bp, edges, target_info, se_names,
        top_k_se=TOP_K_POOL, top_k_targets=5,
    )
    scm_ranked = [s for s, _ in scored]

    hybrid = hybrid_rerank(
        bp, scored, explanations, vocab_payload, client=client,
        top_k_scm_candidates=TOP_K_POOL, therapeutic_area=ta,
        action_types=action_types,
    )
    hybrid_ranked, _ = apply_curated_prior_override_v2(
        hybrid.ranked_side_effects, scm_ranked, bp, curated_priors,
    )
    return {
        "drug_id": cf.drug_id,
        "drug_search_name": cf.drug_search_name,
        "therapeutic_area": ta,
        "severity": cf.severity,
        "causal_off_target": cf.causal_off_target_gene,
        "causal_side_effects_umls": list(cf.causal_side_effects_umls),
        "hybrid_top10": hybrid_ranked[:10],
        "skipped": False,
    }


def run_stage(cases, label, fn_out, client, edges, signed_edges,
              target_action_n, target_info, vocab_payload, se_vocab,
              se_names, curated_priors, vocab_set, target_set,
              drugs_by_name):
    print("\n" + "=" * 78)
    print(f"Phase 4.1 — {label} (n={len(cases)})")
    print("=" * 78)
    eligible = [c for c in cases if passes_eligibility(c, vocab_set, target_set)]
    print(f"[{label}] eligible: {len(eligible)}/{len(cases)}")

    results = [None] * len(eligible)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(_process_case, cf, edges, signed_edges,
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
           "per_drug": results}
    with open(RESULTS / fn_out, "w") as f:
        json.dump(out, f, indent=2)
    return results


def mcnemar_one_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    p = 0.0
    for x in range(b, n + 1):
        p += math.comb(n, x) * (0.5 ** n)
    return p


def bootstrap_ci(hits, n, n_boot=1000, seed=42):
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    arr = np.array([1] * hits + [0] * (n - hits))
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot.append(arr[idx].mean())
    boot.sort()
    return (boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)])


def main():
    print("=" * 78)
    print("Phase 4.1: distinct-concept ranking prompt - Sonnet Hybrid")
    print("=" * 78)

    client = SonnetClient()
    edges = json.load(open(RESULTS / "scm_edges_blended_j.json"))
    signed = json.load(open(RESULTS / "scm_edges_signed.json"))
    signed_edges = signed.get("edges", {})
    target_action_n = signed.get("target_action_n_drugs", {})
    vocab_payload = json.load(open(RESULTS / "side_effect_vocab.json"))
    se_vocab = vocab_payload["umls_ids"]
    se_names = vocab_payload["display_names"]
    vocab_set = set(se_vocab)
    tv = json.load(open(RESULTS / "target_vocab.json"))
    target_info = {t["uniprot"]: t for t in tv["targets"]}
    target_set = set(target_info.keys())
    cat = json.load(open(RESULTS / "catalog.json"))
    drugs_by_name = {(d["drug_name"] or "").lower(): d for d in cat["drugs"]}
    curated_priors = load_curated_priors_for_override()

    # Run both stages
    main_results = run_stage(
        EXPANDED_SAFETY_CASES_V5, "MAIN", "phase_4_1_main.json",
        client, edges, signed_edges, target_action_n, target_info,
        vocab_payload, se_vocab, se_names, curated_priors,
        vocab_set, target_set, drugs_by_name,
    )
    ood_results = run_stage(
        EXPANDED_OOD_CASES_V3, "OOD", "phase_4_1_ood.json",
        client, edges, signed_edges, target_action_n, target_info,
        vocab_payload, se_vocab, se_names, curated_priors,
        vocab_set, target_set, drugs_by_name,
    )

    # ---------- Analyze ----------
    print("\n" + "=" * 78)
    print("Phase 4.1 RESULTS (cluster-aware on Phase 3 collapse)")
    print("=" * 78)

    all_new = [dict(r, dataset="main") for r in main_results] + \
               [dict(r, dataset="ood") for r in ood_results]
    base_main = json.load(open(RESULTS / "sprint_k_safety_sonnet.json"))["per_drug"]
    base_ood = json.load(open(RESULTS / "sprint_k_ood_safety_sonnet.json"))["per_drug"]
    base_all = [dict(r, dataset="main") for r in base_main] + \
                [dict(r, dataset="ood") for r in base_ood]

    new_by = {r["drug_id"]: r for r in all_new if not r.get("skipped")}
    base_by = {r["drug_id"]: r for r in base_all if not r.get("skipped")}
    paired_ids = set(new_by) & set(base_by)

    def hit_cluster(rec, k):
        return hit_at_k_clustered(
            set(rec.get("causal_side_effects_umls") or []),
            collapse_top_k(rec.get("hybrid_top10") or []), k,
        )

    def compute_subset(ids):
        out = {"n": len(ids)}
        for k in (1, 3, 5, 10):
            new_h = sum(1 for i in ids if hit_cluster(new_by[i], k))
            base_h = sum(1 for i in ids if hit_cluster(base_by[i], k))
            b = sum(1 for i in ids
                     if hit_cluster(new_by[i], k) and not hit_cluster(base_by[i], k))
            c = sum(1 for i in ids
                     if hit_cluster(base_by[i], k) and not hit_cluster(new_by[i], k))
            p_new = mcnemar_one_sided(b, c)
            new_ci = bootstrap_ci(new_h, len(ids))
            out[f"hit@{k}"] = {
                "new": new_h, "base": base_h,
                "new_rate": new_h / max(len(ids), 1),
                "base_rate": base_h / max(len(ids), 1),
                "ci_new": new_ci,
                "b": b, "c": c, "p_new_better": p_new,
            }
        return out

    oi_ids = [i for i in paired_ids
              if new_by[i].get("therapeutic_area") in ("Oncology", "Immunology")]
    full_ids = list(paired_ids)
    oi_summary = compute_subset(oi_ids)
    full_summary = compute_subset(full_ids)

    def print_subset(label, sm):
        print(f"\n[{label}] n={sm['n']}")
        for k in (1, 3, 5, 10):
            d = sm[f"hit@{k}"]
            print(f"  hit@{k}: base={d['base']}/{sm['n']} ({d['base_rate']:.1%}) -> "
                  f"new={d['new']}/{sm['n']} ({d['new_rate']:.1%}) "
                  f"[CI {d['ci_new'][0]:.0%}-{d['ci_new'][1]:.0%}]  "
                  f"b={d['b']} c={d['c']} p={d['p_new_better']:.4f}")

    print_subset("Onc + Immuno", oi_summary)
    print_subset("Full Sprint K", full_summary)

    print("\nPer-TA (Onc+Immuno only):")
    for ta in ("Oncology", "Immunology"):
        ta_ids = [i for i in paired_ids
                   if new_by[i].get("therapeutic_area") == ta]
        if ta_ids:
            ta_summary = compute_subset(ta_ids)
            print_subset(ta, ta_summary)

    # Decision per pre-reg
    h1 = oi_summary["hit@1"]["new_rate"]
    h3 = oi_summary["hit@3"]["new_rate"]
    h10 = oi_summary["hit@10"]["new_rate"]
    full_h10 = full_summary["hit@10"]["new_rate"]
    p1 = oi_summary["hit@1"]["p_new_better"]
    no_reg = full_h10 >= 0.87

    if h1 >= 0.78 and h3 >= 0.91 and no_reg:
        decision = "STRONG WIN"
    elif h1 >= 0.74 and h3 >= 0.90 and no_reg:
        decision = "MODERATE WIN"
    elif h1 >= 0.70 and h1 < 0.74:
        decision = "NULL"
    elif h1 < 0.70 or not no_reg:
        decision = "LOSS"
    else:
        decision = "NULL"

    print("\n" + "=" * 78)
    print(f"DECISION per pre-reg: {decision}")
    print(f"  OI hit@1: {h1:.1%} (target STRONG >=78%, MODERATE >=74%)")
    print(f"  OI hit@3: {h3:.1%} (target STRONG >=91%, MODERATE >=90%)")
    print(f"  OI hit@10: {h10:.1%}")
    print(f"  Full hit@10: {full_h10:.1%} (regression check >=87%)")
    print(f"  Paired McNemar OI hit@1: p={p1:.4f}")

    out = {
        "decision": decision,
        "oi_summary": oi_summary,
        "full_summary": full_summary,
    }
    out_path = RESULTS / "phase_4_1_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
