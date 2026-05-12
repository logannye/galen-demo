"""Phase 4.2: Held-out generalization test — apply Phase 4.1 stack to L.1.

Re-runs Sprint L's L.1 (n=80 SIDER held-out test drugs with rare-SE GT)
using the Phase 4.1 production prompt + Phase 3 cluster collapse.

Compares paired against the Sprint L baseline (Sprint K prompt, no cluster).

Pre-registered in docs/PHASE_4_2_PRE_REGISTRATION.md.
"""
from __future__ import annotations

import json
import math
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
from ..demo.scm_explainer import explain_predictions
from ..llm import SonnetClient
from ..pipeline.run_sprint_f_eval import best_causal_rank
from ..pipeline.run_sprint_l import _build_l1_cases
from ..scm.scoring import score_drug_side_effects_signed

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
N_WORKERS = 16

AFFINITY_MODE = "log_sigmoid"
TOP_K_POOL = 100


def _process_l1_case(
    cf, edges, signed_edges, target_action_n, target_info,
    vocab_payload, se_vocab, se_names, client, curated_priors,
):
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
    scm_rank = best_causal_rank(scm_ranked, causal)

    # Phase 4.1 production prompt (already merged in llm_hybrid_reranker)
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
        "n_targets": len(bp),
        "scm_rank": scm_rank,
        "hybrid_rank": hybrid_rank,
        "hybrid_top10": hybrid_ranked[:10],
        "skipped": False,
    }


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
    print("Phase 4.2: Apply Phase 4.1 stack to Sprint L L.1 held-out (n=80)")
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

    # Load L.1 cases (deterministic seed=42, same drugs as Sprint L)
    l1_cases = _build_l1_cases()
    print(f"\n[L.1] running n={len(l1_cases)}")

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

    out_path = RESULTS / "phase_4_2_l1_safety.json"
    with open(out_path, "w") as f:
        json.dump({"n_cases": sum(1 for r in results if not r.get("skipped")),
                   "per_drug": results}, f, indent=2)

    # ---------- Analyze ----------
    print("\n" + "=" * 78)
    print("Phase 4.2 RESULTS — L.1 held-out generalization")
    print("=" * 78)

    # Load Sprint L baseline (Sprint K prompt, no cluster)
    base = json.load(open(RESULTS / "sprint_l_external_safety.json"))["per_drug"]
    base_by = {r["drug_id"]: r for r in base if not r.get("skipped")}
    new_by = {r["drug_id"]: r for r in results if not r.get("skipped")}
    paired = list(set(base_by) & set(new_by))
    n = len(paired)
    print(f"\nPaired n={n}")

    def hit_literal(rec, k):
        top10 = rec.get("hybrid_top10") or []
        gt = set(rec.get("causal_se_umls") or [])
        return bool(set(top10[:k]) & gt)

    def hit_cluster_aware(rec, k):
        return hit_at_k_clustered(
            set(rec.get("causal_se_umls") or []),
            collapse_top_k(rec.get("hybrid_top10") or []), k,
        )

    print("\n=== LITERAL hit@K (apples-to-apples with Sprint L L.1 baseline) ===")
    for k in (1, 3, 5, 10):
        new_h = sum(1 for i in paired if hit_literal(new_by[i], k))
        base_h = sum(1 for i in paired if hit_literal(base_by[i], k))
        b = sum(1 for i in paired
                 if hit_literal(new_by[i], k) and not hit_literal(base_by[i], k))
        c = sum(1 for i in paired
                 if hit_literal(base_by[i], k) and not hit_literal(new_by[i], k))
        p = mcnemar_one_sided(b, c)
        ci = bootstrap_ci(new_h, n)
        print(f"  hit@{k}: Sprint L = {base_h}/{n} ({base_h/n:.1%}) → "
              f"Phase 4.2 = {new_h}/{n} ({new_h/n:.1%}) [CI {ci[0]:.0%}-{ci[1]:.0%}] "
              f"b={b} c={c} p={p:.4f}")

    print("\n=== CLUSTER-AWARE hit@K (Phase 3+4.1 production metric) ===")
    cluster_results = {}
    for k in (1, 3, 5, 10):
        new_h = sum(1 for i in paired if hit_cluster_aware(new_by[i], k))
        base_h = sum(1 for i in paired if hit_cluster_aware(base_by[i], k))
        b = sum(1 for i in paired
                 if hit_cluster_aware(new_by[i], k) and not hit_cluster_aware(base_by[i], k))
        c = sum(1 for i in paired
                 if hit_cluster_aware(base_by[i], k) and not hit_cluster_aware(new_by[i], k))
        p = mcnemar_one_sided(b, c)
        ci = bootstrap_ci(new_h, n)
        print(f"  hit@{k}: Sprint L = {base_h}/{n} ({base_h/n:.1%}) → "
              f"Phase 4.2 = {new_h}/{n} ({new_h/n:.1%}) [CI {ci[0]:.0%}-{ci[1]:.0%}] "
              f"b={b} c={c} p={p:.4f}")
        cluster_results[f"hit@{k}"] = {
            "new": new_h, "base": base_h, "rate": new_h/n,
            "b": b, "c": c, "p": p, "ci": ci,
        }

    # Decision per pre-reg (on cluster-aware hit@10)
    h10_cluster = cluster_results["hit@10"]["rate"]
    p10 = cluster_results["hit@10"]["p"]
    if h10_cluster >= 0.55 and p10 <= 0.01:
        decision = "STRONG GENERALIZATION"
    elif h10_cluster >= 0.45 and p10 <= 0.05:
        decision = "MODERATE GENERALIZATION"
    elif h10_cluster >= 0.40 and h10_cluster < 0.45:
        decision = "NULL (partial generalization)"
    elif h10_cluster <= 0.40:
        decision = "LOSS (over-fitting detected)"
    else:
        decision = "NULL"

    print("\n" + "=" * 78)
    print(f"DECISION per pre-reg: {decision}")
    print(f"  L.1 cluster-aware hit@10: {h10_cluster:.1%} (target STRONG ≥55%, MODERATE ≥45%)")
    print(f"  Paired McNemar p={p10:.4f}")

    summary = {
        "decision": decision,
        "n_paired": n,
        "cluster_aware": cluster_results,
    }
    out2 = RESULTS / "phase_4_2_summary.json"
    with open(out2, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
