"""Phase 4.3: External onc/immuno held-out validation.

Runs Phase 4.1 production stack on 30 onc/immuno drugs NOT in any
prior benchmark (SIDER 247-train, V5 main, V3 OOD). GT from OnSIDES
at pred1 >= 4.5.

Pre-registered in docs/PHASE_4_3_PRE_REGISTRATION.md.
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
from ..demo.scm_explainer import explain_predictions
from ..llm import SonnetClient
from ..pipeline.run_sprint3_clinical_failures import lookup_chembl_molregno
from ..pipeline.run_sprint_f_eval import best_causal_rank
from ..scm.scoring import score_drug_side_effects_signed

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"
N_WORKERS = 16
AFFINITY_MODE = "log_sigmoid"
TOP_K_POOL = 100


def reconstruct_binding(name, molregno=None):
    if molregno:
        try:
            conn = sqlite3.connect(CHEMBL_DB)
            bp = query_binding_profile(conn, molregno)
            conn.close()
            if bp:
                return bp, molregno
        except Exception:
            pass
    mr = lookup_chembl_molregno(name)
    if mr:
        try:
            conn = sqlite3.connect(CHEMBL_DB)
            bp = query_binding_profile(conn, mr)
            conn.close()
            if bp:
                return bp, mr
        except Exception:
            pass
    bp = get_biologic_binding(name) or []
    return bp, None


def _process_case(
    case, edges, signed_edges, target_action_n, target_info,
    vocab_payload, se_vocab, se_names, client, curated_priors,
):
    name = case['name']
    bp, molregno = reconstruct_binding(name, case.get('molregno'))
    if not bp:
        return {"name": name, "skipped": True, "reason": "no binding profile"}

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
    gt = case['gt_umls']
    scm_rank = best_causal_rank(scm_ranked, tuple(gt))

    # Drug-name override for ambiguous targets (VEGFA can be onc or ophth)
    DRUG_NAME_OVERRIDES = {
        'bevacizumab': 'Oncology', 'ranibizumab': 'Ophthalmology',
        'faricimab': 'Ophthalmology', 'tacrolimus': 'Immunology',
        'pimecrolimus': 'Immunology', 'fremanezumab': 'Neurology',
        'alemtuzumab': 'Oncology',
    }

    # Therapeutic area heuristic: oncology if primary target is onc-class,
    # immunology if immuno-class, otherwise Other.
    ONC_TARGETS = {"EGFR","ERBB2","KDR","FLT3","FLT1","MET","ALK","BRAF","RAF1",
                    "MAP2K1","MAP2K2","CDK4","CDK6","MTOR","PARP1","TOP2A",
                    "PDCD1","CD274","CTLA4","LAG3","CD19","MS4A1","TNFRSF17",
                    "NECTIN4","TACSTD2","CD22","TNFRSF8","KRAS","KIT","SYK",
                    "PIK3CA","PIK3CD","PIK3CG","IDH1","IDH2","BTK","BCL2",
                    "PDGFRA","JAK2","ABL1","ROS1","RET","FGFR1","FGFR2",
                    "FGFR3","AURKA","NTRK1","NTRK2","ESR1","AR","HDAC1",
                    "EZH2","CD52","DLL3"}
    IMMUNO_TARGETS = {"TNF","IL17A","IL17RA","IL6","IL6R","IL23A","IL23R","IL4R",
                       "IL5","IL5RA","IL13","IL13RA","IL31RA",
                       "JAK1","JAK3","TYK2","S1PR1","ITGA4","C5","CD20","CD40",
                       "FKBP1A","IGHE","IL1B","IL1R1","IL1RN","IL2RA",
                       "C5AR1","C1S","C3","CSF1R","TLR4","CD11A","ITGAL"}
    OPHTHALMOLOGY_TARGETS = {"VEGFA"}  # VEGFA used intraocularly is ophth, systemic is onc
    pt = case.get('primary_target', '')
    nm = case.get('name', '').lower()
    if nm in DRUG_NAME_OVERRIDES:
        ta = DRUG_NAME_OVERRIDES[nm]
    elif pt in ONC_TARGETS:
        ta = "Oncology"
    elif pt in IMMUNO_TARGETS:
        ta = "Immunology"
    else:
        ta = "Other"

    hybrid = hybrid_rerank(
        bp, scored, explanations, vocab_payload, client=client,
        top_k_scm_candidates=TOP_K_POOL, therapeutic_area=ta,
        action_types=action_types,
    )
    hybrid_ranked, _ = apply_curated_prior_override_v2(
        hybrid.ranked_side_effects, scm_ranked, bp, curated_priors,
    )
    hybrid_rank = best_causal_rank(hybrid_ranked, tuple(gt))

    return {
        "name": name,
        "primary_target": pt,
        "therapeutic_area": ta,
        "gt_umls": gt,
        "n_gt": len(gt),
        "n_targets": len(bp),
        "scm_rank": scm_rank,
        "hybrid_rank": hybrid_rank,
        "hybrid_top10": hybrid_ranked[:10],
        "skipped": False,
    }


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
    print("Phase 4.3: External onc/immuno held-out validation (n=30)")
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

    cases = json.load(open(RESULTS / "phase_4_3_cases.json"))["cases"]
    print(f"\nLoaded {len(cases)} held-out onc/immuno cases")

    results = [None] * len(cases)
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(_process_case, c, edges, signed_edges,
                       target_action_n, target_info, vocab_payload, se_vocab,
                       se_names, client, curated_priors): i
            for i, c in enumerate(cases)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"  case {idx} FAILED: {e}", flush=True)
                results[idx] = {"name": cases[idx]["name"],
                                "skipped": True, "reason": str(e)}
            done += 1
            if done % 5 == 0:
                el = time.monotonic() - t0
                print(f"  {done}/{len(cases)} ({el/60:.1f}m)", flush=True)
    results = [r for r in results if r is not None]
    print(f"\nDone in {(time.monotonic() - t0)/60:.1f}m")

    out_path = RESULTS / "phase_4_3_results.json"
    with open(out_path, "w") as f:
        json.dump({"n_cases": sum(1 for r in results if not r.get("skipped")),
                   "per_drug": results}, f, indent=2)

    # ---------- Analyze ----------
    print("\n" + "=" * 78)
    print("Phase 4.3 RESULTS")
    print("=" * 78)
    valid = [r for r in results if not r.get("skipped")]
    n = len(valid)

    def hit_lit(rec, k):
        return bool(set(rec.get("hybrid_top10", [])[:k]) & set(rec.get("gt_umls", [])))

    def hit_clust(rec, k):
        return hit_at_k_clustered(
            set(rec.get("gt_umls", [])),
            collapse_top_k(rec.get("hybrid_top10", [])), k,
        )

    print(f"\nValid cases: {n}/{len(results)} (some may lack binding profile)")
    print(f"\n{'metric':<8s} {'LITERAL':<25s} {'CLUSTER-AWARE':<25s}")
    summary = {}
    for k in (1, 3, 5, 10):
        h_lit = sum(1 for r in valid if hit_lit(r, k))
        h_clu = sum(1 for r in valid if hit_clust(r, k))
        ci_clu = bootstrap_ci(h_clu, n)
        print(f"  hit@{k:<3d}  {h_lit:>2d}/{n} ({h_lit/n:.1%})            "
              f"{h_clu:>2d}/{n} ({h_clu/n:.1%}) [CI {ci_clu[0]:.0%}-{ci_clu[1]:.0%}]")
        summary[f"hit@{k}"] = {
            "literal": h_lit, "cluster": h_clu,
            "literal_rate": h_lit/n, "cluster_rate": h_clu/n,
            "cluster_ci": ci_clu,
        }

    # Per-TA breakdown
    print("\nPer-TA breakdown (cluster-aware):")
    for ta in ("Oncology", "Immunology", "Other"):
        sub = [r for r in valid if r.get("therapeutic_area") == ta]
        if not sub:
            continue
        print(f"  {ta} (n={len(sub)})")
        for k in (1, 3, 5, 10):
            h = sum(1 for r in sub if hit_clust(r, k))
            print(f"    hit@{k}: {h}/{len(sub)} ({h/len(sub):.1%})")

    # Per-drug detail
    print("\nPer-drug detail (sorted by hybrid_rank):")
    for r in sorted(valid, key=lambda x: x.get("hybrid_rank") or 999):
        h_clu_3 = hit_clust(r, 3)
        rk = r.get("hybrid_rank")
        print(f"  {r['name']:<20s} target={r.get('primary_target','?'):<8s} "
              f"TA={r.get('therapeutic_area','?'):<11s} "
              f"n_gt={r.get('n_gt'):>3d} hybrid_rank={rk} "
              f"cluster_hit@3={'✓' if h_clu_3 else 'X'}")

    # Decision
    h1 = summary["hit@1"]["cluster_rate"]
    h3 = summary["hit@3"]["cluster_rate"]
    h10 = summary["hit@10"]["cluster_rate"]
    if h3 >= 0.90 and h1 >= 0.75 and h10 >= 0.95:
        decision = "STRONG GENERALIZATION"
    elif h3 >= 0.85 and h1 >= 0.70 and h10 >= 0.90:
        decision = "MODERATE GENERALIZATION"
    elif h3 < 0.80 or h1 < 0.60:
        decision = "LOSS (over-fitting detected)"
    else:
        decision = "NULL (partial generalization)"

    print("\n" + "=" * 78)
    print(f"DECISION per pre-reg: {decision}")
    print(f"  Cluster-aware hit@1:  {h1:.1%} (target STRONG >=75%, MODERATE >=70%)")
    print(f"  Cluster-aware hit@3:  {h3:.1%} (target STRONG >=90%, MODERATE >=85%)")
    print(f"  Cluster-aware hit@10: {h10:.1%} (target STRONG >=95%, MODERATE >=90%)")

    out2 = RESULTS / "phase_4_3_summary.json"
    with open(out2, "w") as f:
        json.dump({"n": n, "summary": summary, "decision": decision}, f, indent=2)
    print(f"\n[save] {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
