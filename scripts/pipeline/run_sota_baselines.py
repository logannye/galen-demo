"""Sprint 2.2: SOTA baseline ladder — full ablation.

Compares the SCM against:
  - Random (cached from Sprint 1)
  - Base rate (training set frequency)
  - Drug-Jaccard k-NN (k=5)
  - max-α, sum-α, mean-α aggregation (uses SAME α as SCM)
  - LogReg on binary target indicators
  - RF-ECFP on Morgan fingerprints (chemistry-only)
  - GCN-style matrix-factorization on drug-target-SE graph
  - LLM-drug-blind (cached from Sprint 1)
  - LLM-with-name (cached from Sprint 1)
  - SCM (noisy-OR, cached from Sprint 1)

Outputs:
  results/sprint2_2_per_drug.json     — per-drug metrics across all methods
  results/sprint2_2_summary.json      — aggregate + Wilcoxon p-values + CIs
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from ..baselines.gcn_baseline import (
    fit_decagon_style_embeddings, rank_test_drug_gcn,
)
from ..baselines.logreg_baseline import (
    build_se_label_matrix, build_target_feature_matrix,
    rank_test_drug_logreg, train_logreg_models,
)
from ..baselines.rf_ecfp_baseline import (
    build_ecfp_matrix, fetch_smiles_for_drugs,
    rank_test_drug_rf, train_rf_models,
)
from ..baselines.simple_baselines import (
    rank_by_alpha_aggregation, rank_by_base_rate,
    rank_by_jaccard_knn,
)
from ..evaluation.metrics import per_drug_metrics


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def _wilcoxon_one_sided(diffs: list[float]) -> tuple[int, float, float]:
    """One-sided paired Wilcoxon signed-rank, H1: mean diff > 0."""
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


def main() -> int:
    print("=" * 78)
    print("Sprint 2.2: SOTA baseline ladder — full ablation (no new LLM calls)")
    print("=" * 78)

    # Load catalog + vocab + Sprint 1 results
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    drugs = cat["drugs"]
    train_drugs = [d for d in drugs if d["split"] == "train"]
    test_drugs = [d for d in drugs if d["split"] == "test"]
    print(f"[1/8] train={len(train_drugs)}, test={len(test_drugs)}")

    with open(RESULTS / "side_effect_vocab.json") as f:
        vocab_payload = json.load(f)
    se_vocab = vocab_payload["umls_ids"]
    print(f"      side-effect vocab: {len(se_vocab)}")

    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    target_list = sorted({t["uniprot"] for t in tv["targets"]})
    print(f"      target vocab: {len(target_list)}")

    with open(RESULTS / "scm_edges.json") as f:
        edges = json.load(f)
    print(f"      SCM α edges: {len(edges)} targets")

    with open(RESULTS / "sprint1_per_drug.json") as f:
        s1 = json.load(f)
    s1_by_cid = {r["cid"]: r for r in s1["per_drug"]}

    # ---- Build training feature/label matrices for sklearn baselines
    print("[2/8] Building feature/label matrices...")
    X_train_targets = build_target_feature_matrix(train_drugs, target_list)
    Y_train = build_se_label_matrix(train_drugs, se_vocab)
    X_test_targets = build_target_feature_matrix(test_drugs, target_list)
    print(f"      X_train_targets {X_train_targets.shape}, Y_train {Y_train.shape}")
    print(f"      X_test_targets  {X_test_targets.shape}")
    base_rate_array = np.clip(Y_train.mean(axis=0), 1e-6, 1.0 - 1e-6)

    # ---- LogReg per-side-effect (target-binary features)
    print("[3/8] Training LogReg models per side effect...")
    t0 = time.monotonic()
    lr_models = train_logreg_models(X_train_targets, Y_train, C=1.0)
    n_fit_lr = sum(1 for m in lr_models.values() if m is not None)
    print(f"      LogReg fit: {n_fit_lr}/{len(se_vocab)} side effects ({time.monotonic()-t0:.1f}s)")

    # ---- RF-ECFP on chemistry
    print("[4/8] Fetching SMILES + computing ECFP fingerprints...")
    smiles_map = fetch_smiles_for_drugs(drugs)
    print(f"      SMILES fetched for {len(smiles_map)}/{len(drugs)} drugs")
    X_train_ecfp, train_ecfp_idx = build_ecfp_matrix(train_drugs, smiles_map)
    X_test_ecfp, test_ecfp_idx = build_ecfp_matrix(test_drugs, smiles_map)
    print(f"      X_train_ecfp {X_train_ecfp.shape}, X_test_ecfp {X_test_ecfp.shape}")
    print(f"      training RF models per side effect (this can take a few minutes)...")
    t0 = time.monotonic()
    Y_train_ecfp = Y_train[train_ecfp_idx]
    rf_models = train_rf_models(X_train_ecfp, Y_train_ecfp,
                                  n_estimators=50, max_depth=8)
    n_fit_rf = sum(1 for m in rf_models.values() if m is not None)
    print(f"      RF fit: {n_fit_rf}/{len(se_vocab)} side effects ({time.monotonic()-t0:.1f}s)")

    # ---- GCN-style embeddings
    print("[5/8] Fitting Decagon-style embeddings...")
    target_emb, se_emb, W_bridge = fit_decagon_style_embeddings(
        X_train_targets.astype(np.float32), Y_train, dim=64,
    )
    print(f"      target_emb {target_emb.shape}, se_emb {se_emb.shape}, "
          f"W_bridge {W_bridge.shape}")

    # ---- Base rate ranking (constant per drug)
    print("[6/8] Computing baseline rankings on test set...")
    base_rate_ranking = rank_by_base_rate(train_drugs, se_vocab)

    # ---- For each test drug, compute all baseline rankings
    results: list[dict] = []
    t_start = time.monotonic()
    test_ecfp_set = set(test_ecfp_idx)

    for i, d in enumerate(test_drugs):
        cid = d["cid"]
        gold = set(d["side_effects_in_vocab"])
        s1_rec = s1_by_cid.get(cid, {})
        n_targets = d["n_targets"]

        rankings: dict[str, list[str]] = {}

        # Cached from Sprint 1
        rankings["scm"] = s1_rec.get("scm_top10", []) + [
            s for s in se_vocab if s not in s1_rec.get("scm_top10", [])
        ] if "scm_top10" in s1_rec else []  # fallback never triggered; sprint1 has top10
        rankings["llm_drug_blind"] = s1_rec.get("llm_drug_blind_top10", [])
        rankings["llm_with_name"] = s1_rec.get("llm_with_name_top10", [])

        # NB: Sprint 1 saved only top-10 for LLMs; use top-10 ranking with everything
        # else trailing. The metrics functions handle short rankings fine for
        # P@10 / R@10. For AP we want the full ranking — recompute SCM fully
        # here and load full LLM rankings only if available.
        # We DO have full SCM ranking via re-evaluation:
        from ..scm.scoring import top_k_predictions
        scm_full = top_k_predictions(
            d["binding_profile"], edges, se_vocab, k=len(se_vocab),
        )
        rankings["scm"] = scm_full

        # Simple baselines
        rankings["base_rate"] = base_rate_ranking
        rankings["jaccard_knn5"] = rank_by_jaccard_knn(d, train_drugs, se_vocab, k=5)
        rankings["max_alpha"] = rank_by_alpha_aggregation(d, edges, se_vocab, op="max")
        rankings["sum_alpha"] = rank_by_alpha_aggregation(d, edges, se_vocab, op="sum")
        rankings["mean_alpha"] = rank_by_alpha_aggregation(d, edges, se_vocab, op="mean")

        # LogReg
        rankings["logreg_targets"] = rank_test_drug_logreg(
            X_test_targets[i], lr_models, se_vocab, base_rate_array,
        )

        # RF-ECFP — only if we have a fingerprint for this drug
        if i in test_ecfp_set:
            test_idx_in_ecfp = test_ecfp_idx.index(i)
            rankings["rf_ecfp"] = rank_test_drug_rf(
                X_test_ecfp[test_idx_in_ecfp], rf_models,
                se_vocab, base_rate_array,
            )
        else:
            rankings["rf_ecfp"] = base_rate_ranking  # fallback

        # GCN-style
        rankings["gcn_decagon"] = rank_test_drug_gcn(
            X_test_targets[i].astype(np.float32), W_bridge, se_emb, se_vocab,
        )

        # Compute metrics for each method
        method_metrics: dict[str, dict[str, float]] = {}
        for name, rnk in rankings.items():
            if rnk:
                method_metrics[name] = per_drug_metrics(rnk, gold)
            else:
                method_metrics[name] = per_drug_metrics([], gold)

        # Also copy LLM AP / P@10 from Sprint 1 (we have the actual numbers stored)
        if s1_rec:
            method_metrics["llm_drug_blind"] = s1_rec["llm_drug_blind_metrics"]
            method_metrics["llm_with_name"] = s1_rec["llm_with_name_metrics"]
            method_metrics["random"] = s1_rec["random_metrics"]

        results.append({
            "cid": cid, "drug_name": d["drug_name"],
            "n_targets": n_targets,
            "n_side_effects_gold": len(gold),
            "metrics": method_metrics,
        })

        if (i + 1) % 20 == 0:
            print(f"      processed {i+1}/{len(test_drugs)} drugs "
                  f"({(time.monotonic() - t_start) / 60:.1f}m)")

    # Save per-drug results
    per_drug_out = RESULTS / "sprint2_2_per_drug.json"
    with open(per_drug_out, "w") as f:
        json.dump({"n_cases": len(results), "per_drug": results}, f, indent=2)
    print(f"\n[7/8] saved per-drug: {per_drug_out}")

    # Aggregate
    method_names = [
        "random", "base_rate", "jaccard_knn5",
        "max_alpha", "mean_alpha", "sum_alpha",
        "logreg_targets", "rf_ecfp", "gcn_decagon",
        "llm_drug_blind", "llm_with_name",
        "scm",
    ]

    def _summary(method: str) -> dict[str, float]:
        out = {}
        for key in ("ap", "p@10", "p@20", "p@50", "r@10", "r@20", "r@50", "rr"):
            vals = [r["metrics"][method][key] for r in results
                    if method in r["metrics"]]
            if vals:
                out[key] = {
                    "mean": statistics.mean(vals),
                    "median": statistics.median(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                }
        return out

    summaries = {m: _summary(m) for m in method_names}

    # Wilcoxon: SCM vs each other method on AP
    wilcoxon = {}
    for m in method_names:
        if m == "scm":
            continue
        diffs = []
        for r in results:
            if m in r["metrics"]:
                diffs.append(r["metrics"]["scm"]["ap"] - r["metrics"][m]["ap"])
        n, z, p = _wilcoxon_one_sided(diffs)
        wilcoxon[f"scm_vs_{m}_AP_one_sided"] = {
            "n": n, "z": z, "p_one_sided": p,
            "mean_diff": statistics.mean(diffs) if diffs else 0.0,
            "scm_wins": sum(1 for d in diffs if d > 0),
            "ties": sum(1 for d in diffs if d == 0),
            "method_wins": sum(1 for d in diffs if d < 0),
        }

    sum_out = {
        "n_cases": len(results),
        "summaries": summaries,
        "wilcoxon_scm_vs_each_AP": wilcoxon,
    }
    sum_path = RESULTS / "sprint2_2_summary.json"
    with open(sum_path, "w") as f:
        json.dump(sum_out, f, indent=2, default=str)
    print(f"[8/8] saved summary: {sum_path}")

    # Console table
    print("\n" + "=" * 78)
    print(f"Sprint 2.2 SOTA LADDER (n={len(results)})")
    print("=" * 78)
    print(f"\n{'method':<22s} {'AP (MAP)':<10s} {'P@10':<10s} {'R@10':<10s} {'MRR':<10s}")
    for m in method_names:
        if m not in summaries or not summaries[m]:
            continue
        s = summaries[m]
        if "ap" in s:
            print(f"{m:<22s} {s['ap']['mean']:<10.4f} {s.get('p@10', {}).get('mean', 0):<10.4f} "
                  f"{s.get('r@10', {}).get('mean', 0):<10.4f} {s.get('rr', {}).get('mean', 0):<10.4f}")

    print(f"\nWilcoxon SCM vs each baseline (one-sided H1: SCM > baseline):")
    for k, w in wilcoxon.items():
        print(f"  {k}: n={w['n']} z={w['z']:+.2f} "
              f"mean_diff={w['mean_diff']:+.4f} "
              f"p={w['p_one_sided']:.4g} "
              f"SCM_wins={w['scm_wins']}/{w['scm_wins']+w['method_wins']+w['ties']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
