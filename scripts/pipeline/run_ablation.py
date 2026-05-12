"""Sprint 1 ablation runner: SCM vs LLM-drug-blind vs LLM-with-name vs random.

Pre-registered protocol: docs/PRE_REGISTRATION.md (commit dbc2f4e).

Parallelized via ThreadPoolExecutor(max_workers=16). Bedrock LLM calls
are I/O-bound; per-drug work is independent.

Outputs:
  results/sprint1_per_drug.json — per-drug metrics across all four arms
  results/sprint1_summary.json  — aggregate stats + Wilcoxon p-values + bootstrap CIs
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..baselines.llm_baselines import (
    rank_side_effects_llm_drug_blind, rank_side_effects_llm_with_name,
)
from ..baselines.random_baseline import random_rank_side_effects
from ..evaluation.metrics import per_drug_metrics
from ..llm import SonnetClient
from ..scm.scoring import top_k_predictions


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
N_WORKERS = 16
TOP_K_RANK = 50


def _process_one_drug(
    idx: int, total: int, drug: dict,
    edges: dict, vocab_payload: dict, vocab_list: list[str],
    client: SonnetClient, t_start: float,
) -> dict:
    cid = drug["cid"]
    name = drug["drug_name"]
    bp = drug["binding_profile"]
    gold = set(drug["side_effects_in_vocab"])

    # SCM
    scm_ranked = top_k_predictions(
        bp, edges, vocab_list, k=TOP_K_RANK,
    )
    # LLM-drug-blind
    llm_blind = rank_side_effects_llm_drug_blind(
        bp, vocab_payload, client=client, top_k=TOP_K_RANK,
    )
    # LLM-with-name
    llm_name = rank_side_effects_llm_with_name(
        name, bp, vocab_payload, client=client, top_k=TOP_K_RANK,
    )
    # Random
    random_ranked = random_rank_side_effects(cid, vocab_list, k=TOP_K_RANK)

    rec = {
        "cid": cid, "drug_name": name,
        "n_targets": drug["n_targets"],
        "n_side_effects_gold": len(gold),
        "scm_metrics": per_drug_metrics(scm_ranked, gold),
        "llm_drug_blind_metrics": per_drug_metrics(
            llm_blind.ranked_side_effects, gold),
        "llm_with_name_metrics": per_drug_metrics(
            llm_name.ranked_side_effects, gold),
        "random_metrics": per_drug_metrics(random_ranked, gold),
        "scm_top10": scm_ranked[:10],
        "llm_drug_blind_top10": llm_blind.ranked_side_effects[:10],
        "llm_with_name_top10": llm_name.ranked_side_effects[:10],
        "llm_drug_blind_conf": llm_blind.confidence,
        "llm_with_name_conf": llm_name.confidence,
    }

    elapsed = time.monotonic() - t_start
    eta = elapsed * (total - idx) / max(idx, 1)
    print(
        f"[s1] {idx:>3d}/{total} {name[:24]:<24s} "
        f"SCM_AP={rec['scm_metrics']['ap']:.3f} "
        f"LLM_blind_AP={rec['llm_drug_blind_metrics']['ap']:.3f} "
        f"LLM_name_AP={rec['llm_with_name_metrics']['ap']:.3f} "
        f"rnd_AP={rec['random_metrics']['ap']:.3f} "
        f"({elapsed/60:.1f}m / ETA {eta/60:.1f}m)",
        flush=True,
    )
    return rec


def _summarize(records: list[dict], method_key: str) -> dict:
    metric_keys = ("ap", "p@10", "p@20", "p@50", "r@10", "r@20", "r@50", "rr")
    out = {}
    for k in metric_keys:
        vals = [r[method_key][k] for r in records]
        out[k] = {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    return out


def _bootstrap_ci(
    records: list[dict], method_key: str, metric: str,
    n_boot: int = 5000, seed: int = 42,
) -> dict:
    import random
    rng = random.Random(seed)
    vals = [r[method_key][metric] for r in records]
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "ci95": [0, 0], "ci99": [0, 0]}
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(vals) for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return {
        "mean": sum(boots) / n_boot,
        "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]],
        "ci99": [boots[int(0.005 * n_boot)], boots[int(0.995 * n_boot)]],
    }


def _wilcoxon_signed_rank(
    records: list[dict], method_a: str, method_b: str, metric: str = "ap",
) -> dict:
    """One-sided paired Wilcoxon signed-rank test: H1: A > B.

    Returns p-value via normal approximation (n=200 large enough).
    """
    diffs = [r[method_a][metric] - r[method_b][metric] for r in records]
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "W": 0, "z": 0.0, "p_one_sided_a_gt_b": 0.5}

    # Rank by absolute value
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
    W_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    # H1: A > B → W+ should be large
    W = W_plus
    mean_W = n * (n + 1) / 4
    var_W = n * (n + 1) * (2 * n + 1) / 24
    z = (W - mean_W) / math.sqrt(var_W) if var_W > 0 else 0.0
    # One-sided p (A > B)
    from math import erf, sqrt
    p_one = 0.5 * (1 - erf(z / sqrt(2)))
    return {
        "n": n, "W_plus": W_plus, "W_minus": W_minus,
        "z": z, "p_one_sided_a_gt_b": p_one,
    }


def main(test_only: int | None = None) -> int:
    print("=" * 78)
    print("SCM Off-Target Safety Sprint 1: ablation (SCM vs LLM-blind vs LLM-name vs random)")
    print("=" * 78)

    # Load catalog (test split only)
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    test_drugs = [d for d in cat["drugs"] if d["split"] == "test"]
    if test_only is not None:
        test_drugs = test_drugs[:test_only]
    print(f"[1/3] loaded test drugs: {len(test_drugs)}")

    # Load SCM edges
    with open(RESULTS / "scm_edges.json") as f:
        edges = json.load(f)
    with open(RESULTS / "side_effect_vocab.json") as f:
        vocab_payload = json.load(f)
    vocab_list = vocab_payload["umls_ids"]
    print(f"[2/3] loaded edges ({len(edges)} targets) and vocab ({len(vocab_list)} side effects)")

    print(f"[3/3] running parallel ablation (workers={N_WORKERS})...")
    client = SonnetClient()
    t_start = time.monotonic()
    results: list[dict] = [None] * len(test_drugs)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(_process_one_drug, i + 1, len(test_drugs), drug,
                      edges, vocab_payload, vocab_list, client, t_start): i
            for i, drug in enumerate(test_drugs)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                print(f"[s1] FAILED drug idx {idx}: {e}", flush=True)
                results[idx] = {
                    "cid": test_drugs[idx]["cid"],
                    "drug_name": test_drugs[idx]["drug_name"],
                    "error": str(e),
                }
    results = [r for r in results if r is not None and "error" not in r]
    print(f"\n[s1] completed {len(results)}/{len(test_drugs)} drugs")

    per_drug_path = RESULTS / "sprint1_per_drug.json"
    with open(per_drug_path, "w") as f:
        json.dump({"n_cases": len(results), "per_drug": results}, f, indent=2)
    print(f"[s1] saved per-drug: {per_drug_path}")

    # Aggregate per method
    summary = {}
    for arm_key in ("scm_metrics", "llm_drug_blind_metrics",
                     "llm_with_name_metrics", "random_metrics"):
        arm_name = arm_key.replace("_metrics", "")
        summary[arm_name] = _summarize(results, arm_key)

    # Bootstrap CIs for primary endpoint MAP
    bootstrap_cis = {}
    for arm_key in ("scm_metrics", "llm_drug_blind_metrics",
                     "llm_with_name_metrics", "random_metrics"):
        arm_name = arm_key.replace("_metrics", "")
        bootstrap_cis[arm_name] = {
            "ap_map": _bootstrap_ci(results, arm_key, "ap"),
            "p@10": _bootstrap_ci(results, arm_key, "p@10"),
            "r@10": _bootstrap_ci(results, arm_key, "r@10"),
        }

    # Wilcoxon signed-rank tests (pre-registered)
    wilcoxon = {
        # Primary: H1: SCM > LLM-drug-blind on AP
        "H1_scm_vs_llm_drug_blind_AP": _wilcoxon_signed_rank(
            results, "scm_metrics", "llm_drug_blind_metrics", "ap"),
        # H2: SCM > random
        "H2_scm_vs_random_AP": _wilcoxon_signed_rank(
            results, "scm_metrics", "random_metrics", "ap"),
        # H3: LLM-with-name > LLM-drug-blind (memorization)
        "H3_llm_name_vs_llm_drug_blind_AP": _wilcoxon_signed_rank(
            results, "llm_with_name_metrics", "llm_drug_blind_metrics", "ap"),
        # Also the reverse direction for transparency
        "reverse_llm_drug_blind_vs_scm_AP": _wilcoxon_signed_rank(
            results, "llm_drug_blind_metrics", "scm_metrics", "ap"),
    }

    # Pre-registered subgroups
    h4_subset = [r for r in results if r["n_targets"] >= 5]
    summary_h4 = {}
    for arm_key in ("scm_metrics", "llm_drug_blind_metrics",
                     "llm_with_name_metrics", "random_metrics"):
        arm_name = arm_key.replace("_metrics", "")
        summary_h4[arm_name] = _summarize(h4_subset, arm_key) if h4_subset else {}
    wilcoxon_h4 = (_wilcoxon_signed_rank(h4_subset, "scm_metrics",
                                          "llm_drug_blind_metrics", "ap")
                    if h4_subset else None)

    sum_path = RESULTS / "sprint1_summary.json"
    aggregate = {
        "n_cases": len(results),
        "summary_all": summary,
        "summary_h4_heavy_polypharmacology": {
            "n": len(h4_subset),
            **summary_h4,
            "wilcoxon_h4_scm_vs_llm_drug_blind_AP": wilcoxon_h4,
        },
        "bootstrap_cis": bootstrap_cis,
        "wilcoxon_signed_rank_one_sided": wilcoxon,
    }
    with open(sum_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"[s1] saved summary: {sum_path}")

    # Console summary
    print("\n" + "=" * 78)
    print(f"Sprint 1 SUMMARY (n={len(results)})")
    print("=" * 78)
    print(f"\n{'method':<18s} {'AP (MAP)':<10s} {'P@10':<10s} {'R@10':<10s} {'MRR':<10s}")
    for arm in ("scm", "llm_drug_blind", "llm_with_name", "random"):
        s = summary[arm]
        print(f"{arm:<18s} {s['ap']['mean']:<10.4f} {s['p@10']['mean']:<10.4f} "
              f"{s['r@10']['mean']:<10.4f} {s['rr']['mean']:<10.4f}")

    print(f"\nWilcoxon signed-rank (paired, one-sided):")
    for name, w in wilcoxon.items():
        print(f"  {name}: n={w['n']} z={w['z']:.3f} p(A>B)={w['p_one_sided_a_gt_b']:.4g}")

    print(f"\nH4 subgroup (≥5 binding targets, n={len(h4_subset)}):")
    for arm in ("scm", "llm_drug_blind", "llm_with_name", "random"):
        s = summary_h4[arm] if summary_h4 else {}
        if s:
            print(f"  {arm:<18s} AP={s['ap']['mean']:.4f}")
    if wilcoxon_h4:
        print(f"  Wilcoxon SCM > LLM-drug-blind H4: p={wilcoxon_h4['p_one_sided_a_gt_b']:.4g}")
    return 0


if __name__ == "__main__":
    n = None
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    sys.exit(main(test_only=n))
