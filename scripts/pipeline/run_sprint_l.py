"""Sprint L: rigor + generalizability sprint.

  - L.1: External held-out validation on SIDER test drugs (200 drugs
         never used in alpha training); ground truth = rare-SE subset
         (more drug-specific than common AEs).
  - L.2: Adversarial / negative-control test on main benchmark via
         binding-profile scramble (random in-vocab targets).
  - L.3: per-class analysis already done (see per_class_analysis.py)
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
import time
from collections import defaultdict
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
from ..data.clinical_safety_benchmark import passes_eligibility
from ..data.clinical_safety_benchmark_v5 import EXPANDED_SAFETY_CASES_V5
from ..demo.scm_explainer import explain_predictions
from ..llm import SonnetClient
from ..pipeline.run_sprint7_eval import classify_ta
from ..pipeline.run_sprint_f_eval import best_causal_rank
from ..scm.scoring import score_drug_side_effects_signed


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
N_WORKERS = 16


# ---------- L.1 : External held-out (SIDER test drugs) ----------

L1_N_SAMPLE = 80  # 80 test drugs is enough for tight CI; budget ~$170


def _compute_se_frequencies():
    """Return {umls: count} across training-set drugs."""
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    train_drugs = [d for d in cat["drugs"] if d["split"] == "train"]
    freq = defaultdict(int)
    for d in train_drugs:
        for s in d.get("side_effects_in_vocab", []):
            freq[s] += 1
    n_train = len(train_drugs)
    return {s: c / n_train for s, c in freq.items()}, n_train


def _build_l1_cases(max_drugs: int = L1_N_SAMPLE, max_specific_ses: int = 8):
    """Build external held-out cases from SIDER test drugs.

    Ground truth = drug's RARE SEs (training frequency < 0.30 in vocab).
    More drug-specific than common AEs like headache/nausea.
    """
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    test_drugs = [d for d in cat["drugs"] if d["split"] == "test"]
    print(f"[L.1] SIDER test pool: {len(test_drugs)} drugs")

    freq, n_train = _compute_se_frequencies()
    print(f"[L.1] training drugs: {n_train}; SEs with freq<0.30: "
          f"{sum(1 for c in freq.values() if c < 0.30)}")

    # For each test drug, pick its SEs with training-freq < 0.30
    cases = []
    for d in test_drugs:
        ses = d.get("side_effects_in_vocab", [])
        rare = [s for s in ses if freq.get(s, 0) < 0.30]
        if len(rare) >= 1:
            rare_sorted = sorted(rare, key=lambda s: freq.get(s, 0))[:max_specific_ses]
            cases.append({
                "drug_id": f"l1_{d.get('drug_name', '?')}",
                "drug_search_name": d.get("drug_name", ""),
                "binding_profile": d.get("binding_profile", []),
                "molregno": d.get("molregno"),
                "causal_se_umls": tuple(rare_sorted),
                "all_ses": list(ses),
                "split": "test",
            })

    # Sample
    rng = random.Random(42)
    rng.shuffle(cases)
    cases = cases[:max_drugs]
    print(f"[L.1] sampled n={len(cases)} test drugs with rare-SE ground truth")
    return cases


# ---------- L.2 : Adversarial scramble ----------

L2_N_SAMPLE = 50


def _scramble_binding(bp: list[dict], all_uniprots: list[str], seed: int):
    """Scramble binding profile: same number of targets, randomly drawn."""
    rng = random.Random(seed)
    n = len(bp)
    new_uniprots = rng.sample(all_uniprots, min(n, len(all_uniprots)))
    return [
        {
            "uniprot": u,
            "gene_symbol": "?",
            "target_pref_name": "?",
            "standard_type": "Kd",
            "standard_value_nm": 100.0,
            "source": "scrambled",
        }
        for u in new_uniprots
    ]


def _build_l2_cases(max_n: int = L2_N_SAMPLE):
    """Pick subset of main benchmark cases; we'll scramble their binding."""
    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    vocab_set = set(v["umls_ids"])
    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    target_set = {t["uniprot"] for t in tv["targets"]}
    eligible = [c for c in EXPANDED_SAFETY_CASES_V5
                if passes_eligibility(c, vocab_set, target_set)]
    rng = random.Random(42)
    rng.shuffle(eligible)
    return eligible[:max_n]


# ---------- Shared processing ----------

def _process_l1_case(
    cf: dict, edges: dict, signed_edges: dict, target_action_n: dict,
    target_info: dict, vocab_payload: dict,
    se_vocab: list[str], se_names: dict,
    client, curated_priors: dict,
) -> dict:
    bp = cf["binding_profile"]
    if not bp:
        return {"drug_id": cf["drug_id"], "skipped": True}

    causal = tuple(cf["causal_se_umls"])

    action_types = {}
    if cf.get("molregno") is not None:
        try:
            action_types = load_action_types(cf["molregno"])
        except Exception:
            pass

    scored = score_drug_side_effects_signed(
        bp, edges, signed_edges, action_types, target_action_n, se_vocab,
        min_drugs_for_signed=3, affinity_mode="log_sigmoid",
    )
    explanations = explain_predictions(
        scored[:100], bp, edges, target_info, se_names,
        top_k_se=100, top_k_targets=5,
    )
    scm_ranked = [s for s, _ in scored]
    scm_rank = best_causal_rank(scm_ranked, causal)

    hybrid = hybrid_rerank(
        bp, scored, explanations, vocab_payload, client=client,
        top_k_scm_candidates=100, therapeutic_area="Other",
        action_types=action_types,
    )
    hybrid_ranked, promotions = apply_curated_prior_override_v2(
        hybrid.ranked_side_effects, scm_ranked, bp, curated_priors,
    )
    hybrid_rank = best_causal_rank(hybrid_ranked, causal)

    llm_blind = rank_side_effects_llm_drug_blind(
        bp, vocab_payload, client=client, top_k=50,
    )
    llm_blind_rank = best_causal_rank(
        llm_blind.ranked_side_effects, causal,
    )

    llm_name = rank_side_effects_llm_with_name(
        cf["drug_search_name"], bp, vocab_payload, client=client, top_k=50,
    )
    llm_name_rank = best_causal_rank(
        llm_name.ranked_side_effects, causal,
    )

    return {
        "drug_id": cf["drug_id"],
        "drug_search_name": cf["drug_search_name"],
        "causal_se_umls": list(causal),
        "n_targets": len(bp),
        "n_rare_ses": len(causal),
        "scm_rank": scm_rank,
        "hybrid_rank": hybrid_rank,
        "llm_drug_blind_rank": llm_blind_rank,
        "llm_with_name_rank": llm_name_rank,
        "hybrid_top10": hybrid_ranked[:10],
    }


def _process_l2_case(
    cf, scrambled_bp: list[dict], edges: dict, signed_edges: dict,
    target_action_n: dict, target_info: dict, vocab_payload: dict,
    se_vocab: list[str], se_names: dict, client,
    curated_priors: dict,
) -> dict:
    """Score scrambled binding against ORIGINAL causal AEs."""
    causal = cf.causal_side_effects_umls

    action_types = {}
    scored = score_drug_side_effects_signed(
        scrambled_bp, edges, signed_edges, action_types, target_action_n,
        se_vocab, min_drugs_for_signed=3, affinity_mode="log_sigmoid",
    )
    explanations = explain_predictions(
        scored[:100], scrambled_bp, edges, target_info, se_names,
        top_k_se=100, top_k_targets=5,
    )
    scm_ranked = [s for s, _ in scored]
    scm_rank = best_causal_rank(scm_ranked, causal)

    ta = classify_ta(cf.drug_id, cf.severity, cf.causal_off_target_gene)
    hybrid = hybrid_rerank(
        scrambled_bp, scored, explanations, vocab_payload, client=client,
        top_k_scm_candidates=100, therapeutic_area=ta,
        action_types=action_types,
    )
    hybrid_ranked, _ = apply_curated_prior_override_v2(
        hybrid.ranked_side_effects, scm_ranked, scrambled_bp, curated_priors,
    )
    hybrid_rank = best_causal_rank(hybrid_ranked, causal)

    llm_blind = rank_side_effects_llm_drug_blind(
        scrambled_bp, vocab_payload, client=client, top_k=50,
    )
    llm_blind_rank = best_causal_rank(
        llm_blind.ranked_side_effects, causal,
    )

    # LLM-with-name: drug name leaks even with scrambled targets (sanity)
    llm_name = rank_side_effects_llm_with_name(
        cf.drug_search_name, scrambled_bp, vocab_payload,
        client=client, top_k=50,
    )
    llm_name_rank = best_causal_rank(
        llm_name.ranked_side_effects, causal,
    )

    return {
        "drug_id": cf.drug_id,
        "drug_search_name": cf.drug_search_name,
        "causal_se_umls": list(causal),
        "scm_rank": scm_rank,
        "hybrid_rank": hybrid_rank,
        "llm_drug_blind_rank": llm_blind_rank,
        "llm_with_name_rank": llm_name_rank,
    }


def hit(records: list[dict], key: str, k: int) -> int:
    return sum(1 for r in records if r is not None and r.get(key) is not None
               and r[key] <= k)


def _bootstrap_ci(records: list[dict], key: str, k: int,
                    n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = sum(1 for r in records if r is not None)
    if n == 0:
        return (0.0, 0.0)
    hits = np.array([1 if (r is not None and r.get(key) is not None
                            and r[key] <= k) else 0 for r in records if r is not None])
    boot_means = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(hits[idx].mean())
    boot_means.sort()
    return (boot_means[int(0.025 * n_boot)], boot_means[int(0.975 * n_boot)])


def _mcnemar_one_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    p = 0.0
    for x in range(b, n + 1):
        p += math.comb(n, x) * (0.5 ** n)
    return p


def main() -> int:
    print("=" * 78)
    print("Sprint L: Rigor + Generalizability (audible)")
    print("  L.1: SIDER test-drug external held-out (rare-SE ground truth)")
    print("  L.2: Adversarial / negative-control scramble")
    print("=" * 78)

    client = SonnetClient()

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
    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    target_info = {t["uniprot"]: t for t in tv["targets"]}
    all_uniprots = list(target_info.keys())

    curated_priors = load_curated_priors_for_override()

    # ============= L.1 External =============
    print("\n" + "=" * 78)
    print("L.1: SIDER test-drug external held-out")
    print("=" * 78)
    l1_cases = _build_l1_cases()
    print(f"[L.1] running n={len(l1_cases)} test drugs")

    l1_results: list[dict] = [None] * len(l1_cases)
    t_start = time.monotonic()
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
                l1_results[idx] = fut.result()
            except Exception as e:
                print(f"[L.1] case {idx} FAILED: {e}", flush=True)
            done += 1
            if done % 10 == 0:
                elapsed = time.monotonic() - t_start
                print(f"  [L.1] {done}/{len(l1_cases)} ({elapsed/60:.1f}m)",
                      flush=True)
    l1_results = [r for r in l1_results if r is not None and not r.get("skipped")]
    print(f"[L.1] done n={len(l1_results)} in {(time.monotonic() - t_start)/60:.1f}m")

    with open(RESULTS / "sprint_l_external_safety.json", "w") as f:
        json.dump({"n_cases": len(l1_results), "per_drug": l1_results}, f, indent=2)

    n = len(l1_results)
    h10 = hit(l1_results, "hybrid_rank", 10)
    h_blind = hit(l1_results, "llm_drug_blind_rank", 10)
    h_name = hit(l1_results, "llm_with_name_rank", 10)
    h_scm = hit(l1_results, "scm_rank", 10)
    lo, hi = _bootstrap_ci(l1_results, "hybrid_rank", 10)

    print(f"\n[L.1] SIDER external held-out (n={n})")
    print(f"  SCM-blended hit@10:       {h_scm}/{n} ({h_scm/n:.1%})")
    print(f"  LLM-drug-blind hit@10:    {h_blind}/{n} ({h_blind/n:.1%})")
    print(f"  LLM-with-name hit@10:     {h_name}/{n} ({h_name/n:.1%})")
    print(f"  Hybrid hit@10:            {h10}/{n} ({h10/n:.1%}) [95%CI: {lo:.1%}-{hi:.1%}]")
    diff = (h10 - h_name) / n * 100
    print(f"  Hybrid vs LLM-with-name:  {diff:+.1f}pp drug-blind")

    # ============= L.2 Adversarial =============
    print("\n\n" + "=" * 78)
    print("L.2: Adversarial / negative-control scramble")
    print("=" * 78)
    l2_cases = _build_l2_cases(max_n=L2_N_SAMPLE)
    print(f"[L.2] scrambling n={len(l2_cases)} main benchmark cases")

    # Pre-compute scrambled binding profiles
    scrambled_bps = []
    for i, cf in enumerate(l2_cases):
        with open(RESULTS / "catalog.json") as f:
            cat = json.load(f)
        drugs_by_name = {(d["drug_name"] or "").lower(): d for d in cat["drugs"]}
        cat_drug = drugs_by_name.get(cf.drug_search_name.lower())
        if cat_drug is not None:
            n_targets = len(cat_drug["binding_profile"])
        else:
            n_targets = 5
        scrambled_bps.append(_scramble_binding(
            [{"uniprot": "?"} for _ in range(n_targets)], all_uniprots,
            seed=42 + i,
        ))

    l2_results: list[dict] = [None] * len(l2_cases)
    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(_process_l2_case, cf, scrambled_bps[i], edges,
                       signed_edges, target_action_n, target_info,
                       vocab_payload, se_vocab, se_names, client,
                       curated_priors): i
            for i, cf in enumerate(l2_cases)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                l2_results[idx] = fut.result()
            except Exception as e:
                print(f"[L.2] case {idx} FAILED: {e}", flush=True)
            done += 1
            if done % 10 == 0:
                elapsed = time.monotonic() - t_start
                print(f"  [L.2] {done}/{len(l2_cases)} ({elapsed/60:.1f}m)",
                      flush=True)
    l2_results = [r for r in l2_results if r is not None]
    print(f"[L.2] done n={len(l2_results)} in {(time.monotonic() - t_start)/60:.1f}m")

    with open(RESULTS / "sprint_l_adversarial_safety.json", "w") as f:
        json.dump({"n_cases": len(l2_results), "per_drug": l2_results}, f, indent=2)

    n2 = len(l2_results)
    h10_a = hit(l2_results, "hybrid_rank", 10)
    h_scm_a = hit(l2_results, "scm_rank", 10)
    h_name_a = hit(l2_results, "llm_with_name_rank", 10)
    print(f"\n[L.2] Adversarial scramble (n={n2})")
    print(f"  SCM scrambled hit@10:        {h_scm_a}/{n2} ({h_scm_a/n2:.1%})")
    print(f"  Hybrid scrambled hit@10:     {h10_a}/{n2} ({h10_a/n2:.1%})")
    print(f"  LLM-with-name scrambled hit@10: {h_name_a}/{n2} ({h_name_a/n2:.1%}) "
          f"(should be near baseline — uses drug name not targets)")

    # Compute drop vs original benchmark
    sprint_k_path = RESULTS / "sprint_k_safety_sonnet.json"
    if sprint_k_path.exists():
        with open(sprint_k_path) as f:
            sk = json.load(f)
        # Find paired hit@10 on the L2 subset
        l2_ids = {cf.drug_id for cf in l2_cases}
        k_paired = [r for r in sk.get("per_drug", []) if r["drug_id"] in l2_ids
                     and not r.get("skipped")]
        h10_orig = sum(1 for r in k_paired
                        if r["hybrid_rank"] is not None and r["hybrid_rank"] <= 10)
        n_paired = len(k_paired)
        if n_paired > 0:
            print(f"\n[L.2] Original (Sprint K) Hybrid hit@10 on same drugs: "
                  f"{h10_orig}/{n_paired} ({h10_orig/n_paired:.1%})")
            drop_pp = (h10_orig - h10_a) / n_paired * 100
            print(f"[L.2] Drop from scramble: {drop_pp:+.1f}pp "
                  f"(positive = drop in performance)")

    # ============= Save summary =============
    summary = {
        "l1_external": {
            "n": n,
            "scm_hit10": h_scm,
            "llm_blind_hit10": h_blind,
            "llm_name_hit10": h_name,
            "hybrid_hit10": h10,
            "hybrid_hit10_rate": h10 / n if n > 0 else 0.0,
            "hybrid_95ci": [lo, hi],
            "diff_vs_llm_name_pp": diff,
        },
        "l2_adversarial": {
            "n": n2,
            "scm_scrambled_hit10": h_scm_a,
            "hybrid_scrambled_hit10": h10_a,
            "llm_name_scrambled_hit10": h_name_a,
            "hybrid_scrambled_hit10_rate": h10_a / n2 if n2 > 0 else 0.0,
        },
    }
    with open(RESULTS / "sprint_l_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {RESULTS / 'sprint_l_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
