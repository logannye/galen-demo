"""Day 0 metrics + stratification + failure-mode analysis.

Computes:
  - Aggregate cluster-aware hit@1/3/5/10 with bootstrap 95% CI for:
    - curated-bp pipeline (control)
    - SMILES → TargetNet pipeline (test)
    - stored LLM-drug-blind ranks (Sprint E baseline, for context)
    - stored LLM-with-name ranks (memorization upper bound)
    - stored RF-ECFP ranks (chemistry SOTA baseline)
  - McNemar paired test: SMILES vs curated (the primary Day 0 question)
  - Per-stratum hit-rate tables (severity × TA × polypharmacology)
  - Failure-mode categorization for cases where the SMILES pipeline misses

Output: results/day0_analysis.json + console pretty-print + CSV table.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts.baselines.ae_cluster_postprocess import (
    hit_at_k_clustered, collapse_top_k,
)

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def bootstrap_ci(h: int, n: int, alpha: float = 0.05, n_iter: int = 2000):
    """Bootstrap 95% CI for hit-rate proportion."""
    import random
    rng = random.Random(42)
    if n == 0:
        return (0.0, 0.0)
    samples = []
    for _ in range(n_iter):
        x = sum(1 for _ in range(n) if rng.random() < (h / n))
        samples.append(x / n)
    samples.sort()
    lo = samples[int((alpha / 2) * n_iter)]
    hi = samples[int((1 - alpha / 2) * n_iter)]
    return (lo, hi)


def mcnemar_paired(both: int, a_only: int, b_only: int, neither: int):
    """McNemar's exact test on paired-binary outcomes.

    Returns p-value (two-sided, exact binomial for the discordant pairs).
    """
    n_disc = a_only + b_only
    if n_disc == 0:
        return 1.0
    # Exact binomial: under H0, each discordant pair is 50/50.
    # P(|deviation| as extreme) = 2 * binom_cdf(min(a_only, b_only), n_disc, 0.5)
    m = min(a_only, b_only)
    p = 0.0
    for k in range(m + 1):
        p += math.comb(n_disc, k) * (0.5 ** n_disc)
    return min(1.0, 2 * p)


def hit_clustered(gt_umls: list[str], top10: list[str], k: int) -> bool:
    """Cluster-aware hit at K."""
    return hit_at_k_clustered(set(gt_umls), collapse_top_k(top10), k)


def hit_rate(records, getter, k):
    """Compute (hits, n_evaluable) where evaluable = has gt + has ranked_list."""
    n = len(records)
    h = 0
    for r in records:
        top10 = getter(r)
        if top10 is None:
            continue
        if hit_clustered(r["gt_umls"], top10, k):
            h += 1
    return h, n


def rank_to_top10(rank, fallback_top10):
    """Stored baselines have rank but not top10. We can't compute
    cluster-aware hit@K without the actual UMLS ranking. So for the
    stored baselines we approximate hit@K as `rank <= K` (literal hit)."""
    return None  # signal to use rank-only fallback


def stored_hit_at_k(records, rank_key, k):
    """For stored baselines (rank only, no UMLS top-10): hit@K = rank <= K."""
    h = 0
    n = 0
    for r in records:
        rk = r.get("stored_baselines", {}).get(rank_key)
        if rk is None:
            continue
        n += 1
        if rk <= k:
            h += 1
    return h, n


def main() -> None:
    res = json.load(open(RESULTS / "day0_validation_results.json"))
    drugs = res["per_drug"]
    n = len(drugs)
    print(f"Day 0 analysis: n = {n}")

    # ---- Aggregate hit rates ----
    print("\n" + "=" * 90)
    print("AGGREGATE — cluster-aware hit@K (current production stack, both pipelines)")
    print("=" * 90)

    pipelines = {
        "Curated bp (control)": lambda r: r["curated"]["hybrid_top10"],
        "SMILES → TargetNet (Week 2)": lambda r: r["smiles"]["hybrid_top10"],
    }
    summary: dict = {"n": n, "pipelines": {}}
    for name, getter in pipelines.items():
        row = {}
        for k in (1, 3, 5, 10):
            h, _ = hit_rate(drugs, getter, k)
            lo, hi = bootstrap_ci(h, n)
            row[f"hit@{k}"] = h
            row[f"hit@{k}_rate"] = h / n
            row[f"hit@{k}_ci"] = [lo, hi]
        summary["pipelines"][name] = row
        print(f"\n  {name}:")
        for k in (1, 3, 5, 10):
            h = row[f"hit@{k}"]
            lo, hi = row[f"hit@{k}_ci"]
            print(f"    hit@{k}: {h}/{n} ({h/n:.1%}) [CI {lo:.0%}-{hi:.0%}]")

    # ---- Stored baselines (literal hit, no cluster expansion) ----
    print("\n" + "=" * 90)
    print("STORED BASELINES (Sprint E literal rank ≤ K; no cluster expansion)")
    print("=" * 90)
    summary["stored_baselines"] = {}
    for label, key in [
        ("LLM drug-blind (memorization-free)", "llm_drug_blind_rank"),
        ("LLM with-name (memorization upper bound)", "llm_with_name_rank"),
        ("RF-ECFP (chemistry-only baseline)", "rf_ecfp_rank"),
    ]:
        print(f"\n  {label}:")
        row = {}
        for k in (1, 3, 5, 10):
            h, n_eval = stored_hit_at_k(drugs, key, k)
            row[f"hit@{k}"] = h
            row[f"n_eval"] = n_eval
            row[f"hit@{k}_rate"] = (h / n_eval) if n_eval > 0 else 0.0
            print(f"    hit@{k}: {h}/{n_eval} ({h/max(n_eval,1):.1%})")
        summary["stored_baselines"][label] = row

    # ---- McNemar paired test: SMILES vs Curated ----
    print("\n" + "=" * 90)
    print("PAIRED COMPARISON — SMILES (test) vs Curated (control), cluster-aware")
    print("=" * 90)
    for k in (1, 3, 5, 10):
        both = a_only = b_only = neither = 0
        for r in drugs:
            ca = hit_clustered(r["gt_umls"], r["curated"]["hybrid_top10"], k)
            sm = hit_clustered(r["gt_umls"], r["smiles"]["hybrid_top10"], k)
            if ca and sm:
                both += 1
            elif ca and not sm:
                a_only += 1
            elif sm and not ca:
                b_only += 1
            else:
                neither += 1
        p = mcnemar_paired(both, a_only, b_only, neither)
        # delta = SMILES - Curated (positive = SMILES wins)
        delta = (both + b_only) - (both + a_only)
        print(f"\n  hit@{k}:")
        print(f"    both correct:   {both}")
        print(f"    only curated:   {a_only}")
        print(f"    only SMILES:    {b_only}")
        print(f"    neither:        {neither}")
        print(f"    Δ (SMILES − curated): {'+' if delta >= 0 else ''}{delta} drugs")
        print(f"    McNemar p-value: {p:.4f}")

    # ---- Stratification: severity ----
    print("\n" + "=" * 90)
    print("STRATIFIED — by severity tier")
    print("=" * 90)
    severities = sorted({r["severity"] for r in drugs})
    summary["by_severity"] = {}
    for sev in severities:
        sub = [r for r in drugs if r["severity"] == sev]
        ns = len(sub)
        print(f"\n  Severity '{sev}' (n={ns}):")
        sev_data = {"n": ns}
        for name, getter in pipelines.items():
            print(f"    {name}:")
            sev_data[name] = {}
            for k in (1, 3, 10):
                h, _ = hit_rate(sub, getter, k)
                lo, hi = bootstrap_ci(h, ns)
                sev_data[name][f"hit@{k}"] = h
                sev_data[name][f"hit@{k}_ci"] = [lo, hi]
                print(f"      hit@{k}: {h}/{ns} ({h/max(ns,1):.1%}) "
                      f"[CI {lo:.0%}-{hi:.0%}]")
        summary["by_severity"][sev] = sev_data

    # ---- Stratification: TA ----
    print("\n" + "=" * 90)
    print("STRATIFIED — by therapeutic area")
    print("=" * 90)
    tas = sorted({r["therapeutic_area"] for r in drugs if r["therapeutic_area"]})
    summary["by_ta"] = {}
    for ta in tas:
        sub = [r for r in drugs if r["therapeutic_area"] == ta]
        ns = len(sub)
        if ns < 5:
            continue
        print(f"\n  TA '{ta}' (n={ns}):")
        ta_data = {"n": ns}
        for name, getter in pipelines.items():
            print(f"    {name}:")
            ta_data[name] = {}
            for k in (1, 3, 10):
                h, _ = hit_rate(sub, getter, k)
                lo, hi = bootstrap_ci(h, ns)
                ta_data[name][f"hit@{k}"] = h
                ta_data[name][f"hit@{k}_ci"] = [lo, hi]
                print(f"      hit@{k}: {h}/{ns} ({h/ns:.1%}) [CI {lo:.0%}-{hi:.0%}]")
        summary["by_ta"][ta] = ta_data

    # ---- Polypharmacology level ----
    print("\n" + "=" * 90)
    print("STRATIFIED — by polypharmacology (n_binding_targets_orig)")
    print("=" * 90)
    bins = [
        ("≤3 targets",    lambda r: r["n_binding_targets_orig"] <= 3),
        ("4-10 targets",  lambda r: 4 <= r["n_binding_targets_orig"] <= 10),
        ("11-30 targets", lambda r: 11 <= r["n_binding_targets_orig"] <= 30),
        ("31+ targets",   lambda r: r["n_binding_targets_orig"] >= 31),
    ]
    summary["by_polypharm"] = {}
    for bin_name, bin_fn in bins:
        sub = [r for r in drugs if bin_fn(r)]
        ns = len(sub)
        if ns < 5:
            continue
        print(f"\n  {bin_name} (n={ns}):")
        pp_data = {"n": ns}
        for name, getter in pipelines.items():
            print(f"    {name}:")
            pp_data[name] = {}
            for k in (1, 3, 10):
                h, _ = hit_rate(sub, getter, k)
                lo, hi = bootstrap_ci(h, ns)
                pp_data[name][f"hit@{k}"] = h
                pp_data[name][f"hit@{k}_ci"] = [lo, hi]
                print(f"      hit@{k}: {h}/{ns} ({h/ns:.1%}) [CI {lo:.0%}-{hi:.0%}]")
        summary["by_polypharm"][bin_name] = pp_data

    # ---- Failure-mode analysis ----
    print("\n" + "=" * 90)
    print("FAILURE-MODE ANALYSIS — drugs the SMILES pipeline misses at hit@10")
    print("=" * 90)
    misses = []
    for r in drugs:
        if not hit_clustered(r["gt_umls"], r["smiles"]["hybrid_top10"], 10):
            misses.append(r)
    print(f"\n  Total SMILES misses at hit@10: {len(misses)}/{n} ({len(misses)/n:.1%})")

    # Did curated nail it where SMILES missed?
    curated_saves = [r for r in misses
                     if hit_clustered(r["gt_umls"], r["curated"]["hybrid_top10"], 10)]
    print(f"  ... of those, curated DID hit: {len(curated_saves)} "
          f"(SMILES-specific failures)")
    print(f"  ... and curated ALSO missed: {len(misses) - len(curated_saves)} "
          f"(pipeline-wide failures)")

    # Stratify failures by cause
    failure_cats = {
        "target_not_in_substrate": [],
        "no_predicted_targets": [],
        "causal_target_not_predicted": [],
        "other": [],
    }
    for r in misses:
        bp_size = r["smiles"]["n_targets"]
        causal = r.get("causal_off_target", "")
        # We don't have the TargetNet output here, only n_targets — so we
        # approximate: if bp_size == 0, no predicted targets; else we
        # report 'other' since we can't introspect further.
        if bp_size == 0:
            failure_cats["no_predicted_targets"].append(r["drug_search_name"])
        else:
            failure_cats["other"].append(r["drug_search_name"])
    print(f"\n  Failure categories (approximate):")
    for cat, names in failure_cats.items():
        if names:
            print(f"    {cat}: {len(names)} — {names[:8]}{'...' if len(names)>8 else ''}")
    summary["failure_modes"] = {k: len(v) for k, v in failure_cats.items()}
    summary["smiles_misses_list"] = [
        {
            "drug": r["drug_search_name"],
            "severity": r["severity"],
            "causal_off_target": r["causal_off_target"],
            "causal_se_text": r["causal_se_text"],
            "smiles_bp_size": r["smiles"]["n_targets"],
            "curated_bp_size": r["curated"]["n_targets"],
            "curated_hit@10": hit_clustered(r["gt_umls"], r["curated"]["hybrid_top10"], 10),
        }
        for r in misses
    ]

    out = RESULTS / "day0_analysis.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
