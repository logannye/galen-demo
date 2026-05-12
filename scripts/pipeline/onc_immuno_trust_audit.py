"""Phase 3.1 audit: onc/immuno product-performance trust metrics.

Computes (on existing Sprint K results, no new LLM calls):

  - hit@K and PPV@K for K = 1, 3, 5, 10 on onc + immuno subset
  - Confidence-conditioned PPV: at confidence ≥ {0.3, 0.5, 0.7, 0.85}, what's PPV?
  - Per-TA stratification: Oncology vs Immunology
  - Per-severity stratification: black_box vs not
  - Calibration ECE within onc + immuno only (per-prediction level)
  - Curated-prior-override impact: how many of the hit cases had a promotion?
  - Mechanism class breakdown: which AE classes does the system hit / miss?

Surfaces the single biggest product-performance gap for the onc/immuno
deployment target.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def load_records(in_scope_tas=("Oncology", "Immunology")):
    """Load Sprint K main + OOD, filter to onc + immuno."""
    main = json.load(open(RESULTS / "sprint_k_safety_sonnet.json"))["per_drug"]
    ood = json.load(open(RESULTS / "sprint_k_ood_safety_sonnet.json"))["per_drug"]
    all_records = []
    for r in main:
        r = dict(r)
        r["dataset"] = "main"
        all_records.append(r)
    for r in ood:
        r = dict(r)
        r["dataset"] = "ood"
        all_records.append(r)
    onc_immuno = [r for r in all_records
                   if r.get("therapeutic_area") in in_scope_tas
                   and not r.get("skipped")]
    return all_records, onc_immuno


def hit_at_k(rank, k):
    return rank is not None and rank <= k


def compute_hit_rates(records):
    """hit@K and PPV@K. Note: with single GT per drug, PPV@K = hit@K / K
    in the optimistic case where the single GT slot is in top-K and other slots
    are 'unknown' (not 'wrong'). We report both as classic precision/recall.
    """
    n = len(records)
    out = {}
    for k in (1, 3, 5, 10):
        h = sum(1 for r in records if hit_at_k(r.get("hybrid_rank"), k))
        out[f"hit@{k}"] = h
        out[f"hit@{k}_rate"] = h / max(n, 1)
    out["n"] = n
    return out


def confidence_ppv(records, thresholds=(0.30, 0.50, 0.70, 0.85)):
    """At each confidence threshold, of cases where top-K confidence ≥ T,
    what fraction had GT in top-K?

    A 'confident prediction' = the rank-1 prediction has confidence ≥ T.
    """
    out = {}
    for t in thresholds:
        confident = []
        for r in records:
            top10 = r.get("confidence_top10") or []
            if not top10:
                continue
            top1_conf = top10[0].get("confidence", 0.0)
            if top1_conf >= t:
                confident.append(r)
        n = len(confident)
        # Hit@3 within confident-cases — i.e. when the system is confident,
        # how often does the top-3 actually contain a GT?
        h3 = sum(1 for r in confident if hit_at_k(r.get("hybrid_rank"), 3))
        h10 = sum(1 for r in confident if hit_at_k(r.get("hybrid_rank"), 10))
        out[f"conf_ge_{t:.2f}"] = {
            "n_confident": n,
            "pct_confident": n / max(len(records), 1),
            "ppv_at_3": h3 / max(n, 1),
            "ppv_at_10": h10 / max(n, 1),
        }
    return out


def per_ta_breakdown(records):
    by_ta = defaultdict(list)
    for r in records:
        by_ta[r.get("therapeutic_area") or "?"].append(r)
    out = {}
    for ta, recs in by_ta.items():
        out[ta] = compute_hit_rates(recs)
    return out


def per_severity_breakdown(records):
    by_sev = defaultdict(list)
    for r in records:
        by_sev[r.get("severity") or "?"].append(r)
    out = {}
    for sev, recs in by_sev.items():
        out[sev] = compute_hit_rates(recs)
    return out


def per_dataset_breakdown(records):
    by_ds = defaultdict(list)
    for r in records:
        by_ds[r.get("dataset") or "?"].append(r)
    out = {}
    for ds, recs in by_ds.items():
        out[ds] = compute_hit_rates(recs)
    return out


def calibration_per_prediction(records, n_bins=10):
    """Per-PREDICTION calibration (10 predictions per drug × n drugs).

    For each prediction (rank, umls, confidence), is_hit = 1 if umls is in
    GT for that drug, else 0. Bin by confidence; compare bin-mean
    confidence vs bin-mean hit rate.

    Returns (ECE, Brier, per-bin detail).
    """
    preds = []
    for r in records:
        gt = set(r.get("causal_side_effects_umls") or [])
        for p in (r.get("confidence_top10") or []):
            umls = p.get("umls")
            conf = p.get("confidence", 0.0)
            preds.append({"hit": 1 if umls in gt else 0, "conf": conf})

    if not preds:
        return {"ECE": None, "Brier": None, "n": 0, "bins": []}

    # Brier
    brier = sum((p["conf"] - p["hit"]) ** 2 for p in preds) / len(preds)

    # ECE
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bin_data = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [p for p in preds if lo <= p["conf"] < (hi if i < n_bins - 1 else hi + 1e-9)]
        n_b = len(in_bin)
        if n_b == 0:
            bin_data.append({"bin": [lo, hi], "n": 0, "mean_conf": 0,
                              "mean_hit": 0, "gap": 0})
            continue
        mc = sum(p["conf"] for p in in_bin) / n_b
        mh = sum(p["hit"] for p in in_bin) / n_b
        bin_data.append({"bin": [lo, hi], "n": n_b, "mean_conf": mc,
                          "mean_hit": mh, "gap": abs(mc - mh)})

    total = sum(b["n"] for b in bin_data)
    ece = sum(b["n"] * b["gap"] for b in bin_data) / max(total, 1)

    return {
        "ECE": ece,
        "Brier": brier,
        "n_predictions": len(preds),
        "n_hits": sum(p["hit"] for p in preds),
        "hit_rate": sum(p["hit"] for p in preds) / len(preds),
        "bins": bin_data,
    }


def curated_override_impact(records):
    """How many hits used a curated-prior promotion?"""
    with_promotion = [r for r in records
                       if r.get("n_promotions") and r["n_promotions"] > 0]
    n = len(records)
    n_promoted = len(with_promotion)
    n_promoted_hits = sum(1 for r in with_promotion
                           if hit_at_k(r.get("hybrid_rank"), 10))
    return {
        "n_total": n,
        "n_with_promotion": n_promoted,
        "pct_with_promotion": n_promoted / max(n, 1),
        "hit10_rate_among_promoted": n_promoted_hits / max(n_promoted, 1),
        "n_promoted_hits": n_promoted_hits,
    }


def mechanism_class_breakdown(records):
    """Group by causal_off_target gene to see which mechanism classes
    the system hits vs misses."""
    by_target = defaultdict(lambda: {"n": 0, "hits": 0})
    for r in records:
        gene = r.get("causal_off_target") or "?"
        by_target[gene]["n"] += 1
        if hit_at_k(r.get("hybrid_rank"), 10):
            by_target[gene]["hits"] += 1

    # Sort by n descending, list classes with n >= 3
    rows = []
    for gene, d in by_target.items():
        rows.append({
            "target": gene,
            "n": d["n"],
            "hit10": d["hits"],
            "hit10_rate": d["hits"] / d["n"],
        })
    rows.sort(key=lambda x: (-x["n"], x["target"]))
    return rows


def main():
    print("=" * 78)
    print("Phase 3.1 Onc/Immuno Trust Audit")
    print("(no new LLM calls — analysis of saved Sprint K results)")
    print("=" * 78)

    all_records, oi_records = load_records()
    print(f"\nTotal Sprint K records (main + OOD): {len(all_records)}")
    print(f"Onc + Immuno subset: {len(oi_records)}")

    # ---------- 1. Overall hit/PPV ----------
    print("\n" + "=" * 78)
    print("1. Hit-rate / PPV across onc+immuno (n={})".format(len(oi_records)))
    print("=" * 78)
    overall = compute_hit_rates(oi_records)
    print(f"\n  hit@1  : {overall['hit@1']}/{overall['n']} = {overall['hit@1_rate']:.1%}")
    print(f"  hit@3  : {overall['hit@3']}/{overall['n']} = {overall['hit@3_rate']:.1%}")
    print(f"  hit@5  : {overall['hit@5']}/{overall['n']} = {overall['hit@5_rate']:.1%}")
    print(f"  hit@10 : {overall['hit@10']}/{overall['n']} = {overall['hit@10_rate']:.1%}")

    # ---------- 2. Per-TA ----------
    print("\n" + "=" * 78)
    print("2. Per-TA breakdown")
    print("=" * 78)
    per_ta = per_ta_breakdown(oi_records)
    for ta, d in per_ta.items():
        print(f"\n  {ta} (n={d['n']})")
        for k in (1, 3, 5, 10):
            print(f"    hit@{k}: {d[f'hit@{k}']}/{d['n']} ({d[f'hit@{k}_rate']:.1%})")

    # ---------- 3. Per-severity ----------
    print("\n" + "=" * 78)
    print("3. Per-severity breakdown")
    print("=" * 78)
    per_sev = per_severity_breakdown(oi_records)
    for sev in sorted(per_sev.keys()):
        d = per_sev[sev]
        print(f"\n  {sev} (n={d['n']})")
        for k in (1, 3, 5, 10):
            print(f"    hit@{k}: {d[f'hit@{k}']}/{d['n']} ({d[f'hit@{k}_rate']:.1%})")

    # ---------- 4. Per-dataset (main vs OOD) ----------
    print("\n" + "=" * 78)
    print("4. Per-dataset breakdown (main vs OOD)")
    print("=" * 78)
    per_ds = per_dataset_breakdown(oi_records)
    for ds in ("main", "ood"):
        if ds not in per_ds:
            continue
        d = per_ds[ds]
        print(f"\n  {ds} (n={d['n']})")
        for k in (1, 3, 5, 10):
            print(f"    hit@{k}: {d[f'hit@{k}']}/{d['n']} ({d[f'hit@{k}_rate']:.1%})")

    # ---------- 5. Confidence-conditioned PPV ----------
    print("\n" + "=" * 78)
    print("5. Confidence-conditioned PPV (top-1 prediction confidence)")
    print("=" * 78)
    conf_ppv = confidence_ppv(oi_records)
    print(f"\n  {'threshold':<12s} {'n_conf':>8s} {'%':>6s}  {'PPV@3':>7s} {'PPV@10':>8s}")
    for t in (0.30, 0.50, 0.70, 0.85):
        d = conf_ppv[f"conf_ge_{t:.2f}"]
        print(f"  ≥ {t:.2f}     {d['n_confident']:>8d} {d['pct_confident']:>5.1%}  "
              f"{d['ppv_at_3']:>6.1%} {d['ppv_at_10']:>7.1%}")

    # ---------- 6. Per-prediction calibration ----------
    print("\n" + "=" * 78)
    print("6. Per-prediction calibration (within onc+immuno only)")
    print("=" * 78)
    cal = calibration_per_prediction(oi_records)
    print(f"\n  n predictions: {cal['n_predictions']}")
    print(f"  overall hit rate: {cal['hit_rate']:.2%}")
    print(f"  Brier: {cal['Brier']:.4f}")
    print(f"  ECE  : {cal['ECE']:.4f}")
    print(f"\n  Calibration curve (10 bins):")
    print(f"  {'bin':<14s} {'n':>5s} {'mean_conf':>10s} {'mean_hit':>10s} {'gap':>7s}")
    for b in cal["bins"]:
        if b["n"] == 0:
            continue
        print(f"  [{b['bin'][0]:.2f},{b['bin'][1]:.2f}) {b['n']:>5d}  "
              f"{b['mean_conf']:>9.3f}  {b['mean_hit']:>9.3f}  {b['gap']:>6.3f}")

    # ---------- 7. Curated-prior override impact ----------
    print("\n" + "=" * 78)
    print("7. Curated-prior override impact")
    print("=" * 78)
    cp = curated_override_impact(oi_records)
    print(f"\n  Drugs with ≥1 promotion: {cp['n_with_promotion']}/{cp['n_total']} "
          f"({cp['pct_with_promotion']:.1%})")
    print(f"  Hit@10 rate among promoted: {cp['n_promoted_hits']}/{cp['n_with_promotion']} "
          f"({cp['hit10_rate_among_promoted']:.1%})")

    # ---------- 8. Mechanism class breakdown ----------
    print("\n" + "=" * 78)
    print("8. Mechanism class breakdown (top targets by case count)")
    print("=" * 78)
    mech = mechanism_class_breakdown(oi_records)
    print(f"\n  {'target':<14s} {'n':>4s} {'hit@10':>8s} {'rate':>7s}")
    for row in mech[:20]:
        if row["n"] < 2:
            continue
        print(f"  {row['target']:<14s} {row['n']:>4d}  {row['hit10']:>6d}  "
              f"{row['hit10_rate']:>6.1%}")

    # ---------- 9. Save full output ----------
    out = {
        "n_onc_immuno": len(oi_records),
        "overall": overall,
        "per_ta": per_ta,
        "per_severity": per_sev,
        "per_dataset": per_ds,
        "confidence_ppv": conf_ppv,
        "calibration": cal,
        "curated_override": cp,
        "mechanism_classes": mech,
    }
    out_path = RESULTS / "phase_3_onc_immuno_audit.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
