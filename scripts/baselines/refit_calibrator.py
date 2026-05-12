"""Sprint J fix: refit confidence calibrator on per-prediction labels.

Bug in v1 (confidence_scorer.fit_calibrator_from_sprint_i): trained on
DRUG-LEVEL hit@10 labels but applied to per-prediction-position scoring.
This produced overconfident predictions (Brier 0.53, ECE 0.62).

Fix: train on PER-PREDICTION labels (predicted_umls in causal_side_effects_umls).
Use Sprint J results which now store causal_side_effects_umls in per_drug.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def refit_on_sprint_j() -> dict:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {}

    feats, labels = [], []
    for arm in ("sonnet", "opus"):
        for fn in (f"sprint_j_safety_{arm}.json",
                    f"sprint_j_ood_safety_{arm}.json"):
            path = RESULTS / fn
            if not path.exists():
                continue
            with open(path) as f:
                d = json.load(f)
            for r in d.get("per_drug", []):
                if r.get("skipped"):
                    continue
                causal = set(r.get("causal_side_effects_umls", []))
                if not causal:
                    continue
                scm_rank_lookup_max = 50  # cap
                # Per-prediction: each of the top-10 hybrid predictions
                confs = r.get("confidence_top10", [])
                hybrid_top10 = r.get("hybrid_top10", [])
                # Build SCM rank inverse for predicted UMLS
                # (rough proxy: SCM ranked from scored array; we don't have it
                # explicitly per-UMLS here — use placeholder 0.0)
                drug_in_cat = (
                    1.0 if not r.get("biologic_recovery") else 0.0
                )
                # llm_with_name ranks: estimate from llm_with_name_rank field
                llm_name_rank = r.get("llm_with_name_rank")
                llm_top10 = set()  # we don't store the full list; skip
                override_promoted_umls = {p["umls"] for p in r.get("promotions", [])}

                for c in confs:
                    rank = c["rank"]
                    umls = c["umls"]
                    label = 1 if umls in causal else 0
                    feats.append([
                        1.0 / max(1, rank),
                        0.0,  # scm_rank_inverse — not directly available
                        1.0 if umls in llm_top10 else 0.0,
                        1.0 if umls in override_promoted_umls else 0.0,
                        drug_in_cat,
                    ])
                    labels.append(label)

    if len(feats) < 100:
        print(f"[refit] not enough samples: {len(feats)}; aborting")
        return {}

    X = np.array(feats)
    y = np.array(labels)
    print(f"[refit] n_samples={len(feats)}, hit_rate={y.mean():.3f}")
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(X, y)
    coeffs = {
        "intercept": float(lr.intercept_[0]),
        "hybrid_rank_inverse": float(lr.coef_[0][0]),
        "scm_rank_inverse": float(lr.coef_[0][1]),
        "llm_name_agrees": float(lr.coef_[0][2]),
        "override_applied": float(lr.coef_[0][3]),
        "drug_in_catalog": float(lr.coef_[0][4]),
    }

    out = {"coeffs": coeffs, "n_train": len(feats),
           "hit_rate": float(y.mean())}
    out_path = RESULTS / "confidence_calibrator.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[refit] saved {out_path}")
    print(f"[refit] coeffs:")
    for k, v in coeffs.items():
        print(f"  {k}: {v:.4f}")

    # Recompute calibration on saved Sprint J data
    from .confidence_scorer import (
        compute_confidence, brier_score, expected_calibration_error,
    )
    for arm in ("sonnet", "opus"):
        for fn in (f"sprint_j_safety_{arm}.json",
                    f"sprint_j_ood_safety_{arm}.json"):
            path = RESULTS / fn
            if not path.exists():
                continue
            with open(path) as f:
                d = json.load(f)
            probs, labels_list = [], []
            for r in d.get("per_drug", []):
                if r.get("skipped"):
                    continue
                causal = set(r.get("causal_side_effects_umls", []))
                drug_in_cat = (
                    1.0 if not r.get("biologic_recovery") else 0.0
                )
                override_promoted_umls = {p["umls"] for p in r.get("promotions", [])}
                for c in r.get("confidence_top10", []):
                    rank = c["rank"]
                    umls = c["umls"]
                    new_conf = compute_confidence(
                        predicted_umls=umls,
                        hybrid_rank=rank,
                        scm_rank=None,
                        llm_with_name_top10=None,
                        override_applied=(umls in override_promoted_umls),
                        drug_in_catalog=bool(drug_in_cat),
                        coeffs=coeffs,
                    )
                    probs.append(new_conf)
                    labels_list.append(1 if umls in causal else 0)
            if probs:
                brier = brier_score(probs, labels_list)
                ece = expected_calibration_error(probs, labels_list)
                print(f"[{fn}] refit Brier={brier:.4f} ECE={ece:.4f} "
                      f"mean_conf={sum(probs)/len(probs):.3f} "
                      f"hit_rate={sum(labels_list)/len(labels_list):.3f}")

    return coeffs


if __name__ == "__main__":
    refit_on_sprint_j()
