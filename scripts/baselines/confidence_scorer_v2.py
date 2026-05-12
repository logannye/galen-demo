"""Sprint K.2: enriched confidence calibrator.

Adds features to Sprint J's calibrator:
  - max α value of top contributing target for this AE
  - number of distinct evidence sources for this (target, AE) edge
  - target-class one-hot (kinase / GPCR / ion-channel / nuclear-receptor / other)
  - whether the AE was in the SCM's top-3 (very strong substrate signal)

Original 5 features:
  - hybrid_rank_inverse
  - scm_rank_inverse
  - llm_name_agrees
  - override_applied
  - drug_in_catalog

New 4+ features added in v2.

Trains on per-prediction labels from Sprint J data (correct labeling
target per Sprint J's mid-eval fix).
"""
from __future__ import annotations

import json
import math
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


# Target-class buckets for one-hot (gene_symbol pattern → class)
TARGET_CLASSES = (
    "kinase",       # genes ending in K, having a kinase name, etc.
    "gpcr",         # ADR, HTR, CHR, DR, ADRB, etc.
    "ion_channel",  # KCN, SCN, CACNA, etc.
    "nuclear_recpt", # NR3C, ESR, AR, PPAR, etc.
    "immune",       # IL, CD, ITGA, TNF, PD, CTLA, etc.
    "transporter",  # SLC, ABC, etc.
    "other",
)


def classify_target(gene_symbol: str) -> str:
    g = (gene_symbol or "").upper()
    if not g:
        return "other"
    if any(g.startswith(p) for p in ("KCN", "SCN", "CACN", "TRP", "HCN")):
        return "ion_channel"
    if any(g.startswith(p) for p in ("ADR", "HTR", "CHR", "DR", "OPR",
                                      "HR", "MTNR", "EDN", "AGT", "ACE",
                                      "GHR", "S1PR", "GPR")):
        return "gpcr"
    if any(g.startswith(p) for p in ("NR3", "NR1", "ESR", "AR", "PPAR",
                                      "VDR", "THR", "RAR", "RXR")):
        return "nuclear_recpt"
    if any(g.startswith(p) for p in ("IL", "CD", "ITG", "TNF", "PD",
                                      "CTLA", "LAG", "FCG", "HLA",
                                      "TNFRSF", "TNFSF", "CCR", "CXCR",
                                      "IFNAR", "IGHE", "IGH", "C5",
                                      "C3", "C1")):
        return "immune"
    if any(g.startswith(p) for p in ("SLC", "ABC", "ATP")):
        return "transporter"
    if g.endswith("K") or g.endswith("RK") or g.endswith("KK"):
        return "kinase"
    if g in ("EGFR", "ERBB2", "ERBB3", "ERBB4", "KIT", "KDR", "FLT1",
             "PDGFRA", "PDGFRB", "MET", "ALK", "ROS1", "RET", "BRAF",
             "RAF1", "BTK", "ABL1", "JAK1", "JAK2", "JAK3", "TYK2",
             "CDK4", "CDK6", "MTOR", "PIK3CA", "PIK3CG", "AKT1",
             "MAP2K1", "MAP2K2", "TOP2A", "TOP2B", "PARP1", "BCL2",
             "FGFR1", "FGFR2", "FGFR3", "FGFR4", "IDH1", "IDH2",
             "NTRK1", "NTRK2", "NTRK3", "EZH2", "HMGCR"):
        return "kinase"  # use kinase bucket for "drug target catalytic"
    return "other"


def featurize_prediction(
    predicted_umls: str,
    hybrid_rank: int,
    scm_rank: int | None,
    llm_with_name_top10: set[str] | None,
    override_applied: bool,
    drug_in_catalog: bool,
    top_target_gene: str | None,
    top_target_alpha: float,
    n_evidence_sources: int,
    in_scm_top3: bool,
) -> list[float]:
    hybrid_rank_inv = 1.0 / max(1, hybrid_rank)
    scm_rank_inv = (
        1.0 / max(1, scm_rank) if scm_rank is not None and scm_rank <= 50 else 0.0
    )
    llm_agrees = (
        1.0 if (llm_with_name_top10 is not None
                and predicted_umls in llm_with_name_top10) else 0.0
    )
    override_flag = 1.0 if override_applied else 0.0
    catalog_flag = 1.0 if drug_in_catalog else 0.0
    alpha = max(0.0, min(1.0, top_target_alpha or 0.0))
    n_src = min(8, n_evidence_sources or 0) / 8.0
    scm3 = 1.0 if in_scm_top3 else 0.0

    # Target-class one-hot
    cls = classify_target(top_target_gene or "") if top_target_gene else "other"
    one_hot = [1.0 if cls == c else 0.0 for c in TARGET_CLASSES]

    return [
        hybrid_rank_inv, scm_rank_inv, llm_agrees,
        override_flag, catalog_flag,
        alpha, n_src, scm3,
        *one_hot,
    ]


def feature_names() -> list[str]:
    return [
        "hybrid_rank_inverse", "scm_rank_inverse", "llm_name_agrees",
        "override_applied", "drug_in_catalog",
        "top_target_alpha", "n_evidence_sources_norm", "in_scm_top3",
        *[f"target_class__{c}" for c in TARGET_CLASSES],
    ]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_confidence_v2(features: list[float], coeffs: dict) -> float:
    names = feature_names()
    logit = coeffs.get("intercept", 0.0)
    for f, n in zip(features, names):
        logit += f * coeffs.get(n, 0.0)
    return _sigmoid(logit)


def fit_calibrator_v2() -> dict:
    """Fit logistic regression on Sprint J data with v2 features.

    Per-prediction labels (label = 1 if predicted_umls in causal_se_umls).
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("[refit] sklearn missing")
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
                drug_in_cat = (
                    1.0 if not r.get("biologic_recovery") else 0.0
                )
                override_promoted = {p["umls"] for p in r.get("promotions", [])}
                hybrid_top10 = r.get("hybrid_top10", [])
                # SCM top-3 is the first 3 hybrid_top10 entries that had
                # SCM contribution (we don't have full SCM list; approximate)
                # Use scm_rank field if present
                scm_rank = r.get("scm_rank")
                # For each prediction in confidence_top10
                for c in r.get("confidence_top10", []):
                    rank = c["rank"]
                    umls = c["umls"]
                    label = 1 if umls in causal else 0
                    # Best-effort feature extraction (some fields not stored)
                    feats.append(featurize_prediction(
                        predicted_umls=umls,
                        hybrid_rank=rank,
                        scm_rank=scm_rank,
                        llm_with_name_top10=None,
                        override_applied=(umls in override_promoted),
                        drug_in_catalog=bool(drug_in_cat),
                        top_target_gene=r.get("causal_off_target"),
                        top_target_alpha=0.5,  # placeholder; we don't have per-AE alpha
                        n_evidence_sources=3,  # placeholder
                        in_scm_top3=(scm_rank is not None and scm_rank <= 3
                                      and umls in hybrid_top10[:3]),
                    ))
                    labels.append(label)

    if len(feats) < 100:
        print(f"[refit v2] not enough samples: {len(feats)}")
        return {}

    X = np.array(feats)
    y = np.array(labels)
    print(f"[refit v2] n={len(feats)} samples; hit_rate={y.mean():.3f}")
    lr = LogisticRegression(C=1.0, max_iter=3000)
    lr.fit(X, y)

    names = feature_names()
    coeffs = {"intercept": float(lr.intercept_[0])}
    for i, n in enumerate(names):
        coeffs[n] = float(lr.coef_[0][i])

    out = {"coeffs": coeffs, "n_train": len(feats), "version": 2,
            "hit_rate": float(y.mean())}
    out_path = RESULTS / "confidence_calibrator_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[refit v2] saved {out_path}")

    # Evaluate on Sprint J data (same data for now; held-out testing
    # would require Sprint K data)
    from .confidence_scorer import brier_score, expected_calibration_error
    for arm in ("sonnet", "opus"):
        for fn in (f"sprint_j_safety_{arm}.json",
                    f"sprint_j_ood_safety_{arm}.json"):
            path = RESULTS / fn
            if not path.exists():
                continue
            with open(path) as f:
                d = json.load(f)
            probs, lbls = [], []
            for r in d.get("per_drug", []):
                if r.get("skipped"):
                    continue
                causal = set(r.get("causal_side_effects_umls", []))
                drug_in_cat = 1.0 if not r.get("biologic_recovery") else 0.0
                override_promoted = {p["umls"] for p in r.get("promotions", [])}
                scm_rank = r.get("scm_rank")
                hybrid_top10 = r.get("hybrid_top10", [])
                for c in r.get("confidence_top10", []):
                    rank = c["rank"]
                    umls = c["umls"]
                    feats_pred = featurize_prediction(
                        predicted_umls=umls,
                        hybrid_rank=rank,
                        scm_rank=scm_rank,
                        llm_with_name_top10=None,
                        override_applied=(umls in override_promoted),
                        drug_in_catalog=bool(drug_in_cat),
                        top_target_gene=r.get("causal_off_target"),
                        top_target_alpha=0.5,
                        n_evidence_sources=3,
                        in_scm_top3=(scm_rank is not None and scm_rank <= 3
                                      and umls in hybrid_top10[:3]),
                    )
                    conf = compute_confidence_v2(feats_pred, coeffs)
                    probs.append(conf)
                    lbls.append(1 if umls in causal else 0)
            if probs:
                brier = brier_score(probs, lbls)
                ece = expected_calibration_error(probs, lbls)
                print(f"[{fn}] v2: Brier={brier:.4f} ECE={ece:.4f} "
                      f"mean_conf={sum(probs)/len(probs):.3f}")

    return coeffs


def load_calibrator_v2() -> dict:
    path = RESULTS / "confidence_calibrator_v2.json"
    if path.exists():
        with open(path) as f:
            d = json.load(f)
        return d.get("coeffs", {})
    return {}


if __name__ == "__main__":
    fit_calibrator_v2()
