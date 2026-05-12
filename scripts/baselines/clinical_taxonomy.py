"""Phase 7: clinical taxonomy lookups — severity tier + organ system + counterfactual.

Provides:
  - severity_tier(umls): "critical" / "serious" / "common"
  - organ_system(umls): one of 14 organ systems
  - counterfactual_removal_delta(profile, edges, ae_umls, target_uniprot):
        how much P(AE | drug) changes if target_uniprot is removed
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"


@lru_cache(maxsize=1)
def _load_severity() -> dict:
    with open(RESULTS / "ae_severity_map.json") as f:
        m = json.load(f)
    return {
        "critical": set(m["critical"]),
        "serious": set(m["serious"]),
        "tiers": m["tiers"],
    }


@lru_cache(maxsize=1)
def _load_organ_system() -> dict:
    with open(RESULTS / "ae_organ_system_map.json") as f:
        m = json.load(f)
    return {
        "umls_to_system": m["umls_to_system"],
        "systems": m["systems"],
    }


SEVERITY_PRIORITY = {"critical": 0, "serious": 1, "common": 2}
SEVERITY_COLOR = {
    "critical": "#dc2626",  # red
    "serious": "#ea580c",   # orange
    "common": "#64748b",    # gray
}
SEVERITY_LABEL = {
    "critical": "CRITICAL (BBW-level)",
    "serious": "SERIOUS (W&P)",
    "common": "COMMON (AR section)",
}


def severity_tier(umls: str) -> str:
    """Returns 'critical' / 'serious' / 'common' for an UMLS code."""
    s = _load_severity()
    if umls in s["critical"]:
        return "critical"
    if umls in s["serious"]:
        return "serious"
    return "common"


def severity_color(umls: str) -> str:
    """Hex color for the severity tier (red/orange/gray)."""
    return SEVERITY_COLOR[severity_tier(umls)]


def severity_label(umls: str) -> str:
    """Human-readable label (CRITICAL/SERIOUS/COMMON with parenthetical)."""
    return SEVERITY_LABEL[severity_tier(umls)]


# ---------- Organ system ----------

ORGAN_SYSTEM_DISPLAY = {
    "cardiac": "Cardiac",
    "hepatic": "Hepatic",
    "renal": "Renal",
    "hematologic": "Hematologic",
    "pulmonary": "Pulmonary",
    "gi": "Gastrointestinal",
    "neuro_psych": "Neurologic / Psychiatric",
    "dermatologic": "Dermatologic",
    "endocrine": "Endocrine",
    "immune_infection": "Immune / Infectious",
    "musculoskeletal": "Musculoskeletal",
    "ophthalmologic": "Ophthalmologic",
    "reproductive": "Reproductive",
    "general": "General / Constitutional",
}


def organ_system(umls: str) -> str:
    """Returns organ system id (e.g. 'cardiac'); 'general' if not classified."""
    m = _load_organ_system()
    return m["umls_to_system"].get(umls, "general")


def organ_system_display(umls: str) -> str:
    """Returns the display name for the organ system."""
    return ORGAN_SYSTEM_DISPLAY[organ_system(umls)]


# ---------- Counterfactual mitigation analysis ----------

def _affinity_to_beta(value_nm: float, mode: str = "log_sigmoid") -> float:
    """Match scoring.py affinity weighting. log_sigmoid is Phase K production."""
    if value_nm is None or value_nm <= 0:
        return 1.0
    if mode == "flat":
        return 1.0
    # log_sigmoid: β = sigmoid(-log10(Kd_nM / 100)) — full weight at <10nM, ~0.5 at 100nM
    log_kd = math.log10(value_nm / 100.0)
    return 1.0 / (1.0 + math.exp(log_kd))


def compute_full_score(
    binding_profile: list[dict], edges: dict, ae_umls: str,
    affinity_mode: str = "log_sigmoid",
) -> float:
    """Noisy-OR P(AE | drug) using full binding profile."""
    prob_none = 1.0
    for t in binding_profile:
        uniprot = t.get("uniprot")
        if not uniprot:
            continue
        alpha = edges.get(uniprot, {}).get(ae_umls, 0.0)
        if alpha <= 0:
            continue
        beta = _affinity_to_beta(t.get("standard_value_nm"), affinity_mode)
        prob_none *= (1.0 - alpha * beta)
    return 1.0 - prob_none


def counterfactual_removal_delta(
    binding_profile: list[dict], edges: dict, ae_umls: str,
    target_uniprot: str, affinity_mode: str = "log_sigmoid",
) -> dict:
    """If target_uniprot is removed from binding_profile, how much does
    P(AE) change?

    Returns:
      {
        "p_with": original P(AE | full profile),
        "p_without": new P(AE | profile minus target),
        "delta": p_with - p_without (positive = target was contributing),
        "pct_attributable": delta / p_with (fraction of risk attributable),
      }
    """
    p_with = compute_full_score(binding_profile, edges, ae_umls, affinity_mode)
    filtered = [t for t in binding_profile if t.get("uniprot") != target_uniprot]
    p_without = compute_full_score(filtered, edges, ae_umls, affinity_mode)
    delta = p_with - p_without
    pct = delta / max(p_with, 1e-9) if p_with > 0 else 0.0
    return {
        "p_with": p_with,
        "p_without": p_without,
        "delta": delta,
        "pct_attributable": pct,
    }


def rank_targets_by_attribution(
    binding_profile: list[dict], edges: dict, ae_umls: str,
    top_n: int = 5, affinity_mode: str = "log_sigmoid",
) -> list[dict]:
    """For a given AE, rank binding targets by how much each contributes.

    For non-saturated P (< 0.99): use raw delta in P.
    For saturated P (>= 0.99): use log-odds contribution
        (-log(1 - α × β)) so we can still differentiate.

    Returns top_n records:
      [{"uniprot", "gene_symbol", "p_with", "p_without", "delta",
        "pct_attributable", "kd_nm", "log_odds_contrib"}, ...]
    sorted by log_odds_contrib descending (which equals delta-rank when not saturated).
    """
    # Compute per-target log-odds contribution and counterfactual delta
    p_full = compute_full_score(binding_profile, edges, ae_umls, affinity_mode)
    saturated = p_full >= 0.99

    rows = []
    for t in binding_profile:
        u = t.get("uniprot")
        if not u:
            continue
        alpha = edges.get(u, {}).get(ae_umls, 0.0)
        if alpha <= 0:
            continue
        beta = _affinity_to_beta(t.get("standard_value_nm"), affinity_mode)
        # Log-odds contribution: -log(1 - α × β) ≥ 0
        contrib = alpha * beta
        if contrib >= 1.0:
            log_odds = 20.0  # effectively saturating contribution
        else:
            log_odds = -math.log(max(1.0 - contrib, 1e-12))
        # Counterfactual: P(AE | drug minus this target)
        cf = counterfactual_removal_delta(
            binding_profile, edges, ae_umls, u, affinity_mode,
        )
        rows.append({
            "uniprot": u,
            "gene_symbol": t.get("gene_symbol", u),
            "kd_nm": t.get("standard_value_nm", 0.0),
            "alpha": alpha,
            "beta": beta,
            "log_odds_contrib": log_odds,
            "p_with": cf["p_with"],
            "p_without": cf["p_without"],
            "delta": cf["delta"],
            "pct_attributable": (cf["delta"] / max(cf["p_with"], 1e-9)),
        })

    # Sort by log-odds contribution (works for both saturated and non-saturated)
    # — log-odds-contrib equals -log(1 - α×β), monotonic with α×β
    # When duplicates by uniprot (multi-binding entries), keep highest contrib
    by_uniprot: dict[str, dict] = {}
    for r in rows:
        u = r["uniprot"]
        if u not in by_uniprot or r["log_odds_contrib"] > by_uniprot[u]["log_odds_contrib"]:
            by_uniprot[u] = r
    final = sorted(by_uniprot.values(), key=lambda x: -x["log_odds_contrib"])

    # For saturated cases, recompute pct_attributable based on log-odds share
    if saturated:
        total_log_odds = sum(r["log_odds_contrib"] for r in final)
        if total_log_odds > 0:
            for r in final:
                r["pct_attributable"] = r["log_odds_contrib"] / total_log_odds

    return final[:top_n]
