"""Per-target attribution for SCM side-effect predictions.

For each predicted (drug, side-effect S) pair, decompose the noisy-OR
score into per-target contributions so the user can see WHICH targets
are driving each predicted side effect.

Attribution(T → S | drug X) = α(S | T) × β(Ki_T)
                              / Σ_T' α(S | T') × β(Ki_T')

This is the normalized direct contribution of each binding target to the
predicted probability of side effect S. The biopharma use case:
"this drug is predicted to cause hepatotoxicity; the dominant contributing
targets are CYP3A4 (52%) and PPARG (31%). Can we engineer those off-targets
out of our lead compound?"
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..scm.scoring import affinity_weight


@dataclass(frozen=True)
class TargetAttribution:
    uniprot: str
    gene_symbol: str
    target_pref_name: str
    standard_value_nm: float
    alpha: float           # raw α(S | T) edge weight
    beta: float            # affinity weight
    contribution: float    # α × β, the unnormalized contribution
    contribution_pct: float  # normalized % among this drug's binding targets


@dataclass(frozen=True)
class SideEffectAttribution:
    side_effect_id: str
    side_effect_name: str
    scm_score: float       # noisy-OR aggregated probability
    rank: int
    in_sider_gold: bool | None  # None if unknown
    top_targets: list[TargetAttribution]  # ranked by contribution


def attribute_side_effect(
    side_effect_id: str,
    binding_profile: list[dict],
    edges: dict[str, dict[str, float]],
    target_vocab_info: dict[str, dict],
    *,
    affinity_mode: str = "flat",
    top_k_targets: int = 5,
) -> list[TargetAttribution]:
    """Decompose the SCM score for one side effect into per-target attributions.

    Returns the top-K contributing targets sorted by contribution.
    """
    contributions: list[TargetAttribution] = []
    for t in binding_profile:
        uniprot = t.get("uniprot")
        if not uniprot or uniprot not in edges:
            continue
        alpha = edges[uniprot].get(side_effect_id, 0.0)
        beta = affinity_weight(t.get("standard_value_nm", 1000.0), mode=affinity_mode)
        contrib = alpha * beta
        info = target_vocab_info.get(uniprot, {})
        contributions.append(TargetAttribution(
            uniprot=uniprot,
            gene_symbol=info.get("gene_symbol") or t.get("gene_symbol", uniprot),
            target_pref_name=info.get("target_pref_name") or t.get("target_pref_name", ""),
            standard_value_nm=t.get("standard_value_nm", 0.0),
            alpha=alpha,
            beta=beta,
            contribution=contrib,
            contribution_pct=0.0,
        ))
    total = sum(c.contribution for c in contributions)
    if total <= 0:
        return contributions[:top_k_targets]
    normalized = [
        TargetAttribution(
            uniprot=c.uniprot,
            gene_symbol=c.gene_symbol,
            target_pref_name=c.target_pref_name,
            standard_value_nm=c.standard_value_nm,
            alpha=c.alpha,
            beta=c.beta,
            contribution=c.contribution,
            contribution_pct=c.contribution / total,
        )
        for c in contributions
    ]
    normalized.sort(key=lambda c: c.contribution, reverse=True)
    return normalized[:top_k_targets]


def explain_predictions(
    ranked_side_effects: list[tuple[str, float]],
    binding_profile: list[dict],
    edges: dict[str, dict[str, float]],
    target_vocab_info: dict[str, dict],
    se_display_names: dict[str, str],
    *,
    gold_set: set[str] | None = None,
    top_k_se: int = 20,
    top_k_targets: int = 5,
    affinity_mode: str = "flat",
) -> list[SideEffectAttribution]:
    """For the top-K predicted side effects, return per-target attribution."""
    out: list[SideEffectAttribution] = []
    for rank, (se_id, score) in enumerate(ranked_side_effects[:top_k_se], start=1):
        targets = attribute_side_effect(
            se_id, binding_profile, edges, target_vocab_info,
            affinity_mode=affinity_mode, top_k_targets=top_k_targets,
        )
        out.append(SideEffectAttribution(
            side_effect_id=se_id,
            side_effect_name=se_display_names.get(se_id, se_id),
            scm_score=score,
            rank=rank,
            in_sider_gold=(se_id in gold_set) if gold_set is not None else None,
            top_targets=targets,
        ))
    return out
