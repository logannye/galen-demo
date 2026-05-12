"""SCM scoring: given a drug's polypharmacology binding profile, rank side effects.

The structural function is a weighted noisy-OR over per-target contributions:

  P(side_effect S | drug X) = 1 - Π_i [1 - α(S | T_i) × β(Ki_i)]

where:
  α(S | T_i) = training-set conditional probability that drugs binding T_i
               manifest side effect S (learned in scripts/scm/edge_learning.py)
  β(Ki_i)    = affinity occupancy factor in [0, 1]; default = 1.0 for the
               Sprint 1 simplification (all ≤10 μM binding treated equal)

This is a 2-layer structural causal model:
  Drug → Target (binding) → Side Effect (manifestation)

The model has no drug-identity inputs — only the binding profile. This is
the architectural requirement that lets us compare it apples-to-apples
against the drug-blind LLM baseline.
"""
from __future__ import annotations

import math


def affinity_weight(standard_value_nm: float, *, mode: str = "flat") -> float:
    """Affinity-occupancy factor for noisy-OR aggregation.

    Sprint 1: flat (every ≤10 μM binding treated as full occupancy).
    Sprint 2 extension: sigmoid on log-affinity.
    """
    if mode == "flat":
        return 1.0
    if mode == "log_sigmoid":
        log10_um = math.log10(max(standard_value_nm, 1.0) / 1000.0)
        return 1.0 / (1.0 + math.exp(2.0 * (log10_um - 1.0)))
    raise ValueError(f"unknown affinity weight mode: {mode}")


def score_drug_side_effects(
    binding_profile: list[dict],
    edges: dict[str, dict[str, float]],
    side_effect_vocab: list[str],
    *,
    affinity_mode: str = "flat",
) -> list[tuple[str, float]]:
    """Score each side effect for a drug; return list ranked by score desc.

    binding_profile: list of {uniprot, standard_value_nm, ...} records.
    edges:           {uniprot: {se_umls: alpha}} from edge_learning.
    side_effect_vocab: list of UMLS side-effect ids.
    """
    # Per-side-effect noisy-OR product accumulator (log space for stability)
    log1m_prod: dict[str, float] = {s: 0.0 for s in side_effect_vocab}

    for t in binding_profile:
        uniprot = t.get("uniprot")
        if not uniprot or uniprot not in edges:
            continue
        # Week 2 Day 2: if the target dict carries a TargetNet binding
        # probability, use it as the beta factor (probability-weighted
        # noisy-OR). Otherwise fall back to the affinity-based weight.
        # The continuous beta means low-confidence predicted targets
        # contribute proportionally less, recovering the hit@1 that
        # binary top-K inclusion was diluting.
        pp = t.get("predicted_prob")
        if pp is not None:
            beta = float(pp)
        else:
            beta = affinity_weight(t.get("standard_value_nm", 1000.0),
                                    mode=affinity_mode)
        edge_row = edges[uniprot]
        for s in side_effect_vocab:
            alpha = edge_row.get(s, 0.0) * beta
            if alpha >= 1.0:
                # avoid log(0)
                alpha = 0.999
            log1m_prod[s] += math.log(1.0 - alpha)

    scored = [(s, 1.0 - math.exp(log1m_prod[s])) for s in side_effect_vocab]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def top_k_predictions(
    binding_profile: list[dict],
    edges: dict[str, dict[str, float]],
    side_effect_vocab: list[str],
    k: int = 50,
    *,
    affinity_mode: str = "flat",
) -> list[str]:
    """Top-K side-effect UMLS ids by SCM score."""
    scored = score_drug_side_effects(
        binding_profile, edges, side_effect_vocab,
        affinity_mode=affinity_mode,
    )
    return [s for s, _ in scored[:k]]


def score_drug_side_effects_signed(
    binding_profile: list[dict],
    edges: dict[str, dict[str, float]],
    signed_edges: dict | None,
    action_types: dict[str, str] | None,
    target_action_n_drugs: dict[str, dict[str, int]] | None,
    side_effect_vocab: list[str],
    *,
    min_drugs_for_signed: int = 3,
    affinity_mode: str = "flat",
) -> list[tuple[str, float]]:
    """Sprint 8B scoring with signed-α when action is known and sample is sufficient.

    Per-target decision:
      - If signed_edges[T][A] exists and target_action_n_drugs[T][A] >=
        min_drugs_for_signed, use signed α(S | T, A).
      - Else, fall back to action-agnostic edges[T][S] (the 8B blended).
    """
    log1m_prod: dict[str, float] = {s: 0.0 for s in side_effect_vocab}
    if signed_edges is None:
        signed_edges = {}
    if action_types is None:
        action_types = {}
    if target_action_n_drugs is None:
        target_action_n_drugs = {}

    for t in binding_profile:
        uniprot = t.get("uniprot")
        if not uniprot:
            continue
        # Week 2 Day 2: probability-weighted beta (TargetNet) supersedes
        # affinity-based weight when available.
        pp = t.get("predicted_prob")
        if pp is not None:
            beta = float(pp)
        else:
            beta = affinity_weight(t.get("standard_value_nm", 1000.0),
                                    mode=affinity_mode)
        action = action_types.get(uniprot, "unknown")
        # Decide which edge row to use
        edge_row = None
        if action != "unknown" and uniprot in signed_edges:
            n_action = (target_action_n_drugs.get(uniprot, {}) or {}).get(action, 0)
            if n_action >= min_drugs_for_signed:
                edge_row = signed_edges[uniprot].get(action)
        if edge_row is None:
            if uniprot not in edges:
                continue
            edge_row = edges[uniprot]

        for s in side_effect_vocab:
            alpha = edge_row.get(s, 0.0) * beta
            if alpha >= 1.0:
                alpha = 0.999
            log1m_prod[s] += math.log(max(1.0 - alpha, 1e-9))

    scored = [(s, 1.0 - math.exp(log1m_prod[s])) for s in side_effect_vocab]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
