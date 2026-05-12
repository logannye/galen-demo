"""Sprint H/I Track 1: SCM-Curated-Prior Vote with Confidence Routing.

Post-process the Hybrid LLM re-ranker's output. When the SCM substrate
has a STRONG curated prior signal (α ≥ 0.85) that the LLM re-ranker
moved below top-3, override and promote it.

Motivated by Sprint G OOD finding: on truly novel drugs, SCM-alone
beat Hybrid (73% vs 68%). The LLM re-ranker sometimes overrides
strong curated-prior signal in favor of more "common" predictions.

Sprint H override rule (v1):
  - SCM rank ≤ 10 (curated prior is "relevant")
  - AE not in Hybrid top-3
  - Cap at 3 promotions per case

Sprint I override rule (v2 — REFINED):
  - SCM rank ≤ 3 (curated prior is "very strong SCM signal")
  - Hybrid rank > 5 (LLM clearly missed it; not just below top-3)
  - Cap at 1 promotion per case
  - Effect: less aggressive; minimizes displacement of correct
    rank-8-10 hits.
"""
from __future__ import annotations

STRONG_THRESHOLD = 0.85
MAX_PROMOTIONS = 3


def load_curated_priors_for_override():
    """Returns {uniprot: {umls: alpha}} from results/scm_edges_curated_priors.json."""
    import json
    from pathlib import Path
    workspace = Path(__file__).resolve().parent.parent.parent
    results = workspace / "results"
    with open(results / "scm_edges_curated_priors.json") as f:
        p = json.load(f)
    return p.get("priors", {})


def apply_curated_prior_override(
    hybrid_ranked: list[str],
    scm_top10: list[str],
    binding_profile: list[dict],
    curated_priors: dict[str, dict[str, float]],
    strong_threshold: float = STRONG_THRESHOLD,
    max_promotions: int = MAX_PROMOTIONS,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Apply curated-prior override to Hybrid ranking.

    Returns:
      new_ranked: list of UMLS IDs with promotions applied
      promoted: list of (umls, alpha) for diagnostic logging
    """
    bound_uniprots = {t.get("uniprot") for t in binding_profile if t.get("uniprot")}
    scm_top10_set = set(scm_top10)
    promotions = []
    seen = set()

    for u in bound_uniprots:
        if u not in curated_priors:
            continue
        prior_row = curated_priors[u]
        for ae, alpha in prior_row.items():
            if alpha < strong_threshold:
                continue
            if ae not in scm_top10_set:
                continue
            if ae in hybrid_ranked[:3]:
                continue
            if ae in seen:
                continue
            seen.add(ae)
            promotions.append((ae, alpha))

    # Sort by α desc, take top max_promotions
    promotions.sort(key=lambda x: -x[1])
    promotions = promotions[:max_promotions]
    promote_set = {p for p, _ in promotions}

    if not promotions:
        return hybrid_ranked, []

    # Build new ranking: promoted items first (high-α first), then
    # original ranking minus promoted items
    promote_list = [p for p, _ in promotions]
    remaining = [r for r in hybrid_ranked if r not in promote_set]
    return promote_list + remaining, promotions


# ============================================================================
# Sprint I v2: refined override criteria (tighter, less aggressive)
# ============================================================================

V2_SCM_RANK_THRESHOLD = 3     # SCM rank ≤ 3 (very strong)
V2_HYBRID_RANK_THRESHOLD = 5  # Hybrid rank > 5 (clearly missed)
V2_MAX_PROMOTIONS = 1         # cap at 1 (minimize displacement)


def apply_curated_prior_override_v2(
    hybrid_ranked: list[str],
    scm_ranked: list[str],
    binding_profile: list[dict],
    curated_priors: dict[str, dict[str, float]],
    strong_threshold: float = STRONG_THRESHOLD,
    scm_rank_threshold: int = V2_SCM_RANK_THRESHOLD,
    hybrid_rank_threshold: int = V2_HYBRID_RANK_THRESHOLD,
    max_promotions: int = V2_MAX_PROMOTIONS,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Sprint I refined override.

    Stricter criteria than v1:
      - AE must be at SCM rank ≤ scm_rank_threshold (default 3) —
        proves the substrate has VERY strong signal for it
      - AE must be at Hybrid rank > hybrid_rank_threshold (default 5)
        — the LLM clearly missed it, not just below top-3
      - Cap at max_promotions (default 1) — minimize ripple to main

    Returns (new_ranked, promotions).
    """
    bound_uniprots = {t.get("uniprot") for t in binding_profile if t.get("uniprot")}
    # SCM rank lookup: index in scm_ranked, +1 for 1-indexed
    scm_rank_lookup = {ae: i + 1 for i, ae in enumerate(scm_ranked)}
    hybrid_rank_lookup = {ae: i + 1 for i, ae in enumerate(hybrid_ranked)}

    promotions = []
    seen = set()
    for u in bound_uniprots:
        if u not in curated_priors:
            continue
        prior_row = curated_priors[u]
        for ae, alpha in prior_row.items():
            if alpha < strong_threshold:
                continue
            if ae in seen:
                continue
            scm_rank = scm_rank_lookup.get(ae)
            if scm_rank is None or scm_rank > scm_rank_threshold:
                continue
            hybrid_rank = hybrid_rank_lookup.get(ae)
            # If AE not in Hybrid output at all, treat as "missed" → promote
            if hybrid_rank is not None and hybrid_rank <= hybrid_rank_threshold:
                continue
            seen.add(ae)
            promotions.append((ae, alpha, scm_rank))

    # Sort by (scm_rank asc, α desc) so SCM #1 wins ties
    promotions.sort(key=lambda x: (x[2], -x[1]))
    promotions = promotions[:max_promotions]
    promote_pairs = [(p[0], p[1]) for p in promotions]
    promote_set = {p[0] for p in promotions}

    if not promotions:
        return hybrid_ranked, []

    promote_list = [p[0] for p in promotions]
    remaining = [r for r in hybrid_ranked if r not in promote_set]
    return promote_list + remaining, promote_pairs
