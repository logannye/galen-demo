"""Lift-normalized SCM scoring.

The frequency-based α(S|T) gets dominated by COMMON side effects (Nausea,
Headache, Dizziness) which have high α for many targets. For clinically-
meaningful rare-tox prediction we need to rank by LIFT over the base
rate:

  lift_score(S | drug) = NoisyOR(S | drug) / base_rate(S)

Where base_rate(S) = (training drugs with S) / (total training drugs).

This converts "probability" into "specificity" — predictions are favored
when they're MORE COMMON for this drug than across the population.

This is the key inference-time fix for clinical-failure prediction.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .scoring import affinity_weight


def load_base_rates() -> dict[str, float]:
    """Compute base_rate(S) from the training drug catalog."""
    workspace = Path(__file__).resolve().parent.parent.parent
    with open(workspace / "results/catalog.json") as f:
        cat = json.load(f)
    with open(workspace / "results/side_effect_vocab.json") as f:
        v = json.load(f)
    vocab = v["umls_ids"]

    train_drugs = [d for d in cat["drugs"] if d["split"] == "train"]
    n_train = len(train_drugs)
    counts: dict[str, int] = {s: 0 for s in vocab}
    for d in train_drugs:
        for s in d.get("side_effects_in_vocab", []):
            if s in counts:
                counts[s] += 1
    base_rates = {s: max(c / n_train, 1.0 / n_train) for s, c in counts.items()}
    return base_rates


def lift_score_drug(
    binding_profile: list[dict],
    edges: dict[str, dict[str, float]],
    side_effect_vocab: list[str],
    base_rates: dict[str, float],
    *,
    affinity_mode: str = "flat",
) -> list[tuple[str, float]]:
    """Score with LIFT-normalized noisy-OR.

    lift_score(S | drug) = NoisyOR(S | drug) / base_rate(S)
    """
    # Compute noisy-OR per side effect (log-space)
    log1m_prod: dict[str, float] = {s: 0.0 for s in side_effect_vocab}
    for t in binding_profile:
        u = t.get("uniprot")
        if not u or u not in edges:
            continue
        beta = affinity_weight(t.get("standard_value_nm", 1000.0),
                                mode=affinity_mode)
        edge_row = edges[u]
        for s in side_effect_vocab:
            alpha = edge_row.get(s, 0.0) * beta
            if alpha >= 1.0:
                alpha = 0.999
            log1m_prod[s] += math.log(1.0 - alpha)

    # Convert to noisy-OR probability, then to lift
    scored: list[tuple[str, float]] = []
    for s in side_effect_vocab:
        p = 1.0 - math.exp(log1m_prod[s])
        b = max(base_rates.get(s, 1e-3), 1e-3)
        lift = p / b
        scored.append((s, lift))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def lift_top_k_predictions(
    binding_profile: list[dict],
    edges: dict[str, dict[str, float]],
    side_effect_vocab: list[str],
    base_rates: dict[str, float],
    k: int = 50,
    *,
    affinity_mode: str = "flat",
) -> list[str]:
    scored = lift_score_drug(
        binding_profile, edges, side_effect_vocab, base_rates,
        affinity_mode=affinity_mode,
    )
    return [s for s, _ in scored[:k]]
