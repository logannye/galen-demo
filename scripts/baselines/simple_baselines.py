"""Simple baselines for the SOTA ladder.

These test specific architectural claims:

  base_rate          — predicts global side-effect prevalence. Tests if SCM is
                       just learning prior frequencies.
  drug_jaccard_knn   — nearest-neighbor by target Jaccard. Tests how much
                       of the SCM advantage is just "find nearest training drug."
  max_alpha          — max α(S|T) over drug's binding targets (no aggregation).
  sum_alpha          — sum α(S|T) (additive aggregation).
  mean_alpha         — mean α(S|T) (averaging aggregation).

The α-variant baselines USE THE SAME α(S|T) as the SCM but with different
aggregation operators. They isolate the contribution of noisy-OR specifically.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def rank_by_base_rate(
    training_drugs: list[dict], side_effect_vocab: list[str],
) -> list[str]:
    """Rank side effects by overall prevalence in the training set."""
    counts: dict[str, int] = defaultdict(int)
    for d in training_drugs:
        for s in d.get("side_effects_in_vocab", []):
            counts[s] += 1
    return sorted(side_effect_vocab, key=lambda s: counts.get(s, 0), reverse=True)


def rank_by_jaccard_knn(
    test_drug: dict, training_drugs: list[dict],
    side_effect_vocab: list[str], *, k: int = 5,
) -> list[str]:
    """k-NN: find top-k training drugs by target Jaccard, score each side
    effect by the Jaccard-weighted average of training-drug labels."""
    test_targets = {t["uniprot"] for t in test_drug["binding_profile"]}
    sims = []
    for tr in training_drugs:
        tr_targets = {t["uniprot"] for t in tr["binding_profile"]}
        if not (test_targets | tr_targets):
            continue
        j = len(test_targets & tr_targets) / max(len(test_targets | tr_targets), 1)
        sims.append((tr, j))
    sims.sort(key=lambda x: x[1], reverse=True)
    top_k = sims[:k]
    if not top_k:
        return list(side_effect_vocab)
    total_weight = sum(j for _, j in top_k) or 1.0
    score: dict[str, float] = {s: 0.0 for s in side_effect_vocab}
    for tr, j in top_k:
        for s in tr.get("side_effects_in_vocab", []):
            if s in score:
                score[s] += j / total_weight
    return sorted(side_effect_vocab, key=lambda s: score[s], reverse=True)


def rank_by_alpha_aggregation(
    test_drug: dict, edges: dict[str, dict[str, float]],
    side_effect_vocab: list[str], *, op: str = "max",
) -> list[str]:
    """Aggregate per-target α(S|T) via {max, sum, mean} over binding targets."""
    bound_targets = [t["uniprot"] for t in test_drug["binding_profile"]
                       if t["uniprot"] in edges]
    if not bound_targets:
        return list(side_effect_vocab)

    score: dict[str, float] = {}
    for s in side_effect_vocab:
        vals = [edges[u].get(s, 0.0) for u in bound_targets]
        if op == "max":
            score[s] = max(vals)
        elif op == "sum":
            score[s] = sum(vals)
        elif op == "mean":
            score[s] = sum(vals) / len(vals)
        else:
            raise ValueError(f"unknown op {op}")
    return sorted(side_effect_vocab, key=lambda s: score[s], reverse=True)
