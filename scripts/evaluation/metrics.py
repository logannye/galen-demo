"""Multi-label retrieval metrics for side-effect prediction.

Each held-out drug has a labeled side-effect set (gold) and each method
produces a ranked list of side effects from the vocabulary. We compute
standard IR metrics per drug, then aggregate across drugs.
"""
from __future__ import annotations


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if k <= 0 or not ranked:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for s in top if s in gold)
    return hits / k


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold or not ranked:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for s in top if s in gold)
    return hits / len(gold)


def average_precision(ranked: list[str], gold: set[str]) -> float:
    """AP: mean of precision@k at each k where a relevant doc is retrieved."""
    if not gold or not ranked:
        return 0.0
    hits = 0
    sum_p = 0.0
    for i, s in enumerate(ranked, start=1):
        if s in gold:
            hits += 1
            sum_p += hits / i
    if hits == 0:
        return 0.0
    return sum_p / len(gold)


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    for i, s in enumerate(ranked, start=1):
        if s in gold:
            return 1.0 / i
    return 0.0


def per_drug_metrics(ranked: list[str], gold: set[str]) -> dict[str, float]:
    return {
        "p@10": precision_at_k(ranked, gold, 10),
        "p@20": precision_at_k(ranked, gold, 20),
        "p@50": precision_at_k(ranked, gold, 50),
        "r@10": recall_at_k(ranked, gold, 10),
        "r@20": recall_at_k(ranked, gold, 20),
        "r@50": recall_at_k(ranked, gold, 50),
        "ap": average_precision(ranked, gold),
        "rr": reciprocal_rank(ranked, gold),
    }
