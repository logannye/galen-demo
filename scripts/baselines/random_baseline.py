"""Random baseline: deterministic per-drug random side-effect ranking."""
from __future__ import annotations

import hashlib
import random


def random_rank_side_effects(
    drug_cid: str, side_effect_vocab: list[str], *,
    k: int = 50, base_seed: int = 42,
) -> list[str]:
    """Deterministic random top-K side effects."""
    h = hashlib.md5(f"{drug_cid}|{base_seed}".encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)
    rng = random.Random(seed)
    pool = list(side_effect_vocab)
    rng.shuffle(pool)
    return pool[:k]
