"""Sprint 8B: Sparse-target PPI backoff.

For targets where we have few training drugs binding (|drugs binding T| < K),
the per-target α(S|T) is Laplace-smoothed to near-base-rate (≈1/247) for
most side effects. This means the SCM has NO useful signal for novel/rare
targets — they all rank similarly.

Solution: when |drugs binding T| < K, transfer α from PPI-neighbors of T.
The intuition: PPI-similar proteins tend to participate in the same
pathways and therefore drugs affecting them often have similar AE
profiles.

Backoff formula (per pre-registration):
  if n_drugs(T) >= K_FULL: α_backed(S|T) = α(S|T)   # no backoff needed
  if K_MIN <= n_drugs(T) < K_FULL:
    α_backed(S|T) = α(S|T)*(1-w) + Σ_{T'} weight(T,T') * α(S|T') * w
    where w ∈ [0, 0.5] grows with sparsity, and Σ over PPI-neighbors
  if n_drugs(T) < K_MIN:
    α_backed(S|T) = neighbor-only weighted avg, capped at 0.5

Output: results/scm_edges_ppi_backed.json (drop-in replacement for SIDER α
when target is sparse).
"""
from __future__ import annotations

import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"

K_FULL = 5       # n_drugs >= K_FULL → no backoff
K_MIN = 1        # n_drugs >= K_MIN → partial backoff (else neighbor-only)
MAX_BACKOFF_W = 0.5   # cap on backoff contribution per pre-reg


def main() -> int:
    print("=" * 78)
    print("Sprint 8B: PPI-neighbor backoff for sparse-target α")
    print("=" * 78)

    with open(RESULTS / "scm_edges.json") as f:
        sider_alpha = json.load(f)
    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    se_vocab = v["umls_ids"]
    with open(RESULTS / "scm_target_priors.json") as f:
        priors = json.load(f)
    target_n_drugs = priors["target_n_drugs"]

    with open(RESULTS / "string_ppi_neighbors.json") as f:
        ppi = json.load(f)
    neighbors_map = ppi["neighbors"]

    print(f"[load] SIDER α: {len(sider_alpha)} targets")
    print(f"[load] target n_drugs map: {len(target_n_drugs)}")
    print(f"[load] PPI neighbors: {len(neighbors_map)}")

    # Compute backoff weights and apply
    backed: dict[str, dict[str, float]] = {}
    n_full = 0
    n_partial = 0
    n_neighbor_only = 0
    n_no_neighbors = 0

    for u, alpha_row in sider_alpha.items():
        n_u = target_n_drugs.get(u, 0)
        if n_u >= K_FULL:
            backed[u] = alpha_row
            n_full += 1
            continue
        neighbors = neighbors_map.get(u, [])
        if not neighbors:
            backed[u] = alpha_row  # nothing to do
            n_no_neighbors += 1
            continue

        # Sparsity-weighted backoff w: at K_FULL → 0; at 0 drugs → MAX
        w = MAX_BACKOFF_W * (K_FULL - n_u) / K_FULL
        w = min(MAX_BACKOFF_W, max(0.0, w))

        # Build neighbor-weighted α
        # Normalize neighbor weights to sum=1 among neighbors that have α
        valid_neighbors = [
            (nbr["uniprot"], nbr["weight"])
            for nbr in neighbors
            if nbr["uniprot"] in sider_alpha
            and target_n_drugs.get(nbr["uniprot"], 0) >= 3
        ]
        if not valid_neighbors:
            backed[u] = alpha_row
            n_no_neighbors += 1
            continue

        total_w = sum(w_n for _, w_n in valid_neighbors)
        if total_w <= 0:
            backed[u] = alpha_row
            n_no_neighbors += 1
            continue

        backed_row: dict[str, float] = {}
        for s in se_vocab:
            base = alpha_row.get(s, 1.0 / 247)
            neighbor_sum = sum(
                (w_n / total_w) * sider_alpha[u_n].get(s, 1.0 / 247)
                for (u_n, w_n) in valid_neighbors
            )
            if n_u >= K_MIN:
                backed_row[s] = (1.0 - w) * base + w * neighbor_sum
                n_partial += 1
            else:
                # Pure neighbor backoff (cap contribution at MAX_BACKOFF_W)
                backed_row[s] = (
                    (1.0 - MAX_BACKOFF_W) * base + MAX_BACKOFF_W * neighbor_sum
                )
                n_neighbor_only += 1
        backed[u] = backed_row

    # n_partial and n_neighbor_only are inflated by per-SE counts; collapse
    # to per-target by checking sparsity again
    n_partial = sum(
        1 for u in backed
        if K_MIN <= target_n_drugs.get(u, 0) < K_FULL
        and any(n["uniprot"] in sider_alpha for n in neighbors_map.get(u, []))
    )
    n_neighbor_only = sum(
        1 for u in backed
        if target_n_drugs.get(u, 0) < K_MIN
        and any(n["uniprot"] in sider_alpha for n in neighbors_map.get(u, []))
    )

    print(f"\n[backoff] full (n≥{K_FULL}, no backoff): {n_full}")
    print(f"[backoff] partial backoff (n∈[{K_MIN},{K_FULL})): {n_partial}")
    print(f"[backoff] neighbor-only (n<{K_MIN}): {n_neighbor_only}")
    print(f"[backoff] no PPI neighbors available: {n_no_neighbors}")

    # Diagnostic: report change magnitude
    delta_examples: list[tuple[float, str, str]] = []
    for u in backed:
        for s in se_vocab[:100]:
            base = sider_alpha.get(u, {}).get(s, 1.0 / 247)
            new = backed[u].get(s, base)
            d = abs(new - base)
            if d > 0.01:
                delta_examples.append((d, u, s))
    delta_examples.sort(reverse=True)
    if delta_examples:
        print(f"\n[diagnostic] {len(delta_examples)} (T, SE) with |Δα| > 0.01")
        print(f"  largest 5 changes (sampled from first 100 SEs):")
        for d, u, s in delta_examples[:5]:
            print(f"    T={u} SE={s}: Δα={d:.3f}")

    out_path = RESULTS / "scm_edges_ppi_backed.json"
    with open(out_path, "w") as f:
        json.dump(backed, f)
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
