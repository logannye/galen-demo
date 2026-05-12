"""AE synonym-cluster post-processing for Hybrid output.

Two operations:
  1. `collapse_top_k(top_k, clusters)`: when multiple members of the same
     cluster appear in top-K, keep only the highest-ranked; push the rest
     out and let other distinct AEs promote up.
  2. `hit_at_k_clustered(gt_set, top_k, clusters, k)`: GT-hit if the GT is
     a cluster member AND ANY member of that cluster is in top-K.

Used at output time for product UX (distinct clinical concepts in top-K)
and at evaluation time for clinically-equivalent crediting.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"


@lru_cache(maxsize=1)
def load_clusters():
    """Returns {umls: cluster_id} mapping for fast lookup."""
    with open(RESULTS / "ae_clusters.json") as f:
        payload = json.load(f)
    clusters = payload["clusters"]
    umls_to_cluster = {}
    cluster_meta = {}
    for cluster_id, cluster in clusters.items():
        rep = cluster["representative_umls"]
        cluster_meta[cluster_id] = {
            "representative_umls": rep,
            "representative_name": cluster["representative_name"],
            "n_members": len(cluster["member_umls"]),
        }
        for umls in cluster["member_umls"]:
            umls_to_cluster[umls] = cluster_id
    return umls_to_cluster, cluster_meta


def get_cluster(umls: str) -> str | None:
    """Returns cluster_id for an UMLS, or None if not clustered."""
    umls_to_cluster, _ = load_clusters()
    return umls_to_cluster.get(umls)


def collapse_top_k(top_k: list[str], pad_pool: list[str] = None) -> list[str]:
    """De-duplicate by cluster membership.

    Args:
      top_k: ordered list of UMLS codes from Hybrid output
      pad_pool: optional list of further-down candidates to fill freed slots

    Returns: collapsed list, same length as input (or shorter if pad_pool empty).

    For each cluster, the FIRST-encountered member is kept; later cluster
    members are dropped. Non-clustered UMLS pass through unchanged.
    """
    umls_to_cluster, _ = load_clusters()
    seen_clusters = set()
    seen_umls = set()
    out = []
    for u in top_k:
        if u in seen_umls:
            continue
        seen_umls.add(u)
        cid = umls_to_cluster.get(u)
        if cid is not None:
            if cid in seen_clusters:
                continue  # duplicate cluster member, skip
            seen_clusters.add(cid)
        out.append(u)
    # Fill freed slots from pad_pool, with same cluster-dedup
    if pad_pool:
        for u in pad_pool:
            if len(out) >= len(top_k):
                break
            if u in seen_umls:
                continue
            seen_umls.add(u)
            cid = umls_to_cluster.get(u)
            if cid is not None and cid in seen_clusters:
                continue
            if cid is not None:
                seen_clusters.add(cid)
            out.append(u)
    return out


def hit_at_k_clustered(
    gt_set,
    top_k: list[str],
    k: int,
) -> bool:
    """Returns True if (literal hit OR any cluster member of GT is in top-K).

    Clinical-equivalent crediting: if the GT is in cluster X and any other
    member of cluster X appears in top-K, that counts as a hit.
    """
    if not gt_set or not top_k:
        return False
    umls_to_cluster, _ = load_clusters()
    # Build set of cluster_ids the GT belongs to
    gt_clusters = set()
    for g in gt_set:
        cid = umls_to_cluster.get(g)
        if cid is not None:
            gt_clusters.add(cid)
    top_k_set = set(top_k[:k])
    # Literal match
    if top_k_set & set(gt_set):
        return True
    # Cluster match
    for u in top_k_set:
        cid = umls_to_cluster.get(u)
        if cid is not None and cid in gt_clusters:
            return True
    return False


def rank_clustered(gt_set, top_k_collapsed: list[str]) -> int | None:
    """Returns the 1-indexed rank of the FIRST hit (literal or cluster) in
    the collapsed top-K. None if no hit."""
    if not gt_set or not top_k_collapsed:
        return None
    umls_to_cluster, _ = load_clusters()
    gt_clusters = {umls_to_cluster.get(g) for g in gt_set if umls_to_cluster.get(g)}
    gt_set = set(gt_set)
    for i, u in enumerate(top_k_collapsed, start=1):
        if u in gt_set:
            return i
        cid = umls_to_cluster.get(u)
        if cid is not None and cid in gt_clusters:
            return i
    return None
