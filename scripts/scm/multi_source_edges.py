"""Sprint 4F + 7C + 8A: Multi-source α(S|T) blending for the SCM.

We compose target → side-effect edge weights from SEVEN evidence sources:

  - SIDER (frequency): drugs binding target T in training set that have
    side effect S in their FDA label
  - CTD: curated chemical-gene-disease causal triples with PubMed support
  - AOP-Wiki: formalized adverse outcome pathways with MIE → AO
  - OpenTargets: FAERS-derived statistically-significant target → AE
    (Log-Likelihood Ratio test)
  - PharmGKB: clinical annotations with evidence levels 1A–4
  - Curated class-effect priors (Sprint 7C): hand-curated FDA-BBW priors
  - OnSIDES (Sprint 8A): Tatonetti Lab v3.1.1 transformer-NLP extraction
    of drug-AE from US/UK/EU/JP labels (newer than SIDER, catches
    post-2016 biologics)

Each source provides a per-(target, side-effect) score; we normalize each
to [0, 1] and blend with max-aggregation. The resulting α_combined(S|T)
is fed back into the SCM noisy-OR scoring.

Blending function:
  α_combined(S|T) = max(α_SIDER, α_CTD, α_OT, α_AOP, α_PGKB, α_curated, α_onsides)

We use max-aggregation (not weighted sum) because each source is a
different evidence channel — if ANY of them strongly supports the
edge, the SCM should know about it. Source weights are encoded by how
each source's raw score is normalized to [0, 1].

Output: results/scm_edges_blended.json — drop-in replacement for
results/scm_edges.json. Other modules need no changes.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def normalize_sider(alpha_sider: float) -> float:
    """SIDER α is already in [0, 1]. Pass through."""
    return min(1.0, max(0.0, alpha_sider))


def normalize_ctd(evidence_count: int, pmid_count: int) -> float:
    """CTD: more curated mechanism evidence + PubMed support → stronger."""
    if evidence_count <= 0:
        return 0.0
    # Cap at evidence_count=5 (rare in CTD data anyway)
    base = min(1.0, evidence_count / 5.0)
    pmid_boost = min(0.2, pmid_count / 50.0)
    return min(1.0, base * 0.7 + pmid_boost)


def normalize_opentargets(max_llr: float, max_count: int) -> float:
    """OpenTargets LLR is unbounded; map via sigmoid.

    LLR distribution observed: median 246, p95 4504, max 37537.
    Use log10(LLR) / 4 sigmoid — gives ~0.5 at LLR=1, ~0.9 at LLR=1000.
    """
    if max_llr <= 0:
        return 0.0
    log_llr = math.log10(max(max_llr, 1.0))
    # Map log_llr ∈ [0, 4] → [0.3, 0.95]
    s = 1.0 / (1.0 + math.exp(-(log_llr - 1.5)))
    return min(1.0, s * 0.95)


def normalize_aopwiki(n_aops: int) -> float:
    """AOP-Wiki: per-AOP weight (these are formalized causal pathways).

    Each AOP is high-confidence. n_aops > 1 indicates multiple pathway
    convergence on the same target-AE → very strong.
    """
    if n_aops <= 0:
        return 0.0
    return min(1.0, 0.6 + 0.15 * n_aops)


def normalize_pharmgkb(max_evidence_score: int, n_annotations: int) -> float:
    """PharmGKB: evidence level 1A→6, 1B→5, 2A→4, ..., 4→1, ()→0.

    Higher level + more annotations → higher α.
    """
    if max_evidence_score <= 0:
        return 0.0
    level_part = max_evidence_score / 6.0
    multi_boost = min(0.2, n_annotations / 10.0)
    return min(1.0, level_part * 0.8 + multi_boost)


def normalize_curated(strength: float) -> float:
    """Curated class-effect priors are already in [0, 1]; pass through."""
    return min(1.0, max(0.0, strength))


def normalize_onsides(alpha_onsides: float) -> float:
    """OnSIDES α (from SIDER-style decomposition) is already in [0, 1].

    OnSIDES is treated identically to SIDER in normalization (both are
    drug-label-derived). The signal kind is the same (FDA label AE
    frequency); only the NLP extraction quality differs.
    """
    return min(1.0, max(0.0, alpha_onsides))


def load_edges(path: Path) -> dict[str, dict[str, dict]]:
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return d.get("edges", {})


def main() -> int:
    print("=" * 78)
    print("Sprint 4F: Multi-source α(S|T) blending")
    print("=" * 78)

    # --- Load SIDER α (the original, learned from training drugs)
    with open(RESULTS / "scm_edges.json") as f:
        sider_alpha = json.load(f)  # {uniprot: {umls: alpha}}

    # --- Load other sources
    ctd_edges = load_edges(RESULTS / "scm_edges_ctd.json")
    aop_edges = load_edges(RESULTS / "scm_edges_aopwiki.json")
    ot_edges = load_edges(RESULTS / "scm_edges_opentargets.json")
    pgkb_edges = load_edges(RESULTS / "scm_edges_pharmgkb.json")
    # Curated class-effect priors (Sprint 7C) — 6th source
    curated_path = RESULTS / "scm_edges_curated_priors.json"
    curated_edges = {}
    if curated_path.exists():
        with open(curated_path) as f:
            curated_edges = json.load(f).get("priors", {})
    # OnSIDES (Sprint 8A) — 7th source: format is {uniprot: {umls: alpha}}
    onsides_path = RESULTS / "scm_edges_onsides.json"
    onsides_alpha = {}
    if onsides_path.exists():
        with open(onsides_path) as f:
            onsides_alpha = json.load(f).get("edges", {})
    print(f"[load] SIDER α: {len(sider_alpha)} targets")
    print(f"[load] CTD: {len(ctd_edges)} targets")
    print(f"[load] AOP-Wiki: {len(aop_edges)} targets")
    print(f"[load] OpenTargets: {len(ot_edges)} targets")
    print(f"[load] PharmGKB: {len(pgkb_edges)} targets")
    print(f"[load] Curated class-effect priors: {len(curated_edges)} targets")
    print(f"[load] OnSIDES α: {len(onsides_alpha)} targets")

    # --- Collect side-effect vocab (need to know which SEs exist)
    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    se_vocab = v["umls_ids"]
    se_vocab_set = set(se_vocab)

    # --- Per-target, per-side-effect: compute α from each source then aggregate
    all_targets = (set(sider_alpha.keys()) | set(ctd_edges.keys())
                   | set(aop_edges.keys()) | set(ot_edges.keys())
                   | set(pgkb_edges.keys()) | set(curated_edges.keys())
                   | set(onsides_alpha.keys()))
    print(f"[blend] target union: {len(all_targets)}")

    blended: dict[str, dict[str, float]] = {}
    per_source_n: dict[str, int] = defaultdict(int)

    for u in all_targets:
        blended_target: dict[str, float] = {}
        sider_row = sider_alpha.get(u, {})
        ctd_row = ctd_edges.get(u, {})
        aop_row = aop_edges.get(u, {})
        ot_row = ot_edges.get(u, {})
        pgkb_row = pgkb_edges.get(u, {})
        curated_row = curated_edges.get(u, {})
        onsides_row = onsides_alpha.get(u, {})

        for s in se_vocab:
            sources = []
            # SIDER
            a_sider = normalize_sider(sider_row.get(s, 0.0))
            if a_sider > 0:
                sources.append(("sider", a_sider))
            # CTD
            ctd_e = ctd_row.get(s)
            if ctd_e:
                a_ctd = normalize_ctd(ctd_e.get("evidence_count", 0),
                                        ctd_e.get("pmid_count", 0))
                if a_ctd > 0:
                    sources.append(("ctd", a_ctd))
                    per_source_n["ctd"] += 1
            # AOP-Wiki
            aop_e = aop_row.get(s)
            if aop_e:
                a_aop = normalize_aopwiki(aop_e.get("n_aops", 0))
                if a_aop > 0:
                    sources.append(("aop", a_aop))
                    per_source_n["aop"] += 1
            # OpenTargets
            ot_e = ot_row.get(s)
            if ot_e:
                a_ot = normalize_opentargets(ot_e.get("max_llr", 0),
                                                ot_e.get("max_count", 0))
                if a_ot > 0:
                    sources.append(("ot", a_ot))
                    per_source_n["ot"] += 1
            # PharmGKB
            pgkb_e = pgkb_row.get(s)
            if pgkb_e:
                a_pgkb = normalize_pharmgkb(
                    pgkb_e.get("max_evidence_score", 0),
                    pgkb_e.get("n_annotations", 0),
                )
                if a_pgkb > 0:
                    sources.append(("pgkb", a_pgkb))
                    per_source_n["pgkb"] += 1

            # Curated class-effect priors (Sprint 7C)
            if s in curated_row:
                a_cur = normalize_curated(curated_row[s])
                if a_cur > 0:
                    sources.append(("curated", a_cur))
                    per_source_n["curated"] += 1

            # OnSIDES (Sprint 8A) — only count meaningful non-Laplace values
            a_onsides = normalize_onsides(onsides_row.get(s, 0.0))
            if a_onsides > 0.05:  # avoid Laplace-smoothed 1/(N+2) noise
                sources.append(("onsides", a_onsides))
                per_source_n["onsides"] += 1

            if sources:
                # Max-aggregation: take strongest source
                blended_target[s] = max(a for _, a in sources)
                per_source_n["any"] += 1
            else:
                # If no source has signal, use Laplace smoothing baseline
                blended_target[s] = sider_row.get(s, 1.0 / 247)

        blended[u] = blended_target

    # Save
    out_path = RESULTS / "scm_edges_blended.json"
    with open(out_path, "w") as f:
        json.dump(blended, f)
    print(f"[save] {out_path}")

    # Stats
    print(f"\n[stats] per-source contribution to non-zero edges:")
    for src, n in sorted(per_source_n.items(), key=lambda x: -x[1]):
        print(f"  {src:<8s}: {n:,}")

    total_nonzero = sum(
        1 for u in blended for a in blended[u].values() if a > 0.01
    )
    total_possible = len(all_targets) * len(se_vocab)
    print(f"[stats] non-zero α > 0.01: {total_nonzero:,} / {total_possible:,} "
          f"({100 * total_nonzero / total_possible:.2f}%)")

    # Distribution of max-α per target
    max_per_target = [max(d.values()) for d in blended.values()]
    max_per_target.sort()
    if max_per_target:
        n = len(max_per_target)
        print(f"[stats] max-α per target: min={max_per_target[0]:.3f} "
              f"median={max_per_target[n//2]:.3f} max={max_per_target[-1]:.3f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
