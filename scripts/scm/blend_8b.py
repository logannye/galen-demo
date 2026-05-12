"""Sprint 8B: Assemble final 8B substrate from PPI-backed SIDER + other sources.

The 8B blend differs from 8A in TWO ways:
  1. SIDER base is replaced by scm_edges_ppi_backed.json (PPI backoff for
     targets with <K_FULL=5 training drugs)
  2. Otherwise identical to 8A (CTD, OT, PharmGKB, AOP-Wiki, curated
     priors, OnSIDES via max aggregation).

Output: results/scm_edges_blended_8b.json — drop-in replacement for
scm_edges_blended.json. Signed-α stays in scm_edges_signed.json and is
applied at scoring time (not pre-aggregated).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .multi_source_edges import (
    normalize_aopwiki, normalize_ctd, normalize_curated, normalize_onsides,
    normalize_opentargets, normalize_pharmgkb, normalize_sider, load_edges,
)


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def main() -> int:
    print("=" * 78)
    print("Sprint 8B: Multi-source α(S|T) blending (PPI-backed SIDER base)")
    print("=" * 78)

    # --- PPI-backed SIDER (Sprint 8B base)
    with open(RESULTS / "scm_edges_ppi_backed.json") as f:
        sider_alpha = json.load(f)

    # --- Other sources (same as 8A)
    ctd_edges = load_edges(RESULTS / "scm_edges_ctd.json")
    aop_edges = load_edges(RESULTS / "scm_edges_aopwiki.json")
    ot_edges = load_edges(RESULTS / "scm_edges_opentargets.json")
    pgkb_edges = load_edges(RESULTS / "scm_edges_pharmgkb.json")
    curated_path = RESULTS / "scm_edges_curated_priors.json"
    curated_edges = {}
    if curated_path.exists():
        with open(curated_path) as f:
            curated_edges = json.load(f).get("priors", {})
    onsides_path = RESULTS / "scm_edges_onsides.json"
    onsides_alpha = {}
    if onsides_path.exists():
        with open(onsides_path) as f:
            onsides_alpha = json.load(f).get("edges", {})
    print(f"[load] PPI-backed SIDER α: {len(sider_alpha)} targets")
    print(f"[load] CTD: {len(ctd_edges)} targets")
    print(f"[load] AOP-Wiki: {len(aop_edges)} targets")
    print(f"[load] OpenTargets: {len(ot_edges)} targets")
    print(f"[load] PharmGKB: {len(pgkb_edges)} targets")
    print(f"[load] Curated class-effect priors: {len(curated_edges)} targets")
    print(f"[load] OnSIDES α: {len(onsides_alpha)} targets")

    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    se_vocab = v["umls_ids"]

    all_targets = (set(sider_alpha.keys()) | set(ctd_edges.keys())
                   | set(aop_edges.keys()) | set(ot_edges.keys())
                   | set(pgkb_edges.keys()) | set(curated_edges.keys())
                   | set(onsides_alpha.keys()))
    print(f"[blend] target union: {len(all_targets)}")

    blended: dict[str, dict[str, float]] = {}
    per_source_n: dict[str, int] = defaultdict(int)

    for u in all_targets:
        sider_row = sider_alpha.get(u, {})
        ctd_row = ctd_edges.get(u, {})
        aop_row = aop_edges.get(u, {})
        ot_row = ot_edges.get(u, {})
        pgkb_row = pgkb_edges.get(u, {})
        curated_row = curated_edges.get(u, {})
        onsides_row = onsides_alpha.get(u, {})
        blended_target: dict[str, float] = {}

        for s in se_vocab:
            sources = []
            a_sider = normalize_sider(sider_row.get(s, 0.0))
            if a_sider > 0:
                sources.append(("sider_ppi_backed", a_sider))
                per_source_n["sider_ppi_backed"] += 1
            ctd_e = ctd_row.get(s)
            if ctd_e:
                a_ctd = normalize_ctd(ctd_e.get("evidence_count", 0),
                                        ctd_e.get("pmid_count", 0))
                if a_ctd > 0:
                    sources.append(("ctd", a_ctd))
                    per_source_n["ctd"] += 1
            aop_e = aop_row.get(s)
            if aop_e:
                a_aop = normalize_aopwiki(aop_e.get("n_aops", 0))
                if a_aop > 0:
                    sources.append(("aop", a_aop))
                    per_source_n["aop"] += 1
            ot_e = ot_row.get(s)
            if ot_e:
                a_ot = normalize_opentargets(ot_e.get("max_llr", 0),
                                                ot_e.get("max_count", 0))
                if a_ot > 0:
                    sources.append(("ot", a_ot))
                    per_source_n["ot"] += 1
            pgkb_e = pgkb_row.get(s)
            if pgkb_e:
                a_pgkb = normalize_pharmgkb(
                    pgkb_e.get("max_evidence_score", 0),
                    pgkb_e.get("n_annotations", 0),
                )
                if a_pgkb > 0:
                    sources.append(("pgkb", a_pgkb))
                    per_source_n["pgkb"] += 1
            if s in curated_row:
                a_cur = normalize_curated(curated_row[s])
                if a_cur > 0:
                    sources.append(("curated", a_cur))
                    per_source_n["curated"] += 1
            a_onsides = normalize_onsides(onsides_row.get(s, 0.0))
            if a_onsides > 0.05:
                sources.append(("onsides", a_onsides))
                per_source_n["onsides"] += 1

            if sources:
                blended_target[s] = max(a for _, a in sources)
                per_source_n["any"] += 1
            else:
                blended_target[s] = sider_row.get(s, 1.0 / 247)

        blended[u] = blended_target

    out_path = RESULTS / "scm_edges_blended_8b.json"
    with open(out_path, "w") as f:
        json.dump(blended, f)
    print(f"[save] {out_path}")

    print(f"\n[stats] per-source contribution to non-zero edges:")
    for src, n in sorted(per_source_n.items(), key=lambda x: -x[1]):
        print(f"  {src:<22s}: {n:,}")

    total_nonzero = sum(
        1 for u in blended for a in blended[u].values() if a > 0.01
    )
    total_possible = len(all_targets) * len(se_vocab)
    print(f"[stats] non-zero α > 0.01: {total_nonzero:,} / {total_possible:,} "
          f"({100 * total_nonzero / total_possible:.2f}%)")

    max_per_target = sorted([max(d.values()) for d in blended.values()])
    if max_per_target:
        n = len(max_per_target)
        print(f"[stats] max-α per target: min={max_per_target[0]:.3f} "
              f"median={max_per_target[n//2]:.3f} max={max_per_target[-1]:.3f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
