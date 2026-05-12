"""Ingest OpenTargets FAERS-derived significant target → adverse event edges.

OpenTargets ingests FDA Adverse Event Reporting System (FAERS) reports
and performs statistical analysis (log-likelihood ratio test) to identify
target-AE associations that are significantly enriched vs baseline.

For each (target, AE) pair, OT provides:
  - LLR (log-likelihood ratio)
  - critval (significance threshold)
  - count (number of FAERS reports)
  - event (MedDRA Preferred Term name)
  - meddraCode

We map:
  - ENSG → UniProt (via our target_vocab.json gene_symbol)
  - event (MedDRA PT) → UMLS via SIDER display-name match

Output: results/scm_edges_opentargets.json
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .ingest_ctd import build_disease_name_map, build_uniprot_gene_map


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
OT_DIR = WORKSPACE / "data/raw/opentargets"


def build_ensg_to_uniprot() -> dict[str, set[str]]:
    """Read OT targets/ part files; map ENSG → set of UniProt accessions.

    Falls back to gene-symbol if needed: our target_vocab has gene_symbol
    per UniProt; OT target files have both ensgId and approvedSymbol.
    """
    u2g, g2u = build_uniprot_gene_map()
    ensg_to_u: dict[str, set[str]] = defaultdict(set)

    targets_dir = OT_DIR / "targets"
    if not targets_dir.exists():
        return ensg_to_u
    files = sorted(targets_dir.glob("part-*.json"))
    print(f"[ot/targets] reading {len(files)} part files...")
    for i, fp in enumerate(files):
        with open(fp) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ensg = rec.get("id")
                sym = rec.get("approvedSymbol", "")
                if not ensg or not sym:
                    continue
                sym_u = sym.strip().upper()
                if sym_u in g2u:
                    ensg_to_u[ensg].update(g2u[sym_u])
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}] mapped {len(ensg_to_u)} ENSGs so far")
    print(f"[ot/targets] {len(ensg_to_u)} ENSG → UniProt mappings")
    return ensg_to_u


def normalize_pt(name: str) -> str:
    """Normalize a MedDRA PT name for matching against SIDER display names."""
    return name.lower().strip()


def build_pt_to_umls() -> dict[str, str]:
    """Build (lowercase MedDRA PT name → UMLS) from SIDER display names."""
    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    out: dict[str, str] = {}
    for u in v["umls_ids"]:
        display = v["display_names"].get(u, u).strip().lower()
        out.setdefault(display, u)
    return out


def main() -> int:
    print("=" * 78)
    print("Sprint 4C: OpenTargets significant target → AE ingest")
    print("=" * 78)

    pt_to_umls = build_pt_to_umls()
    print(f"[setup] SIDER PT vocab: {len(pt_to_umls)} terms")
    ensg_to_u = build_ensg_to_uniprot()

    edges: dict[tuple[str, str], dict] = {}
    ae_dir = OT_DIR / "significant_target_ae"
    files = sorted(ae_dir.glob("part-*.json"))
    print(f"[ot/target_ae] reading {len(files)} part files...")
    total_records = 0
    n_target_mapped = 0
    n_event_mapped = 0
    for fp in files:
        with open(fp) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total_records += 1
                ensg = rec.get("targetId")
                event = rec.get("event", "").strip()
                llr = rec.get("llr", 0)
                critval = rec.get("critval", 0)
                count = rec.get("count", 0)
                if not ensg or not event:
                    continue
                # Map target
                uniprots = ensg_to_u.get(ensg, set())
                if not uniprots:
                    continue
                n_target_mapped += 1
                # Map event → UMLS
                umls = pt_to_umls.get(normalize_pt(event))
                if not umls:
                    continue
                n_event_mapped += 1
                for u in uniprots:
                    key = (u, umls)
                    if key not in edges:
                        edges[key] = {
                            "uniprot": u, "umls": umls,
                            "ot_event": event,
                            "max_llr": float(llr),
                            "max_count": int(count),
                            "n_records": 0,
                        }
                    edges[key]["n_records"] += 1
                    edges[key]["max_llr"] = max(edges[key]["max_llr"], float(llr))
                    edges[key]["max_count"] = max(edges[key]["max_count"], int(count))
    print(f"[ot/target_ae] total records scanned: {total_records:,}")
    print(f"[ot/target_ae] records with target mapped: {n_target_mapped:,}")
    print(f"[ot/target_ae] records with both target+event mapped: {n_event_mapped:,}")
    print(f"[ot/target_ae] unique (target, side-effect) edges: {len(edges):,}")

    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for (u, s), rec in edges.items():
        out[u][s] = rec
    out_path = RESULTS / "scm_edges_opentargets.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_unique_edges": len(edges),
            "n_targets_with_edges": len(out),
            "edges": out,
        }, f, indent=2)
    print(f"[save] {out_path}")

    # Distribution
    llrs = sorted(e["max_llr"] for e in edges.values())
    if llrs:
        n = len(llrs)
        print(f"[stats] LLR distribution: min={llrs[0]:.0f} "
              f"median={llrs[n//2]:.0f} p95={llrs[int(n*0.95)]:.0f} max={llrs[-1]:.0f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
