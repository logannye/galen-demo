"""Ingest PharmGKB clinical annotations → SCM target → side-effect edges.

PharmGKB clinical annotations link gene variants to drug-phenotype
associations with evidence levels 1A (consensus causal) through 4
(preliminary). The Phenotype Category includes "Toxicity" entries which
are direct gene → ADR associations.

Output: results/scm_edges_pharmgkb.json
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from .ingest_ctd import (
    _all_variants, build_disease_name_map, build_uniprot_gene_map,
)


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
PGKB = WORKSPACE / "data/raw/pharmgkb"


# PharmGKB evidence-level → numeric strength score
LEVEL_SCORE = {
    "1A": 6, "1B": 5, "2A": 4, "2B": 3, "3": 2, "4": 1, "": 0,
}


def main() -> int:
    print("=" * 78)
    print("Sprint 4D: PharmGKB clinical annotations ingest")
    print("=" * 78)

    u2g, g2u = build_uniprot_gene_map()
    name_to_umls, _ = build_disease_name_map()
    print(f"[setup] target_vocab: {len(u2g)} uniprots; "
          f"se_vocab: {len(name_to_umls)} aliases")

    edges: dict[tuple[str, str], dict] = {}
    n_rows = 0
    n_toxicity = 0
    n_gene_mapped = 0
    n_phenotype_mapped = 0

    fp = PGKB / "clinical_annotations.tsv"
    if not fp.exists():
        print(f"[error] {fp} not found")
        return 1
    with open(fp) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_rows += 1
            gene = (row.get("Gene") or "").strip().upper()
            level = (row.get("Level of Evidence") or "").strip()
            phenotype_cat = (row.get("Phenotype Category") or "").strip()
            phenotypes = (row.get("Phenotype(s)") or "").strip()
            drugs = (row.get("Drug(s)") or "").strip()
            if not gene or not phenotypes:
                continue
            # Focus on TOXICITY annotations (most relevant for ADR prediction)
            # Also include "Efficacy" since some efficacy issues correspond to ADRs
            is_tox = "toxicity" in phenotype_cat.lower()
            if not is_tox:
                continue
            n_toxicity += 1
            # Map gene → uniprot
            uniprots = g2u.get(gene, set())
            if not uniprots:
                continue
            n_gene_mapped += 1
            # Phenotypes may be comma- or semicolon-separated
            phenotype_list = re.split(r"[;,]", phenotypes)
            for phen in phenotype_list:
                phen = phen.strip()
                if not phen:
                    continue
                # Try each variant
                umls = None
                for variant in _all_variants(phen):
                    if variant in name_to_umls:
                        umls = name_to_umls[variant]
                        break
                if umls is None:
                    continue
                n_phenotype_mapped += 1
                weight = LEVEL_SCORE.get(level, 0)
                for u in uniprots:
                    key = (u, umls)
                    if key not in edges:
                        edges[key] = {
                            "uniprot": u, "umls": umls,
                            "max_evidence_score": 0,
                            "n_annotations": 0,
                            "examples_drugs": [],
                            "examples_phenotypes": [],
                            "best_level": "",
                        }
                    edges[key]["n_annotations"] += 1
                    if weight > edges[key]["max_evidence_score"]:
                        edges[key]["max_evidence_score"] = weight
                        edges[key]["best_level"] = level
                    if drugs and drugs not in edges[key]["examples_drugs"]:
                        edges[key]["examples_drugs"].append(drugs[:100])
                    if phen not in edges[key]["examples_phenotypes"]:
                        edges[key]["examples_phenotypes"].append(phen)

    print(f"[parse] total clinical annotations: {n_rows}")
    print(f"[parse] toxicity-category rows: {n_toxicity}")
    print(f"[parse] toxicity rows with gene mapped: {n_gene_mapped}")
    print(f"[parse] phenotype-mapped rows: {n_phenotype_mapped}")
    print(f"[parse] unique (target, side-effect) edges: {len(edges)}")

    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for (u, s), rec in edges.items():
        out[u][s] = rec
    out_path = RESULTS / "scm_edges_pharmgkb.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_unique_edges": len(edges),
            "n_targets_with_edges": len(out),
            "edges": out,
        }, f, indent=2)
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
