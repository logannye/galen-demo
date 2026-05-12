"""Sprint 8A: Ingest DGIdb 5.0 → drug-target action types.

DGIdb (Drug-Gene Interaction Database) aggregates drug-target action types
from 30+ sources. The interaction_type column classifies binding as
agonist / antagonist / inhibitor / activator / modulator / binder /
blocker / etc.

For Sprint 8A, we use this to surface ACTION TYPE per (drug, target)
to the LLM Hybrid prompt as mechanism context. The α math does NOT
change in 8A — this is metadata, not a new α source.

Pipeline:
  1. Parse interactions.tsv (98k drug-gene records)
  2. Map drug_concept_id (chembl:CHEMBL...) → ChEMBL molregno
     or drug_name → molregno via name lookup
  3. Map gene_concept_id (hgnc:...) → UniProt via target_vocab
  4. Aggregate (chembl_id, uniprot) → set of {interaction_types}
  5. Classify the aggregated action as one of:
     - "inhibit": antagonist, inhibitor, blocker, suppressor
     - "activate": agonist, activator, inducer, stimulator
     - "modulator": modulator, allosteric, partial-agonist
     - "binder": binder (no functional info)
     - "unknown": empty / NULL / contradictory

Output: results/dgidb_action_types.json
  {
    "chembl_id": {
      "uniprot": {
        "action_class": "inhibit" | "activate" | "modulator" | "binder" | "unknown",
        "raw_types": [...],
        "n_sources": int
      }
    }
  }
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from .ingest_ctd import build_uniprot_gene_map


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
DGIDB = WORKSPACE / "data/raw/dgidb"


# Action type classification
INHIBIT_TERMS = {
    "antagonist", "inhibitor", "blocker", "suppressor", "negative modulator",
    "inverse agonist", "channel blocker", "competitive antagonist",
    "allosteric antagonist", "noncompetitive antagonist", "inhibitory",
    "inhibition of synthesis", "antibody",
}
ACTIVATE_TERMS = {
    "agonist", "activator", "inducer", "stimulator", "positive modulator",
    "channel opener", "potentiator", "partial agonist",
    # Note: "partial agonist" classed as activate by default; some are
    # functional antagonists in vivo but we go with the receptor-level
    # classification.
}
MODULATOR_TERMS = {
    "modulator", "allosteric modulator", "regulator", "cofactor",
    "substrate", "product of",
}
BINDER_TERMS = {
    "binder", "ligand", "binding",
}


def classify_action(types: set[str]) -> str:
    """Classify aggregated interaction types."""
    lowered = {t.lower().strip() for t in types if t and str(t).strip()}
    has_inhibit = any(any(term in t for term in INHIBIT_TERMS) for t in lowered)
    has_activate = any(any(term in t for term in ACTIVATE_TERMS) for t in lowered)
    has_modulator = any(any(term in t for term in MODULATOR_TERMS) for t in lowered)
    has_binder = any(any(term in t for term in BINDER_TERMS) for t in lowered)
    if has_inhibit and not has_activate:
        return "inhibit"
    if has_activate and not has_inhibit:
        return "activate"
    if has_modulator:
        return "modulator"
    if has_binder:
        return "binder"
    if has_inhibit and has_activate:
        # Contradictory — treat as modulator
        return "modulator"
    return "unknown"


def parse_chembl_id(concept_id: str) -> str | None:
    """Extract ChEMBL ID from 'chembl:CHEMBL1234567'."""
    if not concept_id:
        return None
    m = re.match(r"chembl:(CHEMBL\d+)", concept_id, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def build_chembl_to_molregno(catalog_drugs: list[dict]) -> dict[str, str]:
    """Match catalog drugs' chembl_id → molregno."""
    out: dict[str, str] = {}
    for d in catalog_drugs:
        chembl = d.get("chembl_id") or d.get("CHEMBL_ID")
        molregno = str(d.get("molregno"))
        if chembl and molregno:
            out[chembl.upper()] = molregno
    return out


def build_drugname_to_molregno(catalog_drugs: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for d in catalog_drugs:
        name = (d.get("drug_name") or "").lower().strip()
        molregno = str(d.get("molregno"))
        if name and molregno:
            out[name] = molregno
    return out


def main() -> int:
    print("=" * 78)
    print("Sprint 8A: DGIdb 5.0 ingest → drug-target action types")
    print("=" * 78)

    u2g, g2u = build_uniprot_gene_map()
    print(f"[setup] target_vocab: {len(u2g)} uniprots; {sum(len(v) for v in g2u.values())} gene aliases")

    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    chembl_to_molregno = build_chembl_to_molregno(cat["drugs"])
    drugname_to_molregno = build_drugname_to_molregno(cat["drugs"])
    print(f"[setup] catalog: {len(chembl_to_molregno)} chembl_ids, "
          f"{len(drugname_to_molregno)} drug_names")

    # Aggregate: (molregno, uniprot) → set of interaction_types
    agg: dict[tuple[str, str], set[str]] = defaultdict(set)
    n_rows = 0
    n_drug_matched = 0
    n_gene_matched = 0
    n_both_matched = 0

    with open(DGIDB / "interactions.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_rows += 1
            gene = (row.get("gene_name") or "").upper().strip()
            drug_concept_id = (row.get("drug_concept_id") or "").strip()
            drug_name = (row.get("drug_name") or "").strip()
            interaction_type = (row.get("interaction_type") or "").strip()

            # Skip non-informative
            if interaction_type in ("", "NULL", "null"):
                continue

            # Map drug to molregno
            molregno = None
            chembl = parse_chembl_id(drug_concept_id)
            if chembl:
                molregno = chembl_to_molregno.get(chembl)
            if molregno is None:
                molregno = drugname_to_molregno.get(drug_name.lower())
            if molregno is None:
                continue
            n_drug_matched += 1

            # Map gene to UniProt
            uniprots = g2u.get(gene, set())
            if not uniprots:
                continue
            n_gene_matched += 1
            n_both_matched += 1
            for u in uniprots:
                agg[(molregno, u)].add(interaction_type)

    print(f"[parse] total rows: {n_rows:,}")
    print(f"[parse] drug-matched rows: {n_drug_matched:,}")
    print(f"[parse] gene-matched rows: {n_gene_matched:,}")
    print(f"[parse] both-matched (drug+gene): {n_both_matched:,}")
    print(f"[parse] unique (drug, target) pairs: {len(agg):,}")

    # Classify and serialize
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    class_counts: dict[str, int] = defaultdict(int)
    for (molregno, u), types in agg.items():
        action_class = classify_action(types)
        class_counts[action_class] += 1
        out[molregno][u] = {
            "action_class": action_class,
            "raw_types": sorted(types),
            "n_sources": len(types),
        }

    print(f"\n[classify] action class distribution:")
    for k, n in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<12s}: {n:,}")

    out_path = RESULTS / "dgidb_action_types.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_drugs": len(out),
            "n_pairs": len(agg),
            "class_counts": dict(class_counts),
            "drug_target_actions": dict(out),
        }, f)
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
