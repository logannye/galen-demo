"""Sprint 8A: Ingest DrugCentral drug-target interactions → action types.

DrugCentral 2021_09_01 provides curated drug-target interactions with:
  - ACT_TYPE: assay type (Ki/IC50/Kd/etc.)
  - ACT_VALUE: nM affinity
  - ACTION_TYPE: free-text action descriptor (Inhibitor/Agonist/Antagonist/etc.)
  - MOA: 1 if target is a known mechanism-of-action for the drug

This complements DGIdb's interaction_types with DrugCentral's expert-
curated action descriptors. We use both to:
  - Surface action context to the LLM Hybrid prompt
  - Identify drug's primary MOA target (for prior generation in 8B)

Pipeline:
  1. Stream drug.target.interaction.tsv.gz
  2. Map DRUG_NAME → catalog molregno (via name match)
  3. Use ACCESSION (UniProt) directly
  4. Aggregate ACTION_TYPE per (molregno, uniprot)
  5. Classify into inhibit/activate/modulator/binder/unknown (same scheme
     as DGIdb for consistency)
  6. Also surface MOA flag (mechanism-of-action target indicator)

Output: results/drugcentral_action_types.json
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from .ingest_dgidb import classify_action


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
DRUGCENTRAL = WORKSPACE / "data/raw/drugcentral"


def main() -> int:
    print("=" * 78)
    print("Sprint 8A: DrugCentral 2021_09_01 ingest → drug-target action types")
    print("=" * 78)

    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    # Build drug_name → molregno
    name_to_molregno: dict[str, str] = {}
    for d in cat["drugs"]:
        name = (d.get("drug_name") or "").lower().strip()
        molregno = str(d.get("molregno"))
        if name and molregno:
            name_to_molregno[name] = molregno
    print(f"[setup] catalog: {len(name_to_molregno)} drugs by name")

    # Validate uniprot vocabulary
    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    target_set = {t["uniprot"] for t in tv["targets"]}
    print(f"[setup] target_vocab: {len(target_set)} uniprots")

    # Stream drug.target.interaction.tsv.gz
    agg: dict[tuple[str, str], dict] = defaultdict(lambda: {"actions": set(), "moa": False})
    n_rows = 0
    n_drug_matched = 0
    n_target_matched = 0
    n_both = 0
    moa_pairs = 0

    fp = DRUGCENTRAL / "drug.target.interaction.tsv.gz"
    with gzip.open(fp, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t", quotechar='"')
        for row in reader:
            n_rows += 1
            drug_name = (row.get("DRUG_NAME") or "").lower().strip().strip('"')
            uniprot = (row.get("ACCESSION") or "").strip().strip('"')
            action_type = (row.get("ACTION_TYPE") or "").strip().strip('"')
            moa = (row.get("MOA") or "").strip().strip('"')

            molregno = name_to_molregno.get(drug_name)
            if molregno:
                n_drug_matched += 1
            if uniprot in target_set:
                n_target_matched += 1
            if not molregno or uniprot not in target_set:
                continue
            n_both += 1
            agg[(molregno, uniprot)]["actions"].add(action_type)
            if moa in ("1", "True", "true", "T"):
                agg[(molregno, uniprot)]["moa"] = True
                moa_pairs += 1

    print(f"[parse] total rows: {n_rows:,}")
    print(f"[parse] drug-name matched: {n_drug_matched:,}")
    print(f"[parse] uniprot-matched: {n_target_matched:,}")
    print(f"[parse] both matched: {n_both:,}")
    print(f"[parse] MOA pairs: {moa_pairs:,}")
    print(f"[parse] unique (drug, target) pairs: {len(agg):,}")

    # Classify
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    class_counts: dict[str, int] = defaultdict(int)
    for (molregno, u), info in agg.items():
        action_class = classify_action(info["actions"])
        class_counts[action_class] += 1
        out[molregno][u] = {
            "action_class": action_class,
            "raw_actions": sorted(a for a in info["actions"] if a),
            "moa": info["moa"],
        }

    print(f"\n[classify] action class distribution:")
    for k, n in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:<12s}: {n:,}")

    out_path = RESULTS / "drugcentral_action_types.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_drugs": len(out),
            "n_pairs": len(agg),
            "n_moa_pairs": moa_pairs,
            "class_counts": dict(class_counts),
            "drug_target_actions": dict(out),
        }, f)
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
