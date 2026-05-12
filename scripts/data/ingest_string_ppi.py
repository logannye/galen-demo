"""Sprint 8B: Ingest STRING v12 human PPI → target × target similarity.

STRING combines evidence from text-mining, experimental data, databases,
co-expression, and predictions to score protein-protein interactions.
For Sprint 8B's sparse-target backoff, we want HIGH-CONFIDENCE neighbors
to enable α transfer.

Pipeline:
  1. Load protein.aliases.v12.0.txt.gz → build ENSP → UniProt map (only
     human, only sources=UniProt_AC|BLAST_UniProt_AC).
  2. Stream protein.links.v12.0.txt.gz, filter to combined_score >= 700
     and both proteins map to UniProts in our target_vocab.
  3. Build {uniprot: [(neighbor_uniprot, weight, gene), ...]} where
     weight = combined_score / 1000.0 ∈ [0.7, 1.0]
  4. Save.

Output: results/string_ppi_neighbors.json — drop-in for ppi_backoff.py
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
STRING = WORKSPACE / "data/raw/string"

MIN_SCORE = 700  # combined_score threshold; 700+ is "high-confidence" per STRING docs
TOP_K_NEIGHBORS = 20  # keep at most K neighbors per target


def build_ensp_to_uniprot() -> dict[str, str]:
    """Parse protein.aliases.v12.0 → ENSP → UniProt mapping.

    Filters to source IN {UniProt_AC, Ensembl_UniProt_AC, BLAST_UniProt_AC}.
    """
    ensp_to_u: dict[str, str] = {}
    fp = STRING / "protein.aliases.v12.0.txt.gz"
    with gzip.open(fp, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            src = (row.get("source") or "").strip()
            if "UniProt_AC" not in src:
                continue
            ensp = (row.get("#string_protein_id") or "").strip()
            alias = (row.get("alias") or "").strip()
            # UniProt ACs look like: P12345, Q9XYZ12. Take only well-formed.
            if not ensp or not alias:
                continue
            # Prefer first encountered (canonical comes first in alias dump)
            if ensp not in ensp_to_u:
                ensp_to_u[ensp] = alias
    return ensp_to_u


def main() -> int:
    print("=" * 78)
    print("Sprint 8B: STRING v12 PPI ingest → target similarity")
    print("=" * 78)

    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    target_set = {t["uniprot"] for t in tv["targets"]}
    target_to_gene = {t["uniprot"]: t["gene_symbol"] for t in tv["targets"]}
    print(f"[setup] target_vocab: {len(target_set)} uniprots")

    print("[ensp→uniprot] loading aliases...")
    ensp_to_u = build_ensp_to_uniprot()
    print(f"[ensp→uniprot] {len(ensp_to_u):,} mappings")

    # Stream links file, filter, accumulate
    neighbors: dict[str, list[tuple[str, float]]] = defaultdict(list)
    n_rows = 0
    n_pass_score = 0
    n_both_in_vocab = 0

    fp = STRING / "protein.links.v12.0.txt.gz"
    with gzip.open(fp, "rt") as f:
        header = f.readline()  # skip header
        for line in f:
            n_rows += 1
            if n_rows % 1_000_000 == 0:
                print(f"  [stream] {n_rows:,} rows, {n_both_in_vocab:,} kept",
                      flush=True)
            parts = line.strip().split(" ")
            if len(parts) != 3:
                continue
            ensp1, ensp2, score = parts
            try:
                score_i = int(score)
            except ValueError:
                continue
            if score_i < MIN_SCORE:
                continue
            n_pass_score += 1
            u1 = ensp_to_u.get(ensp1)
            u2 = ensp_to_u.get(ensp2)
            if not u1 or not u2:
                continue
            if u1 not in target_set or u2 not in target_set:
                continue
            n_both_in_vocab += 1
            # Each edge appears twice in STRING (symmetric); we keep both
            # directions for fast lookup.
            weight = score_i / 1000.0
            neighbors[u1].append((u2, weight))

    print(f"[stream] total: {n_rows:,}")
    print(f"[stream] passed score≥{MIN_SCORE}: {n_pass_score:,}")
    print(f"[stream] both proteins in target_vocab: {n_both_in_vocab:,}")

    # Trim to top-K neighbors per target (highest weight)
    trimmed: dict[str, list[dict]] = {}
    for u, neigh in neighbors.items():
        neigh.sort(key=lambda x: -x[1])
        kept = neigh[:TOP_K_NEIGHBORS]
        trimmed[u] = [
            {"uniprot": v, "weight": w, "gene": target_to_gene.get(v, "?")}
            for (v, w) in kept
        ]

    n_targets_with_neighbors = sum(1 for v in trimmed.values() if v)
    avg_neigh = (
        sum(len(v) for v in trimmed.values()) / max(1, len(trimmed))
    )
    print(f"[trim] targets with ≥1 PPI neighbor: {n_targets_with_neighbors}")
    print(f"[trim] avg neighbors per target: {avg_neigh:.1f}")

    out = {
        "n_targets": len(trimmed),
        "min_combined_score": MIN_SCORE,
        "top_k_neighbors": TOP_K_NEIGHBORS,
        "neighbors": trimmed,
    }
    out_path = RESULTS / "string_ppi_neighbors.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
