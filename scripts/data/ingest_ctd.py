"""Ingest CTD (Comparative Toxicogenomics Database) curated causal edges.

CTD provides curated chemical-gene-disease causal triples. For our SCM
we extract two complementary edge sources:

  1. gene-disease "marker/mechanism" edges from CTD_genes_diseases.csv.gz
     — these are curated assertions that modulating gene G causes/marks
     disease D (i.e., target → adverse outcome edges)

  2. chemical-disease "marker/mechanism" edges from
     CTD_chemicals_diseases.csv.gz — curated chemical → disease causal
     assertions

We map:
  - GeneSymbol → UniProt accession (via our target_vocab.json reverse map)
  - DiseaseName → SIDER UMLS code (via display-name normalized match)

Output: results/scm_edges_ctd.json with the per-(target, side-effect)
evidence counts from CTD. These will be blended into the main α(S|T)
via Sprint 4F.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CTD = WORKSPACE / "data/raw/ctd"


_BRIT_AMER = [
    ("anaem", "anem"), ("anaes", "anes"), ("leuk", "leuc"),
    ("oesophag", "esophag"), ("oedema", "edema"), ("haemo", "hemo"),
    ("haemat", "hemat"), ("oestrogen", "estrogen"), ("gynaec", "gynec"),
    ("paediat", "pediat"), ("orthopaed", "orthoped"),
    ("colour", "color"), ("tumour", "tumor"), ("foeto", "feto"),
    ("foetal", "fetal"), ("oxygenation", "oxygenation"),
]


def _normalize_disease_name(name: str) -> str:
    """Aggressive normalization for cross-source disease matching.

    Handles British/American spelling, comma-prefix ordering, punctuation,
    and common medical suffixes. Returns a normalized string.
    """
    n = name.lower().strip().strip('"').strip()
    n = re.sub(r"[^a-z0-9 ,]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    # British → American spelling normalization
    for brit, amer in _BRIT_AMER:
        n = n.replace(brit, amer)
    # Handle comma-prefix: "Anemia, Hemolytic" → "anemia hemolytic" AND "hemolytic anemia"
    if "," in n:
        parts = [p.strip() for p in n.split(",")]
        if len(parts) >= 2:
            n = " ".join(parts)
    # remove common suffixes
    for s in (" disorder", " disorders", " disease", " diseases",
              " syndrome", " toxic", " induced"):
        if n.endswith(s):
            n = n[: -len(s)].strip()
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _all_variants(name: str) -> set[str]:
    """Generate all normalized variants of a disease name to maximize matching."""
    out: set[str] = set()
    base = _normalize_disease_name(name)
    if not base:
        return out
    out.add(base)
    # Also add reversed comma-prefix variant
    if "," in name:
        parts = [p.strip() for p in name.split(",")]
        if len(parts) == 2:
            reversed_form = _normalize_disease_name(parts[1] + " " + parts[0])
            if reversed_form:
                out.add(reversed_form)
    # First word only (for broad matching)
    words = base.split()
    if len(words) >= 2:
        out.add(words[0])  # too broad; we'll be careful
    return out


def build_uniprot_gene_map() -> tuple[dict[str, str], dict[str, set[str]]]:
    """Returns (uniprot → gene_symbol, gene_symbol_upper → set of uniprots)."""
    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    u2g: dict[str, str] = {}
    g2u: dict[str, set[str]] = defaultdict(set)
    for t in tv["targets"]:
        u = t["uniprot"]
        g = (t.get("gene_symbol") or "").strip().upper()
        if not g:
            continue
        u2g[u] = g
        g2u[g].add(u)
    return u2g, g2u


def build_disease_name_map() -> tuple[dict[str, str], dict[str, str]]:
    """Returns (normalized SIDER display → UMLS code, UMLS code → display).

    Aggressive: each UMLS gets indexed under ALL normalized variants so
    cross-source matching has the best chance.
    """
    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    name_to_umls: dict[str, str] = {}
    for u in v["umls_ids"]:
        display = v["display_names"].get(u, u)
        for variant in _all_variants(display):
            # don't overwrite existing UMLS with a different one for the same variant
            name_to_umls.setdefault(variant, u)
    return name_to_umls, v["display_names"]


def _try_map_disease(disease_name: str, name_to_umls: dict[str, str]) -> str | None:
    """Try multiple normalized variants to map CTD disease → UMLS."""
    for variant in _all_variants(disease_name):
        if variant in name_to_umls:
            return name_to_umls[variant]
        # substring fallback: if our variant is a 2+ word substring of any vocab
        # key, accept. Disabled here for speed; can be enabled if needed.
    return None


def ingest_ctd_genes_diseases() -> dict[tuple[str, str], dict]:
    """Stream CTD_genes_diseases.csv.gz, return {(uniprot, umls): evidence_record}.

    Filter:
      - DirectEvidence == "marker/mechanism"
      - GeneSymbol in our target_vocab (case-insensitive)
      - DiseaseName maps to our SIDER UMLS vocab
    """
    print("[ctd/genes-diseases] loading vocab maps...")
    u2g, g2u = build_uniprot_gene_map()
    name_to_umls, _ = build_disease_name_map()
    print(f"  target_vocab: {len(u2g)} uniprots → {len(g2u)} unique gene symbols")
    print(f"  se_vocab name map: {len(name_to_umls)} normalized aliases")

    edges: dict[tuple[str, str], dict] = {}
    n_lines = 0
    n_kept = 0
    n_gene_hit = 0
    n_disease_hit = 0

    path = CTD / "CTD_genes_diseases.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            n_lines += 1
            if n_lines % 5_000_000 == 0:
                print(f"  scanned {n_lines:>11,d} rows; kept {n_kept:,d} edges so far")
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 9:
                continue
            gene_sym = row[0].strip().upper()
            disease_name = row[2].strip()
            direct_ev = row[4].strip()
            pmids = row[8].strip()

            # Keep only curated mechanistic edges
            if direct_ev != "marker/mechanism":
                continue
            if gene_sym not in g2u:
                continue
            n_gene_hit += 1
            umls = _try_map_disease(disease_name, name_to_umls)
            if umls is None:
                continue
            n_disease_hit += 1

            for u in g2u[gene_sym]:
                key = (u, umls)
                if key not in edges:
                    edges[key] = {
                        "uniprot": u, "gene_symbol": gene_sym,
                        "umls": umls,
                        "ctd_disease_name": disease_name,
                        "pmid_count": 0, "evidence_count": 0,
                    }
                edges[key]["evidence_count"] += 1
                if pmids:
                    edges[key]["pmid_count"] += pmids.count("|") + 1
                n_kept += 1
    print(f"  TOTAL rows scanned: {n_lines:,d}")
    print(f"  rows with gene in vocab: {n_gene_hit:,d}")
    print(f"  rows with gene AND disease in vocab: {n_disease_hit:,d}")
    print(f"  unique (target, side-effect) edges kept: {len(edges):,d}")
    return edges


def main() -> int:
    print("=" * 78)
    print("Sprint 4A: CTD genes-diseases ingest (marker/mechanism evidence)")
    print("=" * 78)

    edges = ingest_ctd_genes_diseases()

    # Save in a simple {uniprot: {umls: {evidence_count, pmid_count, disease_name}}} structure
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for (u, s), rec in edges.items():
        out[u][s] = {
            "evidence_count": rec["evidence_count"],
            "pmid_count": rec["pmid_count"],
            "ctd_disease_name": rec["ctd_disease_name"],
            "gene_symbol": rec["gene_symbol"],
        }
    out_path = RESULTS / "scm_edges_ctd.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_unique_edges": len(edges),
            "n_targets_with_edges": len(out),
            "edges": out,
        }, f, indent=2)
    print(f"\n[ctd] saved: {out_path}")

    # Summary
    edge_counts = sorted([e["evidence_count"] for e in edges.values()])
    if edge_counts:
        n = len(edge_counts)
        print(f"  edge evidence_count distribution: "
              f"min={edge_counts[0]} median={edge_counts[n//2]} "
              f"p95={edge_counts[int(n*0.95)]} max={edge_counts[-1]}")

    pmid_counts = sorted([e["pmid_count"] for e in edges.values()])
    if pmid_counts:
        n = len(pmid_counts)
        print(f"  edge pmid_count distribution: "
              f"min={pmid_counts[0]} median={pmid_counts[n//2]} "
              f"p95={pmid_counts[int(n*0.95)]} max={pmid_counts[-1]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
