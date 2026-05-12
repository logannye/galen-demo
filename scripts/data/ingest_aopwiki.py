"""Ingest AOP-Wiki (Adverse Outcome Pathways) → SCM target → side-effect edges.

AOP-Wiki contains ~584 formalized adverse outcome pathways. Each AOP has:
  - Molecular Initiating Event (MIE) — typically gene/protein modulation
  - Adverse Outcome (AO) — phenotypic adverse effect

For each AOP we extract:
  1. Gene/target name from the MIE title (heuristic: parentheses or
     after "antagonist,"/"agonist,"/"Activation," etc.)
  2. Adverse-effect name from the AO title

Then map:
  - Gene name → UniProt via our target_vocab.json
  - AO title → SIDER UMLS via normalized name matching

Output: results/scm_edges_aopwiki.json

Each edge has weight-of-evidence (curated by AOP-Wiki) and the path
length (#key-events). MIE→AO direct AOPs are highest confidence.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from .ingest_ctd import (
    _all_variants, _normalize_disease_name, build_disease_name_map,
    build_uniprot_gene_map,
)


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
AOPWIKI = WORKSPACE / "data/raw/aopwiki/aop-wiki.xml"

NS = "{http://www.aopkb.org/aop-xml}"


def _localtag(e: ET.Element) -> str:
    return e.tag.split("}")[-1] if "}" in e.tag else e.tag


def _extract_gene_candidates(mie_title: str) -> list[str]:
    """Extract plausible gene/protein symbols from an MIE title."""
    cands: list[str] = []
    # parenthesized symbols
    cands.extend(re.findall(r"\(([A-Za-z0-9\-_]+)\)", mie_title))
    # token after "antagonist," / "agonist," / "Activation," / "Inhibition,"
    for sep in ("antagonist,", "agonist,", "Activation,", "Inhibition,",
                  "Increase,", "Decrease,", "Binding of inhibitor,",
                  "Binding,", "Reduction,"):
        idx = mie_title.lower().find(sep.lower())
        if idx >= 0:
            tail = mie_title[idx + len(sep):].strip()
            # take first capitalized chunk
            m = re.match(r"([A-Za-z0-9\-]+(?:\s+[A-Za-z0-9\-]+){0,2})", tail)
            if m:
                cands.append(m.group(1).strip())
    # the whole title's "core" (everything after a comma)
    parts = [p.strip() for p in mie_title.split(",")]
    if len(parts) >= 2:
        cands.append(parts[-1])  # last comma-delimited segment
    # split on whitespace, take all-uppercase tokens
    cands.extend(re.findall(r"\b([A-Z][A-Z0-9]{2,})\b", mie_title))
    # dedup
    seen = set()
    out: list[str] = []
    for c in cands:
        c = c.strip()
        c = re.sub(r"\s+", " ", c)
        if c and c.upper() not in seen and len(c) >= 2:
            seen.add(c.upper())
            out.append(c)
    return out


def _map_gene_to_uniprot(
    gene_candidates: list[str], g2u: dict[str, set[str]],
) -> set[str]:
    """Try each candidate gene name → UniProt(s)."""
    uniprots: set[str] = set()
    for cand in gene_candidates:
        # exact match
        key = cand.strip().upper()
        if key in g2u:
            uniprots.update(g2u[key])
            continue
        # try aliases: PPAR alpha → PPARA, etc.
        key_compact = re.sub(r"\s+", "", key)
        if key_compact in g2u:
            uniprots.update(g2u[key_compact])
            continue
        # Greek-to-Latin: alpha → A, beta → B, etc.
        replacements = {"ALPHA": "A", "BETA": "B", "GAMMA": "G", "DELTA": "D"}
        normalized = key
        for greek, latin in replacements.items():
            normalized = normalized.replace(greek, latin)
        normalized = re.sub(r"\s+", "", normalized)
        if normalized in g2u:
            uniprots.update(g2u[normalized])
    return uniprots


def main() -> int:
    print("=" * 78)
    print("Sprint 4B: AOP-Wiki ingest")
    print("=" * 78)

    u2g, g2u = build_uniprot_gene_map()
    name_to_umls, _ = build_disease_name_map()
    print(f"[setup] target_vocab: {len(u2g)} uniprots; "
          f"se_vocab: {len(name_to_umls)} normalized aliases")

    tree = ET.parse(AOPWIKI)
    root = tree.getroot()
    kes_by_id = {
        (c.attrib.get("id") or c.attrib.get("key-event-id")): c
        for c in root if _localtag(c) == "key-event"
    }
    aops = [c for c in root if _localtag(c) == "aop"]
    print(f"[parse] AOPs: {len(aops)}; KEs: {len(kes_by_id)}")

    edges: dict[tuple[str, str], dict] = {}
    n_mapped_mie = 0
    n_mapped_ao = 0
    n_both_mapped = 0

    for aop in aops:
        title_el = aop.find(f"{NS}title")
        aop_title = (title_el.text or "") if title_el is not None else ""
        mie = aop.find(f"{NS}molecular-initiating-event")
        ao = aop.find(f"{NS}adverse-outcome")
        if mie is None or ao is None:
            continue
        mie_id = mie.attrib.get("key-event-id")
        ao_id = ao.attrib.get("key-event-id")
        mie_ke = kes_by_id.get(mie_id)
        ao_ke = kes_by_id.get(ao_id)
        if mie_ke is None or ao_ke is None:
            continue
        mie_title = (mie_ke.find(f"{NS}title").text or "").strip()
        ao_title = (ao_ke.find(f"{NS}title").text or "").strip()
        if not mie_title or not ao_title:
            continue

        # Extract genes from MIE title
        gene_candidates = _extract_gene_candidates(mie_title)
        uniprots = _map_gene_to_uniprot(gene_candidates, g2u)
        if not uniprots:
            continue
        n_mapped_mie += 1

        # Map AO title to UMLS
        umls = None
        for variant in _all_variants(ao_title):
            if variant in name_to_umls:
                umls = name_to_umls[variant]
                break
        if umls is None:
            continue
        n_mapped_ao += 1
        n_both_mapped += 1

        for u in uniprots:
            key = (u, umls)
            if key not in edges:
                edges[key] = {
                    "uniprot": u,
                    "umls": umls,
                    "n_aops": 0,
                    "example_mie_titles": [],
                    "example_ao_titles": [],
                    "example_aop_titles": [],
                }
            edges[key]["n_aops"] += 1
            if mie_title not in edges[key]["example_mie_titles"]:
                edges[key]["example_mie_titles"].append(mie_title)
            if ao_title not in edges[key]["example_ao_titles"]:
                edges[key]["example_ao_titles"].append(ao_title)
            if aop_title not in edges[key]["example_aop_titles"]:
                edges[key]["example_aop_titles"].append(aop_title)
    print(f"[parse] AOPs with mapped MIE→gene: {n_mapped_mie}")
    print(f"[parse] AOPs with both MIE→gene AND AO→umls: {n_both_mapped}")
    print(f"[parse] unique (target, side-effect) edges: {len(edges)}")

    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for (u, s), rec in edges.items():
        out[u][s] = rec
    out_path = RESULTS / "scm_edges_aopwiki.json"
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
