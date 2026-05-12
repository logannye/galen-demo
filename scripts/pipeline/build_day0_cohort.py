"""Day 0: assemble the safety-failure validation cohort.

Input: results/sprint_e_safety.json (n=223 from Sprint E/F era)
Output: results/day0_validation_cohort.json
  {
    "cases": [
      {
        "drug_id": ...,
        "drug_search_name": ...,
        "severity": "withdrawn" | "black_box" | "mechanism_established",
        "therapeutic_area": ...,
        "causal_off_target": ...,
        "causal_se_text": "...",
        "gt_umls": [...],            # 1-5 UMLS IDs for the causal SE + cluster siblings
        "molregno": int | None,
        "smiles": str | None,
        "n_binding_targets": int,
        "biologic": bool,            # smiles is None -> biologic
        "umls_resolution": "hybrid_top10" | "fuzzy_match" | "none"
      },
      ...
    ],
    "stats": { ... }
  }

UMLS resolution strategy:
  1) If hybrid_top10 + hybrid_rank in 1..10, use hybrid_top10[rank-1] as the
     primary GT UMLS (this is the EXACT UMLS the LLM ranked as #rank for
     this drug, so we know it represents the causal_se text).
  2) Else fuzzy-match causal_se text against side_effect_vocab.json
     display_names. Take the strongest match (with simple word-overlap score).
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"

# Some causal_se text strings ARE actually multi-AE (e.g. "Torsade de pointes / QT prolonged").
# We split on common separators and try to resolve each.
SPLIT_PAT = re.compile(r"\s*[/;]\s*|\s+and\s+|\s*,\s*")


def lookup_molregno(name: str, cur) -> int | None:
    """Lookup ChEMBL molregno by drug name (case-insensitive, fuzzy on synonym)."""
    if not name:
        return None
    # Direct match on molecule_dictionary.pref_name
    cur.execute(
        "SELECT molregno FROM molecule_dictionary "
        "WHERE LOWER(pref_name) = ? LIMIT 1",
        (name.lower(),),
    )
    row = cur.fetchone()
    if row:
        return int(row[0])
    # Fallback: synonym table
    cur.execute(
        "SELECT ms.molregno FROM molecule_synonyms ms "
        "WHERE LOWER(ms.synonyms) = ? LIMIT 1",
        (name.lower(),),
    )
    row = cur.fetchone()
    if row:
        return int(row[0])
    return None


def lookup_smiles(molregno: int | None, cur) -> str | None:
    if molregno is None:
        return None
    cur.execute(
        "SELECT canonical_smiles FROM compound_structures WHERE molregno = ?",
        (molregno,),
    )
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def fuzzy_umls_match(text: str, display: dict) -> list[str]:
    """Return up to 3 UMLS IDs whose display name overlaps with text.

    Strategy: rank by token Jaccard, breaking ties by substring.
    """
    if not text:
        return []
    txt_tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if not txt_tokens:
        return []
    scored = []
    for u, name in display.items():
        nm_tokens = set(re.findall(r"[a-zA-Z]+", name.lower()))
        if not nm_tokens:
            continue
        # Skip pure stop words
        nm_tokens -= {"of", "the", "in", "and", "or", "a", "an"}
        if not nm_tokens:
            continue
        jac = len(txt_tokens & nm_tokens) / max(1, len(txt_tokens | nm_tokens))
        substr = 1 if text.lower() in name.lower() or name.lower() in text.lower() else 0
        scored.append((u, name, jac, substr))
    # Strong match only: require jac >= 0.5 OR substring match
    strong = [(u, n, j, s) for u, n, j, s in scored if j >= 0.5 or s >= 1]
    if not strong:
        return []
    strong.sort(key=lambda x: (-x[2], -x[3]))
    return [u for u, _, _, _ in strong[:3]]


def main() -> None:
    print("Loading raw cohort + vocab...", flush=True)
    raw_obj = json.load(open(RESULTS / "sprint_e_safety.json"))
    raw = raw_obj.get("per_drug", raw_obj if isinstance(raw_obj, list) else [])
    vocab = json.load(open(RESULTS / "side_effect_vocab.json"))
    display = vocab["display_names"]

    # Load AE clusters for cluster-aware GT expansion (10 clinical-equivalence
    # clusters; e.g. all hepatic-injury variants are in one cluster).
    clusters_data = json.load(open(RESULTS / "ae_clusters.json"))
    umls_to_cluster: dict[str, str] = {}
    cluster_members: dict[str, list[str]] = {}
    for cid, info in clusters_data["clusters"].items():
        members = info.get("member_umls", [])
        cluster_members[cid] = members
        for u in members:
            umls_to_cluster[u] = cid

    print(f"  raw cohort n={len(raw)}", flush=True)

    conn = sqlite3.connect(CHEMBL_DB)
    cur = conn.cursor()

    cases: list[dict] = []
    n_molregno = n_smiles = n_umls_top10 = n_umls_fuzzy = n_umls_none = 0
    t0 = time.time()
    for i, c in enumerate(raw):
        name = c["drug_search_name"]
        # 1. molregno + smiles
        mr = lookup_molregno(name, cur)
        smi = lookup_smiles(mr, cur)
        if mr is not None:
            n_molregno += 1
        if smi is not None:
            n_smiles += 1

        # 2. UMLS resolution
        gt_umls: list[str] = []
        umls_resolution = "none"
        hr = c.get("hybrid_rank")
        ht10 = c.get("hybrid_top10") or []
        if hr is not None and 1 <= hr <= len(ht10):
            seed_umls = ht10[hr - 1]
            gt_umls = [seed_umls]
            umls_resolution = "hybrid_top10"
            n_umls_top10 += 1
        else:
            # Try splitting causal_se into pieces and resolving each
            text = c.get("causal_se") or ""
            pieces = [p.strip() for p in SPLIT_PAT.split(text) if p.strip()]
            seen = set()
            for p in pieces:
                for u in fuzzy_umls_match(p, display):
                    if u not in seen:
                        seen.add(u)
                        gt_umls.append(u)
            if gt_umls:
                umls_resolution = "fuzzy_match"
                n_umls_fuzzy += 1
            else:
                n_umls_none += 1

        # 3. Cluster-expand: include cluster siblings of every gt UMLS
        expanded = set(gt_umls)
        for u in gt_umls:
            cid = umls_to_cluster.get(u)
            if cid is not None:
                expanded.update(cluster_members.get(str(cid), []))
        gt_umls_expanded = sorted(expanded)

        cases.append({
            "drug_id": c["drug_id"],
            "drug_search_name": name,
            "severity": c.get("severity"),
            "therapeutic_area": c.get("therapeutic_area"),
            "causal_off_target": c.get("causal_off_target"),
            "causal_se_text": c.get("causal_se"),
            "gt_umls_primary": gt_umls,
            "gt_umls": gt_umls_expanded,  # cluster-expanded for matching
            "molregno": mr,
            "smiles": smi,
            "n_binding_targets": c.get("n_binding_targets", 0),
            "biologic": smi is None,
            "umls_resolution": umls_resolution,
            # Carry forward stored baselines for comparison
            "stored_scm_rank": c.get("scm_rank"),
            "stored_hybrid_rank": c.get("hybrid_rank"),
            "stored_llm_drug_blind_rank": c.get("llm_drug_blind_rank"),
            "stored_llm_with_name_rank": c.get("llm_with_name_rank"),
            "stored_rf_ecfp_rank": c.get("rf_ecfp_rank"),
        })

    conn.close()

    stats = {
        "n_total": len(cases),
        "n_with_molregno": n_molregno,
        "n_with_smiles": n_smiles,
        "n_umls_resolution_hybrid_top10": n_umls_top10,
        "n_umls_resolution_fuzzy_match": n_umls_fuzzy,
        "n_umls_resolution_none": n_umls_none,
        "n_biologic": sum(1 for c in cases if c["biologic"]),
        "n_smiles_AND_umls": sum(
            1 for c in cases
            if c["smiles"] is not None and c["gt_umls"]
        ),
    }
    print(f"\nResolved in {time.time()-t0:.1f}s")
    print(f"  total cases: {stats['n_total']}")
    print(f"  with molregno: {stats['n_with_molregno']}")
    print(f"  with SMILES: {stats['n_with_smiles']}")
    print(f"  UMLS via hybrid_top10: {stats['n_umls_resolution_hybrid_top10']}")
    print(f"  UMLS via fuzzy match: {stats['n_umls_resolution_fuzzy_match']}")
    print(f"  UMLS unresolved: {stats['n_umls_resolution_none']}")
    print(f"  biologic (no SMILES): {stats['n_biologic']}")
    print(f"  TESTABLE (SMILES AND UMLS): {stats['n_smiles_AND_umls']}")

    # Severity breakdown for testable subset
    from collections import Counter
    testable = [c for c in cases if c["smiles"] and c["gt_umls"]]
    print(f"\nTestable subset severity:")
    for sev, ct in Counter(c["severity"] for c in testable).most_common():
        print(f"  {sev}: {ct}")
    print(f"\nTestable subset TA:")
    for ta, ct in Counter(c["therapeutic_area"] for c in testable).most_common():
        print(f"  {ta}: {ct}")

    payload = {"stats": stats, "cases": cases}
    out = RESULTS / "day0_validation_cohort.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
