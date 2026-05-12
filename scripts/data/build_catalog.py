"""Build the SCM Off-Target Safety drug catalog: SIDER ∩ ChEMBL ≥3 targets ≥5 side effects.

Inputs:
  data/raw/sider/drug_names.tsv  — 1430 SIDER drugs (PubChem CID → name)
  data/raw/sider/meddra_all_se.tsv  — drug-side-effect pairs (CID, ...)
  /Volumes/Databank/databases/chembl_36.db  — local ChEMBL 36 SQLite

Outputs:
  results/catalog.json — drug records with binding profile + side effects + split label
  results/side_effect_vocab.json — top-K UMLS-coded side effects vocabulary
  results/target_vocab.json — UniProt+gene-symbol target vocabulary

Pre-registration:
  Test set is the first 200 drugs after random-shuffle with seed=42 from
  drugs satisfying the eligibility filter. Train set is everything else.
  This seal is committed before any LLM call.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"
WORKSPACE = Path(__file__).resolve().parent.parent.parent
RAW = WORKSPACE / "data/raw"
RESULTS = WORKSPACE / "results"
RESULTS.mkdir(exist_ok=True, parents=True)

# Eligibility filters (matches PRE_REGISTRATION.md §4)
MIN_BINDING_TARGETS = 3
MIN_SIDE_EFFECTS = 5
MAX_AFFINITY_NM = 10_000   # ≤10 μM (i.e., 10000 nM)
SIDE_EFFECT_VOCAB_K = 500  # top-K most prevalent side effects

# Test set
TEST_N = 200
SPLIT_SEED = 42

# Activity types we accept (standard binding measures)
STANDARD_TYPES = ("Ki", "IC50", "Kd")


def _normalize_name(name: str) -> str:
    """Aggressive lower-case + strip punctuation for name matching."""
    n = name.lower().strip()
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"[^a-z0-9 \-]+", "", n)
    return n


def load_sider_drugs() -> dict[str, str]:
    """Return {cid: name} for SIDER drugs."""
    drugs: dict[str, str] = {}
    with open(RAW / "sider/drug_names.tsv") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            cid = parts[0]
            name = parts[1].strip()
            if name:
                drugs[cid] = name
    return drugs


def load_sider_side_effects() -> tuple[dict[str, set[str]], dict[str, str]]:
    """Return (cid → set of UMLS side-effect ids), and (umls_id → display name)."""
    # Column structure of meddra_all_se.tsv:
    #   STITCH compound id (flat), STITCH compound id (stereo),
    #   UMLS concept id (label), MedDRA concept type (LLT/PT),
    #   UMLS concept id (meddra), side effect name
    per_drug: dict[str, set[str]] = defaultdict(set)
    name_map: dict[str, str] = {}
    with open(RAW / "sider/meddra_all_se.tsv") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            cid_flat, _cid_stereo, umls_label, mtype, umls_meddra, se_name = parts[:6]
            # Use Preferred Term (PT) entries to dedupe near-duplicates
            if mtype != "PT":
                continue
            umls = umls_meddra or umls_label
            per_drug[cid_flat].add(umls)
            if umls not in name_map:
                name_map[umls] = se_name
    return dict(per_drug), name_map


def match_sider_to_chembl(
    conn: sqlite3.Connection, sider_drugs: dict[str, str],
) -> dict[str, str]:
    """Map SIDER CID → ChEMBL molregno via name lookup (pref_name + synonyms)."""
    # Build a ChEMBL name → molregno map (lowercase, dedup)
    cur = conn.cursor()
    chembl_name_to_molregno: dict[str, int] = {}

    cur.execute("""
        SELECT molregno, pref_name FROM molecule_dictionary
        WHERE pref_name IS NOT NULL AND max_phase > 0
    """)
    for molregno, pref_name in cur:
        key = _normalize_name(pref_name)
        if key:
            chembl_name_to_molregno.setdefault(key, molregno)

    cur.execute("""
        SELECT molregno, synonyms FROM molecule_synonyms
        WHERE syn_type IN ('INN', 'USAN', 'TRADE_NAME', 'BAN', 'JAN', 'USP', 'FDA')
    """)
    for molregno, syn in cur:
        key = _normalize_name(syn)
        if key:
            chembl_name_to_molregno.setdefault(key, molregno)

    # Match SIDER drugs by name
    cid_to_molregno: dict[str, str] = {}
    for cid, name in sider_drugs.items():
        key = _normalize_name(name)
        if key in chembl_name_to_molregno:
            cid_to_molregno[cid] = str(chembl_name_to_molregno[key])
    return cid_to_molregno


def query_binding_profile(
    conn: sqlite3.Connection, molregno: int,
) -> list[dict]:
    """Return list of binding records {target_id, gene_symbol, target_pref_name,
    standard_type, standard_value_nM} for a drug.

    Filters:
      - Human targets only (organism = 'Homo sapiens')
      - SINGLE PROTEIN target type
      - Standard binding types (Ki, IC50, Kd)
      - Affinity ≤10 μM (10000 nM)
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            td.tid AS tid,
            td.pref_name AS target_pref_name,
            cs.component_id AS component_id,
            cs.accession AS uniprot_accession,
            csyn.component_synonym AS gene_symbol,
            act.standard_type, act.standard_value, act.standard_units,
            act.relation, act.pchembl_value
        FROM activities act
        JOIN assays a ON act.assay_id = a.assay_id
        JOIN target_dictionary td ON a.tid = td.tid
        JOIN target_components tc ON td.tid = tc.tid
        JOIN component_sequences cs ON tc.component_id = cs.component_id
        LEFT JOIN component_synonyms csyn ON cs.component_id = csyn.component_id
            AND csyn.syn_type = 'GENE_SYMBOL'
        WHERE act.molregno = ?
          AND td.organism = 'Homo sapiens'
          AND td.target_type IN ('SINGLE PROTEIN', 'PROTEIN COMPLEX')
          AND act.standard_type IN ('Ki', 'IC50', 'Kd')
          AND act.standard_units = 'nM'
          AND act.standard_value IS NOT NULL
          AND act.standard_value > 0
          AND act.standard_value <= ?
          AND act.relation IN ('=', '<')
        ORDER BY act.standard_value ASC
    """, (molregno, MAX_AFFINITY_NM))
    rows = cur.fetchall()

    # Aggregate by uniprot accession: keep most potent measurement
    best_by_target: dict[str, dict] = {}
    for r in rows:
        tid, target_pref_name, component_id, uniprot, gene_symbol, \
            stype, sval, sunits, rel, pchembl = r
        if not uniprot:
            continue
        rec = {
            "uniprot": uniprot,
            "gene_symbol": gene_symbol or uniprot,
            "target_pref_name": target_pref_name,
            "standard_type": stype,
            "standard_value_nm": float(sval),
            "pchembl_value": float(pchembl) if pchembl else None,
        }
        if (uniprot not in best_by_target
                or rec["standard_value_nm"] < best_by_target[uniprot]["standard_value_nm"]):
            best_by_target[uniprot] = rec
    return list(best_by_target.values())


def build_side_effect_vocab(
    per_drug_se: dict[str, set[str]], se_names: dict[str, str], k: int,
) -> list[str]:
    """Return top-K most prevalent side-effect UMLS ids."""
    counts: dict[str, int] = defaultdict(int)
    for ses in per_drug_se.values():
        for se in ses:
            counts[se] += 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [se for se, _ in ranked[:k]]


def main() -> int:
    print("=" * 78)
    print("SCM Off-Target Safety — drug catalog build")
    print("=" * 78)

    print("[1/5] Loading SIDER drug names + side effects...")
    sider_drugs = load_sider_drugs()
    per_drug_se, se_name_map = load_sider_side_effects()
    print(f"  SIDER drugs (with names): {len(sider_drugs)}")
    print(f"  SIDER drugs (with ≥1 PT side effect): {len(per_drug_se)}")

    print("[2/5] Matching SIDER → ChEMBL by name...")
    conn = sqlite3.connect(CHEMBL_DB)
    cid_to_molregno = match_sider_to_chembl(conn, sider_drugs)
    print(f"  matched by name: {len(cid_to_molregno)}/{len(sider_drugs)}")

    print("[3/5] Building side-effect vocabulary (top-K)...")
    side_effect_vocab = build_side_effect_vocab(
        per_drug_se, se_name_map, SIDE_EFFECT_VOCAB_K,
    )
    print(f"  vocabulary size: {len(side_effect_vocab)}")

    print("[4/5] Querying ChEMBL polypharmacology for each matched drug...")
    catalog: list[dict] = []
    for i, (cid, molregno) in enumerate(cid_to_molregno.items(), start=1):
        if cid not in per_drug_se:
            continue
        se_set = per_drug_se[cid]
        # Filter side effects to those in the vocabulary (avoids head-noise)
        se_in_vocab = sorted(s for s in se_set if s in side_effect_vocab)
        if len(se_in_vocab) < MIN_SIDE_EFFECTS:
            continue
        targets = query_binding_profile(conn, int(molregno))
        if len(targets) < MIN_BINDING_TARGETS:
            continue
        catalog.append({
            "cid": cid,
            "drug_name": sider_drugs[cid],
            "molregno": molregno,
            "binding_profile": targets,
            "n_targets": len(targets),
            "side_effects_in_vocab": se_in_vocab,
            "n_side_effects": len(se_in_vocab),
        })
        if i % 100 == 0:
            print(f"  processed {i}/{len(cid_to_molregno)}; eligible so far: {len(catalog)}")
    conn.close()
    print(f"  catalog size (after filters): {len(catalog)}")
    print(f"  filter: ≥{MIN_BINDING_TARGETS} targets, ≥{MIN_SIDE_EFFECTS} side effects (in vocab)")

    if len(catalog) < TEST_N + 50:
        print(f"  WARNING: catalog ({len(catalog)}) is small for n={TEST_N} test set "
              "+ ≥50 training drugs.")

    print("[5/5] Random 80/20 split (seed=42)...")
    rng = random.Random(SPLIT_SEED)
    indices = list(range(len(catalog)))
    rng.shuffle(indices)
    test_idx = set(indices[:TEST_N])
    for i, rec in enumerate(catalog):
        rec["split"] = "test" if i in test_idx else "train"
    n_test = sum(1 for r in catalog if r["split"] == "test")
    n_train = sum(1 for r in catalog if r["split"] == "train")
    print(f"  test n={n_test}; train n={n_train}")

    # Persist
    out_catalog = RESULTS / "catalog.json"
    with open(out_catalog, "w") as f:
        json.dump({
            "n_drugs": len(catalog),
            "n_test": n_test,
            "n_train": n_train,
            "test_seed": SPLIT_SEED,
            "drugs": catalog,
        }, f, indent=2)
    print(f"  saved catalog: {out_catalog}")

    out_vocab = RESULTS / "side_effect_vocab.json"
    with open(out_vocab, "w") as f:
        json.dump({
            "k": SIDE_EFFECT_VOCAB_K,
            "umls_ids": side_effect_vocab,
            "display_names": {s: se_name_map.get(s, s) for s in side_effect_vocab},
        }, f, indent=2)
    print(f"  saved side-effect vocab: {out_vocab}")

    # Target vocab
    target_vocab: dict[str, dict] = {}
    for d in catalog:
        for t in d["binding_profile"]:
            target_vocab.setdefault(t["uniprot"], {
                "uniprot": t["uniprot"],
                "gene_symbol": t["gene_symbol"],
                "target_pref_name": t["target_pref_name"],
            })
    out_target = RESULTS / "target_vocab.json"
    with open(out_target, "w") as f:
        json.dump({
            "n_targets": len(target_vocab),
            "targets": list(target_vocab.values()),
        }, f, indent=2)
    print(f"  saved target vocab ({len(target_vocab)} targets): {out_target}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
