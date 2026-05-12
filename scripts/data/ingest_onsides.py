"""Sprint 8A: Ingest OnSIDES v3.1.1 → SCM target → side-effect edges.

OnSIDES (Tatonetti Lab, 2024) is a transformer-NLP extraction of drug-AE
associations from FDA SPL labels (DailyMed), UK SmPC (EMC), EU EMA, and
Japanese KEGG labels. It supersedes SIDER's older NLP extraction and
covers post-2016 biologics that SIDER misses.

Pipeline:
  1. Load OnSIDES vocab tables:
     - vocab_meddra_adverse_effect.csv: meddra_id → meddra_name, term_type
     - vocab_rxnorm_ingredient.csv: ingredient_id → ingredient_name
     - product_to_rxnorm.csv: label_id → rxnorm_product_id
     - vocab_rxnorm_ingredient_to_product.csv: product_id → ingredient_id
  2. Stream product_adverse_effect.csv (6.9M rows). For each label-AE pair:
     - Map label → ingredient (via product → ingredient chain)
     - Map meddra → UMLS (via meddra_name → our SE vocab display name)
     - Track (ingredient_name, umls) pairs
  3. For each catalog drug, find its OnSIDES ingredient match (by name)
     and assign the set of OnSIDES-derived AEs.
  4. SIDER-style decomposition: for each (T, S), compute
     α_onsides(S|T) = (# training drugs binding T with S in OnSIDES + 1)
                       / (# training drugs binding T + 2).

Output: results/scm_edges_onsides.json — {uniprot: {umls: alpha}} format,
drop-in compatible with multi_source_edges.py.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from .ingest_ctd import _all_variants, _normalize_disease_name


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
ONSIDES = WORKSPACE / "data/raw/onsides/csv"

# Pre-filter on prediction confidence (pred1 from OnSIDES v3.1.1).
# OnSIDES v3.1.1 release notes: "the confidence threshold is now correctly
# applied", filtering out ~20.5M low-confidence predictions. We further
# filter to high-prob AEs to avoid noise.
MIN_PRED1 = 4.5  # corresponds to ~p > 0.99 after sigmoid


def build_se_display_map() -> dict[str, str]:
    """Returns {normalized_lower_name: umls_id} from our SE vocab.

    Uses aggressive normalization so MedDRA names can match.
    """
    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    names = v["display_names"]
    out: dict[str, str] = {}
    for u, n in names.items():
        variants = _all_variants(n)
        for v_norm in variants:
            if v_norm and v_norm not in out:
                out[v_norm] = u
        # also raw lower
        raw = n.lower().strip()
        if raw and raw not in out:
            out[raw] = u
    return out


def load_label_to_ingredient() -> dict[int, int]:
    """label_id → ingredient_id via product chain.

    product_to_rxnorm.csv: label_id, rxnorm_product_id
    vocab_rxnorm_ingredient_to_product.csv: product_id, ingredient_id

    Note column order: 'product_id, ingredient_id' (product is first).
    """
    # First: product_id → ingredient_id (may be multi)
    p2i: dict[int, list[int]] = defaultdict(list)
    with open(ONSIDES / "vocab_rxnorm_ingredient_to_product.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = int(row["product_id"])
                iid = int(row["ingredient_id"])
            except (ValueError, KeyError):
                continue
            p2i[pid].append(iid)

    # Then: label_id → product_id, expanded through p2i
    l2i: dict[int, int] = {}  # if multi, take first
    with open(ONSIDES / "product_to_rxnorm.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lid = int(row["label_id"])
                pid = int(row["rxnorm_product_id"])
            except (ValueError, KeyError):
                continue
            ings = p2i.get(pid, [])
            if ings:
                # If multi-ingredient combo product, skip — can't attribute
                if len(ings) == 1:
                    l2i[lid] = ings[0]
                # combos get dropped (we lose ~5-10% but cleaner attribution)
    return l2i


def load_ingredient_name_map() -> dict[int, str]:
    """ingredient_id → ingredient_name (lowercase)."""
    out: dict[int, str] = {}
    with open(ONSIDES / "vocab_rxnorm_ingredient.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iid = row["rxnorm_id"]
                if iid.startswith("OMOP"):
                    continue
                iid = int(iid)
            except (ValueError, KeyError):
                continue
            name = (row.get("rxnorm_name") or "").strip().lower()
            if name:
                out[iid] = name
    return out


def load_meddra_to_umls(se_display_map: dict[str, str]) -> dict[int, str]:
    """meddra_id → UMLS (via name match)."""
    out: dict[int, str] = {}
    n_matched = 0
    n_total = 0
    with open(ONSIDES / "vocab_meddra_adverse_effect.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            try:
                mid = int(row["meddra_id"])
            except (ValueError, KeyError):
                continue
            name = (row.get("meddra_name") or "").strip().strip('"')
            if not name:
                continue
            # Try direct
            if name.lower() in se_display_map:
                out[mid] = se_display_map[name.lower()]
                n_matched += 1
                continue
            # Try variants
            for v in _all_variants(name):
                if v in se_display_map:
                    out[mid] = se_display_map[v]
                    n_matched += 1
                    break
    print(f"[meddra→umls] matched {n_matched}/{n_total} MedDRA terms to vocab")
    return out


def stream_drug_ae_pairs(
    l2i: dict[int, int], i2n: dict[int, str], m2u: dict[int, str],
) -> dict[str, set[str]]:
    """Stream product_adverse_effect.csv and accumulate ingredient → {UMLS}.

    Filters: pred1 >= MIN_PRED1, AE has UMLS, label has ingredient mapping.
    """
    out: dict[str, set[str]] = defaultdict(set)
    n_rows = 0
    n_pass = 0
    fp = ONSIDES / "product_adverse_effect.csv"
    with open(fp) as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            if n_rows % 1_000_000 == 0:
                print(f"  [stream] {n_rows:,} rows ({n_pass:,} passed)",
                      flush=True)
            try:
                lid = int(row["product_label_id"])
                mid = int(row["effect_meddra_id"])
                pred1 = float(row.get("pred1") or 0.0)
            except (ValueError, KeyError):
                continue
            if pred1 < MIN_PRED1:
                continue
            iid = l2i.get(lid)
            if iid is None:
                continue
            umls = m2u.get(mid)
            if not umls:
                continue
            name = i2n.get(iid)
            if not name:
                continue
            out[name].add(umls)
            n_pass += 1
    print(f"[stream] total rows: {n_rows:,}; passed filter: {n_pass:,}")
    print(f"[stream] unique ingredients with ≥1 SE: {len(out):,}")
    return dict(out)


def decompose_to_alpha(
    catalog_drugs: list[dict], onsides_drug_ses: dict[str, set[str]],
    se_vocab: list[str], smoothing: float = 1.0,
) -> tuple[dict[str, dict[str, float]], dict[str, int], int]:
    """SIDER-style decomposition.

    Match each TRAINING-split catalog drug to OnSIDES by name. For each
    (T, S), compute α_onsides(S|T) on these matched drugs.
    """
    se_vocab_set = set(se_vocab)
    train_drugs = [d for d in catalog_drugs if d["split"] == "train"]
    n_train = len(train_drugs)
    print(f"[decompose] training drugs: {n_train}")

    # Match catalog drugs to OnSIDES
    matched = 0
    drug_onsides_se: dict[str, set[str]] = {}
    for d in train_drugs:
        dn = (d.get("drug_name") or "").lower().strip()
        if not dn:
            continue
        if dn in onsides_drug_ses:
            drug_onsides_se[d["drug_id"] if d.get("drug_id") else d["molregno"]] = (
                onsides_drug_ses[dn] & se_vocab_set
            )
            matched += 1
    print(f"[decompose] training drugs matched to OnSIDES: {matched}/{n_train}")

    # Build counts per target
    target_n_drugs: dict[str, int] = defaultdict(int)
    target_se_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for d in train_drugs:
        drug_key = d["drug_id"] if d.get("drug_id") else d["molregno"]
        if drug_key not in drug_onsides_se:
            # Drug not in OnSIDES — counts as 0-SE drug (still in denominator)
            ses_to_add: set[str] = set()
        else:
            ses_to_add = drug_onsides_se[drug_key]
        bind_uniprots = {t["uniprot"] for t in d["binding_profile"]}
        for u in bind_uniprots:
            target_n_drugs[u] += 1
            for s in ses_to_add:
                target_se_counts[u][s] += 1

    edges: dict[str, dict[str, float]] = {}
    for u, n_u in target_n_drugs.items():
        counts = target_se_counts.get(u, {})
        edges[u] = {
            s: (counts.get(s, 0) + smoothing) / (n_u + 2 * smoothing)
            for s in se_vocab
        }
    return edges, dict(target_n_drugs), matched


def main() -> int:
    print("=" * 78)
    print("Sprint 8A: OnSIDES v3.1.1 ingest → SCM α edges")
    print("=" * 78)

    se_display_map = build_se_display_map()
    print(f"[setup] SE display name map: {len(se_display_map):,} entries")

    print("[setup] Loading label → ingredient chain...")
    l2i = load_label_to_ingredient()
    print(f"[setup] label → ingredient: {len(l2i):,}")

    i2n = load_ingredient_name_map()
    print(f"[setup] ingredient → name: {len(i2n):,}")

    m2u = load_meddra_to_umls(se_display_map)

    print("[stream] Streaming product_adverse_effect.csv (this takes ~30s)...")
    drug_ses = stream_drug_ae_pairs(l2i, i2n, m2u)

    print("[decompose] Loading catalog...")
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    se_vocab = v["umls_ids"]

    edges, target_n, n_matched = decompose_to_alpha(
        cat["drugs"], drug_ses, se_vocab,
    )

    # Stats
    edge_alphas = [a for u in edges for a in edges[u].values()]
    edge_alphas.sort()
    n_pairs = len(edge_alphas)
    print(f"\n[stats] (target, SE) pairs: {n_pairs:,}")
    print(f"[stats] α distribution: min={edge_alphas[0]:.4f} "
          f"median={edge_alphas[n_pairs // 2]:.4f} "
          f"p95={edge_alphas[int(n_pairs * 0.95)]:.4f} "
          f"max={edge_alphas[-1]:.4f}")
    nonzero = sum(1 for a in edge_alphas if a > 0.05)
    print(f"[stats] non-near-zero α (>0.05): {nonzero:,} "
          f"({100 * nonzero / n_pairs:.1f}%)")

    out = {
        "n_targets": len(edges),
        "n_drugs_matched": n_matched,
        "min_pred1_threshold": MIN_PRED1,
        "edges": edges,
        "target_n_drugs": target_n,
    }
    out_path = RESULTS / "scm_edges_onsides.json"
    with open(out_path, "w") as f:
        # Compact format — these files are large
        json.dump(out, f)
    print(f"[save] {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
