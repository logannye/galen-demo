"""SMILES → binding profile inference via Tanimoto nearest-neighbors in ChEMBL.

For a novel compound not in ChEMBL, we don't have measured bioactivity. We
estimate its binding profile by:
  1. Computing ECFP4 fingerprint from SMILES
  2. Finding the top-K most-similar drugs in the training catalog (Tanimoto)
  3. Aggregating their binding profiles, weighted by similarity
  4. Returning the inferred (target, similarity-weighted-affinity) profile

This is a chemistry-mediated bridge: a novel compound's predicted binding
comes from drugs with similar structure. The biopharma use case: a new
lead compound's structural neighbors in ChEMBL are the prior for its
likely off-targets.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


def _compute_ecfp(smiles):
    """Lazy import of compute_ecfp — pulls in rdkit transitively.

    The cloud demo never calls the SMILES inference path (free-text
    input is rejected), so rdkit doesn't need to be installed. Local
    development and batch eval scripts that DO need it will install
    rdkit separately.
    """
    from ..baselines.rf_ecfp_baseline import compute_ecfp
    return compute_ecfp(smiles)


@dataclass(frozen=True)
class InferredBinding:
    binding_profile: list[dict]
    nearest_drugs: list[dict]   # [{cid, name, tanimoto, ...}]
    method: str                 # "exact_lookup" or "smiles_nn"


def tanimoto(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Tanimoto similarity between two binary fingerprints."""
    if fp1 is None or fp2 is None:
        return 0.0
    a = float(fp1.sum())
    b = float(fp2.sum())
    c = float((fp1 * fp2).sum())
    if a + b - c == 0:
        return 0.0
    return c / (a + b - c)


def infer_binding_from_smiles(
    smiles: str,
    catalog_drugs: list[dict],
    smiles_map: dict[str, str],
    *,
    k_neighbors: int = 5,
    min_tanimoto: float = 0.30,
) -> InferredBinding:
    """Infer a binding profile from SMILES via Tanimoto NN in the catalog.

    Args:
      smiles: query molecule SMILES
      catalog_drugs: drugs with binding_profile (from results/catalog.json)
      smiles_map: {molregno: SMILES} from ChEMBL compound_structures
      k_neighbors: number of nearest neighbors to aggregate
      min_tanimoto: minimum similarity to include a neighbor

    Returns:
      InferredBinding with the aggregated profile + nearest-neighbor info.
    """
    query_fp = compute_ecfp(smiles)
    if query_fp is None:
        raise ValueError(f"Invalid SMILES: {smiles[:50]}")

    # Compute similarity to every catalog drug
    sims = []
    for d in catalog_drugs:
        smi = smiles_map.get(str(d.get("molregno")))
        if not smi:
            continue
        try:
            fp = compute_ecfp(smi)
        except Exception:
            continue
        t = tanimoto(query_fp, fp)
        if t >= min_tanimoto:
            sims.append((d, t))
    sims.sort(key=lambda x: x[1], reverse=True)
    top_k = sims[:k_neighbors]

    if not top_k:
        return InferredBinding(
            binding_profile=[], nearest_drugs=[], method="smiles_nn_no_match",
        )

    # Aggregate binding profiles weighted by Tanimoto
    target_weighted_aff: dict[str, dict] = {}
    for nbr, sim in top_k:
        for t in nbr.get("binding_profile", []):
            uniprot = t.get("uniprot")
            if not uniprot:
                continue
            key = uniprot
            if key not in target_weighted_aff:
                target_weighted_aff[key] = {
                    "uniprot": uniprot,
                    "gene_symbol": t.get("gene_symbol", uniprot),
                    "target_pref_name": t.get("target_pref_name", ""),
                    "sum_weight": 0.0,
                    "sum_weighted_log_nm": 0.0,
                    "support": 0,
                }
            # Weight target affinity by Tanimoto: more-similar neighbors carry more weight
            entry = target_weighted_aff[key]
            entry["sum_weight"] += sim
            log_nm = float(np.log10(max(t.get("standard_value_nm", 1000.0), 0.1)))
            entry["sum_weighted_log_nm"] += sim * log_nm
            entry["support"] += 1

    # Convert to binding_profile records
    profile: list[dict] = []
    for u, entry in target_weighted_aff.items():
        if entry["sum_weight"] <= 0:
            continue
        avg_log_nm = entry["sum_weighted_log_nm"] / entry["sum_weight"]
        avg_nm = 10 ** avg_log_nm
        profile.append({
            "uniprot": entry["uniprot"],
            "gene_symbol": entry["gene_symbol"],
            "target_pref_name": entry["target_pref_name"],
            "standard_type": "inferred",
            "standard_value_nm": avg_nm,
            "n_supporting_neighbors": entry["support"],
            "inference_weight": entry["sum_weight"],
        })
    # Rank by support × weight
    profile.sort(key=lambda r: (r["n_supporting_neighbors"], r["inference_weight"]),
                  reverse=True)

    nearest_drugs = [
        {
            "cid": d["cid"],
            "drug_name": d["drug_name"],
            "tanimoto": float(t),
            "n_targets": d.get("n_targets", 0),
            "n_side_effects": d.get("n_side_effects", 0),
        }
        for d, t in top_k
    ]
    return InferredBinding(
        binding_profile=profile,
        nearest_drugs=nearest_drugs,
        method="smiles_nn",
    )


def lookup_drug_by_name(name: str, catalog_drugs: list[dict]) -> dict | None:
    """Case-insensitive drug-name lookup in the catalog."""
    n = name.strip().lower()
    for d in catalog_drugs:
        if (d.get("drug_name") or "").lower() == n:
            return d
    return None
