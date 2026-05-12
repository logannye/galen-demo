"""Batch SMILES -> full safety prediction pipeline.

Public API:
  predict_batch(compounds, lead_id=None, max_workers=8) -> BatchResult

Input is a list of {"id": str, "smiles": str, "name": str (optional)}.
Output is a comparison-ready result object that the analog-comparison UI
consumes.

Day 3 of Week 2.
"""
from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from scripts.targetnet.predict import (
    predict_target_matrix, _rows_to_profiles, _load_target_info,
)

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"

TARGETNET_TOP_K = 20
TARGETNET_PROB_THRESHOLD = 0.5


@dataclass
class CompoundEntry:
    """One row in a batch run."""
    compound_id: str
    name: str
    smiles: str
    binding_profile: list[dict] = field(default_factory=list)
    predictions: list = field(default_factory=list)  # HybridPrediction list
    n_critical: int = 0
    n_serious: int = 0
    n_common: int = 0
    organ_systems: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass
class BatchResult:
    """All compounds + comparison metadata."""
    compounds: list[CompoundEntry]
    lead_id: str | None = None
    elapsed_seconds: float = 0.0

    def by_id(self, cid: str) -> CompoundEntry | None:
        return next((c for c in self.compounds if c.compound_id == cid), None)


def load_smiles_csv(path: Path) -> list[dict]:
    """Parse CSV; required columns: smiles. Optional: id, name."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            smi = row.get("smiles") or row.get("SMILES")
            if not smi:
                continue
            rows.append({
                "id": (row.get("id") or row.get("ID")
                       or f"compound_{i+1:03d}"),
                "name": row.get("name") or row.get("NAME") or "",
                "smiles": smi.strip(),
            })
    return rows


def load_smiles_sdf(path: Path) -> list[dict]:
    """Parse SDF; uses RDKit to derive canonical SMILES per record."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    supplier = Chem.SDMolSupplier(str(path))
    out = []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        if not smi:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        cid = (name or f"compound_{i+1:03d}").strip()
        out.append({"id": cid, "name": name, "smiles": smi})
    return out


def _compound_decision_support(predictions: list, top_k: int = 10) -> dict:
    """Roll up severity + organ-system counts for a compound."""
    from scripts.baselines.clinical_taxonomy import severity_tier, organ_system

    crit = ser = com = 0
    organ_counts: dict[str, int] = {}
    for p in predictions[:top_k]:
        tier = severity_tier(p.side_effect_umls)
        if tier == "critical":
            crit += 1
        elif tier == "serious":
            ser += 1
        else:
            com += 1
        org = organ_system(p.side_effect_umls)
        organ_counts[org] = organ_counts.get(org, 0) + 1
    return {
        "n_critical": crit, "n_serious": ser, "n_common": com,
        "organ_systems": organ_counts,
    }


def _predict_one(entry, binding_profile, engine, therapeutic_area=""):
    """Run engine.predict_clinical_safety for one compound."""
    if not binding_profile:
        return entry.compound_id, [], "no binding profile inferred"
    try:
        result = engine.predict_clinical_safety(
            entry.smiles, query_type="binding_profile",
            binding_profile=binding_profile,
            therapeutic_area=therapeutic_area,
        )
        return entry.compound_id, result.predictions, None
    except Exception as e:
        return entry.compound_id, [], str(e)


def predict_batch(
    compounds: list[dict],
    *,
    lead_id: str | None = None,
    max_workers: int = 8,
    top_k_targetnet: int = TARGETNET_TOP_K,
    prob_threshold: float = TARGETNET_PROB_THRESHOLD,
    therapeutic_area: str = "",
    engine=None,
    progress_cb=None,
) -> BatchResult:
    """Score a batch of compounds end-to-end.

    Args:
      compounds: list of dicts with keys: id, name (optional), smiles
      lead_id: optional id of the lead compound (used by comparison view)
      max_workers: thread pool size for LLM Hybrid calls
      top_k_targetnet, prob_threshold: TargetNet filtering
      therapeutic_area: optional TA bias for Hybrid LLM
      engine: optional ClinicalSafetyEngine instance (will instantiate if None)
      progress_cb: optional callback fn(i, n, compound_id) called after each finish

    Returns: BatchResult
    """
    from scripts.demo.predict_hybrid import ClinicalSafetyEngine

    t0 = time.monotonic()
    entries = [
        CompoundEntry(
            compound_id=c["id"],
            name=c.get("name", ""),
            smiles=c["smiles"],
        )
        for c in compounds
    ]

    # Phase 1: batched TargetNet for all compounds at once
    target_info = _load_target_info()
    smiles_list = [e.smiles for e in entries]
    uniprots, prob_matrix, valid_mask = predict_target_matrix(smiles_list)
    profiles = _rows_to_profiles(
        uniprots, prob_matrix, top_k_targetnet, prob_threshold, target_info,
    )
    for entry, profile, valid in zip(entries, profiles, valid_mask):
        if not valid:
            entry.error = "invalid SMILES"
            continue
        entry.binding_profile = profile

    # Phase 2: Hybrid LLM scoring per compound (parallel via threads)
    if engine is None:
        engine = ClinicalSafetyEngine()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_predict_one, e, e.binding_profile, engine,
                       therapeutic_area): e
            for e in entries if e.binding_profile and not e.error
        }
        done = 0
        for fut in as_completed(futures):
            entry = futures[fut]
            try:
                cid, predictions, err = fut.result()
            except Exception as ex:
                entry.error = str(ex)
                continue
            if err:
                entry.error = err
                continue
            entry.predictions = predictions
            ds = _compound_decision_support(predictions, top_k=10)
            entry.n_critical = ds["n_critical"]
            entry.n_serious = ds["n_serious"]
            entry.n_common = ds["n_common"]
            entry.organ_systems = ds["organ_systems"]
            done += 1
            if progress_cb:
                progress_cb(done, len(futures), entry.compound_id)

    return BatchResult(
        compounds=entries,
        lead_id=lead_id,
        elapsed_seconds=round(time.monotonic() - t0, 2),
    )


def build_comparison_table(result: BatchResult, top_k_per_compound: int = 5):
    """Build a wide table for the comparison view.

    Rows: compounds (lead first if set).
    Columns: union of top-K predicted critical AEs across the batch +
             severity-tier rollup.

    Returns a tuple (columns, rows) suitable for tabular display.
    """
    from scripts.baselines.clinical_taxonomy import severity_tier

    # Aggregate top critical AEs across compounds
    seen_crit_aes: dict[str, str] = {}  # umls -> display name
    for c in result.compounds:
        for p in c.predictions[:top_k_per_compound]:
            if severity_tier(p.side_effect_umls) == "critical":
                seen_crit_aes[p.side_effect_umls] = p.side_effect_name
    column_aes = sorted(seen_crit_aes, key=lambda u: seen_crit_aes[u])

    columns = ["id", "name", "n_critical", "n_serious", "n_common"] + [
        seen_crit_aes[u] for u in column_aes
    ]

    # Determine lead
    lead = result.by_id(result.lead_id) if result.lead_id else None

    rows = []
    ordered = (
        [lead] + [c for c in result.compounds if c is not lead] if lead
        else result.compounds
    )
    for c in ordered:
        per_ae_rank: dict[str, int | None] = {}
        for u in column_aes:
            # Find this AE's rank in this compound's hybrid predictions
            rank = None
            for p in c.predictions:
                if p.side_effect_umls == u:
                    rank = p.rank
                    break
            per_ae_rank[seen_crit_aes[u]] = rank
        rows.append({
            "id": c.compound_id,
            "name": c.name,
            "n_critical": c.n_critical,
            "n_serious": c.n_serious,
            "n_common": c.n_common,
            # rank lower = predicted more strongly; None = not in top-20
            "ae_ranks": per_ae_rank,
            "is_lead": c is lead,
            "error": c.error,
        })

    # Compute deltas vs lead if a lead exists
    if lead is not None:
        lead_ranks = next((r["ae_ranks"] for r in rows if r["is_lead"]), {})
        for r in rows:
            if r["is_lead"]:
                r["delta_vs_lead"] = None
                continue
            delta = {}
            for ae_name, my_rank in r["ae_ranks"].items():
                lead_rank = lead_ranks.get(ae_name)
                if my_rank is None and lead_rank is None:
                    delta[ae_name] = None
                elif my_rank is None:
                    # lead had it, we don't → better
                    delta[ae_name] = "absent"
                elif lead_rank is None:
                    # we have it, lead doesn't → worse
                    delta[ae_name] = "new"
                else:
                    delta[ae_name] = lead_rank - my_rank  # +N means we're worse
            r["delta_vs_lead"] = delta

    return {"columns": columns, "rows": rows,
            "critical_ae_columns": [seen_crit_aes[u] for u in column_aes]}
