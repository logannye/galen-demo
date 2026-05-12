"""Unified prediction API for the SCM Off-Target Safety demo.

Accepts:
  - drug name (looked up in the SIDER ∩ ChEMBL catalog)
  - SMILES (binding inferred via Tanimoto NN against ChEMBL)
  - explicit binding profile (list of {uniprot, gene_symbol, ...})

Returns:
  PredictionResult with ranked side effects, per-target attribution, the
  drug's binding profile (resolved or inferred), and similar-drug context.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .binding_inference import (
    InferredBinding, infer_binding_from_smiles, lookup_drug_by_name,
)
from .scm_explainer import SideEffectAttribution, explain_predictions
from ..scm.scoring import score_drug_side_effects


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"


@dataclass(frozen=True)
class PredictionResult:
    query: str                              # what the user asked for
    query_type: str                         # "drug_name" | "smiles" | "binding_profile"
    resolved_drug_name: str | None
    binding_profile: list[dict]
    is_inferred: bool                       # True if SMILES → NN
    nearest_drugs: list[dict]               # for SMILES queries
    explanations: list[SideEffectAttribution]
    note: str                                # any helpful warning
    n_targets_used: int


class PredictionEngine:
    def __init__(self):
        with open(RESULTS / "catalog.json") as f:
            self.catalog = json.load(f)
        self.drugs = self.catalog["drugs"]
        with open(RESULTS / "side_effect_vocab.json") as f:
            vocab_payload = json.load(f)
        self.se_vocab = vocab_payload["umls_ids"]
        self.se_names = vocab_payload["display_names"]
        with open(RESULTS / "target_vocab.json") as f:
            tv = json.load(f)
        self.target_info = {t["uniprot"]: t for t in tv["targets"]}
        with open(RESULTS / "scm_edges.json") as f:
            self.edges = json.load(f)
        # Lazy: load SMILES map on demand
        self._smiles_map: dict[str, str] | None = None

    @property
    def smiles_map(self) -> dict[str, str]:
        if self._smiles_map is None:
            conn = sqlite3.connect(CHEMBL_DB)
            cur = conn.cursor()
            molregnos = [int(d["molregno"]) for d in self.drugs]
            placeholders = ",".join("?" * len(molregnos))
            cur.execute(f"""
                SELECT molregno, canonical_smiles
                FROM compound_structures
                WHERE molregno IN ({placeholders})
            """, molregnos)
            self._smiles_map = {str(r[0]): r[1] for r in cur if r[1]}
            conn.close()
        return self._smiles_map

    def predict_from_drug_name(
        self, drug_name: str, *, top_k_se: int = 20, top_k_targets: int = 5,
    ) -> PredictionResult:
        d = lookup_drug_by_name(drug_name, self.drugs)
        if d is None:
            return PredictionResult(
                query=drug_name, query_type="drug_name",
                resolved_drug_name=None, binding_profile=[],
                is_inferred=False, nearest_drugs=[],
                explanations=[],
                note=f"Drug '{drug_name}' not in catalog. Try SMILES input or a known drug name.",
                n_targets_used=0,
            )
        bp = d["binding_profile"]
        scored = score_drug_side_effects(bp, self.edges, self.se_vocab)
        gold = set(d.get("side_effects_in_vocab", []))
        explanations = explain_predictions(
            scored, bp, self.edges, self.target_info, self.se_names,
            gold_set=gold, top_k_se=top_k_se, top_k_targets=top_k_targets,
        )
        return PredictionResult(
            query=drug_name, query_type="drug_name",
            resolved_drug_name=d["drug_name"],
            binding_profile=bp,
            is_inferred=False, nearest_drugs=[],
            explanations=explanations,
            note=f"Found in catalog (split={d.get('split')}). "
                 f"{len(bp)} binding targets, {len(gold)} known side effects.",
            n_targets_used=len(bp),
        )

    def predict_from_smiles(
        self, smiles: str, *,
        k_neighbors: int = 5, min_tanimoto: float = 0.30,
        top_k_se: int = 20, top_k_targets: int = 5,
    ) -> PredictionResult:
        # Restrict NN search to the TRAINING partition (no test-set leakage in demo)
        training = [d for d in self.drugs if d.get("split") == "train"]
        inferred = infer_binding_from_smiles(
            smiles, training, self.smiles_map,
            k_neighbors=k_neighbors, min_tanimoto=min_tanimoto,
        )
        if not inferred.binding_profile:
            return PredictionResult(
                query=smiles, query_type="smiles",
                resolved_drug_name=None,
                binding_profile=[], is_inferred=True,
                nearest_drugs=inferred.nearest_drugs,
                explanations=[],
                note=(f"No structural neighbors found with Tanimoto ≥ {min_tanimoto}. "
                      f"This compound has no similar known drugs in the catalog."),
                n_targets_used=0,
            )
        scored = score_drug_side_effects(inferred.binding_profile, self.edges, self.se_vocab)
        explanations = explain_predictions(
            scored, inferred.binding_profile, self.edges,
            self.target_info, self.se_names,
            gold_set=None, top_k_se=top_k_se, top_k_targets=top_k_targets,
        )
        return PredictionResult(
            query=smiles, query_type="smiles",
            resolved_drug_name=None,
            binding_profile=inferred.binding_profile,
            is_inferred=True,
            nearest_drugs=inferred.nearest_drugs,
            explanations=explanations,
            note=(f"Binding profile inferred via Tanimoto NN from "
                  f"{len(inferred.nearest_drugs)} structural neighbors. "
                  f"{len(inferred.binding_profile)} inferred targets."),
            n_targets_used=len(inferred.binding_profile),
        )

    def predict_from_binding_profile(
        self, binding_profile: list[dict], *,
        top_k_se: int = 20, top_k_targets: int = 5,
    ) -> PredictionResult:
        scored = score_drug_side_effects(binding_profile, self.edges, self.se_vocab)
        explanations = explain_predictions(
            scored, binding_profile, self.edges,
            self.target_info, self.se_names,
            gold_set=None, top_k_se=top_k_se, top_k_targets=top_k_targets,
        )
        return PredictionResult(
            query="<binding_profile>", query_type="binding_profile",
            resolved_drug_name=None,
            binding_profile=binding_profile,
            is_inferred=False, nearest_drugs=[],
            explanations=explanations,
            note=f"Direct binding-profile input. {len(binding_profile)} targets.",
            n_targets_used=len(binding_profile),
        )
