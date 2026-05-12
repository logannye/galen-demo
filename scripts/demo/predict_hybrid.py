"""Hybrid Clinical-Safety prediction API for the demo (Sprint 6).

Wraps the Sprint 5 Hybrid SCM+LLM re-ranker with the Sprint 4 blended α
and the Sprint 6 multi-source evidence explainer. Produces the
biopharma-ready output:

  - 20 ranked clinical-safety predictions
  - Per-prediction mechanism rationale (from LLM)
  - Per-prediction top-3 contributing binding targets (from SCM)
  - Per (target, side-effect) multi-source evidence trace (CTD, OT,
    PharmGKB, AOP-Wiki, SIDER)
  - Whether each prediction was IN the SCM's top-100 (vs LLM-promoted
    from full vocab)
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .binding_inference import (
    InferredBinding, infer_binding_from_smiles, lookup_drug_by_name,
)
from .multisource_explainer import EdgeEvidence, MultiSourceExplainer
from .scm_explainer import SideEffectAttribution, explain_predictions
from ..baselines.llm_hybrid_reranker import hybrid_rerank, load_action_types
from ..data.biologic_binding_profiles import get_biologic_binding
from ..data.build_catalog import query_binding_profile
from ..llm import SonnetClient
from ..pipeline.run_sprint3_clinical_failures import lookup_chembl_molregno
from ..scm.scoring import score_drug_side_effects


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"


_DEMO_PROFILES_CACHE: dict | None = None


def _load_demo_profiles() -> dict:
    """Cached load of pre-computed Quick Demo binding profiles.

    Used in cloud deployments where the external pharmacology library
    is unavailable. Each entry: {drug_lower: {"binding_profile": [...]}}
    """
    global _DEMO_PROFILES_CACHE
    if _DEMO_PROFILES_CACHE is None:
        try:
            with open(RESULTS / "demo_binding_profiles.json") as f:
                _DEMO_PROFILES_CACHE = json.load(f)
        except FileNotFoundError:
            _DEMO_PROFILES_CACHE = {}
    return _DEMO_PROFILES_CACHE


@dataclass(frozen=True)
class HybridPrediction:
    rank: int
    side_effect_umls: str
    side_effect_name: str
    mechanism_rationale: str
    top_targets: list[SideEffectAttribution]    # per-target attribution from SCM
    edge_evidences: list[EdgeEvidence]           # multi-source evidence per top target
    scm_top100: bool                              # was this in SCM's top-100? (else LLM-promoted)
    scm_rank: int | None                          # rank in SCM (or None)


@dataclass(frozen=True)
class ClinicalSafetyResult:
    query: str
    query_type: str
    resolved_drug_name: str | None
    binding_profile: list[dict]
    is_inferred: bool
    nearest_drugs: list[dict]
    predictions: list[HybridPrediction]
    note: str
    n_targets_used: int
    confidence: str


class ClinicalSafetyEngine:
    """Two-mode engine: clinical-safety (Hybrid) + broad-ADR (SCM-Blended)."""

    def __init__(self):
        with open(RESULTS / "catalog.json") as f:
            self.catalog = json.load(f)
        self.drugs = self.catalog["drugs"]
        with open(RESULTS / "side_effect_vocab.json") as f:
            self.vocab_payload = json.load(f)
        self.se_vocab = self.vocab_payload["umls_ids"]
        self.se_names = self.vocab_payload["display_names"]
        with open(RESULTS / "target_vocab.json") as f:
            tv = json.load(f)
        self.target_info = {t["uniprot"]: t for t in tv["targets"]}
        # Load the most-recent production substrate, falling back to
        # older bundles if needed. Newer suffixes (j, i, h, ...) are
        # preferred so the engine picks up the latest deployed edges.
        substrate_candidates = [
            "scm_edges_blended_j.json",
            "scm_edges_blended_i.json",
            "scm_edges_blended_h.json",
            "scm_edges_blended_g.json",
            "scm_edges_blended_f.json",
            "scm_edges_blended_e.json",
            "scm_edges_blended_8b.json",
            "scm_edges_blended.json",
        ]
        for fname in substrate_candidates:
            path = RESULTS / fname
            if path.exists():
                with open(path) as f:
                    self.edges = json.load(f)
                break
        else:
            raise FileNotFoundError(
                "No SCM substrate file found in results/. "
                "Expected one of: " + ", ".join(substrate_candidates)
            )
        self.explainer = MultiSourceExplainer()
        self._llm_client: SonnetClient | None = None
        self._smiles_map: dict[str, str] | None = None

    @property
    def llm_client(self) -> SonnetClient:
        if self._llm_client is None:
            self._llm_client = SonnetClient()
        return self._llm_client

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

    def _resolve_binding_profile(
        self, query: str, query_type: str, k_neighbors: int = 5,
        min_tanimoto: float = 0.30,
    ) -> tuple[list[dict], str | None, bool, list[dict], str]:
        """Returns (binding_profile, resolved_drug_name, is_inferred, nearest_drugs, note)."""
        if query_type == "drug_name":
            # 1. Cloud-mode pre-computed profiles (for hosted deployments
            # where ChEMBL DB is unavailable). Loaded once and cached.
            import os
            if os.environ.get("DEMO_CLOUD_MODE", "").lower() in {"1", "true", "yes"}:
                profiles = _load_demo_profiles()
                key = query.lower().strip()
                if key in profiles:
                    bp = profiles[key]["binding_profile"]
                    return bp, query, False, [], (
                        f"Resolved binding profile. {len(bp)} target(s)."
                    )
                # In cloud mode, refuse free-text input — only the
                # curated Quick Demos are supported.
                return [], None, False, [], (
                    f"'{query}' is not one of the demo compounds. "
                    f"Please pick a Quick Demo from the list above, "
                    f"or contact us to evaluate your own compounds."
                )

            # 2. Local mode: SIDER catalog
            d = lookup_drug_by_name(query, self.drugs)
            if d is not None:
                return d["binding_profile"], d["drug_name"], False, [], (
                    f"Resolved from catalog. {len(d['binding_profile'])} binding targets."
                )
            # 3. ChEMBL lookup (for novel small molecules)
            try:
                molregno = lookup_chembl_molregno(query)
                if molregno:
                    conn = sqlite3.connect(CHEMBL_DB)
                    bp = query_binding_profile(conn, molregno)
                    conn.close()
                    if bp:
                        return bp, query, False, [], (
                            f"Resolved from external pharmacology library. "
                            f"{len(bp)} binding targets."
                        )
            except Exception:
                pass
            # 4. Curated biologic binding profiles
            try:
                biologic_bp = get_biologic_binding(query)
                if biologic_bp:
                    return biologic_bp, query, False, [], (
                        f"Resolved from curated biologic profile. "
                        f"{len(biologic_bp)} target(s)."
                    )
            except Exception:
                pass
            return [], None, False, [], (
                f"Compound '{query}' not found. Try one of the Quick Demos."
            )
        if query_type == "smiles":
            training = [d for d in self.drugs if d.get("split") == "train"]
            inferred = infer_binding_from_smiles(
                query, training, self.smiles_map,
                k_neighbors=k_neighbors, min_tanimoto=min_tanimoto,
            )
            note = (f"Binding profile inferred via Tanimoto NN from "
                    f"{len(inferred.nearest_drugs)} structural neighbors. "
                    f"{len(inferred.binding_profile)} inferred targets.")
            return (inferred.binding_profile, None, True,
                    inferred.nearest_drugs, note)
        if query_type == "binding_profile":
            # Caller passes binding_profile via separate path; this is a marker.
            return [], None, False, [], "Direct binding-profile input."
        return [], None, False, [], f"Unknown query type {query_type}"

    def predict_clinical_safety(
        self, query: str, query_type: str = "drug_name", *,
        binding_profile: list[dict] | None = None,
        k_neighbors: int = 5, min_tanimoto: float = 0.30,
        top_k_scm_candidates: int = 100,
        therapeutic_area: str = "",
    ) -> ClinicalSafetyResult:
        """Run the Hybrid SCM+LLM clinical-safety pipeline."""
        if binding_profile is None:
            bp, resolved_name, is_inferred, nearest, note = (
                self._resolve_binding_profile(query, query_type,
                                                k_neighbors, min_tanimoto)
            )
        else:
            bp = binding_profile
            resolved_name = None
            is_inferred = False
            nearest = []
            note = f"Direct binding profile: {len(bp)} targets."

        if not bp:
            return ClinicalSafetyResult(
                query=query, query_type=query_type,
                resolved_drug_name=resolved_name,
                binding_profile=[], is_inferred=is_inferred,
                nearest_drugs=nearest, predictions=[],
                note=note, n_targets_used=0, confidence="insufficient",
            )

        # 1. SCM scoring with blended α → top-100
        scm_scored = score_drug_side_effects(bp, self.edges, self.se_vocab)
        scm_top100_ids = {s for s, _ in scm_scored[:top_k_scm_candidates]}
        scm_rank_map = {s: i + 1 for i, (s, _) in enumerate(scm_scored)}

        # 2. SCM per-target attribution for top-100
        explanations = explain_predictions(
            scm_scored[:top_k_scm_candidates], bp, self.edges,
            self.target_info, self.se_names,
            top_k_se=top_k_scm_candidates, top_k_targets=5,
        )
        explanations_by_id = {e.side_effect_id: e for e in explanations}

        # 3. Hybrid LLM re-rank (with optional TA bias + action types)
        # Sprint 8A: load merged DGIdb + DrugCentral action types
        action_types = {}
        if resolved_name is not None:
            resolved_drug = drugs_by_name.get(resolved_name.lower()) if False else None
            for d in self.drugs:
                if (d.get("drug_name") or "").lower() == (resolved_name or "").lower():
                    try:
                        action_types = load_action_types(d.get("molregno"))
                    except Exception:
                        pass
                    break
        hybrid = hybrid_rerank(
            bp, scm_scored, explanations, self.vocab_payload,
            client=self.llm_client,
            top_k_scm_candidates=top_k_scm_candidates,
            therapeutic_area=therapeutic_area,
            action_types=action_types,
        )

        # 4. Assemble HybridPrediction list with multi-source evidence
        predictions: list[HybridPrediction] = []
        for i, se_id in enumerate(hybrid.ranked_side_effects, start=1):
            se_name = self.se_names.get(se_id, se_id)
            rationale = hybrid.rationales.get(se_id, "")
            attr_obj = explanations_by_id.get(se_id)
            if attr_obj is not None:
                top_targets_list = attr_obj.top_targets
            else:
                # LLM-promoted prediction (not in SCM's top-100); recompute
                # attribution on the fly from binding profile
                from .scm_explainer import attribute_side_effect
                top_targets_list = attribute_side_effect(
                    se_id, bp, self.edges, self.target_info, top_k_targets=5,
                )
            edge_evs: list[EdgeEvidence] = []
            for t in top_targets_list[:3]:
                ev = self.explainer.explain_edge(t.uniprot, se_id)
                if ev.sources:
                    edge_evs.append(ev)
            predictions.append(HybridPrediction(
                rank=i,
                side_effect_umls=se_id,
                side_effect_name=se_name,
                mechanism_rationale=rationale,
                top_targets=top_targets_list,
                edge_evidences=edge_evs,
                scm_top100=(se_id in scm_top100_ids),
                scm_rank=scm_rank_map.get(se_id),
            ))

        return ClinicalSafetyResult(
            query=query, query_type=query_type,
            resolved_drug_name=resolved_name,
            binding_profile=bp, is_inferred=is_inferred,
            nearest_drugs=nearest, predictions=predictions,
            note=note, n_targets_used=len(bp),
            confidence=hybrid.confidence,
        )
