"""Per-source evidence explainer for (target, side-effect) edges.

When the Hybrid SCM+LLM predicts a side effect, we want to show the
biopharma user the specific evidence supporting each (target → SE) edge:

  - SIDER frequency (n training drugs binding target with this SE label)
  - CTD curated mechanism (evidence_count + PubMed citation count)
  - OpenTargets FAERS (LLR statistical significance + report count)
  - PharmGKB clinical annotations (evidence level 1A through 4)
  - AOP-Wiki (formalized adverse outcome pathways)

The output is the "WHY" behind every prediction — convertible to a
regulatory-grade explanation for FDA submissions and pharma technical
DD conversations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


@dataclass(frozen=True)
class SourceEvidence:
    source: str               # "SIDER" | "CTD" | "OpenTargets" | "PharmGKB" | "AOP-Wiki"
    score: float              # source-specific evidence score in [0, 1]
    detail: str               # human-readable summary
    raw: dict                 # raw fields from the source


@dataclass(frozen=True)
class EdgeEvidence:
    uniprot: str
    gene_symbol: str
    side_effect_umls: str
    side_effect_name: str
    sources: list[SourceEvidence]    # one per supporting source
    strongest_source: str            # source with highest normalized score


class MultiSourceExplainer:
    """Caches the per-source edge files and serves attribution queries."""

    def __init__(self) -> None:
        # All source files are optional. The cloud deploy snapshot may
        # ship a curated subset; we degrade gracefully rather than
        # crash. explain_edge() will simply skip sources whose backing
        # files weren't bundled.
        self.sider_alpha = self._load(RESULTS / "scm_edges.json")
        self.ctd = self._load_nested(RESULTS / "scm_edges_ctd.json")
        self.aop = self._load_nested(RESULTS / "scm_edges_aopwiki.json")
        self.ot = self._load_nested(RESULTS / "scm_edges_opentargets.json")
        self.pgkb = self._load_nested(RESULTS / "scm_edges_pharmgkb.json")
        # Sprint 7C curated class-effect priors
        cp_path = RESULTS / "scm_edges_curated_priors.json"
        self.curated_priors: dict[str, dict[str, float]] = {}
        if cp_path.exists():
            with open(cp_path) as f:
                self.curated_priors = json.load(f).get("priors", {})
        with open(RESULTS / "target_vocab.json") as f:
            tv = json.load(f)
        self.target_info = {t["uniprot"]: t for t in tv["targets"]}
        with open(RESULTS / "side_effect_vocab.json") as f:
            v = json.load(f)
        self.se_display = v["display_names"]
        # SIDER training-drug counts for context
        priors_path = RESULTS / "scm_target_priors.json"
        if priors_path.exists():
            with open(priors_path) as f:
                self.priors = json.load(f)
        else:
            self.priors = {"target_n_drugs": {}}

    @staticmethod
    def _load(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def _load_nested(path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path) as f:
            d = json.load(f)
        return d.get("edges", {})

    def explain_edge(self, uniprot: str, se_umls: str) -> EdgeEvidence:
        """Return all supporting evidence for a (target, SE) edge."""
        target = self.target_info.get(uniprot, {})
        gene = target.get("gene_symbol", uniprot)
        se_name = self.se_display.get(se_umls, se_umls)
        sources: list[SourceEvidence] = []

        # SIDER frequency
        alpha_sider = self.sider_alpha.get(uniprot, {}).get(se_umls, 0.0)
        if alpha_sider > 0:
            n_train_binding = self.priors["target_n_drugs"].get(uniprot, 0)
            n_train_with_se = round(alpha_sider * (n_train_binding + 2)) - 1
            n_train_with_se = max(0, n_train_with_se)
            sources.append(SourceEvidence(
                source="SIDER",
                score=alpha_sider,
                detail=(f"{n_train_with_se}/{n_train_binding} training drugs "
                        f"binding {gene} also have {se_name} in SIDER label "
                        f"(empirical α={alpha_sider:.2f})"),
                raw={"alpha": alpha_sider, "n_train_binding": n_train_binding,
                      "n_train_with_se": n_train_with_se},
            ))

        # CTD curated
        ctd_rec = self.ctd.get(uniprot, {}).get(se_umls)
        if ctd_rec:
            ev = ctd_rec.get("evidence_count", 0)
            pmids = ctd_rec.get("pmid_count", 0)
            sources.append(SourceEvidence(
                source="CTD",
                score=min(1.0, ev / 5.0),
                detail=(f"CTD curated marker/mechanism: {ev} entries with "
                        f"{pmids} PubMed citations linking {gene} to "
                        f"{ctd_rec.get('ctd_disease_name', se_name)}"),
                raw=ctd_rec,
            ))

        # AOP-Wiki
        aop_rec = self.aop.get(uniprot, {}).get(se_umls)
        if aop_rec:
            n_aops = aop_rec.get("n_aops", 0)
            ex_aop = aop_rec.get("example_aop_titles", [])
            sources.append(SourceEvidence(
                source="AOP-Wiki",
                score=min(1.0, 0.6 + 0.15 * n_aops),
                detail=(f"AOP-Wiki: {n_aops} formalized adverse outcome "
                        f"pathway(s) link {gene} → {se_name}. "
                        f"Example: \"{ex_aop[0][:80]}\"" if ex_aop else ""),
                raw=aop_rec,
            ))

        # OpenTargets FAERS
        ot_rec = self.ot.get(uniprot, {}).get(se_umls)
        if ot_rec:
            llr = ot_rec.get("max_llr", 0)
            count = ot_rec.get("max_count", 0)
            sources.append(SourceEvidence(
                source="OpenTargets (FAERS)",
                score=min(1.0, max(0, (llr ** 0.5) / 50.0)),
                detail=(f"OpenTargets FAERS analysis: LLR={llr:.0f} "
                        f"(statistical significance vs baseline), {count} "
                        f"adverse-event reports linking {gene}-targeting drugs "
                        f"to {ot_rec.get('ot_event', se_name)}"),
                raw=ot_rec,
            ))

        # Curated class-effect prior (Sprint 7C) — surfaced as
        # "Clinical pharmacology / FDA black-box class effect"
        cur_strength = self.curated_priors.get(uniprot, {}).get(se_umls)
        if cur_strength is not None and cur_strength > 0:
            level = ("STRONG (FDA BBW / class label)" if cur_strength >= 0.8
                       else "MODERATE (documented class effect)")
            sources.append(SourceEvidence(
                source="Class-effect prior",
                score=float(cur_strength),
                detail=(f"Curated FDA black-box / class-effect prior: "
                        f"{gene} → {se_name} is a documented class-level "
                        f"liability. Strength: {level}."),
                raw={"strength": cur_strength},
            ))

        # PharmGKB
        pgkb_rec = self.pgkb.get(uniprot, {}).get(se_umls)
        if pgkb_rec:
            level = pgkb_rec.get("best_level", "")
            n_ann = pgkb_rec.get("n_annotations", 0)
            drugs = pgkb_rec.get("examples_drugs", [])
            sources.append(SourceEvidence(
                source="PharmGKB",
                score=({"1A": 1.0, "1B": 0.85, "2A": 0.7, "2B": 0.5,
                         "3": 0.35, "4": 0.2}.get(level, 0.1)),
                detail=(f"PharmGKB clinical annotation Level {level}: "
                        f"{n_ann} annotation(s) linking {gene} variants to "
                        f"{se_name}. Example drugs: "
                        f"{', '.join(drugs[:2])[:80]}"),
                raw=pgkb_rec,
            ))

        if sources:
            strongest = max(sources, key=lambda s: s.score).source
        else:
            strongest = "none"
        return EdgeEvidence(
            uniprot=uniprot, gene_symbol=gene,
            side_effect_umls=se_umls, side_effect_name=se_name,
            sources=sources, strongest_source=strongest,
        )

    def explain_prediction(
        self, se_umls: str, top_target_uniprots: list[str], top_n: int = 3,
    ) -> list[EdgeEvidence]:
        """For one predicted side effect, explain its top-N supporting target edges."""
        out = []
        for u in top_target_uniprots[:top_n]:
            ev = self.explain_edge(u, se_umls)
            if ev.sources:
                out.append(ev)
        return out
