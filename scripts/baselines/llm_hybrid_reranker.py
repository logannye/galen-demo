"""Sprint 5: LLM-augmented hybrid SCM re-ranker.

Given a drug's polypharmacology binding profile + the SCM's top-100
candidate side effects (with per-target attribution), the LLM (Sonnet,
drug-blind) re-ranks the candidates using mechanism reasoning.

Anti-contamination: no drug name, no SMILES, no synonyms. The LLM must
reason from binding profile + SCM attribution alone.

The LLM also has access to the FULL side-effect vocabulary — so it can
promote a side effect NOT in SCM's top-100 if mechanism strongly
indicates it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..llm import SonnetClient


@dataclass(frozen=True)
class HybridRanking:
    ranked_side_effects: list[str]  # UMLS IDs, top-20
    rationales: dict[str, str]      # UMLS ID → mechanism rationale
    confidence: str
    raw_response: str = ""


_TA_GUIDANCE = {
    "Oncology": """
THERAPEUTIC AREA: ONCOLOGY (preclinical oncology candidate).

For oncology compounds, give SPECIAL PRIORITY to these documented class-
effect safety liabilities when supported by the binding profile:
  - CARDIOTOXICITY: HER2/ERBB2 → cardiac dysfunction; TOP2 → anthracycline
    cardiomyopathy; VEGFR/KDR → hypertension + heart failure; BTK →
    atrial fibrillation; multiple kinases → QT prolongation
  - HEPATOTOXICITY: TKI class (ALK, BRAF, MEK, EGFR); CYP3A4 metabolism;
    mitochondrial-poison-class
  - HEMATOLOGIC: CDK4/6 → neutropenia; PARP → MDS/AML + anaemia; BCL2 →
    tumor lysis syndrome; cytotoxic mechanisms → pancytopenia
  - PULMONARY: EGFR/ALK/mTOR → interstitial lung disease/pneumonitis;
    BRAF/MEK → pneumonitis
  - SKIN: EGFR → rash/dermatitis; VEGFR → hand-foot syndrome (palmar-
    plantar erythrodysaesthesia)
  - GI: VEGFR/EGFR → diarrhea; TKIs → mucositis
  - IMMUNE-RELATED (irAEs) — for ICI mechanisms (PD-1/PDCD1, PD-L1/CD274,
    CTLA-4, LAG-3): pneumonitis, colitis, hepatitis, hypophysitis/adrenal
    insufficiency, thyroid disorder, myocarditis, demyelination,
    autoimmune disorder
  - NEUROTOXICITY: peripheral neuropathy from vinca/taxane/oxaliplatin
    mechanisms
  - SECONDARY MALIGNANCY: PARP inh → MDS/AML
""",
    "Immunology": """
THERAPEUTIC AREA: IMMUNOLOGY / INFLAMMATION (immunomodulator candidate).

For immunology compounds, give SPECIAL PRIORITY to these documented class-
effect safety liabilities when supported by the binding profile:
  - SERIOUS INFECTIONS: TNF inhibition → TB reactivation + opportunistic
    infections; CD20 depletion → PML + HBV reactivation; ITGA4 (α4β1) →
    PML (BBW); JAK inhibition → herpes zoster + opportunistic infections;
    IL-6R → upper respiratory infections; IL-17 → mucocutaneous candidiasis;
    C5 complement inhibition → meningococcal infection (BBW); S1P1 →
    lymphopenia + opportunistic infections
  - MALIGNANCY RISK: TNF inhibitors → lymphoma; B-cell depleters →
    immunosuppression-associated cancers
  - MACE / CARDIOVASCULAR: JAK inhibitors → MACE (MI + stroke + VTE,
    BBW for tofacitinib class)
  - DEMYELINATION: TNF inhibitors (paradoxical) → demyelination + MS
    flare; S1P modulators → reversible-on-discontinuation effects
  - INFUSION REACTIONS: any biologic → infusion-related reaction +
    anaphylaxis
  - LYMPHOPENIA: S1P modulators (fingolimod class); B-cell depleters
  - CARDIAC: S1P modulators → bradycardia (BBW for fingolimod first dose)
  - OPHTHALMOLOGIC: S1P modulators → macular edema (BBW)
  - PARADOXICAL AUTOIMMUNE: anti-TNF → lupus-like syndrome
  - HEPATOTOXICITY: methotrexate-class, azathioprine, leflunomide,
    JAK inhibitors
""",
    "Cardiovascular & metabolic": """
THERAPEUTIC AREA: CARDIOVASCULAR / METABOLIC.

For CV/metabolic compounds, prioritize:
  - QT prolongation (hERG/KCNH2 — most common off-target liability)
  - Hepatotoxicity (statins; thiazolidinediones)
  - Rhabdomyolysis (statin class via HMGCR + SLCO1B1)
  - Hypoglycemia (insulin secretagogues, GLP-1 agonists)
  - Lactic acidosis (metformin-class)
  - Edema (TZD, calcium channel blockers)
""",
    "CNS & psychiatry": """
THERAPEUTIC AREA: CNS / PSYCHIATRY.

For CNS compounds, prioritize:
  - QT prolongation (antipsychotic class via hERG)
  - Extrapyramidal disorder, akathisia, tardive dyskinesia (DRD2 antagonism)
  - Metabolic syndrome (atypical antipsychotics)
  - Suicidality (antidepressant BBW)
  - Seizure threshold (bupropion, tramadol classes)
  - Serotonin syndrome (SSRI/SNRI + MAOI/triptan DDI)
  - Hyperprolactinemia (DRD2 blockers)
""",
    "Other": "",
    "": "",
}


_HYBRID_PROMPT = """You are a clinical pharmacologist analyzing a preclinical drug compound. \
You are given:
  (a) The compound's polypharmacology binding profile — target gene symbols, \
UniProt IDs, and binding affinities (Ki/IC50/Kd in nM). The compound's \
IDENTITY IS WITHHELD.
  (b) A structural causal model's top-100 ranked side effects with per-target \
attribution showing which of the compound's binding targets contribute most \
to each prediction.
  (c) The full candidate side-effect vocabulary (you may promote a side effect \
NOT in the SCM's top-100 if mechanism strongly indicates it).
  (d) MECHANISM CASCADE HINTS — for the compound's binding targets that have \
established multi-step pathway cascades (e.g. PD-1 inhibition → T-cell activation \
→ multi-organ irAEs), explicit cascade context is provided below to guide \
mechanism reasoning.

{ta_guidance}

Your task: re-rank these candidates into a TOP-20 ranked list, using \
mechanism reasoning about the binding profile. Give priority to:
  - Causally-specific predictions where a known target-AE mechanism applies \
(e.g., hERG block → QT prolongation; COX-2 selectivity → CV events; \
PPARγ activation → MI/HF; 5-HT2B agonism → cardiac valvulopathy; \
mitochondrial / CYP3A4 → hepatotoxicity)
  - Multi-target convergence on a single side effect (multiple binding \
targets contribute → higher confidence)
  - Off-target liabilities that the binding profile clearly exposes

DE-PRIORITIZE (rank lower):
  - Common nuisance side effects that appear high for almost every drug \
(headache, dizziness, nausea, dermatitis) UNLESS the binding profile \
specifically supports them
  - Predictions where the SCM attribution doesn't clearly point to a \
mechanistic target

RANKING DISCIPLINE — distinct clinical concepts in top-10:
  - The TOP-10 must show DISTINCT clinical concepts. Do NOT use multiple \
slots for anatomic or severity variants of the SAME condition.
  - Example violations to AVOID:
      * 5 candidiasis sites (oropharyngeal / oesophageal / GI / genital / \
vulvovaginal) in top-5 — collapse to ONE candidiasis/fungal-infection slot \
and free the others for distinct AEs
      * 3 hepatic variants (hepatotoxicity / hepatic failure / ALT increased) \
in top-3 — collapse to ONE hepatic-injury slot
      * 2-3 cardiac failure variants — collapse to ONE
  - Show the FAMILY representative, then move on to a different mechanism.
  - This applies to the AE clusters below (use them as guidance — surface \
ONE per cluster within top-10).

RANK-1 mechanism-decisive rule (the SINGLE most important slot):
  - Rank #1 is for THE SINGLE most mechanism-decisive AE for this binding \
profile. Be DECISIVE — pick the one AE most directly attributable to the \
strongest-binding mechanistic target.
  - If multiple targets converge on one AE → that AE is rank #1.
  - If one target has a famous mechanism (KCNH2→QT, KDR→hypertension, \
HER2→cardiotox, ICI→pneumonitis, IL17A→candidiasis, JAK→VTE, S1PR1→bradycardia, \
PARP→cytopenia/MDS, EGFR→rash, BCL2→TLS) → that AE is rank #1.
  - Use ranks 2-3 for DISTINCT mechanism families, not for rank-1 variants.

CRITICAL CONSTRAINTS:
  - The compound's identity is WITHHELD. Reason from binding profile + \
mechanism only. Do not guess the drug.
  - Use UMLS IDs from the vocabulary only.
  - Return exactly 20 ranked side effects.
  - In top-10: distinct clinical concepts only (one per AE cluster family).
  - In top-3: must be 3 different mechanism categories (not 3 variants of \
the same mechanism).

Output format: a JSON object:
  - ranked_side_effects: list of 20 UMLS IDs in ranked order (most likely first)
  - rationales: dict mapping each UMLS ID → 1-2 sentence mechanism rationale
  - confidence: "high" | "medium" | "low"

Return ONLY the JSON object.

BINDING PROFILE (target | UniProt | affinity):
{binding_block}

MECHANISM CASCADE HINTS (for binding targets with curated multi-step pathways):
{cascade_block}

AE CLUSTERS (treat each cluster as ONE clinical concept — surface only \
ONE representative per cluster in your top-10):
{cluster_block}

SCM TOP-100 CANDIDATES (UMLS | display name | SCM score | top contributing target):
{scm_candidates_block}

FULL CANDIDATE VOCABULARY (UMLS | display name):
{vocab_block}

JSON output:"""


def _format_reactome_cascade_block(binding_profile: list[dict],
                                     max_lines: int = 10) -> str:
    """Sprint K.3: surface curated pathway cascades to LLM prompt.

    For each binding target with a known pathway-mediated AE cascade
    in our curated PATHWAY_TO_AE map (via ingest_reactome.py), produce
    a multi-line cascade hint. Helps biologic / multi-step mechanism
    cases where the LLM should reason about cascades, not just direct
    target-AE binding.
    """
    # Curated cascade templates by gene_symbol (subset of high-value)
    CASCADES = {
        "PDCD1": "PD-1 inhibition → T-cell activation → multi-organ irAE cascade (pneumonitis / colitis / hepatitis / endocrinopathies / myocarditis)",
        "CD274": "PD-L1 inhibition → same irAE cascade as PD-1 (slight skew toward pneumonitis)",
        "CTLA4": "CTLA-4 inhibition → early T-cell co-stimulation block → strong colitis + hepatitis + hypophysitis",
        "LAG3": "LAG-3 inhibition → T-cell exhaustion reversal → irAE spectrum (typically milder than PD-1)",
        "TNF": "TNF blockade → impaired granuloma maintenance → TB reactivation; impaired neoantigen surveillance → lymphoma; demyelination paradox",
        "IL6R": "IL-6R blockade → impaired acute-phase response → atypical infections + GI perforation risk + hepatic effects",
        "MS4A1": "CD20 depletion → prolonged B-cell aplasia → hypogammaglobulinemia → PML / HBV reactivation / infection",
        "CD19": "CD19 CAR-T → cytokine storm cascade → CRS / ICANS / HLH / prolonged cytopenias + B-cell aplasia",
        "TNFRSF17": "BCMA targeting → similar CRS/ICANS cascade to CD19 CAR-T plus T-cell-engager mechanism",
        "S1PR1": "S1P modulation → lymphocyte sequestration → first-dose bradycardia + lymphopenia + macular oedema + PML",
        "ITGA4": "α4 integrin blockade → impaired CNS leukocyte trafficking → PML BBW + opportunistic infection",
        "C5": "C5 inhibition → MAC complex disruption → meningococcal infection BBW (encapsulated organisms)",
        "C3": "C3 inhibition → broader complement deficiency → meningococcal + Pneumococcal + Haemophilus infections",
        "JAK1": "JAK inhibition → IFN/IL-6/STAT-blockade → opportunistic infection + lymphoma + VTE + MACE",
        "BTK": "BTK inhibition → B-cell signalling block + platelet GPVI off-target → AFib + bleeding + infection",
        "KCNH2": "hERG/IKr block → APD prolongation → QT prolongation → Torsade de pointes",
        "PTGS2": "COX-2 selective inhibition → prostacyclin imbalance → arterial thrombotic events (MI / stroke)",
        "ERBB2": "HER2 blockade → cardiomyocyte stress (esp. with anthracyclines) → LV dysfunction / cardiomyopathy",
        "TOP2A": "TOP2A/B trapping → mitochondrial DNA breaks in cardiomyocytes → anthracycline cardiotoxicity",
        "PARP1": "PARP inhibition → impaired DNA repair + replication stress in hematopoietic cells → MDS / AML / cytopenias",
        "BCL2": "BCL2 inhibition → rapid apoptosis of CLL cells → tumor lysis syndrome at first dose",
        "KDR": "VEGFR2 blockade → endothelial dysfunction → hypertension + proteinuria + hand-foot + bleeding",
        "TACSTD2": "TROP2 ADC + SN-38 / DXd payload → systemic chemo-mimetic AEs (diarrhea / neutropenia / ILD)",
        "NECTIN4": "Nectin-4 ADC + MMAE payload → epithelial toxicity + hyperglycemia + SCAR (SJS/TEN) BBW",
        "TNFRSF8": "CD30 ADC + MMAE payload → peripheral neuropathy + PML BBW",
        "CD33": "CD33 ADC + calicheamicin payload → DNA damage in hepatic sinusoidal endothelial cells → VOD/SOS BBW",
        "GLP1R": "GLP-1R activation → delayed gastric emptying + pancreatic stimulation → pancreatitis + thyroid C-cell signal",
        "MTOR": "mTOR inhibition → hyperglycemia + immunosuppression-mediated pneumonitis + opportunistic infection",
        "IDH1": "IDH1m blockade → reverses 2-HG accumulation → differentiation syndrome (cytokine release-like)",
        "IDH2": "IDH2m blockade → differentiation syndrome (similar to IDH1m)",
        "ALK": "ALK kinase inhibition → cardiac off-target (QT / bradycardia) + hepatic + ILD",
        "MAP2K1": "MEK inhibition → RAS pathway block in cardiomyocytes → LV dysfunction + cardiomyopathy",
        "BRAF": "BRAF inhibition → paradoxical RAS activation in normal tissues → pyrexia + rash + cutaneous SCC",
        "EGFR": "EGFR inhibition → impaired epidermal differentiation → acneiform rash + diarrhea + ILD",
        "TYK2": "TYK2 inhibition → impaired IL-23 / IFN-α signalling → infection + HZV reactivation",
        "AKT1": "AKT inhibition → impaired insulin signalling + glucose homeostasis → severe hyperglycemia",
        "ROS1": "ROS1 inhibition → off-target hepatic + CNS effects + ILD",
        "SOST": "Sclerostin inhibition → osteoblast activation but CV signal → MI + stroke (BBW)",
        "F2": "Thrombin inhibition → systemic anticoagulation → major bleeding BBW",
        "F10": "Factor Xa inhibition → systemic anticoagulation → major bleeding BBW",
        "TFPI": "TFPI inhibition (rebalance therapy) → procoagulant tilt → paradox thrombosis (BBW)",
        "F3": "Tissue factor ADC + MMAE → bleeding + ocular toxicity (TF on ocular tissue)",
        "SLC5A2": "SGLT2 inhibition → glucosuria → euglycemic DKA + GU mycotic infections + AKI",
        "PIK3CG": "PI3K-δ inhibition → impaired regulatory T-cell function → severe colitis + hepatitis + opportunistic infection",
        "PIK3CA": "PI3K-α inhibition → hyperglycemia (insulin signalling block) + rash + diarrhea",
        "CDK4": "CDK4/6 inhibition → cell cycle arrest in bone marrow → neutropenia + VTE",
        "SLCO1B1": "OATP1B1 substrate displacement → statin accumulation → rhabdomyolysis (BBW for cerivastatin class)",
        "HMGCR": "HMG-CoA reductase inhibition → muscle mitochondrial dysfunction → myopathy / rhabdomyolysis",
        "SCN5A": "Cardiac Na+ channel block → APD shortening + slowed conduction → proarrhythmia (CAST-type)",
        "ESR1": "Estrogen receptor blockade (SERM) or agonism (estrogen) → VTE / endometrial hyperplasia (agonist) / hot flashes",
        "GHR": "Growth hormone receptor activation → insulin resistance → hyperglycemia + soft-tissue swelling",
        "EDNRA": "Endothelin-A blockade → fluid retention + hepatic injury (BBW) + teratogenicity",
        "PPARA": "PPAR-α activation → hepatic transaminitis + weight gain + GI",
        "PPARG": "PPAR-γ activation → fluid retention + heart failure (BBW) + bone fractures + bladder cancer signal",
        "DRD2": "D2 receptor blockade → EPS + tardive dyskinesia + hyperprolactinemia; partial agonists differ",
        "SLC6A4": "Serotonin transporter inhibition (SSRI) → serotonin syndrome with MAOIs/triptans + GI bleed + pediatric suicidality BBW",
        "ACE": "ACE inhibition → reduced AngII + bradykinin accumulation → cough + angioedema + hyperK + AKI",
        "AGTR1": "AT1 receptor blockade → reduced AngII signalling → AKI + angioedema + hyperK",
        "OPRM1": "MOR agonism → respiratory depression + constipation + dependence; antagonism → withdrawal",
        "HRH3": "Histamine H3 inverse agonism → wakefulness + headache + QT effects",
        "MYH7": "Cardiac myosin inhibition (HCM Rx) → reduced contractility → LV dysfunction / cardiac failure",
        "ATP4A": "Gastric K+/H+ ATPase block → hypoacidity → C diff + B12 malabsorption (chronic)",
        "NR3C1": "Glucocorticoid receptor activation → metabolic syndrome + immunosuppression + mood + osteoporosis",
        "INSR": "Insulin receptor agonism → hypoglycemia (overdose risk)",
        "ADRB2": "β-blockade → bradycardia + bronchospasm; β-agonism → tachycardia + tremor + hypokalemia",
        "GABRA1": "GABA-A potentiation → sedation + dependence + withdrawal seizures",
        "DHFR": "DHFR inhibition → impaired folate metabolism → hepatic + mucositis + cytopenia + pneumonitis (BBW)",
        "ALDH1A1": "ALDH inhibition → acetaldehyde accumulation → flush + hepatic + cardiac effects",
        "ATP6V1A": "DHODH inhibition (teriflunomide) → impaired pyrimidine synthesis → hepatic failure BBW + cytopenia",
    }

    if not binding_profile:
        return ""
    lines = []
    for t in binding_profile[:max_lines]:
        gene = (t.get("gene_symbol") or "").upper()
        if gene in CASCADES:
            lines.append(f"  • {gene}: {CASCADES[gene]}")
    if not lines:
        return ""
    return "\n".join(lines)


def _format_binding_block(
    binding_profile: list[dict], max_targets: int = 30,
    action_types: dict[str, str] | None = None,
) -> str:
    """Format binding profile for LLM prompt.

    action_types: optional dict {uniprot: "inhibit"/"activate"/"modulator"/"binder"}
    from DGIdb 5.0 + DrugCentral merge (Sprint 8A). When present, the
    action class is surfaced to the LLM as mechanism context.
    """
    lines = []
    for t in binding_profile[:max_targets]:
        gene = t.get("gene_symbol") or t.get("uniprot", "?")
        uniprot = t.get("uniprot", "?")
        stype = t.get("standard_type", "?")
        sval = t.get("standard_value_nm", 0)
        target_name = t.get("target_pref_name", "")
        action_str = ""
        if action_types and uniprot in action_types:
            act = action_types[uniprot]
            if act and act != "unknown":
                action_str = f" [action: {act}]"
        lines.append(
            f"  {gene} | {uniprot} | {stype}={sval:.1f}nM{action_str} [{target_name}]"
        )
    return "\n".join(lines)


def _format_scm_candidates_block(
    scm_scored: list[tuple[str, float]],
    explanations: list,  # list of SideEffectAttribution
    display_names: dict[str, str],
    top_k: int = 100,
) -> str:
    """Format SCM top-K with attribution."""
    lines = []
    explanations_by_id = {e.side_effect_id: e for e in explanations}
    for i, (se_id, score) in enumerate(scm_scored[:top_k], start=1):
        display = display_names.get(se_id, se_id)
        attr = explanations_by_id.get(se_id)
        if attr and attr.top_targets:
            top_t = attr.top_targets[0]
            attr_str = f"{top_t.gene_symbol}({top_t.contribution_pct:.0%})"
        else:
            attr_str = "—"
        lines.append(f"  {i:>3d}. {se_id} | {display:<32s} | score={score:.3f} | top={attr_str}")
    return "\n".join(lines)


def _format_vocab_block(vocab_payload: dict, max_entries: int = 500) -> str:
    umls_ids = vocab_payload["umls_ids"][:max_entries]
    names = vocab_payload["display_names"]
    return "\n".join(f"  {u} | {names.get(u, u)}" for u in umls_ids)


def _format_cluster_block() -> str:
    """Phase 4.1: surface AE clusters to LLM prompt for distinct-concept ranking.

    The clusters are the same ones used by Phase 3 cluster collapse + crediting.
    Reading them inline in the prompt makes the LLM aware which AEs are
    clinically equivalent so it surfaces ONE per cluster in top-10.
    """
    try:
        from .ae_cluster_postprocess import load_clusters
        _, meta = load_clusters()
    except Exception:
        return "  (cluster file unavailable)"
    lines = []
    for cid, m in meta.items():
        rep = m["representative_name"]
        n_mem = m["n_members"]
        # Short label
        short = cid.replace("_", " ").title()
        lines.append(f"  • {short} ({n_mem} variants — surface ONE in top-10): {rep}")
    return "\n".join(lines)


def _parse(raw: str, vocab_set: set[str]) -> HybridRanking:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return HybridRanking(
                ranked_side_effects=[], rationales={},
                confidence="insufficient",
                raw_response=raw,
            )
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return HybridRanking(
                ranked_side_effects=[], rationales={},
                confidence="insufficient",
                raw_response=raw,
            )

    raw_ranked = obj.get("ranked_side_effects", []) or []
    seen: set[str] = set()
    filtered: list[str] = []
    for s in raw_ranked:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if s in vocab_set and s not in seen:
            filtered.append(s)
            seen.add(s)
    rationales = obj.get("rationales", {}) or {}
    if not isinstance(rationales, dict):
        rationales = {}
    return HybridRanking(
        ranked_side_effects=filtered,
        rationales={str(k): str(v) for k, v in rationales.items()},
        confidence=obj.get("confidence", "low"),
        raw_response=raw,
    )


def hybrid_rerank(
    binding_profile: list[dict],
    scm_scored: list[tuple[str, float]],
    explanations: list,
    vocab_payload: dict,
    client: SonnetClient | None = None,
    *,
    top_k_scm_candidates: int = 100,
    max_tokens: int = 3000,
    therapeutic_area: str = "",
    action_types: dict[str, str] | None = None,
) -> HybridRanking:
    """LLM-augmented hybrid re-ranking.

    Args:
      binding_profile: drug binding profile
      scm_scored: list of (umls_id, score) from SCM, ranked desc
      explanations: list of SideEffectAttribution (Sprint 3 explainer)
      vocab_payload: side_effect_vocab.json contents
      therapeutic_area: optional TA bias — one of "Oncology",
        "Immunology", "Cardiovascular & metabolic", "CNS & psychiatry",
        "Other", or empty (no TA bias)
      action_types: optional dict {uniprot: action_class} from DGIdb +
        DrugCentral (Sprint 8A). Surfaces per-target action context
        (inhibit/activate/modulator/binder) to the LLM.
    """
    if client is None:
        client = SonnetClient()
    vocab_set = set(vocab_payload["umls_ids"])
    ta_guidance = _TA_GUIDANCE.get(therapeutic_area, "")
    cascade_block = _format_reactome_cascade_block(binding_profile)
    if not cascade_block:
        cascade_block = "  (no curated multi-step cascades for this binding profile)"
    prompt = _HYBRID_PROMPT.format(
        ta_guidance=ta_guidance,
        binding_block=_format_binding_block(binding_profile, action_types=action_types),
        cascade_block=cascade_block,
        cluster_block=_format_cluster_block(),
        scm_candidates_block=_format_scm_candidates_block(
            scm_scored, explanations, vocab_payload["display_names"],
            top_k=top_k_scm_candidates,
        ),
        vocab_block=_format_vocab_block(vocab_payload),
    )
    resp = client.complete(prompt, max_tokens=max_tokens, temperature=0.0)
    if not resp.succeeded:
        import sys as _sys
        print(
            f"[hybrid_rerank] LLM failed: {getattr(resp, 'error', 'unknown')}",
            file=_sys.stderr, flush=True,
        )
        return HybridRanking(
            ranked_side_effects=[], rationales={},
            confidence="insufficient",
            raw_response=resp.raw_text,
        )
    return _parse(resp.raw_text, vocab_set)


def load_action_types(molregno: str | int) -> dict[str, str]:
    """Load merged DGIdb + DrugCentral action types for a drug.

    Returns {uniprot: action_class} where action_class is one of
    "inhibit", "activate", "modulator", "binder", or "unknown".
    Merge policy: DGIdb wins on conflict (more sources typically),
    DrugCentral fills in gaps DGIdb doesn't cover.
    """
    import json as _json
    from pathlib import Path as _Path
    workspace = _Path(__file__).resolve().parent.parent.parent
    results = workspace / "results"
    merged: dict[str, str] = {}
    molregno = str(molregno)

    dc_path = results / "drugcentral_action_types.json"
    if dc_path.exists():
        with open(dc_path) as f:
            dc = _json.load(f)
        for u, info in (dc.get("drug_target_actions", {}).get(molregno, {}) or {}).items():
            ac = info.get("action_class", "unknown")
            if ac and ac != "unknown":
                merged[u] = ac

    dgidb_path = results / "dgidb_action_types.json"
    if dgidb_path.exists():
        with open(dgidb_path) as f:
            dg = _json.load(f)
        for u, info in (dg.get("drug_target_actions", {}).get(molregno, {}) or {}).items():
            ac = info.get("action_class", "unknown")
            if ac and ac != "unknown":
                merged[u] = ac  # DGIdb wins on conflict

    return merged
