"""Clinical-failure dataset for Sprint 3 evaluation.

15 FDA-withdrawn or restricted drugs with publicly documented causal
off-target + causal side effect. Each entry maps the failure mechanism to:
  - causal_off_target_uniprot: the UniProt accession of the off-target
    that's now known to be responsible for the toxicity
  - causal_side_effects_umls: list of UMLS codes (in our top-500 vocab)
    representing the failure-causing adverse effect

The Sprint 3 evaluation question: did the SCM (and baselines) rank ANY
of these causal side effects in the top-K predictions, AND did the SCM's
top-K contributing targets include the causal off-target?

References: FDA action history, peer-reviewed pharmacology literature.
All UMLS codes confirmed in `results/side_effect_vocab.json` (Sprint 1).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalFailure:
    drug_id: str                          # canonical lowercase name
    drug_search_name: str                 # for ChEMBL lookup
    approval_year: int | None
    withdrawal_year: int | None
    withdrawal_reason: str                # short narrative
    causal_off_target_uniprot: str        # UniProt accession
    causal_off_target_gene: str           # gene symbol
    causal_side_effects_umls: tuple[str, ...]  # in our top-500 vocab
    causal_side_effects_display: str      # human-readable summary
    in_sider_catalog: bool                # already in results/catalog.json?
    notes: str = ""


CLINICAL_FAILURES: tuple[ClinicalFailure, ...] = (
    ClinicalFailure(
        drug_id="terfenadine", drug_search_name="terfenadine",
        approval_year=1985, withdrawal_year=1998,
        withdrawal_reason="hERG block → QT prolongation → fatal torsades (parent compound)",
        causal_off_target_uniprot="Q12809", causal_off_target_gene="KCNH2",
        causal_side_effects_umls=("C0040479", "C0151878", "C0003811"),
        causal_side_effects_display="Torsade de pointes / QT prolonged / Arrhythmia",
        in_sider_catalog=False,
        notes="Classic hERG-block CV liability; replaced by fexofenadine (active metabolite).",
    ),
    ClinicalFailure(
        drug_id="astemizole", drug_search_name="astemizole",
        approval_year=1988, withdrawal_year=1999,
        withdrawal_reason="hERG block → QT prolongation → cardiac arrhythmia",
        causal_off_target_uniprot="Q12809", causal_off_target_gene="KCNH2",
        causal_side_effects_umls=("C0040479", "C0151878", "C0003811"),
        causal_side_effects_display="Torsade de pointes / QT prolonged",
        in_sider_catalog=False,
        notes="Second-generation antihistamine; hERG IC50 ~0.3 nM.",
    ),
    ClinicalFailure(
        drug_id="cisapride", drug_search_name="cisapride",
        approval_year=1993, withdrawal_year=2000,
        withdrawal_reason="hERG block → QT prolongation + DDI risks",
        causal_off_target_uniprot="Q12809", causal_off_target_gene="KCNH2",
        causal_side_effects_umls=("C0151878", "C0040479", "C0003811", "C0085612"),
        causal_side_effects_display="QT prolonged / Torsade de pointes / Ventricular arrhythmia",
        in_sider_catalog=False,
        notes="Gastroprokinetic 5-HT4 agonist; hERG cross-reactivity caused withdrawal.",
    ),
    ClinicalFailure(
        drug_id="troglitazone", drug_search_name="troglitazone",
        approval_year=1997, withdrawal_year=2000,
        withdrawal_reason="Idiosyncratic hepatic failure (mitochondrial / metabolite toxicity)",
        causal_off_target_uniprot="P08684", causal_off_target_gene="CYP3A4",
        causal_side_effects_umls=("C0085605", "C0235378", "C0160390",
                                    "C0023895", "C0019158", "C0151766"),
        causal_side_effects_display="Hepatic failure / Hepatotoxicity / Liver injury",
        in_sider_catalog=True,
        notes="PPARG agonist; idiosyncratic hepatotoxicity not seen with rosi/pio.",
    ),
    ClinicalFailure(
        drug_id="cerivastatin", drug_search_name="cerivastatin",
        approval_year=1997, withdrawal_year=2001,
        withdrawal_reason="Rhabdomyolysis (especially with gemfibrozil DDI via CYP2C8/SLCO1B1)",
        causal_off_target_uniprot="Q9Y6L6", causal_off_target_gene="SLCO1B1",
        causal_side_effects_umls=("C0035410",),
        causal_side_effects_display="Rhabdomyolysis",
        in_sider_catalog=False,
        notes="HMGCR inhibition is the on-target; SLCO1B1/CYP2C8 mediate the DDI that drives blood-level rhabdo.",
    ),
    ClinicalFailure(
        drug_id="rofecoxib", drug_search_name="rofecoxib",
        approval_year=1999, withdrawal_year=2004,
        withdrawal_reason="COX-2 selective → prostacyclin/thromboxane imbalance → MI",
        causal_off_target_uniprot="P35354", causal_off_target_gene="PTGS2",
        causal_side_effects_umls=("C0027051", "C0151744", "C0038454"),
        causal_side_effects_display="Myocardial infarction / Stroke",
        in_sider_catalog=True,
        notes="Vioxx — COX-2 selectivity vs COX-1 sparing tipped vascular balance.",
    ),
    ClinicalFailure(
        drug_id="valdecoxib", drug_search_name="valdecoxib",
        approval_year=2001, withdrawal_year=2005,
        withdrawal_reason="Same COX-2 cardiovascular liability as rofecoxib",
        causal_off_target_uniprot="P35354", causal_off_target_gene="PTGS2",
        causal_side_effects_umls=("C0027051", "C0151744", "C0038454"),
        causal_side_effects_display="Myocardial infarction / CV events",
        in_sider_catalog=True,
        notes="Bextra — withdrawn for both CV (class effect) and SJS skin reactions.",
    ),
    ClinicalFailure(
        drug_id="pergolide", drug_search_name="pergolide",
        approval_year=1988, withdrawal_year=2007,
        withdrawal_reason="5-HT2B agonism → cardiac valvulopathy",
        causal_off_target_uniprot="P41595", causal_off_target_gene="HTR2B",
        causal_side_effects_umls=("C0018802", "C0018801", "C0018799", "C0018790"),
        causal_side_effects_display="Cardiac failure / Cardiac disorder",
        in_sider_catalog=True,
        notes="Dopamine agonist for PD; 5-HT2B side effect drove valvulopathy in long-term use.",
    ),
    ClinicalFailure(
        drug_id="sibutramine", drug_search_name="sibutramine",
        approval_year=1997, withdrawal_year=2010,
        withdrawal_reason="Adrenergic-mediated CV events (MI + stroke)",
        causal_off_target_uniprot="P23975", causal_off_target_gene="SLC6A2",
        causal_side_effects_umls=("C0027051", "C0038454", "C0020538"),
        causal_side_effects_display="Myocardial infarction / Stroke / Hypertension",
        in_sider_catalog=True,
        notes="SNRI weight-loss drug; SCOUT trial showed +CV events.",
    ),
    ClinicalFailure(
        drug_id="rosiglitazone", drug_search_name="rosiglitazone",
        approval_year=1999, withdrawal_year=2010,
        withdrawal_reason="MI / heart failure risk (restricted, not fully withdrawn US)",
        causal_off_target_uniprot="P37231", causal_off_target_gene="PPARG",
        causal_side_effects_umls=("C0027051", "C0018802", "C0018801"),
        causal_side_effects_display="Myocardial infarction / Cardiac failure",
        in_sider_catalog=True,
        notes="PPARG agonist; Nissen 2007 meta-analysis triggered restrictions.",
    ),
    ClinicalFailure(
        drug_id="mibefradil", drug_search_name="mibefradil",
        approval_year=1997, withdrawal_year=1998,
        withdrawal_reason="CYP3A4 inhibition → severe DDI-mediated rhabdo and arrhythmia",
        causal_off_target_uniprot="P08684", causal_off_target_gene="CYP3A4",
        causal_side_effects_umls=("C0035410", "C0003811", "C0085612"),
        causal_side_effects_display="Rhabdomyolysis / Arrhythmia",
        in_sider_catalog=False,
        notes="T-type Ca channel blocker; CYP3A4 inhibition caused DDI catastrophe.",
    ),
    ClinicalFailure(
        drug_id="nefazodone", drug_search_name="nefazodone",
        approval_year=1994, withdrawal_year=2003,
        withdrawal_reason="Idiosyncratic hepatic failure (CYP3A4 + metabolite toxicity)",
        causal_off_target_uniprot="P08684", causal_off_target_gene="CYP3A4",
        causal_side_effects_umls=("C0085605", "C0235378", "C0160390",
                                    "C0023895", "C0019158"),
        causal_side_effects_display="Hepatic failure / Hepatotoxicity",
        in_sider_catalog=True,
        notes="Antidepressant; CYP3A4 inhibition + quinone metabolite drove liver toxicity.",
    ),
    ClinicalFailure(
        drug_id="bromfenac", drug_search_name="bromfenac",
        approval_year=1997, withdrawal_year=1998,
        withdrawal_reason="Severe hepatotoxicity (acyl-glucuronidation reactive metabolite)",
        causal_off_target_uniprot="P35354", causal_off_target_gene="PTGS2",
        causal_side_effects_umls=("C0085605", "C0235378", "C0023895"),
        causal_side_effects_display="Hepatic failure / Hepatotoxicity",
        in_sider_catalog=False,
        notes="NSAID; reactive acyl glucuronide metabolite drove liver injury.",
    ),
    ClinicalFailure(
        drug_id="flecainide", drug_search_name="flecainide",
        approval_year=1985, withdrawal_year=None,
        withdrawal_reason="Black box: proarrhythmia in post-MI patients (CAST trial)",
        causal_off_target_uniprot="Q14524", causal_off_target_gene="SCN5A",
        causal_side_effects_umls=("C0085612", "C0003811", "C0040479",
                                    "C0151878", "C0018790"),
        causal_side_effects_display="Ventricular arrhythmia / Arrhythmia",
        in_sider_catalog=False,
        notes="Class IC antiarrhythmic; CAST trial showed paradoxical mortality post-MI.",
    ),
    ClinicalFailure(
        drug_id="tegaserod", drug_search_name="tegaserod",
        approval_year=2002, withdrawal_year=2007,
        withdrawal_reason="Increased CV ischemic events (MI, stroke, unstable angina)",
        causal_off_target_uniprot="P50406", causal_off_target_gene="HTR6",
        causal_side_effects_umls=("C0027051", "C0038454", "C0151744"),
        causal_side_effects_display="MI / Stroke (CV ischemic events)",
        in_sider_catalog=False,
        notes="5-HT4 agonist for IBS-C; serotonergic CV liability. Note: causal target debated; multiple 5-HT receptor involvement.",
    ),
)


def get_failures_in_catalog() -> list[ClinicalFailure]:
    return [c for c in CLINICAL_FAILURES if c.in_sider_catalog]


def get_failures_not_in_catalog() -> list[ClinicalFailure]:
    return [c for c in CLINICAL_FAILURES if not c.in_sider_catalog]
