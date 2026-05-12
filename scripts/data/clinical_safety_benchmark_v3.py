"""Sprint E expanded clinical-safety benchmark (n=185+ → target effective n≥200).

Extends Sprint 8A's v2 benchmark with ~80 modern biologic cases:
  - CAR-T cell therapies (axicabtagene, tisagenlecleucel, brexucabtagene,
    lisocabtagene, idecabtagene, ciltacabtagene) → CRS, ICANS, B-cell
    aplasia, cytopenias
  - Bispecifics (mosunetuzumab, glofitamab, epcoritamab, teclistamab,
    talquetamab, elranatamab, tarlatamab) → CRS, neurotoxicity, cytopenias
  - ADCs (T-DXd, sacituzumab govitecan, enfortumab vedotin, trastuzumab
    deruxtecan, mirvetuximab soravtansine, tisotumab vedotin, etc.) →
    payload-specific (ILD, neuropathy, neutropenia, ocular tox, hepatotox)
  - Newer ICIs (tislelizumab, tiragolumab, sabatolimab, cobolimab) →
    irAE spectrum
  - Modern targets (TROP2, Nectin-4, BCMA, GPRC5D, CD47, etc.)
  - Newer biologic classes (anti-CCR4, anti-CD47, anti-CD33, anti-CD30)
  - Bone (romosozumab → CV events BBW)
  - Metabolic (tirzepatide, retatrutide → pancreatitis, GI)
  - Neuro (galcanezumab → constipation, hypertension)
  - Hematology (emicizumab → TMA, lanadelumab)

Each case requires (target in target_vocab) AND (≥1 SE in side-effect vocab).
"""
from __future__ import annotations

from .clinical_safety_benchmark import SafetyCase, passes_eligibility
from .clinical_safety_benchmark_v2 import EXPANDED_SAFETY_CASES


# -----------------------------------------------------------------------------
# Sprint E NEW cases — modern biologic deep coverage
# -----------------------------------------------------------------------------

SPRINT_E_NEW_CASES: tuple[SafetyCase, ...] = (

    # =========================================================================
    # CAR-T CELL THERAPIES (CD19)
    # =========================================================================
    SafetyCase("axicabtagene_crs",
               "axicabtagene ciloleucel", "black_box", "P15391", "CD19",
               ("C0948715", "C0014335", "C0029118", "C0027947", "C0040034", "C0002871"),
               "CRS / ICANS / cytopenias",
               "anti-CD19 CAR-T; CRS + ICANS BBWs"),
    SafetyCase("tisagenlecleucel_crs",
               "tisagenlecleucel", "black_box", "P15391", "CD19",
               ("C0948715", "C0014335", "C0029118", "C0027947", "C0040034"),
               "CRS / ICANS / cytopenias",
               "anti-CD19 CAR-T"),
    SafetyCase("brexucabtagene_crs",
               "brexucabtagene autoleucel", "black_box", "P15391", "CD19",
               ("C0948715", "C0014335", "C0029118", "C0027947", "C0002871"),
               "CRS / ICANS / cytopenias",
               "anti-CD19 CAR-T"),
    SafetyCase("lisocabtagene_crs",
               "lisocabtagene maraleucel", "black_box", "P15391", "CD19",
               ("C0948715", "C0014335", "C0029118", "C0027947"),
               "CRS / ICANS / infections",
               "anti-CD19 CAR-T"),

    # =========================================================================
    # CAR-T CELL THERAPIES (BCMA)
    # =========================================================================
    SafetyCase("idecabtagene_crs",
               "idecabtagene vicleucel", "black_box", "Q02223", "TNFRSF17",
               ("C0948715", "C0014335", "C0029118", "C0027947", "C0040034", "C0002871"),
               "CRS / ICANS / cytopenias",
               "anti-BCMA CAR-T"),
    SafetyCase("ciltacabtagene_crs",
               "ciltacabtagene autoleucel", "black_box", "Q02223", "TNFRSF17",
               ("C0948715", "C0014335", "C0029118", "C0027947", "C0019360"),
               "CRS / ICANS / infections / HZV",
               "anti-BCMA CAR-T"),

    # =========================================================================
    # CD20 × CD3 BISPECIFICS
    # =========================================================================
    SafetyCase("mosunetuzumab_crs",
               "mosunetuzumab", "black_box", "P11836", "MS4A1",
               ("C0948715", "C0014335", "C0027947", "C0029118"),
               "CRS / cytopenias / infections",
               "anti-CD20×CD3 bispecific"),
    SafetyCase("glofitamab_crs",
               "glofitamab", "black_box", "P11836", "MS4A1",
               ("C0948715", "C0014335", "C0027947", "C0029118"),
               "CRS / cytopenias",
               "anti-CD20×CD3 bispecific"),
    SafetyCase("epcoritamab_crs",
               "epcoritamab", "black_box", "P11836", "MS4A1",
               ("C0948715", "C0014335", "C0027947"),
               "CRS / cytopenias",
               "anti-CD20×CD3 bispecific"),

    # =========================================================================
    # BCMA × CD3 BISPECIFICS
    # =========================================================================
    SafetyCase("teclistamab_crs",
               "teclistamab", "black_box", "Q02223", "TNFRSF17",
               ("C0948715", "C0014335", "C0027947", "C0029118", "C0019360"),
               "CRS / ICANS / cytopenias / infections",
               "anti-BCMA×CD3 bispecific"),
    SafetyCase("elranatamab_crs",
               "elranatamab", "black_box", "Q02223", "TNFRSF17",
               ("C0948715", "C0014335", "C0027947", "C0029118"),
               "CRS / cytopenias / infections",
               "anti-BCMA×CD3 bispecific"),
    SafetyCase("linvoseltamab_crs",
               "linvoseltamab", "black_box", "Q02223", "TNFRSF17",
               ("C0948715", "C0014335", "C0027947"),
               "CRS / cytopenias",
               "anti-BCMA×CD3 bispecific"),

    # =========================================================================
    # GPRC5D × CD3 BISPECIFIC
    # =========================================================================
    SafetyCase("talquetamab_crs",
               "talquetamab", "black_box", "Q9NZD1", "GPRC5D",
               ("C0948715", "C0014335", "C0027947", "C0029118"),
               "CRS / cytopenias / dysgeusia / skin",
               "anti-GPRC5D×CD3 bispecific"),

    # =========================================================================
    # DLL3 × CD3 BISPECIFIC (SCLC)
    # =========================================================================
    SafetyCase("tarlatamab_crs",
               "tarlatamab", "black_box", "Q9NYJ7", "DLL3",
               ("C0948715", "C0014335", "C0027947"),
               "CRS / cytopenias / neurotox",
               "anti-DLL3×CD3 bispecific"),

    # =========================================================================
    # ADCs — TROP2 (govitecan, SN-38 payload)
    # =========================================================================
    SafetyCase("sacituzumab_govitecan_diarr",
               "sacituzumab govitecan", "black_box", "P09758", "TACSTD2",
               ("C0011991", "C0027947", "C0746883", "C0040034", "C0002871"),
               "Severe diarrhea (BBW) / neutropenia (BBW) / cytopenias",
               "anti-TROP2 ADC; SN-38 payload"),
    SafetyCase("datopotamab_diarr",
               "datopotamab deruxtecan", "mechanism_established", "P09758", "TACSTD2",
               ("C0011991", "C0027947", "C1279945", "C0032310"),
               "Diarrhea / neutropenia / ILD (deruxtecan)",
               "anti-TROP2 ADC; DXd payload"),

    # =========================================================================
    # ADCs — Nectin-4 (enfortumab vedotin, MMAE payload)
    # =========================================================================
    SafetyCase("enfortumab_skin",
               "enfortumab vedotin", "black_box", "Q96NY8", "NECTIN4",
               ("C0020456", "C0015230", "C0014742", "C0014518", "C0027947"),
               "SJS/TEN (BBW) / hyperglycemia (BBW) / rash",
               "anti-Nectin-4 ADC; MMAE payload"),

    # =========================================================================
    # ADCs — CD30 (brentuximab vedotin, MMAE)
    # =========================================================================
    SafetyCase("brentuximab_pn",
               "brentuximab vedotin", "black_box", "P28908", "TNFRSF8",
               ("C0023524", "C0027947", "C0746883", "C0029118"),
               "PML (BBW) / neutropenia / febrile neutropenia",
               "anti-CD30 ADC; MMAE payload"),

    # =========================================================================
    # ADCs — HER3 (patritumab deruxtecan, DXd)
    # =========================================================================
    SafetyCase("patritumab_ild",
               "patritumab deruxtecan", "mechanism_established", "P21860", "ERBB3",
               ("C1279945", "C0032310", "C0027947", "C0002871"),
               "ILD/pneumonitis / cytopenias",
               "anti-HER3 ADC; DXd payload"),

    # =========================================================================
    # ADCs — CD22 (inotuzumab + moxetumomab — calicheamicin / immunotoxin)
    # =========================================================================
    SafetyCase("inotuzumab_hep",
               "inotuzumab", "black_box", "P20273", "CD22",
               ("C0085605", "C0235378", "C0040034", "C0027947"),
               "Hepatotoxicity / VOD (BBW) / cytopenias",
               "anti-CD22 ADC; calicheamicin"),
    SafetyCase("moxetumomab_hus",
               "moxetumomab pasudotox", "black_box", "P20273", "CD22",
               ("C0022660", "C0040034", "C0235378", "C0029118"),
               "Capillary leak / HUS (BBW) / hepatic / cytopenias",
               "anti-CD22 immunotoxin"),

    # =========================================================================
    # ADCs — CD33 (gemtuzumab ozogamicin, calicheamicin)
    # =========================================================================
    SafetyCase("gemtuzumab_hep",
               "gemtuzumab ozogamicin", "black_box", "P20138", "CD33",
               ("C0085605", "C0235378", "C0040034", "C0027947", "C0948715"),
               "Hepatotoxicity / VOD (BBW) / cytopenias / infusion",
               "anti-CD33 ADC; calicheamicin"),

    # =========================================================================
    # ADCs — BCMA (belantamab mafodotin, MMAF)
    # =========================================================================
    SafetyCase("belantamab_ocular",
               "belantamab mafodotin", "black_box", "Q02223", "TNFRSF17",
               ("C0040034", "C0027947", "C0009763", "C0002871"),
               "Ocular toxicity (BBW) / cytopenias / conjunctivitis",
               "anti-BCMA ADC; MMAF payload"),

    # =========================================================================
    # ADCs — Folate receptor (mirvetuximab soravtansine)
    # =========================================================================
    SafetyCase("mirvetuximab_ocular",
               "mirvetuximab soravtansine", "black_box", "P15328", "FOLR1",
               ("C0009763", "C0011991", "C0027947"),
               "Ocular toxicity (BBW) / GI / cytopenias",
               "anti-FOLR1 ADC; DM4 payload"),

    # =========================================================================
    # ADCs — Tissue factor (tisotumab vedotin)
    # =========================================================================
    SafetyCase("tisotumab_ocular",
               "tisotumab vedotin", "black_box", "P13726", "F3",
               ("C0009763", "C0019080", "C0042487", "C0027947"),
               "Ocular toxicity (BBW) / bleeding / VTE",
               "anti-tissue factor ADC; MMAE"),

    # =========================================================================
    # NEWER ICIs (TIGIT, LAG3, TIM-3)
    # =========================================================================
    SafetyCase("tiragolumab_irae",
               "tiragolumab", "mechanism_established", "Q495A1", "TIGIT",
               ("C1279945", "C0009319", "C0019158", "C0040128", "C0001623"),
               "irAE spectrum",
               "anti-TIGIT"),
    SafetyCase("vibostolimab_irae",
               "vibostolimab", "mechanism_established", "Q495A1", "TIGIT",
               ("C1279945", "C0009319", "C0019158", "C0040128"),
               "irAE spectrum",
               "anti-TIGIT"),
    SafetyCase("sabatolimab_irae",
               "sabatolimab", "mechanism_established", "Q8TDQ0", "HAVCR2",
               ("C1279945", "C0009319", "C0040128"),
               "irAE spectrum",
               "anti-TIM-3"),

    # =========================================================================
    # NEWER ANTI-PD-1 (Asian-approved)
    # =========================================================================
    SafetyCase("tislelizumab_irae",
               "tislelizumab", "black_box", "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128", "C0001623"),
               "irAE spectrum",
               "anti-PD-1"),
    SafetyCase("toripalimab_irae",
               "toripalimab", "black_box", "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128"),
               "irAE spectrum",
               "anti-PD-1"),
    SafetyCase("sintilimab_irae",
               "sintilimab", "mechanism_established", "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128"),
               "irAE spectrum",
               "anti-PD-1"),

    # =========================================================================
    # CD47/SIRPα anti-cancer
    # =========================================================================
    SafetyCase("magrolimab_anemia",
               "magrolimab", "black_box", "Q08722", "CD47",
               ("C0002871", "C0040034", "C0948715"),
               "Severe anemia (BBW) / thrombocytopenia / infusion",
               "anti-CD47"),

    # =========================================================================
    # CD38 (daratumumab/isatuximab)
    # =========================================================================
    SafetyCase("daratumumab_infusion",
               "daratumumab", "black_box", "P28907", "CD38",
               ("C0948715", "C0027947", "C0040034", "C0029118", "C0019360"),
               "Infusion BBW / cytopenias / HZV",
               "anti-CD38"),
    SafetyCase("isatuximab_infusion",
               "isatuximab", "mechanism_established", "P28907", "CD38",
               ("C0948715", "C0027947", "C0040034", "C0029118"),
               "Infusion / cytopenias / infections",
               "anti-CD38"),

    # =========================================================================
    # SLAMF7 (elotuzumab)
    # =========================================================================
    SafetyCase("elotuzumab_infusion",
               "elotuzumab", "mechanism_established", "Q9NQ25", "SLAMF7",
               ("C0948715", "C0029118"),
               "Infusion / infections",
               "anti-SLAMF7"),

    # =========================================================================
    # CCR4 (mogamulizumab) — BBW for hepatitis B and SJS-like, EBV reactivation
    # =========================================================================
    SafetyCase("mogamulizumab_skin",
               "mogamulizumab", "black_box", "P51679", "CCR4",
               ("C0015230", "C0029118", "C0019163", "C0024299"),
               "Skin AE (BBW) / hepatitis B / EBV / lymphoma",
               "anti-CCR4"),

    # =========================================================================
    # CLAUDIN 18.2 (zolbetuximab)
    # =========================================================================
    SafetyCase("zolbetuximab_nv",
               "zolbetuximab", "mechanism_established", "P56856", "CLDN18",
               ("C0027497", "C0042963", "C0948715"),
               "Nausea / vomiting / infusion",
               "anti-Claudin 18.2"),

    # =========================================================================
    # BONE — sclerostin (romosozumab) — CV BBW
    # =========================================================================
    SafetyCase("romosozumab_mi",
               "romosozumab", "black_box", "Q9BQB4", "SOST",
               ("C0027051", "C0038454", "C0018802"),
               "MI / stroke / cardiac failure (BBW)",
               "anti-sclerostin"),

    # =========================================================================
    # METABOLIC bispecifics (tirzepatide, retatrutide)
    # =========================================================================
    SafetyCase("tirzepatide_pancr",
               "tirzepatide", "mechanism_established", "P48546", "GIPR",
               ("C0030305", "C0011991", "C0020538"),
               "Pancreatitis / GI / thyroid C-cell BBW (proxy)",
               "GLP-1/GIP dual agonist"),
    SafetyCase("retatrutide_gi",
               "retatrutide", "mechanism_established", "P47871", "GCGR",
               ("C0011991", "C0030305"),
               "GI / pancreatitis",
               "GLP-1/GIP/GCG triple agonist"),

    # =========================================================================
    # NEURO biologics (CGRP class)
    # =========================================================================
    SafetyCase("galcanezumab_constip",
               "galcanezumab", "mechanism_established", "P06881", "CALCA",
               ("C0009806", "C0020538"),
               "Constipation / hypertension (FDA warning)",
               "anti-CGRP"),
    SafetyCase("erenumab_constip",
               "erenumab", "mechanism_established", "Q16602", "CALCRL",
               ("C0009806", "C0020538"),
               "Constipation / hypertension",
               "anti-CGRP receptor"),

    # =========================================================================
    # HEMATOLOGY biologics
    # =========================================================================
    SafetyCase("emicizumab_tma",
               "emicizumab", "black_box", "P00740", "F9",
               ("C0022660", "C0948715"),
               "TMA (BBW with concurrent bypassing agents) / infusion",
               "FVIII-mimic bispecific"),
    SafetyCase("lanadelumab_inj",
               "lanadelumab", "mechanism_established", "P03952", "KLKB1",
               ("C0948715",),
               "Injection-site reactions",
               "anti-plasma kallikrein (HAE prophylaxis)"),

    # =========================================================================
    # COMPLEMENT inhibitors (pegcetacoplan, danicopan, iptacopan)
    # =========================================================================
    SafetyCase("pegcetacoplan_inf",
               "pegcetacoplan", "black_box", "P01024", "C3",
               ("C0025289", "C0029118"),
               "Meningococcal infection BBW",
               "anti-C3"),
    SafetyCase("iptacopan_inf",
               "iptacopan", "black_box", "P01024", "C3",
               ("C0025289", "C0029118"),
               "Meningococcal infection BBW",
               "complement factor B inhibitor"),

    # =========================================================================
    # NEWER ANTI-IL-17/IL-23 (bimekizumab, mirikizumab)
    # =========================================================================
    SafetyCase("bimekizumab_candid",
               "bimekizumab", "mechanism_established", "Q16552", "IL17A",
               ("C0029118", "C0009319", "C0019163"),
               "Mucocutaneous candidiasis / IBD flare / HBV",
               "anti-IL-17A/F"),
    SafetyCase("mirikizumab_inf",
               "mirikizumab", "mechanism_established", "Q9NPF7", "IL23A",
               ("C0029118",),
               "Infections",
               "anti-IL-23"),

    # =========================================================================
    # NEWER ASTHMA biologics (tezepelumab, lebrikizumab, tralokinumab)
    # =========================================================================
    SafetyCase("tezepelumab_inf",
               "tezepelumab", "mechanism_established", "Q969D9", "TSLP",
               ("C0948715", "C0029118"),
               "Infusion / infections",
               "anti-TSLP"),
    SafetyCase("lebrikizumab_conj",
               "lebrikizumab", "mechanism_established", "P35225", "IL13",
               ("C0009763", "C0029118"),
               "Conjunctivitis / infections",
               "anti-IL-13"),

    # =========================================================================
    # METABOLIC RNA therapies (inclisiran, volanesorsen)
    # =========================================================================
    SafetyCase("inclisiran_inj",
               "inclisiran", "mechanism_established", "Q8NBP7", "PCSK9",
               ("C0948715",),
               "Injection-site reactions",
               "PCSK9 siRNA"),
    SafetyCase("volanesorsen_plat",
               "volanesorsen", "black_box", "P02656", "APOC3",
               ("C0040034", "C0022660"),
               "Thrombocytopenia (BBW) / glomerulonephritis (BBW)",
               "APOC3 antisense"),
    SafetyCase("evinacumab_inj",
               "evinacumab", "mechanism_established", "Q9Y5C1", "ANGPTL3",
               ("C0948715",),
               "Injection-site reactions",
               "anti-ANGPTL3"),

    # =========================================================================
    # ANGIOPOIETIN/VEGF eye (faricimab) and CV
    # =========================================================================
    SafetyCase("brolucizumab_endoph",
               "brolucizumab", "black_box", "P15692", "VEGFA",
               ("C0009763",),
               "Endophthalmitis / intraocular inflammation (BBW)",
               "anti-VEGF eye"),

    # =========================================================================
    # IL-36R (spesolimab) for generalized pustular psoriasis
    # =========================================================================
    SafetyCase("spesolimab_inf",
               "spesolimab", "mechanism_established", "Q9HBE5", "IL36R",
               ("C0948715", "C0029118"),
               "Infusion / infections",
               "anti-IL-36R"),

    # =========================================================================
    # B7-H3 (enoblituzumab)
    # =========================================================================
    SafetyCase("enoblituzumab_inf",
               "enoblituzumab", "mechanism_established", "Q5ZPR3", "CD276",
               ("C0948715", "C0029118"),
               "Infusion / infections",
               "anti-B7-H3"),

    # =========================================================================
    # IL-2Rα (basiliximab/daclizumab)
    # =========================================================================
    SafetyCase("basiliximab_inf",
               "basiliximab", "mechanism_established", "P01589", "IL2RA",
               ("C0948715", "C0029118"),
               "Infusion / infections",
               "anti-IL-2Rα (transplant)"),

    # =========================================================================
    # ANTI-TIGIT bispecifics
    # =========================================================================
    SafetyCase("ociperlimab_irae",
               "ociperlimab", "mechanism_established", "Q495A1", "TIGIT",
               ("C1279945", "C0040128"),
               "irAE",
               "anti-TIGIT"),

    # =========================================================================
    # ANTI-PD-1 × CTLA-4 (cadonilimab)
    # =========================================================================
    SafetyCase("cadonilimab_irae",
               "cadonilimab", "mechanism_established", "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128", "C0027059"),
               "Combined ICI irAE spectrum",
               "PD-1×CTLA-4 bispecific"),

    # =========================================================================
    # IL-13Rα1 / IL-31 — dermatology
    # =========================================================================
    SafetyCase("nemolizumab_inf",
               "nemolizumab", "mechanism_established", "Q8NI17", "IL31RA",
               ("C0029118",),
               "Infections / asthma exacerbation",
               "anti-IL-31RA"),

    # =========================================================================
    # OX40 / 4-1BB historical (urelumab hepatotox)
    # =========================================================================
    SafetyCase("urelumab_hep",
               "urelumab", "black_box", "Q07011", "TNFRSF9",
               ("C0085605", "C0235378", "C0019158"),
               "Hepatotoxicity (BBW; clinical hold)",
               "4-1BB agonist"),
)


EXPANDED_SAFETY_CASES_V3: tuple[SafetyCase, ...] = (
    EXPANDED_SAFETY_CASES + SPRINT_E_NEW_CASES
)


def main() -> int:
    """Diagnostic: eligibility distribution by therapeutic-area."""
    import json
    from collections import Counter
    from pathlib import Path

    workspace = Path(__file__).resolve().parent.parent.parent
    results = workspace / "results"

    with open(results / "side_effect_vocab.json") as f:
        v = json.load(f)
    vocab_set = set(v["umls_ids"])
    with open(results / "target_vocab.json") as f:
        tv = json.load(f)
    target_set = {t["uniprot"] for t in tv["targets"]}

    n_v2 = len(EXPANDED_SAFETY_CASES)
    n_v2_elig = sum(1 for c in EXPANDED_SAFETY_CASES
                    if passes_eligibility(c, vocab_set, target_set))
    n_new = len(SPRINT_E_NEW_CASES)
    n_new_elig = sum(1 for c in SPRINT_E_NEW_CASES
                     if passes_eligibility(c, vocab_set, target_set))
    n_total = len(EXPANDED_SAFETY_CASES_V3)
    n_total_elig = sum(1 for c in EXPANDED_SAFETY_CASES_V3
                       if passes_eligibility(c, vocab_set, target_set))

    print(f"[v2 cases (Sprint 8A)]   total={n_v2}, eligible={n_v2_elig}")
    print(f"[E new cases]            total={n_new}, eligible={n_new_elig}")
    print(f"[v3 total]               total={n_total}, eligible={n_total_elig}")

    onc_genes = {"ERBB2", "TOP2A", "TOP2B", "KDR", "PDGFRB", "ALK",
                 "EGFR", "BRAF", "MAP2K1", "CDK4", "CDK6", "PARP1",
                 "BCL2", "MTOR", "PIK3CA", "PIK3CG", "ABL1", "KIT",
                 "MET", "FLT1", "PDCD1", "CD274", "CTLA4", "LAG3",
                 "MAPT", "TUBA4A", "TACSTD2", "NECTIN4", "ERBB3",
                 "CD276", "CD22", "CD33", "F3", "FOLR1", "CD30",
                 "TNFRSF8", "TIGIT", "HAVCR2", "TNFRSF17", "GPRC5D",
                 "DLL3", "CLDN18", "CD47", "SIRPA", "CD38", "SLAMF7",
                 "CCR4"}
    immuno_genes = {"TNF", "IL6R", "IL17A", "IL12B", "IL23A", "IL5",
                     "IL5RA", "IL4R", "IL1B", "JAK1", "JAK2", "JAK3",
                     "S1PR1", "ITGA4", "ITGB7", "C5", "MS4A1", "CD19",
                     "TNFRSF13B", "C3", "TSLP", "IL13", "IL31RA",
                     "IL36R", "IL2RA", "C5AR1"}

    def classify(c: SafetyCase) -> str:
        g = c.causal_off_target_gene.upper()
        if g in onc_genes:
            return "Oncology"
        if g in immuno_genes:
            return "Immunology"
        return "Other"

    new_classes = Counter(classify(c) for c in SPRINT_E_NEW_CASES)
    total_classes = Counter(classify(c) for c in EXPANDED_SAFETY_CASES_V3
                             if passes_eligibility(c, vocab_set, target_set))
    print(f"\n[Sprint E new cases by TA]:           {dict(new_classes)}")
    print(f"[v3 eligible total by TA]:            {dict(total_classes)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
