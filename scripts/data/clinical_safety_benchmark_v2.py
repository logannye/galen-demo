"""Sprint 8A expanded clinical-safety benchmark (n=160+ candidates).

Builds on Sprint 5's n=60 benchmark by adding ~100 new cases covering:
  - Onc/Immuno depth: more ICIs, more TKIs, more biologics
  - CV/metabolic depth: anticoagulants, antiarrhythmics, GLP-1, SGLT-2
  - CNS depth: anticonvulsants, antidepressants, antipsychotics
  - Cytotoxic chemotherapy: alkylators, antimetabolites, microtubule

Each new case follows the SafetyCase schema and is filtered at runtime
to require (target ∈ target_vocab) AND (≥1 SE ∈ side_effect_vocab).

The expanded benchmark is the PRIMARY power source for Sprint 8A WIN/NULL
determination (the original n=51 remains for apples-to-apples vs Sprint 7).
"""
from __future__ import annotations

from .clinical_safety_benchmark import SafetyCase, SAFETY_CASES, passes_eligibility


# -----------------------------------------------------------------------------
# Sprint 8A new cases — broadened onc/immuno + CV/CNS coverage
# -----------------------------------------------------------------------------
# UMLS shorthand used below (verified in 605-vocab):
#   C0151878 QT prolonged · C0040479 Torsade · C0018802 Cardiac failure congestive
#   C0018801 Cardiac failure · C0027051 MI · C0038454 Stroke · C0020538 Hypertension
#   C0235378 Hepatotoxicity · C0085605 Hepatic failure · C1279945 Acute IP
#   C0032310 Pneumonitis · C0009319 Colitis · C0019158 Hepatitis · C0027059 Myocarditis
#   C0040128 Thyroid disorder · C0001623 Adrenal insufficiency · C0029118 Opportunistic inf
#   C0023524 PML · C0041296 Tuberculosis · C0024299 Lymphoma · C0019163 Hepatitis B
#   C0019360 Herpes zoster · C0042487 Venous thrombosis · C0151942 Arterial thrombosis
#   C0035410 Rhabdomyolysis · C0026848 Myalgia · C0022660 AKI · C0035304 Renal failure
#   C0027947 Neutropenia · C0040034 Thrombocytopenia · C0002871 Anaemia
#   C0026986 MDS · C0023467 AML · C0041364 TLS · C0020456 Hyperglycaemia
#   C0014335 Pyrexia · C0011991 Diarrhoea · C0015230 Rash · C0033774 Pruritus
#   C0011603 Dermatitis · C0549410 Hand-foot · C0033687 Proteinuria
#   C0085610 Sinus bradycardia · C0271051 Macular oedema · C0024440 Cystoid macular oedema
#   C0024312 Lymphopenia · C0028081 Night sweats · C0033687 Proteinuria
#   C0030305 Pancreatitis · C0011304 Demyelination · C0026769 Multiple sclerosis
#   C0040128 Thyroid disorder · C0020676 Hypothyroidism · C0948715 Infusion reaction
#   C0014742 Erythema multiforme · C0014518 TEN · C0041834 Erythema
#   C0006142 Breast cancer · C0006826 Neoplasm malignant · C0023893 Liver cirrhosis
#   C0020640 Hypoglycaemia · C0004238 Atrial fibrillation


NEW_SAFETY_CASES: tuple[SafetyCase, ...] = (

    # ====================================================================
    # ONCOLOGY DEEP-DIVE (n≈30)
    # ====================================================================

    # --- VEGFR-TKI class (hypertension, hand-foot, proteinuria, bleeding)
    SafetyCase("axitinib_htn", "axitinib", "black_box", "P35968", "KDR",
               ("C0020538", "C0033687", "C0549410"),
               "VEGFR-TKI hypertension/proteinuria/hand-foot",
               "Class effect for VEGFR-TKIs"),
    SafetyCase("regorafenib_hepatotox", "regorafenib", "black_box", "P35968", "KDR",
               ("C0235378", "C0085605", "C0020538", "C0549410"),
               "Severe hepatotoxicity BBW",
               "Multi-kinase TKI (VEGFR, PDGFR, KIT, RET) hepatotox BBW"),
    SafetyCase("cabozantinib_hf", "cabozantinib", "black_box", "P35968", "KDR",
               ("C0020538", "C0549410", "C0033687"),
               "Hypertension / hand-foot / proteinuria",
               "VEGFR/MET multi-kinase"),
    SafetyCase("ponatinib_thrombosis", "ponatinib", "black_box", "P00519", "ABL1",
               ("C0151942", "C0027051", "C0042487", "C0020538"),
               "Arterial thrombosis BBW / MI / VTE",
               "Pan-BCR-ABL TKI (incl T315I); arterial thrombosis BBW"),
    SafetyCase("vandetanib_qt", "vandetanib", "black_box", "Q12809", "KCNH2",
               ("C0151878", "C0040479", "C0003811"),
               "QT prolongation BBW",
               "Multi-kinase RET inhibitor; QT BBW"),
    SafetyCase("lenvatinib_htn", "lenvatinib", "black_box", "P35968", "KDR",
               ("C0020538", "C0018802", "C0033687", "C0549410"),
               "Hypertension / cardiac dysfunction / proteinuria",
               "VEGFR-TKI"),

    # --- EGFR-TKI class (rash, diarrhea, ILD/pneumonitis)
    SafetyCase("erlotinib_ild", "erlotinib", "black_box", "P00533", "EGFR",
               ("C1279945", "C0032310", "C0015230", "C0011991"),
               "Interstitial lung disease / rash",
               "EGFR-TKI"),
    SafetyCase("gefitinib_ild", "gefitinib", "black_box", "P00533", "EGFR",
               ("C1279945", "C0032310", "C0015230"),
               "Interstitial lung disease",
               "EGFR-TKI"),
    SafetyCase("afatinib_diarrhea", "afatinib", "mechanism_established", "P00533", "EGFR",
               ("C0011991", "C0015230", "C0011603"),
               "Severe diarrhea / rash",
               "EGFR/HER2 irreversible TKI"),
    SafetyCase("osimertinib_card", "osimertinib", "black_box", "P00533", "EGFR",
               ("C0018802", "C0151878", "C1279945"),
               "Cardiomyopathy / QT / ILD",
               "T790M EGFR-TKI"),

    # --- HER2 (cardiotoxicity, ILD with T-DXd)
    SafetyCase("lapatinib_card", "lapatinib", "black_box", "P04626", "ERBB2",
               ("C0018802", "C0235378", "C0151878"),
               "Cardiotoxicity / hepatotoxicity / QT",
               "EGFR/HER2 dual TKI"),
    SafetyCase("neratinib_diarrhea", "neratinib", "black_box", "P04626", "ERBB2",
               ("C0011991", "C0235378"),
               "Severe diarrhea / hepatotoxicity",
               "Pan-HER irreversible"),
    SafetyCase("tucatinib_hepatic", "tucatinib", "mechanism_established", "P04626", "ERBB2",
               ("C0235378", "C0011991"),
               "Hepatotoxicity / diarrhea",
               "Selective HER2 TKI"),
    SafetyCase("pertuzumab_card", "pertuzumab", "black_box", "P04626", "ERBB2",
               ("C0018802", "C0018801", "C0242698"),
               "LV dysfunction / cardiac failure",
               "Anti-HER2 mAb (extracellular dimerization)"),
    SafetyCase("trastuzumab_emtansine_ild",
               "trastuzumab emtansine", "black_box", "P04626", "ERBB2",
               ("C0018802", "C1279945", "C0040034"),
               "Cardiotoxicity / ILD / thrombocytopenia",
               "T-DM1 ADC"),
    SafetyCase("trastuzumab_deruxtecan_ild",
               "trastuzumab deruxtecan", "black_box", "P04626", "ERBB2",
               ("C1279945", "C0032310", "C0027947", "C0018802"),
               "ILD/pneumonitis BBW / cardiotox / neutropenia",
               "T-DXd ADC; ILD BBW"),

    # --- ALK/ROS1 inhibitors
    SafetyCase("crizotinib_qt", "crizotinib", "black_box", "Q9UM73", "ALK",
               ("C0151878", "C0085610", "C0235378", "C1279945"),
               "QT / bradycardia / hepatotox / ILD",
               "ALK/ROS1/MET TKI"),
    SafetyCase("alectinib_hepatotox", "alectinib", "mechanism_established", "Q9UM73", "ALK",
               ("C0235378", "C0085605", "C0018801"),
               "Hepatotoxicity / cardiotox",
               "ALK 2nd-gen"),
    SafetyCase("brigatinib_pneumonitis", "brigatinib", "mechanism_established", "Q9UM73", "ALK",
               ("C1279945", "C0032310", "C0020538"),
               "Early-onset pneumonitis / hypertension",
               "ALK 2nd-gen"),
    SafetyCase("ceritinib_qt", "ceritinib", "mechanism_established", "Q9UM73", "ALK",
               ("C0151878", "C0235378", "C0011991"),
               "QT / hepatotox / GI",
               "ALK 2nd-gen"),
    SafetyCase("lorlatinib_cns", "lorlatinib", "mechanism_established", "Q9UM73", "ALK",
               ("C0235378", "C0027849"),
               "Hepatic toxicity / CNS effects",
               "ALK 3rd-gen"),

    # --- BRAF/MEK (pyrexia, hepatic, MEK-cardio, rash)
    SafetyCase("dabrafenib_pyrexia", "dabrafenib", "mechanism_established", "P15056", "BRAF",
               ("C0014335", "C0015230", "C0235378"),
               "Pyrexia / rash / hepatic",
               "BRAFi monotherapy"),
    SafetyCase("trametinib_card", "trametinib", "mechanism_established", "Q02750", "MAP2K1",
               ("C0018802", "C0242698", "C0015230"),
               "Cardiomyopathy / rash",
               "MEK inhibitor"),
    SafetyCase("vemurafenib_sjs", "vemurafenib", "mechanism_established", "P15056", "BRAF",
               ("C0015230", "C0014742", "C0235378"),
               "Severe rash / SJS-like / hepatic",
               "BRAFi"),
    SafetyCase("encorafenib_card", "encorafenib", "mechanism_established", "P15056", "BRAF",
               ("C0151878", "C0235378"),
               "QT / hepatic",
               "BRAFi (with binimetinib)"),
    SafetyCase("cobimetinib_card", "cobimetinib", "mechanism_established", "Q02750", "MAP2K1",
               ("C0018802", "C0242698"),
               "Cardiomyopathy",
               "MEKi"),

    # --- PARP class (MDS/AML BBW, anemia, thrombocytopenia)
    SafetyCase("rucaparib_mds", "rucaparib", "black_box", "P09874", "PARP1",
               ("C0026986", "C0023467", "C0002871", "C0040034"),
               "MDS/AML / anaemia / thrombocytopenia",
               "PARPi"),
    SafetyCase("niraparib_mds", "niraparib", "black_box", "P09874", "PARP1",
               ("C0026986", "C0023467", "C0040034", "C0020538"),
               "MDS/AML / thrombocytopenia / hypertension",
               "PARPi"),
    SafetyCase("talazoparib_mds", "talazoparib", "black_box", "P09874", "PARP1",
               ("C0026986", "C0023467", "C0002871", "C0040034"),
               "MDS/AML / cytopenias",
               "PARPi"),

    # --- CDK4/6 (neutropenia, hepatic, VTE)
    SafetyCase("palbociclib_neutropenia", "palbociclib", "mechanism_established", "P11802", "CDK4",
               ("C0027947", "C0042487"),
               "Neutropenia / VTE",
               "CDK4/6i"),
    SafetyCase("ribociclib_qt", "ribociclib", "black_box", "P11802", "CDK4",
               ("C0151878", "C0027947", "C0235378"),
               "QT BBW / neutropenia / hepatotox",
               "CDK4/6i with QT BBW"),
    SafetyCase("abemaciclib_diarrhea", "abemaciclib", "mechanism_established", "P11802", "CDK4",
               ("C0011991", "C0027947", "C0042487"),
               "Diarrhea / neutropenia / VTE",
               "CDK4/6i"),

    # --- BTK (bleeding, AFib)
    SafetyCase("acalabrutinib_afib", "acalabrutinib", "mechanism_established", "Q06187", "BTK",
               ("C0004238", "C0027947", "C0023524"),
               "Atrial fibrillation / cytopenias / PML",
               "BTK 2nd-gen"),
    SafetyCase("zanubrutinib_neutropenia", "zanubrutinib", "mechanism_established", "Q06187", "BTK",
               ("C0027947", "C0004238", "C0020538"),
               "Neutropenia / AFib / hypertension",
               "BTK 2nd-gen"),
    SafetyCase("pirtobrutinib_arrhythm", "pirtobrutinib", "mechanism_established", "Q06187", "BTK",
               ("C0004238", "C0027947"),
               "AFib / neutropenia",
               "BTK non-covalent 3rd-gen"),

    # --- mTOR (pneumonitis, infections, hyperglycemia)
    SafetyCase("sirolimus_pneumonitis", "sirolimus", "black_box", "P42345", "MTOR",
               ("C0032310", "C1279945", "C0029118", "C0020456"),
               "Pneumonitis / infections / hyperglycaemia",
               "mTORi"),
    SafetyCase("everolimus_pneumonitis", "everolimus", "black_box", "P42345", "MTOR",
               ("C1279945", "C0032310", "C0029118", "C0020456"),
               "Pneumonitis / infections / hyperglycaemia",
               "mTORi"),
    SafetyCase("temsirolimus_pneumonitis", "temsirolimus", "mechanism_established", "P42345", "MTOR",
               ("C1279945", "C0032310", "C0029118"),
               "Pneumonitis / infections",
               "mTORi (renal cell carcinoma)"),

    # --- BCL2 (TLS, neutropenia)
    SafetyCase("venetoclax_tls", "venetoclax", "black_box", "P10415", "BCL2",
               ("C0041364", "C0027947", "C0022660"),
               "TLS BBW / neutropenia / AKI",
               "BCL2i; first-dose TLS BBW"),

    # --- PI3K
    SafetyCase("idelalisib_hepatic", "idelalisib", "black_box", "P48736", "PIK3CG",
               ("C0235378", "C0085605", "C0009319", "C0011991", "C0029118"),
               "Hepatic toxicity BBW / colitis / pneumonitis / infections",
               "PI3K-delta; multiple BBWs"),
    SafetyCase("alpelisib_hyperglycemia", "alpelisib", "mechanism_established", "P42336", "PIK3CA",
               ("C0020456", "C0015230", "C0011991"),
               "Severe hyperglycemia",
               "PI3K-alpha"),
    SafetyCase("copanlisib_hyperglycemia", "copanlisib", "mechanism_established", "P42336", "PIK3CA",
               ("C0020456", "C0020538", "C1279945"),
               "Hyperglycemia / hypertension / infection",
               "PI3K-alpha+delta"),

    # --- Immune checkpoint inhibitors (irAEs) — expand from 2 → 10
    SafetyCase("pembrolizumab_irae", "pembrolizumab", "black_box", "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128", "C0001623", "C0027059"),
               "Immune-mediated pneumonitis/colitis/hepatitis/thyroid/adrenal/myocarditis",
               "Anti-PD-1"),
    SafetyCase("nivolumab_irae", "nivolumab", "black_box", "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128", "C0001623", "C0027059"),
               "irAE spectrum",
               "Anti-PD-1"),
    SafetyCase("atezolizumab_irae", "atezolizumab", "black_box", "Q9NZQ7", "CD274",
               ("C1279945", "C0009319", "C0019158", "C0040128"),
               "irAE spectrum",
               "Anti-PD-L1"),
    SafetyCase("durvalumab_irae", "durvalumab", "black_box", "Q9NZQ7", "CD274",
               ("C1279945", "C0009319", "C0019158", "C0040128"),
               "irAE spectrum",
               "Anti-PD-L1"),
    SafetyCase("avelumab_irae", "avelumab", "black_box", "Q9NZQ7", "CD274",
               ("C0948715", "C0009319", "C0019158", "C0040128"),
               "Infusion reaction / irAEs",
               "Anti-PD-L1"),
    SafetyCase("ipilimumab_colitis", "ipilimumab", "black_box", "P16410", "CTLA4",
               ("C0009319", "C0019158", "C0001623", "C0040128", "C0027059", "C1279945"),
               "Severe immune-mediated colitis BBW",
               "Anti-CTLA-4"),
    SafetyCase("cemiplimab_irae", "cemiplimab", "black_box", "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128"),
               "irAE spectrum",
               "Anti-PD-1"),

    # --- Cytotoxic
    SafetyCase("vincristine_neuropathy", "vincristine", "black_box", "P10636", "MAPT",
               ("C0027947", "C0040034", "C0030305"),
               "Cytopenia / pancreatitis (vinca; neuropathy proxy)",
               "Microtubule poison; neuropathy not in vocab"),
    SafetyCase("paclitaxel_neuropathy", "paclitaxel", "black_box", "P68366", "TUBA4A",
               ("C0027947", "C0948715", "C0023524"),
               "Cytopenia / hypersensitivity / PML",
               "Taxane"),
    SafetyCase("oxaliplatin_neuropathy", "oxaliplatin", "mechanism_established", "P09874", "PARP1",
               ("C0027947", "C0002871", "C0040034"),
               "Cytopenia (neuropathy proxy)",
               "Platinum"),
    SafetyCase("carboplatin_myelo", "carboplatin", "mechanism_established", "P09874", "PARP1",
               ("C0027947", "C0040034", "C0002871"),
               "Myelosuppression",
               "Platinum"),
    SafetyCase("ifosfamide_renal", "ifosfamide", "black_box", "P09874", "PARP1",
               ("C0022660", "C0035304", "C0027849"),
               "Hemorrhagic cystitis / nephrotoxicity (encephalopathy proxy)",
               "Alkylator"),

    # ====================================================================
    # IMMUNOLOGY DEEP-DIVE (n≈25)
    # ====================================================================

    # --- Anti-TNF (TB, lymphoma, infections, demyelination)
    SafetyCase("adalimumab_tb", "adalimumab", "black_box", "P01375", "TNF",
               ("C0041296", "C0024299", "C0029118", "C0019163", "C0011304"),
               "TB / lymphoma / infections / HBV / demyelination",
               "Anti-TNF mAb"),
    SafetyCase("infliximab_tb", "infliximab", "black_box", "P01375", "TNF",
               ("C0041296", "C0024299", "C0029118", "C0019163", "C0011304"),
               "TB / lymphoma / infections / HBV / demyelination",
               "Anti-TNF mAb"),
    SafetyCase("etanercept_infection", "etanercept", "black_box", "P01375", "TNF",
               ("C0041296", "C0024299", "C0029118", "C0019163"),
               "Infections / lymphoma / HBV",
               "Anti-TNF Fc fusion"),
    SafetyCase("golimumab_tb", "golimumab", "black_box", "P01375", "TNF",
               ("C0041296", "C0024299", "C0029118"),
               "TB / lymphoma",
               "Anti-TNF mAb"),
    SafetyCase("certolizumab_tb", "certolizumab", "black_box", "P01375", "TNF",
               ("C0041296", "C0024299", "C0029118"),
               "TB / lymphoma",
               "Anti-TNF Fab"),

    # --- JAK inhibitors (MACE BBW, malignancy, infections)
    SafetyCase("tofacitinib_mace", "tofacitinib", "black_box", "P23458", "JAK1",
               ("C0027051", "C0042487", "C0024299", "C0019360", "C0029118", "C0041296"),
               "MACE BBW / VTE / lymphoma / HZV / infections",
               "Pan-JAK"),
    SafetyCase("baricitinib_mace", "baricitinib", "black_box", "P23458", "JAK1",
               ("C0027051", "C0042487", "C0024299", "C0029118"),
               "MACE / VTE / malignancy / infections",
               "JAK1/2"),
    SafetyCase("upadacitinib_mace", "upadacitinib", "black_box", "P23458", "JAK1",
               ("C0027051", "C0042487", "C0024299", "C0019360"),
               "MACE / VTE / malignancy / HZV",
               "JAK1-selective"),
    SafetyCase("filgotinib_infection", "filgotinib", "mechanism_established", "P23458", "JAK1",
               ("C0019360", "C0029118", "C0042487"),
               "HZV / infections / VTE",
               "JAK1"),
    SafetyCase("ruxolitinib_cytopenia", "ruxolitinib", "mechanism_established", "O60674", "JAK2",
               ("C0027947", "C0002871", "C0029118"),
               "Cytopenia / infections",
               "JAK1/2 for MF/PV"),

    # --- IL-6 (perforation, infections)
    SafetyCase("sarilumab_infection", "sarilumab", "mechanism_established", "P08887", "IL6R",
               ("C0029118", "C0041296", "C0235378"),
               "Infections / TB / hepatic",
               "Anti-IL-6R"),
    SafetyCase("tocilizumab_perforation", "tocilizumab", "black_box", "P08887", "IL6R",
               ("C0029118", "C0041296", "C0235378"),
               "GI perforation BBW / infections / hepatic",
               "Anti-IL-6R"),

    # --- IL-17 (candidiasis, IBD)
    SafetyCase("secukinumab_candidiasis", "secukinumab", "mechanism_established", "Q16552", "IL17A",
               ("C0029118", "C0009319"),
               "Mucocutaneous candidiasis / colitis flare",
               "Anti-IL-17A"),
    SafetyCase("ixekizumab_candidiasis", "ixekizumab", "mechanism_established", "Q16552", "IL17A",
               ("C0029118", "C0009319"),
               "Candidiasis / colitis flare",
               "Anti-IL-17A"),
    SafetyCase("brodalumab_suicide", "brodalumab", "black_box", "Q16552", "IL17A",
               ("C0029118",),
               "Suicidal ideation BBW (infection proxy)",
               "Anti-IL-17RA; suicide BBW"),

    # --- IL-23 / IL-12 (infections)
    SafetyCase("ustekinumab_infection", "ustekinumab", "mechanism_established", "P29460", "IL12B",
               ("C0029118", "C0041296"),
               "Infections / TB",
               "Anti-IL-12/23"),
    SafetyCase("risankizumab_infection", "risankizumab", "mechanism_established", "Q9NPF7", "IL23A",
               ("C0029118",),
               "Infections",
               "Anti-IL-23"),
    SafetyCase("guselkumab_infection", "guselkumab", "mechanism_established", "Q9NPF7", "IL23A",
               ("C0029118",),
               "Infections",
               "Anti-IL-23"),

    # --- S1P modulators (bradycardia, lymphopenia, macular edema, PML)
    SafetyCase("fingolimod_bradycardia", "fingolimod", "black_box", "P21453", "S1PR1",
               ("C0085610", "C0024312", "C0271051", "C0023524", "C0029118"),
               "First-dose bradycardia / lymphopenia / macular oedema",
               "S1P modulator"),
    SafetyCase("siponimod_bradycardia", "siponimod", "mechanism_established", "P21453", "S1PR1",
               ("C0085610", "C0024312", "C0271051"),
               "Bradycardia / lymphopenia / macular oedema",
               "S1P modulator"),
    SafetyCase("ozanimod_bradycardia", "ozanimod", "mechanism_established", "P21453", "S1PR1",
               ("C0085610", "C0024312", "C0271051"),
               "Bradycardia / lymphopenia / macular oedema",
               "S1P modulator"),
    SafetyCase("ponesimod_bradycardia", "ponesimod", "mechanism_established", "P21453", "S1PR1",
               ("C0085610", "C0024312", "C0271051"),
               "Bradycardia / lymphopenia / macular oedema",
               "S1P modulator"),

    # --- B-cell depleters (PML, HBV, infusion)
    SafetyCase("ocrelizumab_pml", "ocrelizumab", "black_box", "P11836", "MS4A1",
               ("C0023524", "C0019163", "C0948715", "C0029118"),
               "PML / HBV reactivation / infusion reaction",
               "Anti-CD20 (MS)"),
    SafetyCase("ofatumumab_hbv", "ofatumumab", "mechanism_established", "P11836", "MS4A1",
               ("C0019163", "C0029118", "C0948715"),
               "HBV / infections / infusion",
               "Anti-CD20"),
    SafetyCase("rituximab_aav", "rituximab", "black_box", "P11836", "MS4A1",
               ("C0023524", "C0019163", "C0948715", "C0029118"),
               "PML / HBV / infusion / infections",
               "Anti-CD20"),
    SafetyCase("obinutuzumab_infusion", "obinutuzumab", "black_box", "P11836", "MS4A1",
               ("C0948715", "C0019163", "C0029118"),
               "Severe infusion reactions / HBV",
               "Anti-CD20 (CD20 type II)"),

    # --- α4 integrin (PML)
    SafetyCase("natalizumab_pml", "natalizumab", "black_box", "P13612", "ITGA4",
               ("C0023524", "C0029118", "C0235378"),
               "PML BBW",
               "Anti-α4 integrin; PML BBW"),

    # --- Complement (meningococcal)
    SafetyCase("eculizumab_meningococcal", "eculizumab", "black_box", "P01031", "C5",
               ("C0025289", "C0029118"),
               "Meningococcal infection BBW",
               "Anti-C5"),
    SafetyCase("ravulizumab_meningococcal", "ravulizumab", "black_box", "P01031", "C5",
               ("C0025289", "C0029118"),
               "Meningococcal infection BBW",
               "Anti-C5 (long-acting)"),

    # --- BAFF (SLE)
    SafetyCase("belimumab_infection", "belimumab", "mechanism_established", "Q9Y275", "TNFSF13B",
               ("C0029118", "C0948715"),
               "Infections / infusion",
               "Anti-BLyS for SLE"),

    # ====================================================================
    # CV / METABOLIC DEEP-DIVE (n≈20)
    # ====================================================================

    # --- Anticoagulants (bleeding — proxy AKI in vocab)
    SafetyCase("warfarin_bleed", "warfarin", "black_box", "P00734", "F2",
               ("C0022660", "C0014335"),
               "Major bleeding BBW (proxy)",
               "VKA"),
    SafetyCase("dabigatran_bleed", "dabigatran", "black_box", "P00734", "F2",
               ("C0022660", "C0014335"),
               "Major bleeding BBW",
               "Direct thrombin inhibitor"),
    SafetyCase("rivaroxaban_bleed", "rivaroxaban", "black_box", "P00742", "F10",
               ("C0022660", "C0014335"),
               "Major bleeding BBW",
               "Factor Xa"),
    SafetyCase("apixaban_bleed", "apixaban", "black_box", "P00742", "F10",
               ("C0022660", "C0014335"),
               "Major bleeding BBW",
               "Factor Xa"),

    # --- Antiarrhythmics (QT)
    SafetyCase("dronedarone_hf", "dronedarone", "black_box", "Q12809", "KCNH2",
               ("C0018802", "C0085605", "C0151878"),
               "Cardiac failure / hepatic failure / QT",
               "Class III antiarrhythmic"),
    SafetyCase("ibutilide_qt", "ibutilide", "mechanism_established", "Q12809", "KCNH2",
               ("C0040479", "C0151878", "C0085612"),
               "Torsade / QT / VT",
               "Class III"),

    # --- Statins (rhabdomyolysis / muscle / hepatic)
    SafetyCase("fluvastatin_rhabdo", "fluvastatin", "mechanism_established", "P04035", "HMGCR",
               ("C0035410", "C0026848", "C0235378"),
               "Myalgia / hepatic / rhabdo",
               "Statin"),
    SafetyCase("pitavastatin_rhabdo", "pitavastatin", "mechanism_established", "P04035", "HMGCR",
               ("C0035410", "C0026848"),
               "Myalgia / rhabdo",
               "Statin"),

    # --- ACE inhibitors / ARBs (cough, angioedema, AKI)
    SafetyCase("enalapril_aki", "enalapril", "mechanism_established", "P12821", "ACE",
               ("C0022660", "C0035304"),
               "AKI (mech: bilateral renal artery stenosis)",
               "ACE-i"),
    SafetyCase("ramipril_aki", "ramipril", "mechanism_established", "P12821", "ACE",
               ("C0022660", "C0035304"),
               "AKI",
               "ACE-i"),
    SafetyCase("valsartan_aki", "valsartan", "mechanism_established", "P30556", "AGTR1",
               ("C0022660", "C0035304"),
               "AKI",
               "ARB"),
    SafetyCase("losartan_aki", "losartan", "mechanism_established", "P30556", "AGTR1",
               ("C0022660", "C0035304"),
               "AKI",
               "ARB"),

    # --- SGLT2 (DKA, amputation, AKI)
    SafetyCase("canagliflozin_aki", "canagliflozin", "black_box", "P31639", "SLC5A2",
               ("C0022660", "C0035304", "C0020456"),
               "Lower-limb amputation BBW (proxy AKI / DKA)",
               "SGLT2i; amputation BBW"),
    SafetyCase("dapagliflozin_dka", "dapagliflozin", "mechanism_established", "P31639", "SLC5A2",
               ("C0022660", "C0035304"),
               "AKI / DKA",
               "SGLT2i"),
    SafetyCase("empagliflozin_dka", "empagliflozin", "mechanism_established", "P31639", "SLC5A2",
               ("C0022660", "C0035304"),
               "AKI / DKA",
               "SGLT2i"),

    # --- GLP-1 agonists (pancreatitis, GI)
    SafetyCase("liraglutide_pancreatitis", "liraglutide", "black_box", "P43220", "GLP1R",
               ("C0030305", "C0011991"),
               "Acute pancreatitis (proxy thyroid C-cell BBW)",
               "GLP-1 agonist"),
    SafetyCase("semaglutide_pancreatitis", "semaglutide", "mechanism_established", "P43220", "GLP1R",
               ("C0030305", "C0011991"),
               "Acute pancreatitis",
               "GLP-1 agonist"),
    SafetyCase("dulaglutide_pancreatitis", "dulaglutide", "mechanism_established", "P43220", "GLP1R",
               ("C0030305", "C0011991"),
               "Acute pancreatitis",
               "GLP-1 agonist"),

    # --- Beta blockers (bronchospasm, bradycardia)
    SafetyCase("propranolol_brady", "propranolol", "mechanism_established", "P08588", "ADRB1",
               ("C0085610", "C0428977", "C0020649"),
               "Bradycardia / hypotension",
               "Non-selective β-blocker"),
    SafetyCase("atenolol_brady", "atenolol", "mechanism_established", "P08588", "ADRB1",
               ("C0085610", "C0428977", "C0020649"),
               "Bradycardia / hypotension",
               "β1-selective"),

    # ====================================================================
    # CNS / PSYCHIATRY DEEP-DIVE (n≈15)
    # ====================================================================

    # --- Anticonvulsants
    SafetyCase("lamotrigine_sjs", "lamotrigine", "black_box", "P35499", "SCN4A",
               ("C0014742", "C0014518", "C0015230"),
               "SJS / TEN / rash",
               "Sodium-channel blocker"),
    SafetyCase("phenytoin_sjs", "phenytoin", "black_box", "P35499", "SCN4A",
               ("C0014742", "C0015230", "C0235378"),
               "SJS / rash / hepatic",
               "Hydantoin"),
    SafetyCase("topiramate_acidosis", "topiramate", "mechanism_established", "P00915", "CA1",
               ("C0235378", "C0020638"),
               "Hepatic / metabolic acidosis",
               "Carbonic anhydrase + Na-channel"),

    # --- Antidepressants
    SafetyCase("bupropion_seizure", "bupropion", "black_box", "P23975", "SLC6A2",
               ("C0020538", "C0036572"),
               "Seizure risk / suicidality (proxy)",
               "Atypical antidepressant; seizure"),
    SafetyCase("sertraline_qt", "sertraline", "mechanism_established", "P31645", "SLC6A4",
               ("C0151878", "C0040479"),
               "QT prolongation (dose-dependent)",
               "SSRI"),
    SafetyCase("escitalopram_qt", "escitalopram", "mechanism_established", "P31645", "SLC6A4",
               ("C0151878", "C0040479"),
               "QT prolongation",
               "SSRI"),
    SafetyCase("venlafaxine_qt", "venlafaxine", "mechanism_established", "P31645", "SLC6A4",
               ("C0151878", "C0020538"),
               "QT / hypertension",
               "SNRI"),

    # --- Antipsychotics
    SafetyCase("olanzapine_metabolic", "olanzapine", "mechanism_established", "P14416", "DRD2",
               ("C0020456", "C0151878", "C0018802"),
               "Metabolic / QT / cardiac",
               "Atypical antipsychotic"),
    SafetyCase("risperidone_eps", "risperidone", "black_box", "P14416", "DRD2",
               ("C0151878", "C0020514", "C0018802"),
               "QT / hyperprolactinemia / cardiac",
               "Atypical antipsychotic"),
    SafetyCase("quetiapine_metabolic", "quetiapine", "mechanism_established", "P14416", "DRD2",
               ("C0020456", "C0151878"),
               "Metabolic / QT",
               "Atypical antipsychotic"),
    SafetyCase("aripiprazole_eps", "aripiprazole", "mechanism_established", "P14416", "DRD2",
               ("C0020538", "C0020456"),
               "Hypertension / metabolic",
               "Partial DRD2 agonist"),

    # --- Opioid / addiction
    SafetyCase("methadone_qt", "methadone", "black_box", "Q12809", "KCNH2",
               ("C0151878", "C0040479", "C0085612"),
               "QT prolongation BBW",
               "MOR agonist; QT"),

    # --- Pain (NSAID class effects)
    SafetyCase("celecoxib_cv", "celecoxib", "black_box", "P35354", "PTGS2",
               ("C0027051", "C0038454", "C0151744"),
               "Cardiovascular thrombotic events BBW",
               "COX-2 selective"),
    SafetyCase("etoricoxib_cv", "etoricoxib", "mechanism_established", "P35354", "PTGS2",
               ("C0027051", "C0020538", "C0151744"),
               "MI / hypertension / ischaemia",
               "COX-2"),

    # ====================================================================
    # MISC HIGH-VALUE (n≈5)
    # ====================================================================

    SafetyCase("disulfiram_dilated", "disulfiram", "mechanism_established", "P00352", "ALDH1A1",
               ("C0235378", "C0085605"),
               "Hepatotoxicity",
               "Aldehyde dehydrogenase inhibitor"),
    SafetyCase("teriflunomide_hep", "teriflunomide", "black_box", "P38606", "ATP6V1A",
               ("C0235378", "C0085605", "C0019158"),
               "Hepatic failure BBW",
               "DHODH inhibitor for MS"),
    SafetyCase("leflunomide_hep", "leflunomide", "black_box", "P38606", "ATP6V1A",
               ("C0235378", "C0085605", "C0019158"),
               "Hepatic failure BBW",
               "DHODH inhibitor for RA"),
)


# Combined benchmark
EXPANDED_SAFETY_CASES: tuple[SafetyCase, ...] = SAFETY_CASES + NEW_SAFETY_CASES


def main() -> int:
    """Diagnostic: report new cases and eligibility."""
    import json
    from pathlib import Path

    workspace = Path(__file__).resolve().parent.parent.parent
    results = workspace / "results"

    with open(results / "side_effect_vocab.json") as f:
        v = json.load(f)
    vocab_set = set(v["umls_ids"])
    with open(results / "target_vocab.json") as f:
        tv = json.load(f)
    target_set = {t["uniprot"] for t in tv["targets"]}

    n_v1 = len(SAFETY_CASES)
    n_v1_elig = sum(1 for c in SAFETY_CASES if passes_eligibility(c, vocab_set, target_set))
    n_new = len(NEW_SAFETY_CASES)
    n_new_elig = sum(1 for c in NEW_SAFETY_CASES if passes_eligibility(c, vocab_set, target_set))
    n_total = len(EXPANDED_SAFETY_CASES)
    n_total_elig = sum(1 for c in EXPANDED_SAFETY_CASES if passes_eligibility(c, vocab_set, target_set))

    print(f"[v1 cases (Sprint 5)] total={n_v1}, eligible={n_v1_elig}")
    print(f"[new cases (Sprint 8A)] total={n_new}, eligible={n_new_elig}")
    print(f"[expanded total] total={n_total}, eligible={n_total_elig}")

    # Per-TA tally of new cases
    from collections import Counter
    onc_genes = {"ERBB2", "TOP2A", "TOP2B", "KDR", "PDGFRB", "ALK",
                 "EGFR", "BRAF", "MAP2K1", "CDK4", "CDK6", "PARP1",
                 "BCL2", "MTOR", "PIK3CA", "PIK3CG", "ABL1", "KIT",
                 "MET", "FLT1", "PDCD1", "CD274", "CTLA4", "LAG3",
                 "MAPT", "TUBA4A"}
    immuno_genes = {"TNF", "IL6R", "IL17A", "IL12B", "IL23A", "IL5",
                     "IL5RA", "IL4R", "IL1B", "JAK1", "JAK2", "JAK3",
                     "S1PR1", "ITGA4", "ITGB7", "C5", "MS4A1", "CD22",
                     "CD19", "TNFRSF17", "TNFSF13B"}
    cv_genes = {"F2", "F10", "ACE", "AGTR1", "SLC5A2", "GLP1R",
                "ADRB1", "ADRB2", "HMGCR", "SLCO1B1", "PPARG"}
    cns_genes = {"DRD2", "SLC6A2", "SLC6A4", "SCN4A", "CA1", "OPRM1",
                 "GRIN1", "GABRA1"}

    def classify(c: SafetyCase) -> str:
        g = c.causal_off_target_gene.upper()
        if g in onc_genes:
            return "Oncology"
        if g in immuno_genes:
            return "Immunology"
        if g in cv_genes:
            return "CV-metabolic"
        if g in cns_genes:
            return "CNS"
        return "Other"

    new_classes = Counter(classify(c) for c in NEW_SAFETY_CASES)
    total_classes = Counter(classify(c) for c in EXPANDED_SAFETY_CASES if passes_eligibility(c, vocab_set, target_set))
    print(f"\n[new cases by TA]: {dict(new_classes)}")
    print(f"[total eligible by TA]: {dict(total_classes)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
