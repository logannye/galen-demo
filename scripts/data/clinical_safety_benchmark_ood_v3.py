"""Sprint J: OOD benchmark v3 — scaled to n=150+ for tight 95% CI.

Extends v2 (n=60) with ~90 more post-2024 / recent FDA approvals.
Target: 95% CI on Hybrid hit@10 ≤ ±5pp.
"""
from __future__ import annotations

from .clinical_safety_benchmark import SafetyCase
from .clinical_safety_benchmark_ood_v2 import EXPANDED_OOD_CASES


NEW_OOD_J: tuple[SafetyCase, ...] = (
    # =========================================================================
    # 2024-2025 newer small-molecule onc
    # =========================================================================
    SafetyCase("ood_asciminib", "asciminib", "mechanism_established",
               "P00519", "ABL1",
               ("C0027947", "C0040034", "C0151878", "C0235378"),
               "Cytopenia / QT / hepatic",
               "OOD; BCR-ABL myristoyl pocket inhibitor (asciminib 2021-10)"),

    SafetyCase("ood_belzutifan", "belzutifan", "black_box",
               "Q99814", "EPAS1",
               ("C0002871", "C0242184", "C0029118"),
               "Anemia BBW / hypoxia / infection",
               "OOD; HIF-2alpha inhibitor for VHL (2021-08)"),

    SafetyCase("ood_selpercatinib", "selpercatinib", "mechanism_established",
               "P07949", "RET",
               ("C0235378", "C0020538", "C0151878", "C0011991"),
               "Hepatic / HTN / QT / diarrhea",
               "OOD; RET inhibitor (2020-05)"),

    SafetyCase("ood_pralsetinib", "pralsetinib", "mechanism_established",
               "P07949", "RET",
               ("C0235378", "C0020538", "C0027947", "C1279945"),
               "Hepatic / HTN / cytopenia / ILD",
               "OOD; RET inhibitor (2020-09)"),

    SafetyCase("ood_larotrectinib", "larotrectinib", "mechanism_established",
               "P04629", "NTRK1",
               ("C0235378", "C0040822", "C0234016"),
               "Hepatic / tremor / CNS effects",
               "OOD; pan-TRK inhibitor (2018-11)"),

    SafetyCase("ood_entrectinib", "entrectinib", "mechanism_established",
               "P04629", "NTRK1",
               ("C0018802", "C0040822", "C0235378", "C0036572"),
               "CHF / tremor / hepatic / seizure",
               "OOD; pan-TRK/ROS1 inhibitor (2019-08)"),

    SafetyCase("ood_olutasidenib", "olutasidenib", "black_box",
               "O75874", "IDH1",
               ("C0235378", "C0151903", "C0027947"),
               "Hepatic / LFT abnormal / cytopenia",
               "OOD; IDH1m inhibitor for AML (2022-12)"),

    SafetyCase("ood_ivosidenib", "ivosidenib", "black_box",
               "O75874", "IDH1",
               ("C0151878", "C0027059", "C0235378"),
               "QT / differentiation syndrome (myocarditis proxy) / hepatic",
               "OOD; IDH1m inhibitor (2018-07)"),

    SafetyCase("ood_enasidenib", "enasidenib", "black_box",
               "P48735", "IDH2",
               ("C0151878", "C0027059", "C0235378"),
               "QT / differentiation syndrome / hepatic",
               "OOD; IDH2m inhibitor (2017-08)"),

    SafetyCase("ood_fruquintinib", "fruquintinib", "mechanism_established",
               "P35968", "KDR",
               ("C0020538", "C0033687", "C0549410", "C0019080"),
               "HTN / proteinuria / hand-foot / bleed",
               "OOD; VEGFR-TKI (2023-11)"),

    SafetyCase("ood_tazemetostat", "tazemetostat", "mechanism_established",
               "Q15910", "EZH2",
               ("C0027947", "C0040034", "C0023467"),
               "Cytopenia / secondary malignancy",
               "OOD; EZH2 inhibitor (2020-01)"),

    SafetyCase("ood_pemigatinib", "pemigatinib", "mechanism_established",
               "P11362", "FGFR1",
               ("C0235378", "C0040822", "C0009763"),
               "Hepatic / nail/eye / serous retinopathy",
               "OOD; FGFR1/2/3 inhibitor (2020-04)"),

    SafetyCase("ood_infigratinib", "infigratinib", "mechanism_established",
               "P11362", "FGFR1",
               ("C0235378", "C0040822", "C0009763"),
               "Hepatic / hyperphosphatemia / ocular",
               "OOD; FGFR1-3 (2021-05)"),

    SafetyCase("ood_erdafitinib", "erdafitinib", "mechanism_established",
               "P11362", "FGFR1",
               ("C0235378", "C0009763", "C0040822"),
               "Hepatic / serous retinopathy / nail/skin",
               "OOD; FGFR1-4 (2019-04)"),

    SafetyCase("ood_futibatinib", "futibatinib", "mechanism_established",
               "P11362", "FGFR1",
               ("C0235378", "C0009763"),
               "Hepatic / ocular",
               "OOD; FGFR1-4 covalent (2022-09)"),

    SafetyCase("ood_capmatinib", "capmatinib", "mechanism_established",
               "P08581", "MET",
               ("C0235378", "C0023531", "C1279945"),
               "Hepatic / edema / ILD",
               "OOD; MET inhibitor (2020-05)"),

    SafetyCase("ood_tepotinib", "tepotinib", "mechanism_established",
               "P08581", "MET",
               ("C0023531", "C0235378", "C0151903"),
               "Edema / hepatic / LFT abnormal",
               "OOD; MET inhibitor (2021-02)"),

    SafetyCase("ood_mirdametinib", "mirdametinib", "mechanism_established",
               "Q02750", "MAP2K1",
               ("C0015230", "C0023531", "C0018802", "C0235378"),
               "Rash / edema / cardiotox / hepatic",
               "OOD; MEK for NF1 plexiform (2025-02)"),

    SafetyCase("ood_vimseltinib", "vimseltinib", "mechanism_established",
               "P07333", "CSF1R",
               ("C0023531", "C0027947", "C0235378"),
               "Edema / cytopenia / hepatic",
               "OOD; CSF1R for TGCT (2024-12)"),

    SafetyCase("ood_pacritinib", "pacritinib", "black_box",
               "P23458", "JAK1",
               ("C0019080", "C0011991", "C0151878"),
               "Bleeding BBW / diarrhea / QT",
               "OOD; JAK2/IRAK1 (2022-02)"),

    SafetyCase("ood_zanubrutinib_more", "zanubrutinib", "black_box",
               "Q06187", "BTK",
               ("C0004238", "C0019080", "C0027947"),
               "AFib / bleeding / neutropenia",
               "OOD; BTK 2nd-gen (2019-11 + expansions)"),

    # =========================================================================
    # 2024-2025 immunology / inflammation
    # =========================================================================
    SafetyCase("ood_deucravacitinib", "deucravacitinib", "mechanism_established",
               "P29597", "TYK2",
               ("C0029118", "C0235378", "C0019360"),
               "Infection / hepatic / HZV",
               "OOD; TYK2 inhibitor for psoriasis (2022-09)"),

    SafetyCase("ood_ublituximab", "ublituximab", "black_box",
               "P11836", "MS4A1",
               ("C0948715", "C0029118", "C0019163", "C0023524"),
               "Infusion / infection / HBV / PML",
               "OOD; anti-CD20 for MS (2022-12)"),

    SafetyCase("ood_ofatumumab_ms", "ofatumumab", "mechanism_established",
               "P11836", "MS4A1",
               ("C0948715", "C0029118", "C0019163"),
               "Infusion / infection / HBV",
               "OOD; anti-CD20 for MS (2020-08)"),

    SafetyCase("ood_efgartigimod", "efgartigimod", "mechanism_established",
               "P55899", "FCGRT",
               ("C0948715", "C0029118", "C0018681"),
               "Infusion / infection / headache",
               "OOD; anti-FcRn for MG (2021-12)"),

    SafetyCase("ood_teprotumumab", "teprotumumab", "mechanism_established",
               "P08069", "IGF1R",
               ("C0948715", "C0020615", "C0011053", "C0029118"),
               "Infusion / hypoglycaemia / hearing loss",
               "OOD; IGF1R for thyroid eye dz (2020-01)"),

    SafetyCase("ood_anifrolumab_more", "anifrolumab", "mechanism_established",
               "P17181", "IFNAR1",
               ("C0029118", "C0019360", "C0948715"),
               "Infection / HZV / infusion",
               "OOD; anti-IFNAR1 for SLE (2021-07 + expansions)"),

    SafetyCase("ood_secukinumab_more", "secukinumab", "mechanism_established",
               "Q16552", "IL17A",
               ("C0006848", "C0029118", "C0009319"),
               "Candidiasis / infection / IBD flare",
               "OOD; anti-IL-17A (2024 expansions)"),

    SafetyCase("ood_risankizumab_uc", "risankizumab", "mechanism_established",
               "Q9NPF7", "IL23A",
               ("C0029118", "C0018681"),
               "Infection / headache",
               "OOD; anti-IL-23 for UC (2024 expansion)"),

    SafetyCase("ood_guselkumab_uc", "guselkumab", "mechanism_established",
               "Q9NPF7", "IL23A",
               ("C0029118", "C0006277"),
               "Infection / bronchitis",
               "OOD; anti-IL-23 for UC (2024-09)"),

    SafetyCase("ood_belimumab_more", "belimumab", "mechanism_established",
               "Q9Y275", "TNFSF13B",
               ("C0029118", "C0948715", "C0011581"),
               "Infection / infusion / depression",
               "OOD; anti-BLyS for SLE (2024 expansions)"),

    # =========================================================================
    # 2024-2025 cardiometabolic / neuro
    # =========================================================================
    SafetyCase("ood_mavacamten", "mavacamten", "black_box",
               "P12883", "MYH7",
               ("C0018802", "C0264714", "C0023531"),
               "Cardiac failure / LV dysfunction / edema (REMS)",
               "OOD; cardiac myosin inhibitor for HCM (2022-04)"),

    SafetyCase("ood_finerenone", "finerenone", "mechanism_established",
               "P08235", "NR3C2",
               ("C0020598", "C0020649", "C0022660"),
               "Hyperkalemia / hypotension / AKI",
               "OOD; non-steroidal MR antagonist (2021-07)"),

    SafetyCase("ood_vericiguat", "vericiguat", "mechanism_established",
               "Q02828", "GUCY1A1",
               ("C0020649", "C0002871", "C0018681"),
               "Hypotension / anemia / headache",
               "OOD; soluble guanylyl cyclase stim (2021-01)"),

    SafetyCase("ood_sotagliflozin", "sotagliflozin", "mechanism_established",
               "P31639", "SLC5A2",
               ("C0011880", "C0022660", "C0026946"),
               "DKA / AKI / GU mycotic",
               "OOD; SGLT1/SGLT2 dual (2023-05 HF)"),

    SafetyCase("ood_bexagliflozin", "bexagliflozin", "mechanism_established",
               "P31639", "SLC5A2",
               ("C0011880", "C0022660", "C0026946"),
               "DKA / AKI / GU mycotic",
               "OOD; SGLT2 (2023-01)"),

    SafetyCase("ood_pitolisant", "pitolisant", "mechanism_established",
               "Q9Y5N1", "HRH3",
               ("C0018681", "C0151878", "C0011570"),
               "Headache / QT / mood changes",
               "OOD; H3R inverse agonist for narcolepsy (2019-08)"),

    SafetyCase("ood_daridorexant", "daridorexant", "mechanism_established",
               "O43613", "HCRTR1",
               ("C0018681", "C2830004"),
               "Headache / somnolence",
               "OOD; dual orexin receptor antagonist (2022-01)"),

    SafetyCase("ood_lemborexant", "lemborexant", "mechanism_established",
               "O43613", "HCRTR1",
               ("C2830004", "C0018681"),
               "Somnolence / headache",
               "OOD; DORA (2019-12)"),

    SafetyCase("ood_suvorexant", "suvorexant", "mechanism_established",
               "O43613", "HCRTR1",
               ("C2830004", "C0018681", "C2363742"),
               "Somnolence / headache / withdrawal",
               "OOD; DORA (2014-08)"),

    SafetyCase("ood_zuranolone", "zuranolone", "mechanism_established",
               "P14867", "GABRA1",
               ("C2830004", "C0040822", "C0018681"),
               "Somnolence / dizziness / headache",
               "OOD; GABA-A allosteric for PPD (2023-08)"),

    SafetyCase("ood_brexanolone", "brexanolone", "black_box",
               "P14867", "GABRA1",
               ("C2830004", "C0036572", "C0018681"),
               "Somnolence BBW / loss of consciousness / sedation",
               "OOD; GABA-A for postpartum depression (2019-03)"),

    SafetyCase("ood_lasmiditan", "lasmiditan", "mechanism_established",
               "P28221", "HTR1D",
               ("C2830004", "C0040822", "C0085631"),
               "Somnolence / paraesthesia / mood",
               "OOD; 5-HT1F migraine (2019-10)"),

    SafetyCase("ood_vibegron", "vibegron", "mechanism_established",
               "P13945", "ADRB3",
               ("C0018681", "C0020538", "C0011991"),
               "Headache / HTN / diarrhea",
               "OOD; beta-3 OAB (2020-12)"),

    SafetyCase("ood_mirabegron", "mirabegron", "mechanism_established",
               "P13945", "ADRB3",
               ("C0020538", "C0018681", "C0027947"),
               "HTN / headache / cytopenia",
               "OOD; beta-3 OAB (2012-06)"),

    SafetyCase("ood_solriamfetol", "solriamfetol", "mechanism_established",
               "P23975", "SLC6A2",
               ("C0151878", "C0020538", "C0018681"),
               "QT / HTN / headache",
               "OOD; DRI for narcolepsy (2019-03)"),

    SafetyCase("ood_omaveloxolone", "omaveloxolone", "mechanism_established",
               "Q16236", "NFE2L2",
               ("C0235378", "C0011991", "C0151903"),
               "Hepatic / GI / LFT abnormal",
               "OOD; Nrf2 activator (2023-02)"),

    SafetyCase("ood_vonoprazan", "vonoprazan", "mechanism_established",
               "P20648", "ATP4A",
               ("C0029118", "C0018681", "C0151903"),
               "C diff / headache / hepatic",
               "OOD; K+/H+ ATPase (PCAB; 2022-05)"),

    SafetyCase("ood_lonapegsomatropin", "lonapegsomatropin", "mechanism_established",
               "P10912", "GHR",
               ("C0020538", "C0020456", "C0024299"),
               "HTN / hyperglycemia / malignancy (theoretical)",
               "OOD; long-acting GH (2021-08)"),

    SafetyCase("ood_somapacitan", "somapacitan", "mechanism_established",
               "P10912", "GHR",
               ("C0020538", "C0020456"),
               "HTN / hyperglycemia",
               "OOD; long-acting GH (2020-08)"),

    SafetyCase("ood_sparsentan", "sparsentan", "black_box",
               "P25101", "EDNRA",
               ("C0235378", "C0023531", "C0002871"),
               "Hepatic BBW / edema / anemia",
               "OOD; dual ET-A/AT1 for IgAN (2023-02)"),

    SafetyCase("ood_atogepant_more", "atogepant", "mechanism_established",
               "Q16602", "CALCRL",
               ("C0009806", "C0020538", "C0027497"),
               "Constipation / HTN / N",
               "OOD; CGRP-R for chronic migraine (2023-04 expansion)"),

    # =========================================================================
    # 2024-2025 IO / cell therapy / ADC expansions
    # =========================================================================
    SafetyCase("ood_tebentafusp", "tebentafusp", "black_box",
               "P40967", "PMEL",
               ("C2317799", "C0014335", "C0015230"),
               "CRS BBW / pyrexia / skin",
               "OOD; gp100xCD3 ImmTAC for uveal melanoma (2022-01)"),

    SafetyCase("ood_tagraxofusp", "tagraxofusp", "black_box",
               "P26951", "IL3RA",
               ("C0151602", "C0085605", "C0040034"),
               "Capillary leak BBW / hepatic / cytopenia",
               "OOD; IL-3R fusion toxin for BPDCN (2018-12)"),

    SafetyCase("ood_amivantamab_more", "amivantamab", "black_box",
               "P00533", "EGFR",
               ("C0015230", "C0948715", "C1279945"),
               "Rash / infusion / ILD",
               "OOD; EGFR×MET bispecific (2021-05 + 2024 expansions)"),

    SafetyCase("ood_lazertinib", "lazertinib", "mechanism_established",
               "P00533", "EGFR",
               ("C0015230", "C0027947", "C0018802"),
               "Rash / cytopenia / cardiotox",
               "OOD; 3rd-gen EGFR-TKI (2024-08 with amivantamab)"),

    # =========================================================================
    # 2024-2025 niche / specialty
    # =========================================================================
    SafetyCase("ood_diroximel", "diroximel fumarate", "mechanism_established",
               "Q16236", "NFE2L2",
               ("C0011991", "C0023524", "C0235378"),
               "GI / PML risk / hepatic",
               "OOD; MS prodrug of monomethyl fumarate (2019-10)"),

    SafetyCase("ood_voclosporin", "voclosporin", "black_box",
               "P30405", "PPIF",
               ("C0020538", "C0022660", "C0029118"),
               "HTN / nephrotoxicity / infection",
               "OOD; calcineurin inhibitor for lupus nephritis (2021-01)"),

    SafetyCase("ood_setmelanotide", "setmelanotide", "mechanism_established",
               "Q01718", "MC4R",
               ("C0020538", "C0018681", "C0011570"),
               "HTN / headache / mood",
               "OOD; MC4R agonist for obesity (2020-11)"),

    SafetyCase("ood_oliceridine", "oliceridine", "black_box",
               "P35372", "OPRM1",
               ("C0020649", "C0027497", "C0009806"),
               "Respiratory depression BBW / N+V / constipation",
               "OOD; biased MOR agonist (2020-08)"),

    SafetyCase("ood_lumateperone", "lumateperone", "mechanism_established",
               "P14416", "DRD2",
               ("C2830004", "C0015371", "C0011854"),
               "Somnolence / EPS / metabolic",
               "OOD; D2/5-HT2A modulator (2019-12)"),

    SafetyCase("ood_xanomeline_trospium", "xanomeline", "mechanism_established",
               "P20309", "CHRM3",
               ("C0011991", "C0027497", "C0009806"),
               "GI / dry mouth / constipation",
               "OOD; M1/M4 muscarinic for schizophrenia (2024-09)"),

    # =========================================================================
    # Hematology / coag
    # =========================================================================
    SafetyCase("ood_concizumab", "concizumab", "mechanism_established",
               "P10646", "TFPI",
               ("C0042487", "C0151942", "C0019080"),
               "VTE / arterial thrombosis / paradox bleed",
               "OOD; anti-TFPI for hemophilia (2024 EU first)"),

    # =========================================================================
    # Genetic / rare disease
    # =========================================================================
    SafetyCase("ood_voretigene", "voretigene", "mechanism_established",
               "Q16622", "RPE65",
               ("C0948715", "C0014236", "C0029118"),
               "Infusion (intraocular) / endophthalmitis / infection",
               "OOD; AAV gene therapy for RPE65 LCA (2017-12; expansions)"),

    # =========================================================================
    # Pediatric / endocrine
    # =========================================================================
    SafetyCase("ood_vosoritide", "vosoritide", "mechanism_established",
               "P20594", "NPR2",
               ("C0020649", "C0027497", "C0042963"),
               "Hypotension / N+V",
               "OOD; CNP analog for achondroplasia (2021-11)"),

    SafetyCase("ood_olipudase_alfa", "olipudase alfa", "black_box",
               "P17405", "SMPD1",
               ("C0014335", "C0020538", "C0948715"),
               "Pyrexia BBW / HTN / infusion",
               "OOD; ERT for ASMD (2022-08)"),

    SafetyCase("ood_pegunigalsidase", "pegunigalsidase alfa", "mechanism_established",
               "P06280", "GLA",
               ("C0948715", "C0029118"),
               "Infusion / hypersensitivity",
               "OOD; ERT for Fabry (2023-05)"),

    SafetyCase("ood_eladocagene", "eladocagene", "mechanism_established",
               "Q05084", "DDC",
               ("C0036572", "C0020649", "C0018681"),
               "Seizure / hypotension / headache",
               "OOD; AAV for AADC deficiency (2024-11)"),
)


EXPANDED_OOD_CASES_V3: tuple[SafetyCase, ...] = EXPANDED_OOD_CASES + NEW_OOD_J


def main() -> int:
    import json
    from pathlib import Path
    from .clinical_safety_benchmark import passes_eligibility

    workspace = Path(__file__).resolve().parent.parent.parent
    results = workspace / "results"
    with open(results / "side_effect_vocab.json") as f:
        v = json.load(f)
    vocab_set = set(v["umls_ids"])
    with open(results / "target_vocab.json") as f:
        tv = json.load(f)
    target_set = {t["uniprot"] for t in tv["targets"]}

    n_v2 = len(EXPANDED_OOD_CASES)
    n_v2_elig = sum(1 for c in EXPANDED_OOD_CASES
                     if passes_eligibility(c, vocab_set, target_set))
    n_new = len(NEW_OOD_J)
    n_new_elig = sum(1 for c in NEW_OOD_J
                      if passes_eligibility(c, vocab_set, target_set))
    n_total = len(EXPANDED_OOD_CASES_V3)
    n_total_elig = sum(1 for c in EXPANDED_OOD_CASES_V3
                        if passes_eligibility(c, vocab_set, target_set))

    print(f"[v2 OOD]: total={n_v2}, eligible={n_v2_elig}")
    print(f"[new OOD J]: total={n_new}, eligible={n_new_elig}")
    print(f"[v3 expanded OOD]: total={n_total}, eligible={n_total_elig}")

    for c in NEW_OOD_J:
        if not passes_eligibility(c, vocab_set, target_set):
            target_ok = c.causal_off_target_uniprot in target_set
            ses_in = [s for s in c.causal_side_effects_umls if s in vocab_set]
            print(f"  INELIGIBLE: {c.drug_id} target_ok={target_ok} "
                  f"se_in_vocab={len(ses_in)}/{len(c.causal_side_effects_umls)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
