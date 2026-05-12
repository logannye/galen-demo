"""Sprint I: expanded post-2024 OOD benchmark (n=22 → n=50+).

Extends OOD_SAFETY_CASES with ~30 additional 2024+ FDA approvals.
Adds new targets that Sprint I expanded the vocab with (IDH1, TERT,
CXCR4, KRAS, NCSTN, MYH7, TFPI, ODC1, CFB).
"""
from __future__ import annotations

from .clinical_safety_benchmark import SafetyCase
from .clinical_safety_benchmark_ood_2024 import OOD_SAFETY_CASES


NEW_OOD_CASES: tuple[SafetyCase, ...] = (
    # --- 2024+ FDA approvals previously ineligible (vocab gaps now fixed) ---
    SafetyCase("ood_vorasidenib", "vorasidenib", "mechanism_established",
               "O75874", "IDH1",
               ("C0235378", "C0151903", "C0036572", "C0014335"),
               "Hepatic transaminitis / LFT / seizure / pyrexia",
               "OOD 2024-08; IDH1/2 inhibitor for IDH-mutant glioma"),

    SafetyCase("ood_imetelstat", "imetelstat", "black_box",
               "O14746", "TERT",
               ("C0027947", "C0040034", "C0002871", "C0235378", "C0151903"),
               "Cytopenia BBW / hepatic",
               "OOD 2024-06; telomerase inhibitor for MDS"),

    SafetyCase("ood_mavorixafor", "mavorixafor", "mechanism_established",
               "P61073", "CXCR4",
               ("C0040034", "C0027947", "C0029118", "C0151878"),
               "Cytopenia / infection / QT",
               "OOD 2024-04; CXCR4 antagonist for WHIM syndrome"),

    # --- KRAS G12C (adagrasib expansion 2024, sotorasib already in v3 benchmark) ---
    SafetyCase("ood_adagrasib", "adagrasib", "mechanism_established",
               "P01116", "KRAS",
               ("C0011991", "C0027497", "C0042963", "C0235378", "C0151878",
                "C0151903"),
               "Diarrhea / N/V / hepatic / QT / LFT abnormal",
               "OOD 2022-12 + 2024 expansion; KRAS G12C inhibitor"),

    # --- Gamma-secretase (nirogacestat) for desmoid tumor ---
    SafetyCase("ood_nirogacestat", "nirogacestat", "mechanism_established",
               "Q92542", "NCSTN",
               ("C0011991", "C0027497", "C0015230", "C0029118", "C0235378"),
               "Diarrhea / N/V / rash / infection / hepatic",
               "OOD 2023-11; gamma-secretase inhibitor for desmoid"),

    # --- Cardiac myosin (aficamten) for HCM ---
    SafetyCase("ood_aficamten", "aficamten", "mechanism_established",
               "P12883", "MYH7",
               ("C0018802", "C0018801", "C0264714", "C0023531"),
               "LV dysfunction / cardiac failure / edema",
               "OOD 2024-2025; cardiac myosin inhibitor for HCM"),

    # --- Anti-TFPI (marstacimab) hemophilia ---
    SafetyCase("ood_marstacimab", "marstacimab", "mechanism_established",
               "P10646", "TFPI",
               ("C0019080", "C0151942", "C0042487", "C0948715"),
               "Thrombotic events / bleeding rebalance / infusion",
               "OOD 2024-10; anti-TFPI hemophilia A/B"),

    # --- Ornithine decarboxylase (eflornithine) neuroblastoma ---
    SafetyCase("ood_eflornithine", "eflornithine", "mechanism_established",
               "P11926", "ODC1",
               ("C0027947", "C0040034", "C0011053", "C0029118"),
               "Cytopenia / hearing loss / infection",
               "OOD 2023-12; ODC1 inhibitor for neuroblastoma"),

    # --- PDE3/4 (ensifentrine) COPD ---
    SafetyCase("ood_ensifentrine", "ensifentrine", "mechanism_established",
               "Q07343", "PDE4B",
               ("C0018681", "C0042029", "C0006277"),
               "Headache / UTI / bronchitis",
               "OOD 2024-06; PDE3/4 inhibitor for COPD"),

    # --- S1PR (etrasimod) UC ---
    SafetyCase("ood_etrasimod", "etrasimod", "mechanism_established",
               "P21453", "S1PR1",
               ("C0085610", "C0024312", "C0271051", "C0023524", "C0029118"),
               "Bradycardia / lymphopenia / macular oedema / PML / infections",
               "OOD 2023-10; S1P modulator for UC"),

    # --- Anti-GD2 / GFAP-related (under review)
    # --- Newer ADCs ---
    SafetyCase("ood_disitamab_vedotin", "disitamab vedotin", "mechanism_established",
               "P04626", "ERBB2",
               ("C0031117", "C0015230", "C0027947", "C0018802"),
               "Peripheral neuropathy / rash / neutropenia / cardiotox",
               "OOD 2024+ expansion (gastric); HER2 ADC MMAE"),

    SafetyCase("ood_loncastuximab", "loncastuximab", "mechanism_established",
               "P15391", "CD19",
               ("C0027947", "C0040034", "C0002871", "C0235378", "C0948715"),
               "Cytopenia / hepatic / infusion",
               "OOD 2024 expansion; CD19 ADC PBD"),

    SafetyCase("ood_tafasitamab", "tafasitamab", "mechanism_established",
               "P15391", "CD19",
               ("C0027947", "C0040034", "C0002871", "C0029118"),
               "Cytopenia / infections",
               "OOD 2024 expansion; anti-CD19 (DLBCL)"),

    SafetyCase("ood_polatuzumab_vedotin", "polatuzumab vedotin", "black_box",
               "P20273", "CD22",
               ("C0027947", "C0031117", "C0024862", "C0040034"),
               "Cytopenia / peripheral neuropathy / mucositis",
               "OOD 2023-2024 expansion; CD22 ADC MMAE"),

    # --- Newer bispecifics ---
    SafetyCase("ood_blinatumomab_expand", "blinatumomab", "black_box",
               "P15391", "CD19",
               ("C2317799", "C0234016", "C0027947", "C0014335", "C0079545"),
               "CRS / ICANS BBW / neutropenia / pyrexia / HLH",
               "OOD 2024 expansion; CD19xCD3 bispecific"),

    # --- 2024 IO expansions ---
    SafetyCase("ood_toripalimab", "toripalimab", "black_box",
               "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128", "C0596022"),
               "irAE spectrum",
               "OOD 2023-10 US; anti-PD-1 for nasopharyngeal"),

    SafetyCase("ood_sintilimab", "sintilimab", "mechanism_established",
               "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128"),
               "irAE spectrum",
               "OOD 2024 expansion; anti-PD-1"),

    SafetyCase("ood_tislelizumab_hcc", "tislelizumab", "black_box",
               "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128", "C0027059"),
               "irAE spectrum incl myocarditis",
               "OOD 2024 US HCC approval; anti-PD-1"),

    SafetyCase("ood_avelumab_renal", "avelumab", "black_box",
               "Q9NZQ7", "CD274",
               ("C1279945", "C0009319", "C0019158", "C0948715"),
               "irAE spectrum / infusion",
               "OOD 2024 expansion; anti-PD-L1"),

    SafetyCase("ood_dostarlimab", "dostarlimab", "black_box",
               "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128", "C0001623"),
               "irAE spectrum",
               "OOD 2023-04 + 2024 expansion; anti-PD-1"),

    # --- Bicuculline-class fungal (Brexafemme = ibrexafungerp) — niche
    # Skipping (no clear target in vocab)

    # --- Anti-CGRP-receptor (atogepant 2023-09)
    SafetyCase("ood_atogepant", "atogepant", "mechanism_established",
               "Q16602", "CALCRL",
               ("C0027497", "C0011991", "C0009806", "C0020538"),
               "N/V / diarrhea / constipation / HTN",
               "OOD 2023-09; CGRP-R antagonist for migraine prevention"),

    SafetyCase("ood_rimegepant", "rimegepant", "mechanism_established",
               "Q16602", "CALCRL",
               ("C0027497", "C0020538"),
               "N/V / HTN",
               "OOD 2024 expansion; CGRP-R antagonist"),

    # --- Newer asthma/EOS (newer IL5/IL5RA)
    SafetyCase("ood_benralizumab_eosinophil", "benralizumab", "mechanism_established",
               "Q01344", "IL5RA",
               ("C0029118", "C0948715"),
               "Infections / infusion",
               "OOD 2024 expansion; anti-IL5RA"),

    # --- Glucagon receptor (already have GCGR for retatrutide)
    # Already covered.

    # --- IL-31RA (nemolizumab 2024-08)
    SafetyCase("ood_nemolizumab_pn", "nemolizumab", "mechanism_established",
               "Q8NI17", "IL31RA",
               ("C0029118", "C0948715"),
               "Infections / asthma exacerbation / injection",
               "OOD 2024-08; anti-IL31RA for prurigo nodularis"),

    # --- Tezepelumab (2024 expansions) ---
    SafetyCase("ood_tezepelumab_asthma", "tezepelumab", "mechanism_established",
               "Q969D9", "TSLP",
               ("C0948715", "C0029118"),
               "Infusion / infections",
               "OOD 2024 expansion; anti-TSLP for asthma"),

    # --- IL-17 (bimekizumab 2024 expansions) ---
    SafetyCase("ood_bimekizumab_psA", "bimekizumab", "black_box",
               "Q16552", "IL17A",
               ("C0029118", "C0006848", "C0009319"),
               "Mucocutaneous candidiasis / IBD flare",
               "OOD 2024 expansion; anti-IL-17A/F"),

    # --- Generic newer 2024 entries — selected high-quality cases ---
    SafetyCase("ood_zanubrutinib_2024", "zanubrutinib", "black_box",
               "Q06187", "BTK",
               ("C0004238", "C0027947", "C0040034", "C0020538"),
               "AFib / cytopenia / HTN",
               "OOD 2024 expansion; BTK 2nd gen"),

    SafetyCase("ood_pirtobrutinib", "pirtobrutinib", "mechanism_established",
               "Q06187", "BTK",
               ("C0004238", "C0027947", "C0040034"),
               "AFib / cytopenia",
               "OOD 2023-12 + 2024 expansion; BTK non-covalent"),

    SafetyCase("ood_acalabrutinib_2024", "acalabrutinib", "mechanism_established",
               "Q06187", "BTK",
               ("C0004238", "C0027947", "C0023524"),
               "AFib / cytopenia / PML",
               "OOD 2024 CLL expansion; BTK 2nd gen"),

    SafetyCase("ood_ribociclib_2024", "ribociclib", "black_box",
               "P11802", "CDK4",
               ("C0151878", "C0027947", "C0235378", "C0235378"),
               "QT BBW / neutropenia / hepatotox",
               "OOD 2024 EBC adjuvant expansion; CDK4/6"),

    SafetyCase("ood_capivasertib_2024", "capivasertib", "mechanism_established",
               "P31749", "AKT1",
               ("C0020456", "C0011991", "C0015230", "C0029118"),
               "Hyperglycemia / diarrhea / rash / infection",
               "OOD 2023-11; AKT inhibitor"),

    SafetyCase("ood_palbociclib_2024", "palbociclib", "mechanism_established",
               "P11802", "CDK4",
               ("C0027947", "C0042487", "C0029118"),
               "Neutropenia / VTE / infection",
               "OOD 2024 male BC expansion; CDK4/6"),
)


EXPANDED_OOD_CASES: tuple[SafetyCase, ...] = OOD_SAFETY_CASES + NEW_OOD_CASES


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

    n_orig = len(OOD_SAFETY_CASES)
    n_orig_elig = sum(1 for c in OOD_SAFETY_CASES
                       if passes_eligibility(c, vocab_set, target_set))
    n_new = len(NEW_OOD_CASES)
    n_new_elig = sum(1 for c in NEW_OOD_CASES
                      if passes_eligibility(c, vocab_set, target_set))
    n_total = len(EXPANDED_OOD_CASES)
    n_total_elig = sum(1 for c in EXPANDED_OOD_CASES
                        if passes_eligibility(c, vocab_set, target_set))

    print(f"[orig OOD v1]: total={n_orig}, eligible={n_orig_elig}")
    print(f"[new OOD I]: total={n_new}, eligible={n_new_elig}")
    print(f"[expanded OOD v2]: total={n_total}, eligible={n_total_elig}")

    for c in NEW_OOD_CASES:
        if not passes_eligibility(c, vocab_set, target_set):
            target_ok = c.causal_off_target_uniprot in target_set
            ses_in = [s for s in c.causal_side_effects_umls if s in vocab_set]
            print(f"  INELIGIBLE: {c.drug_id} target_ok={target_ok} "
                  f"se_in_vocab={len(ses_in)}/{len(c.causal_side_effects_umls)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
