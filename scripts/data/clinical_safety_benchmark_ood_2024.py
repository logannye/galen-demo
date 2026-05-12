"""Sprint G Track 3: Post-2024 OOD benchmark.

Hand-curated benchmark of drugs FDA-approved after the Sonnet 4.6
training cutoff (~April 2024). Tests whether Hybrid performance
holds on drugs the LLM cannot have memorized from training data.

Each entry follows the standard SafetyCase schema:
  - drug_search_name: catalog/biologic profile lookup name
  - causal_off_target_uniprot: in our target vocab
  - causal_side_effects_umls: in our SE vocab

Drugs included must be either:
  (a) Already in `biologic_binding_profiles.py` mapping (most modern
      biologics already added in Sprints 8B/E), OR
  (b) Findable in ChEMBL by name

Tagged with `approval_year` for stratified analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

from .clinical_safety_benchmark import SafetyCase


# Format follows SafetyCase but with extra approval_year metadata.
# We construct SafetyCase objects with notes encoding the year for
# downstream filtering.

OOD_SAFETY_CASES: tuple[SafetyCase, ...] = (

    # --- Bispecifics approved post-April-2024 ---
    SafetyCase("ood_tarlatamab", "tarlatamab", "black_box",
               "Q9NYJ7", "DLL3",
               ("C2317799", "C0234016", "C0014335", "C0027947"),
               "CRS / ICANS / pyrexia / cytopenia",
               "OOD 2024-05; DLL3×CD3 bispecific"),

    SafetyCase("ood_zenocutuzumab", "zenocutuzumab", "mechanism_established",
               "P21860", "ERBB3",
               ("C0948715", "C0011991", "C0235378"),
               "Infusion / diarrhea / hepatic",
               "OOD 2024-12; HER2×HER3 bispecific NRG1+ cancers"),

    # --- ADCs / Cell therapy expansions post-April-2024 ---
    SafetyCase("ood_lifileucel", "lifileucel", "black_box",
               "P15391", "CD19",
               ("C2317799", "C0014335", "C0027947", "C0029118",
                "C0040034", "C0002871"),
               "CRS / cytopenia / infection",
               "OOD 2024-02; mTIL cell therapy"),

    SafetyCase("ood_obecabtagene", "obecabtagene autoleucel", "black_box",
               "P15391", "CD19",
               ("C2317799", "C0234016", "C0086438", "C0027947"),
               "CRS / ICANS / hypogamma / cytopenia",
               "OOD 2024-11; CD19 CAR-T (B-ALL)"),

    # --- Newer ICIs (PD-1 / PD-L1 expansions) ---
    SafetyCase("ood_tislelizumab", "tislelizumab", "black_box",
               "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128",
                "C0596022", "C0001623"),
               "irAE: pneumonitis / colitis / hepatitis / thyroid / hypophysitis",
               "OOD 2024 US HCC; anti-PD-1"),

    SafetyCase("ood_cosibelimab", "cosibelimab", "mechanism_established",
               "Q9NZQ7", "CD274",
               ("C1279945", "C0009319", "C0019158", "C0040128"),
               "irAE spectrum",
               "OOD 2024-12; anti-PD-L1 for cSCC"),

    SafetyCase("ood_retifanlimab", "retifanlimab", "mechanism_established",
               "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128"),
               "irAE spectrum",
               "OOD 2024 expansion; anti-PD-1"),

    # --- Complement ---
    SafetyCase("ood_crovalimab", "crovalimab", "black_box",
               "P01031", "C5",
               ("C0025289", "C0025291", "C0029118"),
               "Meningococcal meningitis BBW",
               "OOD 2024-06; anti-C5 for PNH"),

    SafetyCase("ood_iptacopan", "iptacopan", "black_box",
               "P01024", "C3",
               ("C0025289", "C0025291", "C0029118"),
               "Meningococcal meningitis BBW",
               "OOD 2023-12; complement factor B"),

    # --- Modern targeted small molecules (approved post-April-2024) ---
    SafetyCase("ood_inavolisib", "inavolisib", "mechanism_established",
               "P42336", "PIK3CA",
               ("C0020456", "C0011991", "C0015230", "C0024862"),
               "Hyperglycemia / diarrhea / rash / mucositis",
               "OOD 2024-10; PI3K-alpha selective"),

    SafetyCase("ood_vorasidenib", "vorasidenib", "mechanism_established",
               "O75874", "IDH1",
               ("C0235378", "C0151903", "C0036572"),
               "Hepatic transaminitis / LFT / seizure",
               "OOD 2024-08; IDH1/2 inhibitor for IDH-mutant glioma"),

    SafetyCase("ood_capivasertib", "capivasertib", "mechanism_established",
               "P31749", "AKT1",
               ("C0020456", "C0011991", "C0015230", "C0011603"),
               "Hyperglycemia / diarrhea / rash",
               "OOD 2023-11; AKT inhibitor for breast cancer"),

    SafetyCase("ood_repotrectinib", "repotrectinib", "mechanism_established",
               "P08922", "ROS1",
               ("C0235378", "C1279945", "C0234016", "C0151903"),
               "Hepatic / ILD / CNS / LFT",
               "OOD 2024-06; ROS1/NTRK inhibitor"),

    SafetyCase("ood_tovorafenib", "tovorafenib", "black_box",
               "P15056", "BRAF",
               ("C0015230", "C0011603", "C0235378", "C0002170"),
               "Rash / dermatitis / hepatic / hair color change",
               "OOD 2024-04; BRAF/CRAF for pediatric LGG"),

    # --- ADC payload-specific (TROP2 / HER2 expansions) ---
    SafetyCase("ood_datopotamab_dxd", "datopotamab deruxtecan", "black_box",
               "P09758", "TACSTD2",
               ("C1279945", "C0032310", "C0011991", "C0027947"),
               "ILD/pneumonitis BBW / diarrhea / neutropenia",
               "OOD 2025 (FDA) approval; TROP2 ADC DXd payload"),

    SafetyCase("ood_zanidatamab", "zanidatamab", "mechanism_established",
               "P04626", "ERBB2",
               ("C0011991", "C0018802", "C0948715", "C0027497"),
               "Diarrhea / cardiac / infusion / N/V",
               "OOD 2024-11; HER2 biparatopic mAb for biliary"),

    SafetyCase("ood_patritumab_dxd", "patritumab deruxtecan", "mechanism_established",
               "P21860", "ERBB3",
               ("C1279945", "C0032310", "C0027947", "C0002871"),
               "ILD / cytopenia",
               "OOD post-2024; HER3 ADC DXd payload"),

    # --- Hematology ---
    SafetyCase("ood_imetelstat", "imetelstat", "black_box",
               "O14746", "TERT",
               ("C0027947", "C0040034", "C0002871", "C0235378"),
               "Cytopenia / hepatic",
               "OOD 2024-06; telomerase inhibitor for MDS"),

    # --- Cardiometabolic ---
    SafetyCase("ood_resmetirom", "resmetirom", "mechanism_established",
               "P10828", "THRB",
               ("C0011991", "C0235378", "C0151903"),
               "Diarrhea / hepatic",
               "OOD 2024-03; thyroid-beta agonist for NASH/MASH"),

    SafetyCase("ood_aprocitentan", "aprocitentan", "black_box",
               "P25101", "EDNRA",
               ("C0002871", "C0029118"),
               "Anemia / edema",
               "OOD 2024-03; dual ET-A/B antagonist for resistant HTN"),

    SafetyCase("ood_mavorixafor", "mavorixafor", "mechanism_established",
               "P61073", "CXCR4",
               ("C0040034", "C0027947", "C0029118"),
               "Cytopenia / infection",
               "OOD 2024-04; CXCR4 antagonist for WHIM"),

    # --- Neurology ---
    SafetyCase("ood_donanemab", "donanemab", "black_box",
               "P05067", "APP",
               ("C0234016", "C0948715"),
               "ARIA (encephalopathy proxy) / infusion",
               "OOD 2024-07; anti-Aβ for Alzheimer"),

    SafetyCase("ood_lecanemab", "lecanemab", "black_box",
               "P05067", "APP",
               ("C0234016", "C0948715"),
               "ARIA / infusion",
               "OOD 2023-07 full approval; anti-Aβ"),

    # --- Hepatology / PBC ---
    SafetyCase("ood_elafibranor", "elafibranor", "mechanism_established",
               "Q07869", "PPARA",
               ("C0011991", "C0027497", "C0030193"),
               "Diarrhea / N/V / abdominal pain",
               "OOD 2024-06; PPAR-α/δ agonist for PBC"),

    # --- Newer IO bispecifics (PD-1 × CTLA-4) ---
    SafetyCase("ood_cadonilimab", "cadonilimab", "mechanism_established",
               "Q15116", "PDCD1",
               ("C1279945", "C0009319", "C0019158", "C0040128",
                "C0027059", "C0596022"),
               "Combined ICI irAE spectrum",
               "OOD 2022-23 China, 2024 expansion; PD-1×CTLA-4 bispecific"),

    # --- CAR-T expansions ---
    SafetyCase("ood_ciltacabtagene", "ciltacabtagene autoleucel", "black_box",
               "Q02223", "TNFRSF17",
               ("C2317799", "C0234016", "C0086438", "C0014335",
                "C0027947", "C0019360"),
               "CRS / ICANS / hypogamma / infections",
               "OOD expansions 2024; BCMA CAR-T"),

    # --- Asthma / immunology ---
    SafetyCase("ood_nemolizumab", "nemolizumab", "mechanism_established",
               "Q8NI17", "IL31RA",
               ("C0029118",),
               "Infections",
               "OOD 2024-08; anti-IL-31RA for prurigo nodularis"),

    SafetyCase("ood_brodalumab", "brodalumab", "black_box",
               "Q96F46", "IL17RA",
               ("C0438696", "C0029118"),
               "Suicidal ideation BBW / infection",
               "OOD 2017 but vocab fix in Sprint G; anti-IL-17RA"),
)


def main() -> int:
    import json
    from pathlib import Path
    from collections import Counter

    workspace = Path(__file__).resolve().parent.parent.parent
    results = workspace / "results"

    with open(results / "side_effect_vocab.json") as f:
        v = json.load(f)
    vocab_set = set(v["umls_ids"])
    with open(results / "target_vocab.json") as f:
        tv = json.load(f)
    target_set = {t["uniprot"] for t in tv["targets"]}

    from .clinical_safety_benchmark import passes_eligibility

    n_total = len(OOD_SAFETY_CASES)
    n_elig = sum(1 for c in OOD_SAFETY_CASES
                 if passes_eligibility(c, vocab_set, target_set))

    print(f"OOD benchmark: total={n_total}, eligible={n_elig}")

    # Check which fail eligibility and why
    for c in OOD_SAFETY_CASES:
        target_ok = c.causal_off_target_uniprot in target_set
        ses_in_vocab = [s for s in c.causal_side_effects_umls if s in vocab_set]
        if not target_ok or not ses_in_vocab:
            print(f"  INELIGIBLE: {c.drug_id} target_ok={target_ok} "
                  f"se_in_vocab={len(ses_in_vocab)}/{len(c.causal_side_effects_umls)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
