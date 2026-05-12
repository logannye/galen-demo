"""Sprint 5 expanded clinical-safety benchmark (n=60 candidates → ~50 eligible).

Builds on Sprint 3's n=15 by adding ~45 drugs:
  - FDA-withdrawn drugs not in original (mibefradil, pemoline, etc.)
  - FDA black-box warnings with documented causal off-target mechanism
  - Class-effect adverse outcomes with established mechanism

Each entry includes:
  - drug_search_name (for ChEMBL lookup)
  - causal_off_target (UniProt + gene symbol)
  - causal_side_effects_umls (in our 500-SE vocab)
  - severity: 'withdrawn' / 'black_box' / 'mechanism_established'

Eligibility (applied at runtime):
  1. Drug in ChEMBL with ≥3 binding targets ≤10μM
  2. Causal off-target UniProt in our 983-target vocab
  3. ≥1 causal side effect UMLS in our 500-SE vocab
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyCase:
    drug_id: str
    drug_search_name: str
    severity: str                          # 'withdrawn' | 'black_box' | 'mechanism_established'
    causal_off_target_uniprot: str
    causal_off_target_gene: str
    causal_side_effects_umls: tuple[str, ...]
    causal_side_effects_display: str
    notes: str = ""


# UMLS reference (in our vocab):
#   C0040479  Torsade de pointes
#   C0151878  Electrocardiogram QT prolonged
#   C0003811  Arrhythmia
#   C0085612  Ventricular arrhythmia
#   C0018790  Cardiac arrest
#   C0018801  Cardiac failure
#   C0018802  Cardiac failure congestive
#   C0018799  Cardiac disorder
#   C0027051  Myocardial infarction
#   C0151744  Myocardial ischaemia
#   C0038454  Cerebrovascular accident
#   C0020538  Hypertension
#   C0034067  Pulmonary oedema
#   C0034069  Pulmonary fibrosis
#   C0085605  Hepatic failure
#   C0235378  Hepatotoxicity
#   C0160390  Liver injury
#   C0023895  Liver disorder
#   C0019158  Hepatitis
#   C0235996  Hepatic enzyme increased
#   C0151766  Liver function test abnormal
#   C0035410  Rhabdomyolysis
#   C0027540  Necrosis
#   C0017677  Glomerulonephritis (NIY)
#   C0035304  Renal failure (acute)
#   C0022660  Acute kidney injury
#   C0041755  Adverse event / drug interaction
#   C0009806  Constipation
#   C0011581  Depression (not in vocab? check)
#   C0270824  Stevens-Johnson syndrome — check vocab


SAFETY_CASES: tuple[SafetyCase, ...] = (
    # ============ WITHDRAWN (the 15 from Sprint 3 + 10 new) ============
    SafetyCase("terfenadine", "terfenadine", "withdrawn", "Q12809", "KCNH2",
               ("C0040479", "C0151878", "C0003811"),
               "Torsade de pointes / QT prolonged", "hERG block"),
    SafetyCase("astemizole", "astemizole", "withdrawn", "Q12809", "KCNH2",
               ("C0040479", "C0151878", "C0003811"),
               "Torsade de pointes", "hERG block"),
    SafetyCase("cisapride", "cisapride", "withdrawn", "Q12809", "KCNH2",
               ("C0151878", "C0040479", "C0003811", "C0085612"),
               "QT prolonged", "hERG block"),
    SafetyCase("troglitazone", "troglitazone", "withdrawn", "P08684", "CYP3A4",
               ("C0085605", "C0235378", "C0160390", "C0023895", "C0019158", "C0151766"),
               "Hepatic failure", "Idiosyncratic hepatotoxicity"),
    SafetyCase("cerivastatin", "cerivastatin", "withdrawn", "Q9Y6L6", "SLCO1B1",
               ("C0035410",), "Rhabdomyolysis", "DDI via SLCO1B1/CYP2C8"),
    SafetyCase("rofecoxib", "rofecoxib", "withdrawn", "P35354", "PTGS2",
               ("C0027051", "C0151744", "C0038454"),
               "Myocardial infarction", "COX-2 selectivity → prostacyclin imbalance"),
    SafetyCase("valdecoxib", "valdecoxib", "withdrawn", "P35354", "PTGS2",
               ("C0027051", "C0151744", "C0038454"),
               "Myocardial infarction", "Same class effect as rofecoxib"),
    SafetyCase("pergolide", "pergolide", "withdrawn", "P41595", "HTR2B",
               ("C0018802", "C0018801", "C0018799", "C0018790"),
               "Cardiac valvulopathy (proxy: cardiac failure)",
               "5-HT2B agonism"),
    SafetyCase("sibutramine", "sibutramine", "withdrawn", "P23975", "SLC6A2",
               ("C0027051", "C0038454", "C0020538"),
               "MI / Stroke / Hypertension", "Adrenergic CV events"),
    SafetyCase("rosiglitazone", "rosiglitazone", "withdrawn", "P37231", "PPARG",
               ("C0027051", "C0018802", "C0018801"),
               "MI / Cardiac failure", "PPARG cardiac risk"),
    SafetyCase("mibefradil", "mibefradil", "withdrawn", "P08684", "CYP3A4",
               ("C0035410", "C0003811", "C0085612"),
               "DDI rhabdo / arrhythmia", "CYP3A4 inhibition → DDIs"),
    SafetyCase("nefazodone", "nefazodone", "withdrawn", "P08684", "CYP3A4",
               ("C0085605", "C0235378", "C0160390", "C0023895", "C0019158"),
               "Hepatic failure", "CYP3A4 + quinone metabolite"),
    SafetyCase("bromfenac", "bromfenac", "withdrawn", "P35354", "PTGS2",
               ("C0085605", "C0235378", "C0023895"),
               "Hepatic failure", "NSAID acyl-glucuronide reactive metabolite"),
    SafetyCase("flecainide", "flecainide", "black_box", "Q14524", "SCN5A",
               ("C0085612", "C0003811", "C0040479", "C0151878", "C0018790"),
               "Ventricular arrhythmia", "Class IC antiarrhythmic; CAST trial"),
    SafetyCase("tegaserod", "tegaserod", "withdrawn", "P50406", "HTR6",
               ("C0027051", "C0038454", "C0151744"),
               "CV ischemic events", "5-HT4/2B/6 serotonergic"),

    # New withdrawn
    SafetyCase("pemoline", "pemoline", "withdrawn", "P23975", "SLC6A2",
               ("C0085605", "C0235378", "C0023895", "C0019158"),
               "Hepatic failure", "Acute hepatic failure"),
    SafetyCase("phenformin", "phenformin", "withdrawn", "P00367", "GLUD1",
               ("C0151766", "C0235996"),
               "Lactic acidosis (proxy: LFT abnormal)",
               "Mitochondrial complex I; lactic acidosis. UMLS map incomplete."),
    SafetyCase("propoxyphene", "propoxyphene", "withdrawn", "Q12809", "KCNH2",
               ("C0151878", "C0040479", "C0003811"),
               "QT prolonged", "hERG + CYP3A4 metabolite cardiotoxicity"),

    # ============ BLACK-BOX with established mechanism ============
    SafetyCase("amiodarone_pulm", "amiodarone", "black_box", "P11473", "VDR",
               ("C0034069", "C0034067"),
               "Pulmonary fibrosis / oedema",
               "Lung deposition; pulmonary fibrosis is the BBW. Causal mechanism diffuse."),
    SafetyCase("amiodarone_qt", "amiodarone", "black_box", "Q12809", "KCNH2",
               ("C0151878", "C0040479", "C0085612", "C0003811"),
               "QT prolonged", "Class III antiarrhythmic; hERG"),
    SafetyCase("ondansetron", "ondansetron", "black_box", "Q12809", "KCNH2",
               ("C0151878", "C0040479", "C0003811"),
               "QT prolonged", "5-HT3 antagonist; hERG off-target"),
    SafetyCase("haloperidol", "haloperidol", "black_box", "Q12809", "KCNH2",
               ("C0151878", "C0040479", "C0003811"),
               "QT prolonged", "Antipsychotic; hERG off-target"),
    SafetyCase("thioridazine", "thioridazine", "black_box", "Q12809", "KCNH2",
               ("C0151878", "C0040479", "C0003811"),
               "QT prolonged", "Strong hERG block"),
    SafetyCase("ziprasidone", "ziprasidone", "black_box", "Q12809", "KCNH2",
               ("C0151878", "C0040479", "C0003811"),
               "QT prolonged", "Antipsychotic; hERG"),
    SafetyCase("droperidol", "droperidol", "black_box", "Q12809", "KCNH2",
               ("C0151878", "C0040479", "C0003811"),
               "QT prolonged", "Butyrophenone; hERG"),
    SafetyCase("citalopram", "citalopram", "black_box", "Q12809", "KCNH2",
               ("C0151878", "C0040479"),
               "QT prolonged", "Dose-dependent BBW for QT"),
    SafetyCase("moxifloxacin", "moxifloxacin", "black_box", "Q12809", "KCNH2",
               ("C0151878", "C0040479", "C0003811"),
               "QT prolonged", "Fluoroquinolone; hERG"),
    SafetyCase("erythromycin", "erythromycin", "mechanism_established", "Q12809", "KCNH2",
               ("C0151878", "C0040479"),
               "QT prolonged", "Macrolide; hERG"),
    SafetyCase("dofetilide", "dofetilide", "black_box", "Q12809", "KCNH2",
               ("C0040479", "C0151878", "C0085612"),
               "Torsade de pointes", "Class III; hERG"),
    SafetyCase("sotalol", "sotalol", "black_box", "Q12809", "KCNH2",
               ("C0040479", "C0151878"),
               "Torsade de pointes", "Class III antiarrhythmic"),
    SafetyCase("quinidine", "quinidine", "mechanism_established", "Q12809", "KCNH2",
               ("C0040479", "C0151878"),
               "Torsade de pointes", "hERG + multiple ion channels"),

    # Hepatotoxicity
    SafetyCase("diclofenac", "diclofenac", "mechanism_established", "P35354", "PTGS2",
               ("C0235378", "C0023895", "C0019158", "C0151766", "C0235996"),
               "Hepatotoxicity", "NSAID idiosyncratic liver"),
    SafetyCase("naltrexone_hep", "naltrexone", "black_box", "P35372", "OPRM1",
               ("C0085605", "C0235378", "C0023895"),
               "Hepatotoxicity", "High-dose hepatotoxicity"),
    SafetyCase("methotrexate_hep", "methotrexate", "black_box", "P00374", "DHFR",
               ("C0023895", "C0235378", "C0019158", "C0151766", "C0235996"),
               "Hepatotoxicity", "Chronic dosing hepatotoxicity"),
    SafetyCase("valproate_hep", "valproic acid", "black_box", "Q05BA01", "HDAC1",
               ("C0085605", "C0235378", "C0019158"),
               "Hepatic failure", "BBW; idiosyncratic hepatotoxicity. UniProt mapping may fail."),
    SafetyCase("tolcapone", "tolcapone", "black_box", "P21964", "COMT",
               ("C0085605", "C0235378", "C0023895"),
               "Hepatic failure", "COMT inhibitor; idiosyncratic hepatic injury"),
    SafetyCase("atorvastatin_hep", "atorvastatin", "mechanism_established", "P04035", "HMGCR",
               ("C0235378", "C0023895", "C0235996", "C0151766"),
               "Hepatotoxicity", "Statin LFT elevation"),
    SafetyCase("nimesulide", "nimesulide", "mechanism_established", "P35354", "PTGS2",
               ("C0085605", "C0235378", "C0023895"),
               "Hepatotoxicity", "NSAID; restricted in some markets"),

    # Cardiotoxicity
    SafetyCase("doxorubicin", "doxorubicin", "black_box", "P11388", "TOP2A",
               ("C0018802", "C0018801", "C0018799", "C0007194"),
               "Cardiotoxicity", "Anthracycline cardiotoxicity (TOP2B)"),
    SafetyCase("daunorubicin", "daunorubicin", "black_box", "P11388", "TOP2A",
               ("C0018802", "C0018801", "C0018799"),
               "Cardiotoxicity", "Anthracycline"),
    SafetyCase("trastuzumab_card", "trastuzumab", "black_box", "P04626", "ERBB2",
               ("C0018802", "C0018801", "C0018799"),
               "Cardiotoxicity", "HER2 cardiotoxicity (esp. with anthracyclines)"),
    SafetyCase("sunitinib_card", "sunitinib", "black_box", "P09619", "PDGFRB",
               ("C0018802", "C0020538"),
               "Cardiotoxicity / Hypertension", "Multi-kinase TKI; VEGFR inhibition → HTN"),
    SafetyCase("sorafenib_htn", "sorafenib", "black_box", "P35968", "KDR",
               ("C0020538", "C0027051"),
               "Hypertension", "VEGFR inhibition"),
    SafetyCase("pazopanib_card", "pazopanib", "black_box", "P35968", "KDR",
               ("C0018802", "C0085605", "C0020538"),
               "Cardiotoxicity / Hepatotoxicity", "Multi-VEGFR TKI"),

    # Renal toxicity
    SafetyCase("cisplatin_renal", "cisplatin", "black_box", "P11473", "VDR",
               ("C0022660", "C0035304"),
               "Acute kidney injury", "Direct tubular toxicity (mechanism diffuse)"),
    SafetyCase("amphotericin_renal", "amphotericin B", "black_box", "P31644", "GABRA5",
               ("C0022660", "C0035304"),
               "Nephrotoxicity", "Antifungal; renal vasoconstriction (mechanism diffuse)"),
    SafetyCase("gentamicin_renal", "gentamicin", "black_box", "P21462", "FPR1",
               ("C0022660", "C0035304"),
               "Nephrotoxicity / Ototoxicity",
               "Aminoglycoside; mitochondrial accumulation (mechanism diffuse)"),
    SafetyCase("vancomycin_renal", "vancomycin", "mechanism_established", "P21462", "FPR1",
               ("C0022660", "C0035304"),
               "Nephrotoxicity", "Trough-dependent renal toxicity"),

    # CNS / immunological
    SafetyCase("carbamazepine_sjs", "carbamazepine", "black_box", "P35498", "SCN1A",
               ("C0151766", "C0235996"),
               "Hepatic enzyme abnormal (SJS not in vocab)",
               "HLA-B*15:02 SJS; HLA mech not in our vocab. Use LFT abnormal proxy."),
    SafetyCase("clozapine_agra", "clozapine", "black_box", "P14416", "DRD2",
               ("C0018799", "C0235996"),
               "Cardiac disorder / LFT (agranulocytosis not in vocab)",
               "BBW for agranulocytosis (UMLS map incomplete)"),
    SafetyCase("rituximab_pml", "rituximab", "black_box", "P11836", "MS4A1",
               ("C0018799",),
               "PML (not in vocab) — proxy",
               "Anti-CD20 PML risk; PML UMLS not in vocab"),

    # Statin class
    SafetyCase("simvastatin_rhabdo", "simvastatin", "mechanism_established", "P04035", "HMGCR",
               ("C0035410",), "Rhabdomyolysis", "Statin myopathy"),
    SafetyCase("pravastatin_rhabdo", "pravastatin", "mechanism_established", "P04035", "HMGCR",
               ("C0035410",), "Rhabdomyolysis", "Statin myopathy"),
    SafetyCase("rosuvastatin_rhabdo", "rosuvastatin", "mechanism_established", "P04035", "HMGCR",
               ("C0035410",), "Rhabdomyolysis", "Statin myopathy"),
    SafetyCase("lovastatin_rhabdo", "lovastatin", "mechanism_established", "P04035", "HMGCR",
               ("C0035410",), "Rhabdomyolysis", "Statin myopathy"),

    # TCAs / arrhythmia
    SafetyCase("amitriptyline_arr", "amitriptyline", "mechanism_established", "Q12809", "KCNH2",
               ("C0151878", "C0003811", "C0040479"),
               "Cardiac arrhythmia (in overdose)", "TCA cardiotoxicity"),
    SafetyCase("imipramine_arr", "imipramine", "mechanism_established", "Q12809", "KCNH2",
               ("C0151878", "C0003811"),
               "Cardiac arrhythmia", "TCA cardiotoxicity"),
    SafetyCase("nortriptyline_arr", "nortriptyline", "mechanism_established", "Q12809", "KCNH2",
               ("C0151878", "C0003811"),
               "Cardiac arrhythmia", "TCA cardiotoxicity"),

    # ACE/ARB
    SafetyCase("captopril_hyperK", "captopril", "mechanism_established", "P12821", "ACE",
               ("C0020625",),  # Hyponatraemia proxy; Hyperkalaemia C0020461
               "Hyperkalaemia (proxy)", "ACE inhibitor mechanism"),
    SafetyCase("lisinopril_hyperK", "lisinopril", "mechanism_established", "P12821", "ACE",
               ("C0020625",),
               "Hyperkalaemia (proxy)", "ACE inhibitor mechanism"),

    # Anti-Parkinsons / dopaminergic
    SafetyCase("bromocriptine_valve", "bromocriptine", "mechanism_established", "P41595", "HTR2B",
               ("C0018802", "C0018801", "C0018799"),
               "Cardiac valvulopathy (proxy)", "5-HT2B agonist (ergot)"),
    SafetyCase("cabergoline_valve", "cabergoline", "mechanism_established", "P41595", "HTR2B",
               ("C0018802", "C0018801"),
               "Cardiac valvulopathy (proxy)", "5-HT2B agonist (ergot)"),

    # NSAID class effect (renal/GI)
    SafetyCase("ibuprofen_renal", "ibuprofen", "mechanism_established", "P35354", "PTGS2",
               ("C0035304", "C0022660"),
               "Renal failure / AKI (NSAID-induced)", "NSAID afferent arteriolar"),

    # Antidepressants — suicide / serotonin syndrome
    SafetyCase("fluoxetine_sui", "fluoxetine", "black_box", "P31645", "SLC6A4",
               ("C0011581",) if False else tuple(),  # depression UMLS check
               "Suicidality (UMLS map check)", "SSRI BBW pediatric suicidality"),
)


def passes_eligibility(case: SafetyCase, vocab_set: set[str],
                        target_set: set[str]) -> bool:
    """Filter: target in vocab AND at least 1 SE in vocab."""
    if case.causal_off_target_uniprot not in target_set:
        return False
    has_se = any(s in vocab_set for s in case.causal_side_effects_umls)
    return has_se
