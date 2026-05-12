"""Sprint F: Benchmark v4 — augment v3 cases with new specific UMLS codes.

Sprint F's expanded SE vocab includes specific UMLS codes for AEs that
v3 only referenced via proxies (e.g., CRS-proxy C0948715 → CRS proper
C2317799). For each v3 case, we add the new specific UMLS codes to
the `causal_side_effects_umls` tuple — keeping the proxies for backward
compatibility while enabling matches on the specific terms.

We don't add new CASES in v4 (that work was done in v3). v4 is purely
a relabeling pass.
"""
from __future__ import annotations

from .clinical_safety_benchmark import SafetyCase, passes_eligibility
from .clinical_safety_benchmark_v3 import EXPANDED_SAFETY_CASES_V3


# Mapping: v3 SafetyCase drug_id pattern → additional UMLS codes to add
# to the causal_side_effects_umls tuple.
# Pattern: substring match on drug_id.
AUGMENTATIONS: list[tuple[str, tuple[str, ...]]] = [
    # CAR-T / bispecific CRS+ICANS pattern — add specific CRS, encephalopathy,
    # HLH, hypogammaglobulinemia UMLS
    ("_crs", ("C2317799", "C0234016", "C0079545")),

    # ICI irAE pattern — add hypophysitis, T1DM, CMV
    ("_irae", ("C0596022", "C0011860")),

    # CD19 CAR-T specifically: add hypogammaglobulinemia
    ("axicabtagene", ("C0086438",)),
    ("tisagenlecleucel", ("C0086438",)),
    ("brexucabtagene", ("C0086438",)),
    ("lisocabtagene", ("C0086438",)),
    ("obecabtagene", ("C0086438",)),

    # Vedotin (MMAE) ADCs — add specific peripheral neuropathy
    ("brentuximab_pn", ("C0031117",)),
    ("enfortumab_skin", ("C0038325", "C2700346")),  # SJS, DRESS

    # CD33 / inotuzumab / gemtuzumab calicheamicin — VOD
    ("gemtuzumab_hep", ("C0080226",)),
    ("inotuzumab_hep", ("C0080226",)),

    # Anti-VEGF eye (brolucizumab) — retinal vasculitis, endophthalmitis
    ("brolucizumab", ("C0152114", "C0014236")),

    # SJS pattern — add specific SJS UMLS where TEN was used as proxy
    ("carbamazepine_sjs", ("C0038325",)),
    ("lamotrigine_sjs", ("C0038325", "C2700346")),
    ("phenytoin_sjs", ("C0038325",)),
    ("vemurafenib_sjs", ("C0038325",)),

    # Anticoagulant bleeding — add specific haemorrhage
    ("warfarin_bleed", ("C0019080", "C0017181")),
    ("dabigatran_bleed", ("C0019080", "C0017181")),
    ("rivaroxaban_bleed", ("C0019080", "C0017181")),
    ("apixaban_bleed", ("C0019080", "C0017181")),

    # PARP class — add specific AML / MDS already covered;
    # add anaemia variants
    # (existing tuples mostly have these)

    # JAK MACE pattern — add specific deep vein thrombosis / PE
    ("tofacitinib_mace", ("C0034065", "C0149871")),
    ("baricitinib_mace", ("C0034065", "C0149871")),
    ("upadacitinib_mace", ("C0034065", "C0149871")),

    # ICI pneumonitis cases — already have C1279945 / C0032310; no change

    # Anti-CD20 PML — add CMV reactivation possibility
    ("rituximab_pml", ("C0010823",)),
    ("rituximab_aav", ("C0010823",)),
    ("ocrelizumab_pml", ("C0010823",)),

    # IL-6 perforation / sepsis pattern
    ("tocilizumab_perforation", ("C0036690",)),

    # Atopic dermatitis biologics — add conjunctivitis specifics
    ("dupilumab_eczema", ("C0042164",)),  # uveitis is rare but reported

    # Sclerostin (romosozumab) — already strong; add MI subtype
    # (existing tuples mostly have these)

    # GLP-1/GIP — add pancreatitis specific (already have C0030305)

    # Anti-EGFR ILD — specific ILD codes already in tuples

    # ADC ocular — already have C0009763 + ophthalmic codes
    ("belantamab_ocular", ("C0042164", "C0014236")),  # uveitis, endophthalmitis
    ("mirvetuximab_ocular", ("C0042164", "C0014236")),
    ("tisotumab_ocular", ("C0042164",)),
]


def _augment(case: SafetyCase) -> SafetyCase:
    """Return a new SafetyCase with augmented AE tuple."""
    extra_codes: tuple[str, ...] = ()
    for pattern, codes in AUGMENTATIONS:
        if pattern in case.drug_id.lower():
            extra_codes = extra_codes + codes
    if not extra_codes:
        return case
    # Deduplicate + preserve order
    existing = set(case.causal_side_effects_umls)
    new_codes = tuple(c for c in extra_codes if c not in existing)
    if not new_codes:
        return case
    return SafetyCase(
        drug_id=case.drug_id,
        drug_search_name=case.drug_search_name,
        severity=case.severity,
        causal_off_target_uniprot=case.causal_off_target_uniprot,
        causal_off_target_gene=case.causal_off_target_gene,
        causal_side_effects_umls=case.causal_side_effects_umls + new_codes,
        causal_side_effects_display=case.causal_side_effects_display,
        notes=case.notes,
    )


EXPANDED_SAFETY_CASES_V4: tuple[SafetyCase, ...] = tuple(
    _augment(c) for c in EXPANDED_SAFETY_CASES_V3
)


def main() -> int:
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

    n_v3 = len(EXPANDED_SAFETY_CASES_V3)
    n_v3_elig = sum(1 for c in EXPANDED_SAFETY_CASES_V3
                    if passes_eligibility(c, vocab_set, target_set))
    n_v4 = len(EXPANDED_SAFETY_CASES_V4)
    n_v4_elig = sum(1 for c in EXPANDED_SAFETY_CASES_V4
                    if passes_eligibility(c, vocab_set, target_set))

    n_augmented = sum(
        1 for v3, v4 in zip(EXPANDED_SAFETY_CASES_V3, EXPANDED_SAFETY_CASES_V4)
        if v3.causal_side_effects_umls != v4.causal_side_effects_umls
    )

    print(f"[v3 cases]: total={n_v3}, eligible={n_v3_elig}")
    print(f"[v4 cases]: total={n_v4}, eligible={n_v4_elig}")
    print(f"[augmented]: {n_augmented} cases got new specific UMLS codes")

    # Per-TA tally
    from collections import Counter
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

    total_classes = Counter(classify(c) for c in EXPANDED_SAFETY_CASES_V4
                             if passes_eligibility(c, vocab_set, target_set))
    print(f"[v4 eligible by TA]: {dict(total_classes)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
