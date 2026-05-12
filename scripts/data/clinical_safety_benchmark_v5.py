"""Sprint G Track 1: Benchmark v5 — augment v4 with Sprint G new UMLS codes.

Augmentations identified from Sprint F miss analysis: 12 cases where
the Hybrid was predicting a semantically correct AE but the benchmark
used a proxy/different UMLS that didn't match.
"""
from __future__ import annotations

from .clinical_safety_benchmark import SafetyCase, passes_eligibility
from .clinical_safety_benchmark_v4 import EXPANDED_SAFETY_CASES_V4


# Sprint G augmentations from miss analysis (drug_id substring → new UMLS codes)
AUGMENTATIONS_G: list[tuple[str, tuple[str, ...]]] = [
    # SGLT2 → DKA proper
    ("dapagliflozin_dka", ("C0011880",)),
    ("empagliflozin_dka", ("C0011880",)),
    ("canagliflozin_aki", ("C0011880",)),

    # Antipsychotics → EPS / akathisia / metabolic / QT
    ("aripiprazole_eps", ("C0015371", "C0392156", "C0011854", "C0151878")),
    ("risperidone_eps", ("C0015371", "C0392156")),
    ("olanzapine_metabolic", ("C0011854", "C0015371")),
    ("quetiapine_metabolic", ("C0011854",)),

    # Clozapine → agranulocytosis proper
    ("clozapine_agra", ("C0001824", "C0027947")),

    # Brodalumab → suicidal ideation BBW
    ("brodalumab_suicide", ("C0438696",)),

    # Topiramate → metabolic acidosis (already in vocab; benchmark gap)
    ("topiramate_acidosis", ("C0220981",)),

    # SSRIs / SNRIs → serotonin syndrome
    ("sertraline_qt", ("C0036875",)),
    ("escitalopram_qt", ("C0036875",)),
    ("venlafaxine_qt", ("C0036875",)),
    ("fluoxetine_sui", ("C0438696", "C0036875")),
    ("bupropion_seizure", ("C0036572", "C0438696")),

    # Complement inhibitors → meningococcal meningitis specifically
    ("eculizumab_meningococcal", ("C0025291", "C0025289")),
    ("ravulizumab_meningococcal", ("C0025291", "C0025289")),
    ("pegcetacoplan_inf", ("C0025291",)),
    ("iptacopan_inf", ("C0025291",)),
    ("crovalimab", ("C0025291",)),

    # MEKi → cardiomyopathy
    ("cobimetinib_card", ("C0007193", "C0878544")),
    ("trametinib_card", ("C0007193", "C0878544")),
    ("encorafenib_card", ("C0007193",)),

    # Anthracyclines → dilated cardiomyopathy
    ("doxorubicin", ("C0007193",)),
    ("daunorubicin", ("C0007193",)),

    # Anticonvulsants → drug eruption / SCAR
    ("phenytoin_sjs", ("C0151654",)),
    ("lamotrigine_sjs", ("C0151654",)),
    ("carbamazepine_sjs", ("C0151654",)),

    # Anti-VEGF eye → visual impairment + ophthalmic
    ("brolucizumab_endoph", ("C0234518",)),
]


def _augment(case: SafetyCase) -> SafetyCase:
    extra: tuple[str, ...] = ()
    for pattern, codes in AUGMENTATIONS_G:
        if pattern in case.drug_id.lower():
            extra = extra + codes
    if not extra:
        return case
    existing = set(case.causal_side_effects_umls)
    new = tuple(c for c in extra if c not in existing)
    if not new:
        return case
    return SafetyCase(
        drug_id=case.drug_id,
        drug_search_name=case.drug_search_name,
        severity=case.severity,
        causal_off_target_uniprot=case.causal_off_target_uniprot,
        causal_off_target_gene=case.causal_off_target_gene,
        causal_side_effects_umls=case.causal_side_effects_umls + new,
        causal_side_effects_display=case.causal_side_effects_display,
        notes=case.notes,
    )


EXPANDED_SAFETY_CASES_V5: tuple[SafetyCase, ...] = tuple(
    _augment(c) for c in EXPANDED_SAFETY_CASES_V4
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

    n_v4 = len(EXPANDED_SAFETY_CASES_V4)
    n_v4_elig = sum(1 for c in EXPANDED_SAFETY_CASES_V4
                    if passes_eligibility(c, vocab_set, target_set))
    n_v5 = len(EXPANDED_SAFETY_CASES_V5)
    n_v5_elig = sum(1 for c in EXPANDED_SAFETY_CASES_V5
                    if passes_eligibility(c, vocab_set, target_set))

    n_augmented = sum(
        1 for v4, v5 in zip(EXPANDED_SAFETY_CASES_V4, EXPANDED_SAFETY_CASES_V5)
        if v4.causal_side_effects_umls != v5.causal_side_effects_umls
    )
    print(f"[v4]: total={n_v4}, eligible={n_v4_elig}")
    print(f"[v5]: total={n_v5}, eligible={n_v5_elig}")
    print(f"[augmented]: {n_augmented} cases")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
