"""Sprint G Track 1: vocab additions identified from Sprint F miss analysis.

Each new UMLS code addresses a specific Hybrid miss where the LLM was
predicting a semantically correct AE that wasn't in our vocab.
"""
from __future__ import annotations

import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


NEW_SE_TERMS_G: list[tuple[str, str]] = [
    # --- Track 1 vocab fixes from Sprint F miss analysis ---
    ("C0438696", "Suicidal ideation"),         # brodalumab BBW
    ("C0001824", "Agranulocytosis"),            # clozapine BBW
    ("C0011880", "Diabetic ketoacidosis"),      # SGLT2 BBW
    ("C0007193", "Cardiomyopathy dilated"),     # anthracycline / MEKi
    ("C0878544", "Cardiomyopathy"),             # general
    ("C0025291", "Meningococcal meningitis"),   # eculizumab / iptacopan BBW
    ("C0036875", "Serotonin syndrome"),         # SSRI / SNRI BBW
    ("C0151654", "Drug eruption"),              # broad SCAR umbrella
    ("C0011854", "Diabetes mellitus"),          # general; existing C0011860 is T1DM
    ("C0013182", "Drug-induced liver injury"),  # broader hepatic
    ("C0151744", "Myocardial ischaemia"),        # may already exist
    ("C0231807", "Ventricular dysfunction"),    # cardiomyopathy proxy
    ("C0009450", "Infection"),                  # general (high-prevalence)
    ("C0085631", "Agitation"),                  # antipsychotic withdrawal
    ("C2363742", "Drug withdrawal syndrome"),
    ("C2939193", "Long QT syndrome"),
    ("C0023786", "Lysosomal storage disease"),  # ERT mAbs

    # --- Track 2 vocab for Other subgroup ---
    ("C0857533", "Long-term effects on cardiac function"),
    ("C0234518", "Visual impairment"),          # may exist
    ("C0011570", "Mental depression"),
    ("C0011581", "Depression"),
    ("C0024874", "Mydriasis"),
    ("C0026766", "Multiple organ dysfunction syndrome"),
    ("C0027947", "Neutropenia"),                # likely already
    ("C0017205", "Gaucher disease"),

    # --- Cardiac specifics ---
    ("C0265279", "Premature ventricular contractions"),
    ("C0344434", "Cardiac flutter"),
    ("C0085620", "Cardiac dysrhythmia"),

    # --- Renal specifics ---
    ("C0151746", "Renal tubular acidosis"),

    # --- GI specifics ---
    ("C0021368", "Bowel perforation"),
    ("C0157654", "Acute pancreatitis"),

    # --- Psychiatric ---
    ("C0497327", "Anxiety"),
    ("C0085631_b", "Restlessness"),  # may need actual UMLS

    # --- Neurological specifics ---
    ("C0085631_c", "Ataxia"),
    ("C0011570_b", "Cognitive disorder"),

    # --- Immunology specifics ---
    ("C0085655", "Polymyositis"),  # may already
    ("C0019095", "Hemolytic uremic syndrome"),

    # --- Endocrine specifics ---
    ("C0853286", "Adrenocortical insufficiency"),

    # --- Drug-induced ---
    ("C0151903", "Hepatic enzyme abnormal"),
]


def main() -> int:
    print("=" * 78)
    print("Sprint G Track 1: SE vocab additions from miss analysis")
    print("=" * 78)

    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    existing_ids = set(v["umls_ids"])
    print(f"[load] existing vocab: {len(existing_ids)}")

    added = 0
    skipped = 0
    for umls, name in NEW_SE_TERMS_G:
        # Skip suffix-tagged placeholders (don't have valid UMLS)
        if "_b" in umls or "_c" in umls:
            continue
        if umls in existing_ids:
            skipped += 1
            continue
        v["umls_ids"].append(umls)
        v["display_names"][umls] = name
        added += 1

    print(f"[expand] added: {added}; skipped: {skipped}; new total: {len(v['umls_ids'])}")

    v["n_added_sprint_g"] = added
    with open(RESULTS / "side_effect_vocab.json", "w") as f:
        json.dump(v, f, indent=2)
    print(f"[save] {RESULTS / 'side_effect_vocab.json'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
