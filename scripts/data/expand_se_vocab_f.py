"""Sprint F: Expand side-effect vocab with ~100 modern biologic AE terms.

Adds UMLS codes for AEs that:
  1. Are referenced by Sprint E curated priors but were missing from vocab
     (causing those priors to be filtered out)
  2. Are critical for modern biologic safety (CRS, ICANS, HLH, SCAR
     subtypes, ADC payload effects, irAE-specific endocrinopathies)
  3. Improve coverage of common AEs that small-molecule SIDER vocab
     under-represents (vomiting, hemorrhage, pneumonia, fungal/viral
     infections, etc.)
"""
from __future__ import annotations

import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


# Curated list of AE UMLS codes to add to vocab.
# Format: (umls_id, display_name, category_hint)
NEW_SE_TERMS: list[tuple[str, str, str]] = [
    # --- Cytokine release / immune effector syndromes ---
    ("C2317799", "Cytokine release syndrome", "cytokine"),
    ("C0234016", "Encephalopathy", "neuro"),
    ("C0079545", "Hemophagocytic lymphohistiocytosis", "hematologic"),
    ("C2363741", "Macrophage activation syndrome", "immune"),

    # --- Severe cutaneous adverse reactions (SCAR) ---
    ("C0038325", "Stevens-Johnson syndrome", "skin"),
    ("C2700346", "Drug rash with eosinophilia and systemic symptoms", "skin"),
    ("C1740659", "Acute generalized exanthematous pustulosis", "skin"),
    ("C0033581", "Photosensitivity reaction", "skin"),

    # --- ADC / biologic complications ---
    ("C0151602", "Capillary leak syndrome", "vascular"),
    ("C0080226", "Hepatic veno-occlusive disease", "hepatic"),
    ("C0152114", "Retinal vasculitis", "ophthalmic"),
    ("C0014236", "Endophthalmitis", "ophthalmic"),

    # --- irAE-specific endocrinopathies ---
    ("C0596022", "Hypophysitis", "endocrine"),
    ("C0020635", "Hypopituitarism", "endocrine"),
    ("C0011860", "Type 1 diabetes mellitus", "endocrine"),
    ("C0020550", "Hyperthyroidism", "endocrine"),
    ("C0001620", "Adrenalitis", "endocrine"),

    # --- B-cell aplasia / hypogammaglobulinemia (CAR-T) ---
    ("C0086438", "Hypogammaglobulinemia", "immune"),
    ("C0034155", "Pure red cell aplasia", "hematologic"),

    # --- Renal / nephrotoxicity ---
    ("C0017658", "Glomerulonephritis", "renal"),
    ("C0027697", "Nephritis", "renal"),
    ("C0040076", "Tubulointerstitial nephritis", "renal"),
    ("C0035091", "Renal vein thrombosis", "renal"),

    # --- GI / common ---
    ("C0042963", "Vomiting", "GI"),
    ("C0017181", "Gastrointestinal haemorrhage", "GI"),
    ("C0019080", "Haemorrhage", "vascular"),
    ("C0033687", "Proteinuria", "renal"),

    # --- Hematologic malignancies (PARP class) ---
    ("C0023467", "Acute myeloid leukaemia", "malignancy"),
    ("C0026986", "Myelodysplastic syndrome", "malignancy"),

    # --- CNS / neurology ---
    ("C0014038", "Encephalitis", "neuro"),
    ("C0029134", "Optic neuritis", "neuro"),
    ("C0079737", "Demyelinating disease", "neuro"),
    ("C0018378", "Guillain-Barré syndrome", "neuro"),
    ("C0026896", "Myasthenia gravis", "neuro"),
    ("C0085655", "Polymyositis", "muscle"),
    ("C0011633", "Dermatomyositis", "muscle"),
    ("C0036572", "Seizure", "neuro"),
    ("C0038220", "Status epilepticus", "neuro"),
    ("C0003537", "Aphasia", "neuro"),
    ("C0040822", "Tremor", "neuro"),
    ("C0042571", "Vertigo", "neuro"),
    ("C0040264", "Tinnitus", "ear"),
    ("C0011053", "Deafness", "ear"),
    ("C0338656", "Cognitive impairment", "neuro"),

    # --- Ophthalmic ---
    ("C0042164", "Uveitis", "ophthalmic"),
    ("C0022568", "Keratitis", "ophthalmic"),
    ("C0234518", "Visual impairment", "ophthalmic"),

    # --- Critical care / shock ---
    ("C0035222", "Acute respiratory distress syndrome", "respiratory"),
    ("C0036690", "Sepsis", "infection"),
    ("C0036983", "Septic shock", "infection"),
    ("C0036980", "Cardiogenic shock", "cardiac"),
    ("C0026766", "Multi-organ failure", "critical"),
    ("C0242184", "Hypoxia", "respiratory"),
    ("C0039231", "Tachycardia", "cardiac"),

    # --- Infections (immunosuppression-related) ---
    ("C0032285", "Pneumonia", "infection"),
    ("C0021400", "Influenza", "infection"),
    ("C5203670", "COVID-19", "infection"),
    ("C0010823", "Cytomegalovirus infection", "infection"),
    ("C0004026", "Aspergillosis", "infection"),
    ("C0032305", "Pneumocystis jirovecii pneumonia", "infection"),
    ("C0042769", "Viral infection", "infection"),
    ("C0026946", "Fungal infection", "infection"),
    ("C0007642", "Cellulitis", "infection"),
    ("C0006848", "Oral candidiasis", "infection"),
    ("C0006277", "Bronchitis", "respiratory"),
    ("C0037199", "Sinusitis", "respiratory"),
    ("C0040329", "Tonsillitis", "respiratory"),

    # --- Thromboembolic ---
    ("C0034065", "Pulmonary embolism", "vascular"),
    ("C0149871", "Deep vein thrombosis", "vascular"),

    # --- Skin / mucosal ---
    ("C0024862", "Mucositis oral", "GI"),
    ("C0027339", "Nail disorder", "skin"),
    ("C0016436", "Folliculitis", "skin"),
    ("C0001144", "Acne", "skin"),
    ("C0002170", "Alopecia", "skin"),
    ("C0043352", "Xerostomia", "GI"),

    # --- ADC payload-specific neurological ---
    # Peripheral neuropathy variants
    ("C0031117", "Peripheral neuropathy", "neuro"),
    ("C0151313", "Sensory neuropathy", "neuro"),
    ("C0270922", "Motor neuropathy", "neuro"),

    # --- Cardiac (additional) ---
    ("C0231807", "Ventricular dysfunction", "cardiac"),
    ("C0011854", "Diabetes mellitus", "endocrine"),

    # --- Cancer therapy general ---
    ("C0007873", "Asthenia", "general"),
    ("C0235329", "Pyrexia of unknown origin", "infection"),

    # --- Atrial dysrhythmia subtypes ---
    ("C0264714", "Left ventricular failure", "cardiac"),
    ("C0344440", "Bundle branch block", "cardiac"),

    # --- Pregnancy / fertility ---
    ("C0008372", "Cholestasis", "hepatic"),

    # --- GI specific ---
    ("C0009806", "Constipation", "GI"),  # may already exist
    ("C0011168", "Dysphagia", "GI"),
    ("C0019163", "Hepatitis B reactivation", "hepatic"),

    # --- Cardiovascular events ---
    ("C0040961", "Tachyarrhythmia", "cardiac"),
    ("C0085675", "Cardiomyopathy alcoholic", "cardiac"),  # placeholder
    ("C0027059", "Myocarditis", "cardiac"),  # may have
    ("C0152096", "Hypertrophic cardiomyopathy", "cardiac"),  # may have
]


def main() -> int:
    print("=" * 78)
    print("Sprint F: Expand side-effect vocab with biologic AE terms")
    print("=" * 78)

    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)

    existing_ids = set(v["umls_ids"])
    print(f"[load] existing vocab: {len(existing_ids)}")

    added = 0
    skipped = 0
    for umls_id, display, _cat in NEW_SE_TERMS:
        if umls_id in existing_ids:
            skipped += 1
            continue
        v["umls_ids"].append(umls_id)
        v["display_names"][umls_id] = display
        added += 1

    print(f"[expand] candidates: {len(NEW_SE_TERMS)}")
    print(f"[expand] added: {added}")
    print(f"[expand] skipped (already in vocab): {skipped}")
    print(f"[expand] new total: {len(v['umls_ids'])}")

    # Backup
    backup = RESULTS / "side_effect_vocab_pre_sprint_f.json"
    if not backup.exists():
        with open(backup, "w") as f:
            # Save the pre-expansion state (current state minus our additions)
            pre = {**v}
            pre["umls_ids"] = v["umls_ids"][:-added] if added > 0 else v["umls_ids"]
            pre_names = dict(v["display_names"])
            for umls_id, _, _ in NEW_SE_TERMS:
                if umls_id not in existing_ids:
                    pre_names.pop(umls_id, None)
            pre["display_names"] = pre_names
            json.dump(pre, f, indent=2)
        print(f"[backup] {backup}")

    v["n_added_sprint_f"] = added
    out_path = RESULTS / "side_effect_vocab.json"
    with open(out_path, "w") as f:
        json.dump(v, f, indent=2)
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
