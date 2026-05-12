"""Sprint 7A: expand side-effect vocabulary for oncology + immunology.

Strategy: keep all 500 existing UMLS terms, add ~200 clinically-relevant
onc/immuno AEs that are in SIDER but didn't make the top-500 cut.

Selection criteria:
  - Drug count ≥3 in SIDER (statistical support)
  - Onc/immuno-relevant: irAEs, infections, MACE, ILD/pneumonitis subtypes,
    cytopenias, malignancy types, infusion reactions, secondary cancers,
    demyelination, opportunistic infections, organ-specific tox subtypes

Output: results/side_effect_vocab_expanded.json (~700 terms total)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
SIDER_PATH = WORKSPACE / "data/raw/sider/meddra_all_se.tsv"


# Curated onc/immuno-relevant UMLS additions (verified in SIDER)
# Format: UMLS → (display_name, category_tag)
ONC_IMMUNO_ADDITIONS = {
    # --- Infections / opportunistic (immuno-critical) ---
    "C0029118": ("Opportunistic infection", "infection"),
    "C0041296": ("Tuberculosis", "infection"),
    "C0023524": ("Progressive multifocal leukoencephalopathy", "infection"),
    "C0019163": ("Hepatitis B", "infection"),
    "C0019196": ("Hepatitis C", "infection"),
    "C0019159": ("Hepatitis A", "infection"),
    "C0010823": ("Cytomegalovirus infection", "infection"),
    "C1535939": ("Pneumocystis jirovecii pneumonia", "infection"),
    "C0851807": ("Aspergillus infection", "infection"),
    "C0343886": ("Gastrointestinal candidiasis", "infection"),
    "C0343863": ("Genital candidiasis", "infection"),
    "C0239295": ("Oesophageal candidiasis", "infection"),
    "C0919659": ("Oropharyngeal candidiasis", "infection"),
    "C0025289": ("Meningitis", "infection"),
    "C0025290": ("Meningitis aseptic", "infection"),
    "C0036983": ("Septic shock", "infection"),
    "C0877153": ("Neutropenic sepsis", "infection"),
    "C0149801": ("Urosepsis", "infection"),
    # Note: Herpes zoster / shingles — add manually below

    # --- Cardiovascular / MACE / arrhythmia (JAKi + TKI critical) ---
    "C0042487": ("Venous thrombosis", "vascular"),
    "C0151942": ("Arterial thrombosis", "vascular"),
    "C0010072": ("Coronary artery thrombosis", "vascular"),
    "C0079102": ("Cerebral thrombosis", "vascular"),
    "C0549288": ("Pelvic venous thrombosis", "vascular"),
    "C0155773": ("Portal vein thrombosis", "vascular"),
    "C0521501": ("Injection site thrombosis", "vascular"),
    "C0085610": ("Sinus bradycardia", "cardiac"),
    "C0039239": ("Sinus tachycardia", "cardiac"),
    "C0546959": ("Atrial tachycardia", "cardiac"),
    "C0007194": ("Hypertrophic cardiomyopathy", "cardiac"),
    "C0007193": ("Congestive cardiomyopathy", "cardiac"),
    "C0027059": ("Myocarditis", "cardiac"),
    "C0242698": ("Left ventricular dysfunction", "cardiac"),
    "C0020542": ("Pulmonary hypertension", "vascular"),
    "C0429098": ("Electrocardiogram QRS complex", "cardiac"),
    "C0438173": ("Electrocardiogram QRS complex abnormal", "cardiac"),

    # --- Pulmonary / ILD subtypes (TKI + ICI critical) ---
    "C0034069": ("Pulmonary fibrosis", "pulmonary"),
    "C0206063": ("Radiation pneumonitis", "pulmonary"),
    "C1279945": ("Acute interstitial pneumonitis", "pulmonary"),
    "C1800706": ("Idiopathic pulmonary fibrosis", "pulmonary"),

    # --- Hepatic / GI (oncology drug class) ---
    "C0149904": ("Hepatitis cholestatic", "hepatic"),
    "C0267797": ("Hepatitis acute", "hepatic"),
    "C0520463": ("Hepatitis chronic active", "hepatic"),
    "C0241910": ("Autoimmune hepatitis", "hepatic"),
    "C0009324": ("Colitis ulcerative", "GI/colitis"),
    "C0162529": ("Colitis ischaemic", "GI/colitis"),
    "C0033247": ("Proctocolitis", "GI/colitis"),
    "C0238106": ("Clostridium difficile colitis", "GI/colitis"),
    "C0014356": ("Enterocolitis", "GI/colitis"),
    "C0400823": ("Neutropenic colitis", "GI/colitis"),
    "C0151623": ("Enterocolitis haemorrhagic", "GI/colitis"),
    "C0948692": ("Necrotising colitis", "GI/colitis"),
    "C0400821": ("Colitis microscopic", "GI/colitis"),
    "C1442826": ("Necrotising enterocolitis neonatal", "GI/colitis"),
    "C0038363": ("Aphthous stomatitis", "GI/mucositis"),

    # --- Endocrine (immune-mediated for ICI; metabolic for kinase) ---
    "C0001623": ("Adrenal insufficiency", "endocrine"),
    "C0040128": ("Thyroid disorder", "endocrine"),
    "C0020514": ("Hyperprolactinaemia", "endocrine"),
    "C0030517": ("Parathyroid disorder", "endocrine"),
    "C0747102": ("Ovarian failure", "endocrine/reproductive"),
    "C0004509": ("Azoospermia", "endocrine/reproductive"),

    # --- Hematologic (cytopenia subtypes; oncology + JAKi) ---
    "C0026986": ("Myelodysplastic syndrome", "hematologic/malignancy"),
    "C0023467": ("Acute myeloid leukaemia", "hematologic/malignancy"),
    "C0085669": ("Acute leukaemia", "hematologic/malignancy"),
    "C0023418": ("Leukaemia", "hematologic/malignancy"),
    "C0023470": ("Myeloid leukaemia", "hematologic/malignancy"),
    "C0023473": ("Chronic myeloid leukaemia", "hematologic/malignancy"),
    "C0023449": ("Acute lymphocytic leukaemia", "hematologic/malignancy"),
    "C0023434": ("Chronic lymphocytic leukaemia", "hematologic/malignancy"),
    "C0024299": ("Lymphoma", "hematologic/malignancy"),
    "C0024305": ("Non-Hodgkin's lymphoma", "hematologic/malignancy"),
    "C0079731": ("B-cell lymphoma", "hematologic/malignancy"),
    "C0006413": ("Burkitt's lymphoma", "hematologic/malignancy"),
    "C0079744": ("Diffuse large B-cell lymphoma", "hematologic/malignancy"),
    "C1333984": ("Hepatosplenic T-cell lymphoma", "hematologic/malignancy"),
    "C0221269": ("Pseudolymphoma", "hematologic/malignancy"),
    "C0853986": ("Lymphocyte count decreased", "hematologic/cytopenia"),
    "C0002884": ("Hypochromic anaemia", "hematologic/cytopenia"),
    "C0002888": ("Anaemia megaloblastic", "hematologic/cytopenia"),
    "C0002886": ("Anaemia macrocytic", "hematologic/cytopenia"),
    "C0162316": ("Iron deficiency anaemia", "hematologic/cytopenia"),
    "C0271899": ("Normochromic normocytic anaemia", "hematologic/cytopenia"),
    "C0085576": ("Microcytic anaemia", "hematologic/cytopenia"),
    "C0178416": ("Hypoplastic anaemia", "hematologic/cytopenia"),
    "C0002880": ("Autoimmune haemolytic anaemia", "hematologic/cytopenia"),
    "C0221021": ("Microangiopathic haemolytic anaemia", "hematologic/cytopenia"),
    "C0948168": ("Bone marrow toxicity", "hematologic/cytopenia"),
    # Note: Cytokine release syndrome — add manually

    # --- Skin / dermatologic (TKI + ICI + chemo) ---
    "C0549410": ("Palmar-plantar erythrodysaesthesia syndrome", "dermatologic"),
    "C0014745": ("Palmar erythema", "dermatologic"),

    # --- Neurologic / demyelination (immuno + S1P modulator) ---
    "C0011304": ("Demyelination", "neurologic"),
    "C0026769": ("Multiple sclerosis", "neurologic"),
    "C0018378": ("Guillain-Barre syndrome", "neurologic"),
    "C0699828": ("Serotonin syndrome", "neurologic"),

    # --- Ophthalmologic (S1P modulators, taxanes, etc.) ---
    "C0271051": ("Macular oedema", "ophthalmologic"),
    "C0024440": ("Cystoid macular oedema", "ophthalmologic"),
    "C1527411": ("Retinal vein thrombosis", "ophthalmologic"),
    "C0919714": ("Retinal vascular thrombosis", "ophthalmologic"),

    # --- Renal / GU ---
    "C0041364": ("Tumour lysis syndrome", "renal/metabolic"),

    # --- Autoimmune (paradoxical for anti-TNF, etc.) ---
    "C0004364": ("Autoimmune disorder", "autoimmune"),
    "C0919715": ("Lupus-like syndrome", "autoimmune"),

    # --- Metabolic / acid-base ---
    "C0220981": ("Metabolic acidosis", "metabolic"),
    "C0001125": ("Lactic acidosis", "metabolic"),
    "C0025637": ("Methaemoglobinaemia", "metabolic"),

    # --- Bone (anti-resorptives, anti-VEGF, mTOR) ---
    "C0029445": ("Osteonecrosis", "bone"),
    "C2711248": ("Osteonecrosis of jaw", "bone"),

    # --- Reactions ---
    "C0948715": ("Infusion related reaction", "reaction"),
}


def main() -> int:
    print("=" * 78)
    print("Sprint 7A: expanding side-effect vocabulary for oncology + immunology")
    print("=" * 78)

    # Verify each addition exists in SIDER raw
    print("\n[1/3] Verifying additions exist in SIDER...")
    found_in_sider = set()
    se_drug_counts = defaultdict(int)
    with open(SIDER_PATH) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            _, _, _, mtype, umls, _ = parts[:6]
            if mtype != "PT":
                continue
            if umls in ONC_IMMUNO_ADDITIONS:
                found_in_sider.add(umls)
                se_drug_counts[umls] += 1
    not_found = set(ONC_IMMUNO_ADDITIONS) - found_in_sider
    print(f"  {len(found_in_sider)}/{len(ONC_IMMUNO_ADDITIONS)} additions verified in SIDER")
    if not_found:
        print(f"  NOT FOUND ({len(not_found)}): {not_found}")

    # Load existing vocab
    print("\n[2/3] Loading existing vocab...")
    with open(RESULTS / "side_effect_vocab.json") as f:
        existing = json.load(f)
    print(f"  existing: {len(existing['umls_ids'])} terms")

    # Build expanded vocab
    print("\n[3/3] Building expanded vocab...")
    new_umls_ids = list(existing["umls_ids"])
    new_display = dict(existing["display_names"])
    new_categories: dict[str, str] = {}
    added = 0
    for umls, (display, category) in ONC_IMMUNO_ADDITIONS.items():
        if umls in found_in_sider and umls not in new_display:
            new_umls_ids.append(umls)
            new_display[umls] = display
            new_categories[umls] = category
            added += 1
    # Tag categories on existing vocab as 'general'
    for u in existing["umls_ids"]:
        new_categories.setdefault(u, "general")

    print(f"  added {added} new onc/immuno terms")
    print(f"  expanded total: {len(new_umls_ids)} terms")

    # Save expanded vocab
    out = {
        "k": len(new_umls_ids),
        "umls_ids": new_umls_ids,
        "display_names": new_display,
        "categories": new_categories,
        "n_original": len(existing["umls_ids"]),
        "n_added_onc_immuno": added,
    }
    out_path = RESULTS / "side_effect_vocab_expanded.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved: {out_path}")

    # Show category breakdown
    print("\nCategory breakdown of NEW additions:")
    cat_counts = defaultdict(int)
    for c in new_categories.values():
        if c != "general":
            cat_counts[c] += 1
    for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:<30s} {n:>3d}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
