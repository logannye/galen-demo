"""Sprint E: Expand target vocab with ~100 modern biologic targets.

Adds vocabulary for:
  - Bispecific T-cell engager targets (BCMA, GPRC5D, CD3γ/ε, DLL3, CLDN18.2)
  - ADC targets (TROP2, Nectin-4, HER3, MUC16, FOLR1, mesothelin, etc.)
  - Modern IO targets (TIGIT, TIM3, LAG3, OX40, 4-1BB, GITR, ICOS, B7-H3)
  - Cell-therapy targets (CD19, CD20 already; CD22, CD33, CD30, CD38, CD47)
  - Immunology (TSLP, CRTH2, CCR4, SLAMF7, IL36R, IL13/IL13R, TYK2, complement)
  - Neurology biologics (CGRP receptor, APP, α-synuclein, tau)
  - Metabolic biologics (PCSK9, ANGPTL3, APOC3, LPA, FGF21, GIP/GCG receptors)
  - Hematology / coagulation (vWF, F8/F9/F11, C5AR1)

Each target entry: {uniprot, gene_symbol, target_pref_name}
Format matches existing target_vocab.json.
"""
from __future__ import annotations

import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


# Curated list of modern biologic targets.
# Format: (uniprot, gene_symbol, target_pref_name)
BIOLOGIC_TARGETS_TO_ADD: list[tuple[str, str, str]] = [
    # --- CRITICAL: long-standing missing biologic targets (Sprint 7+ curated
    #     priors were silently filtered out because target_vocab didn't include
    #     these). Adding here brings the SCM substrate online for the major
    #     biologic classes.
    ("Q15116", "PDCD1", "Programmed cell death protein 1 (PD-1)"),
    ("P16410", "CTLA4", "Cytotoxic T-lymphocyte protein 4"),
    ("P18627", "LAG3", "Lymphocyte activation gene 3 protein"),
    ("P11836", "MS4A1", "B-lymphocyte antigen CD20"),
    ("P15391", "CD19", "B-lymphocyte antigen CD19"),
    ("P20273", "CD22", "B-cell receptor CD22"),
    ("Q02223", "TNFRSF17", "Tumor necrosis factor receptor superfamily member 17 (BCMA)"),
    ("P31358", "CD52", "CAMPATH-1 antigen"),
    ("P08887", "IL6R", "Interleukin-6 receptor subunit alpha"),
    ("Q16552", "IL17A", "Interleukin-17A"),
    ("Q9NPF7", "IL23A", "Interleukin-23 subunit alpha"),
    ("P29460", "IL12B", "Interleukin-12 subunit beta"),
    ("P05113", "IL5", "Interleukin-5"),
    ("Q01344", "IL5RA", "Interleukin-5 receptor subunit alpha"),
    ("P24394", "IL4R", "Interleukin-4 receptor subunit alpha"),
    ("P14778", "IL1R1", "Interleukin-1 receptor type 1"),
    ("P01589", "IL2RA", "Interleukin-2 receptor subunit alpha"),
    ("Q9Y275", "TNFSF13B", "Tumor necrosis factor ligand superfamily member 13B (BAFF)"),
    ("O14788", "TNFSF11", "Tumor necrosis factor ligand superfamily member 11 (RANKL)"),
    ("P13612", "ITGA4", "Integrin alpha-4"),
    ("P26010", "ITGB7", "Integrin beta-7"),
    ("P13726", "F3", "Tissue factor / coagulation factor III"),
    ("P17181", "IFNAR1", "Interferon alpha/beta receptor 1"),
    ("P01008", "SERPINC1", "Antithrombin-III"),
    ("P21453", "S1PR1", "Sphingosine 1-phosphate receptor 1"),  # already may exist
    ("P10415", "BCL2", "B-cell lymphoma 2 protein"),  # may already exist
    ("P48736", "PIK3CG", "Phosphatidylinositol 4,5-bisphosphate 3-kinase catalytic subunit gamma"),

    # --- Bispecific T-cell engager targets ---
    ("Q9NZD1", "GPRC5D", "G-protein coupled receptor family C group 5 member D"),
    ("Q9NYJ7", "DLL3", "Delta-like protein 3"),
    ("P56856", "CLDN18", "Claudin-18"),
    ("Q9UM44", "CD3D", "T-cell surface glycoprotein CD3 delta chain"),
    ("P07766", "CD3E", "T-cell surface glycoprotein CD3 epsilon chain"),
    ("P09693", "CD3G", "T-cell surface glycoprotein CD3 gamma chain"),

    # --- ADC targets ---
    ("P09758", "TACSTD2", "Tumor-associated calcium signal transducer 2 (TROP2)"),
    ("Q96NY8", "NECTIN4", "Nectin-4"),
    ("P21860", "ERBB3", "Receptor tyrosine-protein kinase erbB-3 (HER3)"),
    ("Q15303", "ERBB4", "Receptor tyrosine-protein kinase erbB-4 (HER4)"),
    ("Q13421", "MSLN", "Mesothelin"),
    ("P15328", "FOLR1", "Folate receptor alpha"),
    ("Q04609", "FOLH1", "Glutamate carboxypeptidase 2 (PSMA)"),
    ("Q13641", "TPBG", "Trophoblast glycoprotein (5T4)"),
    ("Q8WXI7", "MUC16", "Mucin-16 (CA125)"),
    ("Q13433", "SLC39A6", "Zinc transporter ZIP6 (LIV-1)"),
    ("P11279", "LAMP1", "Lysosome-associated membrane glycoprotein 1"),
    ("Q5ZPR3", "CD276", "CD276 antigen (B7-H3)"),
    ("Q7Z7D3", "VTCN1", "V-set domain-containing T-cell activation inhibitor 1 (B7-H4)"),

    # --- Cell surface ADC / cell-therapy targets ---
    ("P28908", "TNFRSF8", "Tumor necrosis factor receptor superfamily member 8 (CD30)"),
    ("P20138", "CD33", "Myeloid cell surface antigen CD33"),
    ("P28907", "CD38", "ADP-ribosyl cyclase/cyclic ADP-ribose hydrolase 1 (CD38)"),
    ("Q08722", "CD47", "Leukocyte surface antigen CD47"),
    ("P78324", "SIRPA", "Tyrosine-protein phosphatase non-receptor type substrate 1 (SIRPα)"),
    ("P08069", "IGF1R", "Insulin-like growth factor 1 receptor"),

    # --- Modern IO checkpoint receptors ---
    ("Q495A1", "TIGIT", "T-cell immunoreceptor with Ig and ITIM domains"),
    ("Q8TDQ0", "HAVCR2", "Hepatitis A virus cellular receptor 2 (TIM-3)"),
    ("Q7Z6A9", "BTLA", "B- and T-lymphocyte attenuator"),
    ("P43489", "TNFRSF4", "Tumor necrosis factor receptor superfamily member 4 (OX40)"),
    ("Q9Y5U5", "TNFRSF18", "Tumor necrosis factor receptor superfamily member 18 (GITR)"),
    ("Q07011", "TNFRSF9", "Tumor necrosis factor receptor superfamily member 9 (4-1BB/CD137)"),
    ("Q9Y6W8", "ICOS", "Inducible T-cell costimulator"),
    ("P25942", "CD40", "Tumor necrosis factor receptor superfamily member 5 (CD40)"),
    ("P29965", "CD40LG", "CD40 ligand"),
    ("P26842", "CD27", "CD27 antigen"),
    ("P26718", "KLRK1", "NKG2-D type II integral membrane protein"),
    ("Q9H7M9", "VSIR", "V-type immunoglobulin domain-containing suppressor of T-cell activation (VISTA)"),

    # --- Cytokine / cytokine receptor (biologic) ---
    ("Q969D9", "TSLP", "Thymic stromal lymphopoietin"),
    ("Q9Y5Y4", "PTGDR2", "Prostaglandin D2 receptor 2 (CRTH2)"),
    ("Q9NQ25", "SLAMF7", "SLAM family member 7"),
    ("P51679", "CCR4", "C-C chemokine receptor type 4"),
    ("Q9HBE5", "IL36R", "Interleukin-36 receptor"),
    ("Q14116", "IL18", "Interleukin-18"),
    ("P40933", "IL15", "Interleukin-15"),
    ("Q13261", "IL15RA", "Interleukin-15 receptor subunit alpha"),
    ("P35225", "IL13", "Interleukin-13"),
    ("P78552", "IL13RA1", "Interleukin-13 receptor subunit alpha-1"),
    ("Q8NI17", "IL31RA", "Interleukin-31 receptor subunit alpha"),
    ("Q99650", "OSMR", "Oncostatin-M-specific receptor subunit beta"),
    ("P29597", "TYK2", "Non-receptor tyrosine-protein kinase TYK2"),
    ("Q96F46", "IL17RA", "Interleukin-17 receptor A"),

    # --- Complement (eculizumab + ravulizumab + others) ---
    ("P02745", "C1QA", "Complement C1q subcomponent subunit A"),
    ("P00736", "C1R", "Complement C1r subcomponent"),
    ("P09871", "C1S", "Complement C1s subcomponent"),
    ("P01024", "C3", "Complement C3"),
    ("P21730", "C5AR1", "C5a anaphylatoxin chemotactic receptor 1"),
    ("P03952", "KLKB1", "Plasma kallikrein"),
    ("P00748", "F12", "Coagulation factor XII"),

    # --- Coagulation / hematology ---
    ("P04275", "VWF", "von Willebrand factor"),
    ("P00451", "F8", "Coagulation factor VIII"),
    ("P00740", "F9", "Coagulation factor IX"),
    ("P03951", "F11", "Coagulation factor XI"),
    ("Q14790", "F2RL1", "Proteinase-activated receptor 2"),

    # --- Bone / sclerostin ---
    ("Q9BQB4", "SOST", "Sclerostin"),
    ("Q9Y6Q6", "TNFRSF11A", "Tumor necrosis factor receptor superfamily member 11A (RANK)"),

    # --- Metabolic biologics ---
    ("Q8NBP7", "PCSK9", "Proprotein convertase subtilisin/kexin type 9"),
    ("Q9Y5C1", "ANGPTL3", "Angiopoietin-related protein 3"),
    ("P02656", "APOC3", "Apolipoprotein C-III"),
    ("P08519", "LPA", "Apolipoprotein(a)"),
    ("Q9NSA1", "FGF21", "Fibroblast growth factor 21"),
    ("P48546", "GIPR", "Glucose-dependent insulinotropic peptide receptor"),
    ("P47871", "GCGR", "Glucagon receptor"),
    ("O15123", "ANGPT2", "Angiopoietin-2"),

    # --- Neurology biologics ---
    ("P06881", "CALCA", "Calcitonin gene-related peptide 1"),
    ("Q16602", "CALCRL", "Calcitonin gene-related peptide type 1 receptor"),
    ("P05067", "APP", "Amyloid-beta A4 protein"),
    ("P37840", "SNCA", "Alpha-synuclein"),

    # --- Asthma/allergy/IgE ---
    ("P01854", "IGHE", "Immunoglobulin epsilon chain C region"),

    # --- BAFF / B-cell ---
    ("Q96RJ3", "TNFRSF13C", "Tumor necrosis factor receptor superfamily member 13C (BAFF-R)"),

    # --- Growth factors / receptors ---
    ("P15692", "VEGFA", "Vascular endothelial growth factor A"),
    ("P49765", "VEGFB", "Vascular endothelial growth factor B"),
    ("P49767", "VEGFC", "Vascular endothelial growth factor C"),
    ("P04085", "PDGFA", "Platelet-derived growth factor subunit A"),
    ("P01127", "PDGFB", "Platelet-derived growth factor subunit B"),
    ("P07333", "CSF1R", "Macrophage colony-stimulating factor 1 receptor"),
    ("P11362", "FGFR1", "Fibroblast growth factor receptor 1"),

    # --- Bispecific / ADC complement targets ---
    ("P30530", "AXL", "Tyrosine-protein kinase receptor UFO (AXL)"),
]


def main() -> int:
    print("=" * 78)
    print("Sprint E: Expand target_vocab with modern biologic targets")
    print("=" * 78)

    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)

    existing = {t["uniprot"] for t in tv["targets"]}
    print(f"[load] existing targets: {len(existing)}")

    added = 0
    skipped = 0
    new_targets = []
    for u, gene, pref_name in BIOLOGIC_TARGETS_TO_ADD:
        if u in existing:
            skipped += 1
            continue
        new_targets.append({
            "uniprot": u,
            "gene_symbol": gene,
            "target_pref_name": pref_name,
        })
        added += 1

    tv["targets"].extend(new_targets)
    tv["n_targets"] = len(tv["targets"])

    print(f"[expand] candidates: {len(BIOLOGIC_TARGETS_TO_ADD)}")
    print(f"[expand] added: {added}")
    print(f"[expand] skipped (already in vocab): {skipped}")
    print(f"[expand] new total: {tv['n_targets']}")

    # Backup + write
    backup = RESULTS / "target_vocab_pre_sprint_e.json"
    if not backup.exists():
        with open(backup, "w") as f:
            json.dump({"n_targets": len(tv["targets"]) - added,
                       "targets": tv["targets"][:-added] if added > 0 else tv["targets"]},
                      f, indent=2)
        print(f"[backup] {backup}")

    out_path = RESULTS / "target_vocab.json"
    with open(out_path, "w") as f:
        json.dump(tv, f, indent=2)
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
