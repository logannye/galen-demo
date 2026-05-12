"""Sprint I: add missing biologic/small-molecule targets for OOD expansion.

Adds targets needed to make post-2024 drugs eligible in the OOD
benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


NEW_TARGETS_I: list[tuple[str, str, str]] = [
    # --- Targets enabling previously-ineligible OOD cases ---
    ("O75874", "IDH1", "Isocitrate dehydrogenase (NADP) cytoplasmic 1"),
    ("O14746", "TERT", "Telomerase reverse transcriptase"),
    ("P61073", "CXCR4", "C-X-C chemokine receptor type 4"),

    # --- 2024+ approvals requiring new targets ---
    ("P01116", "KRAS", "GTPase KRas"),
    ("Q92542", "NCSTN", "Nicastrin"),
    ("P12883", "MYH7", "Myosin-7 cardiac beta isoform"),
    ("P10646", "TFPI", "Tissue factor pathway inhibitor"),
    ("P11926", "ODC1", "Ornithine decarboxylase 1"),

    # --- Other commonly-bound targets not yet in vocab ---
    ("P01116", "KRAS", "GTPase KRas"),
    ("P11362", "FGFR1", "Fibroblast growth factor receptor 1"),  # may be in
    ("P22455", "FGFR4", "Fibroblast growth factor receptor 4"),  # may be in
    ("P21802", "FGFR2", "Fibroblast growth factor receptor 2"),  # may be in
    ("P22607", "FGFR3", "Fibroblast growth factor receptor 3"),  # may be in
    ("Q9UHB4", "NDOR1", "NADPH-dependent diflavin oxidoreductase 1"),
    ("P14735", "IDE", "Insulin-degrading enzyme"),
    ("P55072", "VCP", "Transitional endoplasmic reticulum ATPase"),
    ("Q9Y5K8", "ATP6V1D", "V-type proton ATPase subunit D"),
    ("P02768", "ALB", "Serum albumin (carrier; for SGLT2 etc)"),

    # --- Newer cell-therapy / gene-therapy targets ---
    ("P02775", "PPBP", "Platelet basic protein"),
    ("P29622", "SERPINA4", "Kallistatin"),

    # --- Antineoplastic mAb targets ---
    ("P26006", "ITGA3", "Integrin alpha-3"),
    ("Q9NRD8", "DUOX2", "Dual oxidase 2"),

    # --- Coagulation / hematology specifics ---
    ("P00751", "CFB", "Complement factor B (iptacopan target)"),
    ("P00748", "F12", "Coagulation factor XII"),
    ("P03952", "KLKB1", "Plasma kallikrein (already?)"),

    # --- Immunology ---
    ("Q14213", "EBI3", "IL-27 / IL-35 subunit beta"),
    ("Q14116", "IL18", "Interleukin-18"),

    # --- Drug-development ---
    ("Q9Y6Y9", "LY96", "MD-2 / lymphocyte antigen 96"),

    # --- Aging / senescence ---
    ("P49675", "ALAS1", "Delta-aminolevulinate synthase nonspecific (NB drug)"),
]


def main() -> int:
    print("=" * 78)
    print("Sprint I: target_vocab expansion for OOD eligibility")
    print("=" * 78)

    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    existing = {t["uniprot"] for t in tv["targets"]}
    print(f"[load] existing: {len(existing)}")

    added = 0
    skipped = 0
    for u, gene, pref_name in NEW_TARGETS_I:
        if u in existing:
            skipped += 1
            continue
        tv["targets"].append({
            "uniprot": u,
            "gene_symbol": gene,
            "target_pref_name": pref_name,
        })
        existing.add(u)
        added += 1

    tv["n_targets"] = len(tv["targets"])
    print(f"[expand] added={added}, skipped={skipped}, new total={tv['n_targets']}")
    with open(RESULTS / "target_vocab.json", "w") as f:
        json.dump(tv, f, indent=2)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
