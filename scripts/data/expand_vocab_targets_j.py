"""Sprint J: target vocab expansion for n=150+ OOD scaling."""
from __future__ import annotations

import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


NEW_TARGETS_J: list[tuple[str, str, str]] = [
    # 2024-2025 small-mol oncology drug targets
    ("Q99814", "EPAS1", "Endothelial PAS domain-containing protein 1 (HIF-2α)"),
    ("P07949", "RET", "Proto-oncogene tyrosine-protein kinase receptor RET"),
    ("P04629", "NTRK1", "High affinity nerve growth factor receptor (TrkA)"),
    ("Q16620", "NTRK2", "BDNF/NT-3 growth factors receptor (TrkB)"),
    ("Q16288", "NTRK3", "NT-3 growth factor receptor (TrkC)"),
    ("P48735", "IDH2", "Isocitrate dehydrogenase NADP mitochondrial"),
    ("Q15910", "EZH2", "Histone-lysine N-methyltransferase EZH2"),
    ("P08581", "MET", "Hepatocyte growth factor receptor (already may exist)"),

    # 2024-2025 immunology/neuro
    ("P29597", "TYK2", "Non-receptor tyrosine-protein kinase TYK2 (may exist)"),
    ("Q9Y5N1", "HRH3", "Histamine H3 receptor"),
    ("P14867", "GABRA1", "GABA-A receptor subunit alpha-1"),
    ("O43613", "HCRTR1", "Orexin receptor type 1"),
    ("O43614", "HCRTR2", "Orexin receptor type 2"),
    ("P55899", "FCGRT", "IgG receptor FcRn large subunit p51"),

    # 2024-2025 cardiometabolic
    ("P13945", "ADRB3", "Beta-3 adrenergic receptor"),
    ("Q8TDS4", "HCAR2", "Hydroxycarboxylic acid receptor 2 (niacin)"),
    ("P20648", "ATP4A", "Potassium-transporting ATPase alpha chain 1"),
    ("Q16236", "NFE2L2", "Nuclear factor erythroid 2-related factor 2 (Nrf2)"),
    ("P10912", "GHR", "Growth hormone receptor"),
    ("Q12908", "SLC10A2", "Ileal sodium/bile acid cotransporter (NTCP)"),

    # 2024-2025 specialty
    ("P40967", "PMEL", "Premelanosome protein gp100"),
    ("P39060", "COL18A1", "Endostatin precursor"),
    ("Q13133", "NR1H3", "Oxysterols receptor LXR-alpha"),
]


def main() -> int:
    print("=" * 78)
    print("Sprint J: target vocab expansion")
    print("=" * 78)

    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    existing = {t["uniprot"] for t in tv["targets"]}
    print(f"[load] existing: {len(existing)}")

    added = 0
    skipped = 0
    for u, gene, pref_name in NEW_TARGETS_J:
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
