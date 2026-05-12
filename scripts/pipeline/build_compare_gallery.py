"""Build cached predictions for the Compare-Analogs demo gallery.

Pre-runs curated analog series through the production stack
(SMILES → TargetNet → prob-weighted SCM → Hybrid LLM) and saves the full
batch result so the demo can show instant, polished comparisons without
running the LLM during an investor meeting.

Output: results/compare_gallery.json
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"


# Curated series. Each compound MUST exist in ChEMBL by name so we can
# look up the canonical SMILES. We deliberately pick FDA-approved real
# drugs in a class (not hypothetical analogs) so predictions are
# verifiable and the audience recognizes the names.
CURATED_SERIES = [
    {
        "id": "cdk46",
        "name": "CDK4/6 inhibitors",
        "indication": "HR+ HER2- metastatic breast cancer",
        "context": (
            "Three FDA-approved CDK4/6 inhibitors for hormone-receptor-positive "
            "breast cancer. All three engage CDK4/6 on-target; the differences "
            "in their off-target footprints drive divergent clinical safety "
            "profiles. This is exactly the triage decision a medicinal chemist "
            "or program lead would face when prioritising a candidate."
        ),
        "compounds": [
            {
                "id": "palbociclib",
                "name": "Palbociclib",
                "trade_name": "Ibrance",
                "manufacturer": "Pfizer",
                "year_approved": 2015,
                "known_profile_summary": (
                    "Most narrow off-target footprint; predominantly "
                    "haematologic AEs (severe neutropenia is the dose-limiting "
                    "tox)."
                ),
            },
            {
                "id": "ribociclib",
                "name": "Ribociclib",
                "trade_name": "Kisqali",
                "manufacturer": "Novartis",
                "year_approved": 2017,
                "known_profile_summary": (
                    "Carries FDA BBW for QT prolongation; class-wide neutropenia "
                    "plus a distinct cardiac liability."
                ),
            },
            {
                "id": "abemaciclib",
                "name": "Abemaciclib",
                "trade_name": "Verzenio",
                "manufacturer": "Eli Lilly",
                "year_approved": 2017,
                "known_profile_summary": (
                    "Broader kinome engagement; BBW for venous thromboembolism, "
                    "and severe diarrhea / hepatic injury alongside the class "
                    "haematologic signal."
                ),
            },
        ],
        "insight": (
            "Same on-target mechanism, three distinct safety signatures. A "
            "program lead choosing between these three would face a real "
            "trade-off: palbociclib has the narrowest off-target profile but "
            "the deepest neutropenia; ribociclib adds cardiac risk; "
            "abemaciclib trades narrower neutropenia for GI + thromboembolic "
            "risk. The system surfaces this trade-off from SMILES alone."
        ),
    },
    {
        "id": "vegfr_tkis",
        "name": "VEGFR tyrosine-kinase inhibitors",
        "indication": "Renal cell carcinoma, hepatocellular carcinoma, soft-tissue sarcoma",
        "context": (
            "Three FDA-approved multi-kinase inhibitors that all engage "
            "VEGFR2 (KDR) plus a range of other RTKs. Their off-target "
            "breadths vary widely — sunitinib is the most promiscuous, "
            "pazopanib the most narrow. Different off-target profiles drive "
            "different dose-limiting AEs."
        ),
        "compounds": [
            {
                "id": "sunitinib",
                "name": "Sunitinib",
                "trade_name": "Sutent",
                "manufacturer": "Pfizer",
                "year_approved": 2006,
                "known_profile_summary": (
                    "Most promiscuous VEGFR-TKI; hypertension, fatigue, "
                    "hand-foot syndrome, hypothyroidism, and a distinct "
                    "cardiotoxicity signal (LV dysfunction)."
                ),
            },
            {
                "id": "sorafenib",
                "name": "Sorafenib",
                "trade_name": "Nexavar",
                "manufacturer": "Bayer",
                "year_approved": 2005,
                "known_profile_summary": (
                    "RAF + VEGFR-TKI; hypertension, hand-foot syndrome, "
                    "GI bleeding. Used in HCC and RCC."
                ),
            },
            {
                "id": "pazopanib",
                "name": "Pazopanib",
                "trade_name": "Votrient",
                "manufacturer": "Novartis",
                "year_approved": 2009,
                "known_profile_summary": (
                    "Most selective VEGFR-TKI of the three; QT prolongation "
                    "(BBW), hepatotoxicity (BBW), narrower kinome footprint."
                ),
            },
        ],
        "insight": (
            "All three engage VEGFR2 and predict the class-effect "
            "cardiovascular signal (hypertension, MACE). But the off-target "
            "breadth varies, and that variance drives the differentiated "
            "AE pattern: sunitinib's wider kinome → cardiotoxicity + "
            "hypothyroidism; pazopanib's narrower profile → cleaner CV "
            "story but a distinct hepatic + QT liability."
        ),
    },
]


def _lookup_chembl_smiles(name: str) -> tuple[int | None, str | None]:
    conn = sqlite3.connect(CHEMBL_DB)
    cur = conn.cursor()
    # Prefer pref_name (FDA-style names usually match)
    cur.execute(
        "SELECT mol.molregno, cs.canonical_smiles "
        "FROM molecule_dictionary mol "
        "JOIN compound_structures cs USING(molregno) "
        "WHERE LOWER(mol.pref_name) = ? AND cs.canonical_smiles IS NOT NULL "
        "LIMIT 1",
        (name.lower(),),
    )
    row = cur.fetchone()
    if not row:
        # Fall back to synonyms
        cur.execute(
            "SELECT cs.molregno, cs.canonical_smiles "
            "FROM molecule_synonyms ms "
            "JOIN compound_structures cs ON cs.molregno = ms.molregno "
            "WHERE LOWER(ms.synonyms) = ? AND cs.canonical_smiles IS NOT NULL "
            "LIMIT 1",
            (name.lower(),),
        )
        row = cur.fetchone()
    conn.close()
    if row:
        return int(row[0]), row[1]
    return None, None


def _decision_rollup(predictions, top_k: int = 10) -> dict:
    """Compute critical/serious/common counts + top-3 critical AEs."""
    from scripts.baselines.clinical_taxonomy import severity_tier, organ_system_display
    crit = ser = com = 0
    critical_aes: list[dict] = []
    organ_counter: dict[str, int] = {}
    for p in predictions[:top_k]:
        tier = severity_tier(p.side_effect_umls)
        organ = organ_system_display(p.side_effect_umls)
        organ_counter[organ] = organ_counter.get(organ, 0) + 1
        if tier == "critical":
            crit += 1
            critical_aes.append({
                "rank": p.rank,
                "name": p.side_effect_name,
                "umls": p.side_effect_umls,
                "organ": organ,
            })
        elif tier == "serious":
            ser += 1
        else:
            com += 1
    return {
        "n_critical": crit,
        "n_serious": ser,
        "n_common": com,
        "critical_aes": critical_aes[:5],
        "organ_counts": organ_counter,
    }


def main() -> None:
    from scripts.targetnet.batch_predict import predict_batch
    from scripts.demo.predict_hybrid import ClinicalSafetyEngine

    print("Loading engine...", flush=True)
    engine = ClinicalSafetyEngine()

    output_series = []
    t0_total = time.time()
    for series in CURATED_SERIES:
        print(f"\n=== Series: {series['name']} ===", flush=True)
        # Resolve SMILES for each compound
        compounds = []
        for c in series["compounds"]:
            mr, smi = _lookup_chembl_smiles(c["id"])
            if not smi:
                print(f"  SKIP {c['id']}: no SMILES in ChEMBL", flush=True)
                continue
            compounds.append({
                "id": c["id"],
                "name": c["name"],
                "smiles": smi,
                "molregno": mr,
                "meta": c,
            })
            print(f"  {c['id']}: molregno={mr}, smiles={smi[:60]}...", flush=True)

        # Run batch prediction (cached predictions, instant in demo)
        batch_compounds = [
            {"id": c["id"], "name": c["name"], "smiles": c["smiles"]}
            for c in compounds
        ]
        lead_id = compounds[0]["id"]
        t0 = time.time()
        result = predict_batch(
            batch_compounds, lead_id=lead_id, therapeutic_area="Oncology",
            engine=engine,
        )
        print(f"  predictions done in {time.time()-t0:.1f}s", flush=True)

        # Build per-compound payload
        compound_payloads = []
        for entry, compound_meta in zip(result.compounds, compounds):
            if entry.error or not entry.predictions:
                print(f"  WARN: {entry.compound_id} failed: {entry.error}",
                      flush=True)
                continue
            rollup = _decision_rollup(entry.predictions, top_k=10)
            compound_payloads.append({
                "id": entry.compound_id,
                "name": compound_meta["meta"]["name"],
                "trade_name": compound_meta["meta"].get("trade_name", ""),
                "manufacturer": compound_meta["meta"].get("manufacturer", ""),
                "year_approved": compound_meta["meta"].get("year_approved"),
                "known_profile_summary": compound_meta["meta"].get(
                    "known_profile_summary", "",
                ),
                "smiles": compound_meta["smiles"],
                "n_binding_targets": len(entry.binding_profile),
                "top10_predictions": [
                    {
                        "rank": p.rank,
                        "umls": p.side_effect_umls,
                        "name": p.side_effect_name,
                    }
                    for p in entry.predictions[:10]
                ],
                **rollup,
            })

        output_series.append({
            "id": series["id"],
            "name": series["name"],
            "indication": series["indication"],
            "context": series["context"],
            "insight": series["insight"],
            "compounds": compound_payloads,
        })

    out_path = RESULTS / "compare_gallery.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_series": len(output_series),
            "series": output_series,
        }, f, indent=2)
    print(f"\nWrote {out_path}: {len(output_series)} series, "
          f"{sum(len(s['compounds']) for s in output_series)} compounds "
          f"({time.time()-t0_total:.1f}s total)")


if __name__ == "__main__":
    main()
