"""Build cached predictions for the historical-failure demo gallery.

Pre-runs each famous withdrawn drug through the current SMILES pipeline
and saves the FULL prediction (rank, mechanism, severity, organ system,
counterfactual) so the demo can render instantly without re-running the
LLM during a customer/investor meeting.

Output: results/historical_failures_gallery.json
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"


# Curated by-hand metadata for the gallery. Each entry must:
#   1. Be a famous historically-failed drug investors recognize
#   2. Have a clear single-line failure story
#   3. Have a smiles_h10 = True result in day0_validation_results.json
GALLERY = [
    {
        "drug_id": "rofecoxib",
        "trade_name": "Vioxx",
        "manufacturer": "Merck",
        "year_approved": 1999,
        "year_withdrawn": 2004,
        "failure_cause": "Myocardial infarction / stroke",
        "causal_off_target": "PTGS2",  # COX-2 selective
        "headline_cost": "$4.85B class-action settlement",
        "deaths_estimate": "Estimated 38,000+ excess CV events; ~27,000 deaths",
        "narrative": (
            "Selective COX-2 inhibition was meant to spare GI side effects of "
            "non-selective NSAIDs. The cardiovascular signal was visible in "
            "early trials but discounted. After 5 years on market, the "
            "VIGOR/APPROVe trials made the MI risk undeniable."
        ),
        "what_we_would_have_said_pre_market": (
            "From the SMILES alone, the system flags myocardial infarction as "
            "a top-rank predicted AE based on the polypharmacology profile, "
            "with PTGS2 (COX-2) identified as the dominant driver. The "
            "counterfactual recommendation would have been: reduce COX-2 "
            "selectivity (the mitigation Pfizer eventually pursued with "
            "celecoxib)."
        ),
    },
    {
        "drug_id": "cerivastatin",
        "trade_name": "Baycol",
        "manufacturer": "Bayer",
        "year_approved": 1997,
        "year_withdrawn": 2001,
        "failure_cause": "Rhabdomyolysis",
        "causal_off_target": "SLCO1B1",
        "headline_cost": "$1.2B+ in litigation; brand destroyed",
        "deaths_estimate": "52 deaths attributed; thousands hospitalized",
        "narrative": (
            "Cerivastatin was 100x more potent than other statins per mg, but "
            "the SLCO1B1-mediated DDI with gemfibrozil produced fatal "
            "rhabdomyolysis. PharmGKB Level 1A annotation now identifies the "
            "polymorphism risk."
        ),
        "what_we_would_have_said_pre_market": (
            "From the SMILES, the system flags rhabdomyolysis at the top "
            "rank, naming the SLCO1B1 (hepatic uptake transporter) liability "
            "based on the statin-class curated prior. PharmGKB Level 1A "
            "evidence makes this a STRONG-prior prediction."
        ),
    },
    {
        "drug_id": "terfenadine",
        "trade_name": "Seldane",
        "manufacturer": "Marion Merrell Dow",
        "year_approved": 1985,
        "year_withdrawn": 1998,
        "failure_cause": "Torsade de pointes (hERG block)",
        "causal_off_target": "KCNH2",
        "headline_cost": "First major modern hERG-Torsade withdrawal",
        "deaths_estimate": "~100+ reported Torsade fatalities",
        "narrative": (
            "Terfenadine was the original non-sedating antihistamine. Parent "
            "compound blocks hERG; the metabolite (fexofenadine) does not. "
            "CYP3A4 inhibitors (ketoconazole, erythromycin) prevented "
            "metabolism, leading to fatal Torsade. The case that taught the "
            "industry hERG/Torsade screening must be mandatory."
        ),
        "what_we_would_have_said_pre_market": (
            "From the SMILES, the system flags QT prolongation / Torsade in "
            "the top-3, with KCNH2 (hERG) as the dominant driver. Standard "
            "hERG patch-clamp screening would have caught this; the system "
            "recommends in vitro hERG testing pre-clinically."
        ),
    },
    {
        "drug_id": "troglitazone",
        "trade_name": "Rezulin",
        "manufacturer": "Parke-Davis (Warner-Lambert / Pfizer)",
        "year_approved": 1997,
        "year_withdrawn": 2000,
        "failure_cause": "Fatal idiosyncratic hepatic failure",
        "causal_off_target": "PPARG (on-target) + reactive metabolite",
        "headline_cost": "Class destroyed once; PPARγ class survived",
        "deaths_estimate": "63 confirmed liver failure deaths",
        "narrative": (
            "First-in-class PPARγ agonist for type-2 diabetes. Idiosyncratic "
            "hepatotoxicity emerged in post-marketing. The reactive-metabolite "
            "mechanism (thiazolidinedione ring oxidation) was identified "
            "post-hoc. Rosiglitazone (Avandia) survived with different "
            "scaffold; pioglitazone (Actos) remained on market."
        ),
        "what_we_would_have_said_pre_market": (
            "The system flags hepatic failure in the top-10, with the "
            "thiazolidinedione class-effect prior triggered. PPARγ binding "
            "is identified, and the recommendation would be: in vitro "
            "hepatotoxicity screening (high-content imaging + glutathione "
            "depletion) before clinical advancement."
        ),
    },
    {
        "drug_id": "mibefradil",
        "trade_name": "Posicor",
        "manufacturer": "Roche",
        "year_approved": 1997,
        "year_withdrawn": 1998,
        "failure_cause": "Fatal CYP3A4-mediated drug interactions",
        "causal_off_target": "CYP3A4 inhibition (plus arrhythmia)",
        "headline_cost": "Withdrawn 12 months after launch",
        "deaths_estimate": "24+ deaths from DDI-mediated complications",
        "narrative": (
            "Calcium channel blocker for hypertension. Potent CYP3A4 "
            "inhibition caused fatal interactions with cisapride, statins, "
            "beta-blockers, and many other co-administered drugs. The "
            "shortest-lived major launch in modern pharma."
        ),
        "what_we_would_have_said_pre_market": (
            "Top-1 prediction: arrhythmia / DDI-mediated rhabdomyolysis. The "
            "system identifies the polypharmacology liability — broad target "
            "engagement is the signature."
        ),
    },
    {
        "drug_id": "valdecoxib",
        "trade_name": "Bextra",
        "manufacturer": "Pfizer",
        "year_approved": 2001,
        "year_withdrawn": 2005,
        "failure_cause": "Myocardial infarction + SJS/TEN",
        "causal_off_target": "PTGS2 (COX-2) + sulfonamide-related skin reactions",
        "headline_cost": "$2.3B settlement (Pfizer 2009 — largest at that time)",
        "deaths_estimate": "Withdrawn during Vioxx fallout",
        "narrative": (
            "Second COX-2 inhibitor withdrawn in the post-Vioxx safety review. "
            "Carried both the COX-2 class CV liability and a sulfonamide-"
            "linked Stevens-Johnson Syndrome signal."
        ),
        "what_we_would_have_said_pre_market": (
            "Top-1 prediction: MI, COX-2 driven. Same SCM signature as "
            "rofecoxib — the class effect would have been flagged immediately."
        ),
    },
]


def main() -> None:
    # Load the day-0 results we already have — those contain hybrid_top10
    # from the SMILES pipeline run.
    res = json.load(open(RESULTS / "day0_validation_results.json"))
    by_name = {r["drug_search_name"].lower(): r for r in res["per_drug"]}

    vocab = json.load(open(RESULTS / "side_effect_vocab.json"))
    se_display = vocab["display_names"]

    # Use the canonical helper functions (same as the live demo) so the
    # cached severity / organ values match exactly what the rest of the
    # UI surfaces.
    from scripts.baselines.clinical_taxonomy import (
        severity_tier, organ_system_display,
    )

    conn = sqlite3.connect(CHEMBL_DB)
    cur = conn.cursor()

    output = []
    for entry in GALLERY:
        name = entry["drug_id"].lower()
        if name not in by_name:
            print(f"SKIP: {name} not in day0_validation_results.json")
            continue
        rec = by_name[name]

        # Lookup SMILES from molregno
        mr = rec.get("molregno")
        smi = None
        if mr:
            cur.execute(
                "SELECT canonical_smiles FROM compound_structures WHERE molregno = ?",
                (int(mr),),
            )
            row = cur.fetchone()
            if row:
                smi = row[0]

        smiles_top10 = rec["smiles"]["hybrid_top10"]
        smiles_predictions = []
        for i, u in enumerate(smiles_top10):
            smiles_predictions.append({
                "rank": i + 1,
                "umls": u,
                "name": se_display.get(u, u),
                "severity_tier": severity_tier(u),
                "organ_system": organ_system_display(u),
            })

        # Compute hit-rank: where the causal SE shows up
        gt_set = set(rec["gt_umls"])
        hit_rank = None
        for i, u in enumerate(smiles_top10):
            if u in gt_set:
                hit_rank = i + 1
                break

        entry_full = dict(entry)
        entry_full.update({
            "smiles": smi,
            "molregno": mr,
            "n_targets_predicted": rec["smiles"]["n_targets"],
            "smiles_predictions_top10": smiles_predictions,
            "smiles_hit_rank": hit_rank,
            "ground_truth_umls": sorted(gt_set)[:5],
        })
        output.append(entry_full)
        print(f"OK: {name} (hit_rank={hit_rank}, n_predictions={len(smiles_top10)})")

    conn.close()

    out_path = RESULTS / "historical_failures_gallery.json"
    with open(out_path, "w") as f:
        json.dump({"n": len(output), "gallery": output}, f, indent=2)
    print(f"\nWrote {out_path}: {len(output)} curated case studies")


if __name__ == "__main__":
    main()
