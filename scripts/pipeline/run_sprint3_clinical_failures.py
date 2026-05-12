"""Sprint 3: Clinical-failure prediction benchmark.

For each of the 15 FDA-withdrawn/restricted drugs, run all methods and
check whether the causal toxicity was predicted (top-K rank) AND whether
the SCM's per-target attribution included the known causal off-target.

For drugs IN the SIDER ∩ ChEMBL catalog: leave-one-out — exclude the
drug from SCM training, re-learn α(S|T), then score.

For drugs NOT in the catalog: fetch ChEMBL binding profile directly,
use the FULL training-set SCM, then score.

Outputs:
  results/sprint3_clinical_failures.json — per-drug results
  results/sprint3_summary.json            — aggregate metrics
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from ..baselines.llm_baselines import (
    rank_side_effects_llm_drug_blind, rank_side_effects_llm_with_name,
)
from ..baselines.rf_ecfp_baseline import (
    build_ecfp_matrix, compute_ecfp, fetch_smiles_for_drugs,
    rank_test_drug_rf, train_rf_models,
)
from ..baselines.logreg_baseline import (
    build_se_label_matrix, build_target_feature_matrix,
)
from ..data.build_catalog import query_binding_profile
from ..data.clinical_failures import CLINICAL_FAILURES
from ..llm import SonnetClient
from ..scm.edge_learning import learn_edges
from ..scm.scoring import top_k_predictions


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"
TOP_K_RANK = 50


def best_causal_rank(ranked: list[str], causal_se_ids: tuple[str, ...]) -> int | None:
    """Return the rank (1-indexed) of the BEST (lowest-rank) causal side effect."""
    best = None
    for i, s in enumerate(ranked, start=1):
        if s in causal_se_ids:
            if best is None or i < best:
                best = i
    return best


def lookup_chembl_molregno(name: str) -> int | None:
    conn = sqlite3.connect(CHEMBL_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT molregno FROM molecule_dictionary
        WHERE LOWER(pref_name)=? AND max_phase > 0 LIMIT 1
    """, (name.lower(),))
    r = cur.fetchone()
    if not r:
        cur.execute("""
            SELECT DISTINCT m.molregno FROM molecule_dictionary m
            JOIN molecule_synonyms s ON m.molregno=s.molregno
            WHERE LOWER(s.synonyms)=? LIMIT 1
        """, (name.lower(),))
        r = cur.fetchone()
    conn.close()
    return r[0] if r else None


def fetch_smiles(molregno: int) -> str | None:
    conn = sqlite3.connect(CHEMBL_DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT canonical_smiles FROM compound_structures WHERE molregno=?",
        (molregno,),
    )
    r = cur.fetchone()
    conn.close()
    return r[0] if r and r[0] else None


def main() -> int:
    print("=" * 78)
    print("Sprint 3: Clinical-Failure prediction benchmark")
    print(f"          {len(CLINICAL_FAILURES)} FDA-withdrawn/restricted drugs")
    print("=" * 78)

    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    all_drugs = cat["drugs"]
    drugs_by_name = {(d["drug_name"] or "").lower(): d for d in all_drugs}

    with open(RESULTS / "side_effect_vocab.json") as f:
        vocab_payload = json.load(f)
    se_vocab = vocab_payload["umls_ids"]
    se_names = vocab_payload["display_names"]

    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    target_list = sorted({t["uniprot"] for t in tv["targets"]})
    target_info = {t["uniprot"]: t for t in tv["targets"]}

    # Pre-train RF-ECFP once (same models as Sprint 2.2 — chemistry-only baseline)
    print("\n[setup/1] Training RF-ECFP on full training catalog...")
    train_drugs = [d for d in all_drugs if d["split"] == "train"]
    smiles_map_all = fetch_smiles_for_drugs(all_drugs)
    X_train_ecfp, train_ecfp_idx = build_ecfp_matrix(train_drugs, smiles_map_all)
    Y_train = build_se_label_matrix(train_drugs, se_vocab)
    Y_train_ecfp = Y_train[train_ecfp_idx]
    rf_models = train_rf_models(X_train_ecfp, Y_train_ecfp,
                                 n_estimators=50, max_depth=8)
    base_rate_array = np.clip(Y_train.mean(axis=0), 1e-6, 1.0 - 1e-6)
    print(f"          RF-ECFP fit: {sum(1 for m in rf_models.values() if m is not None)}/{len(se_vocab)} side effects")

    print("[setup/2] Loading SCM α(S|T) edges (full training)...")
    with open(RESULTS / "scm_edges.json") as f:
        full_edges = json.load(f)

    client = SonnetClient()

    # ---- Per-failure evaluation
    results: list[dict] = []
    t_start = time.monotonic()
    for i, cf in enumerate(CLINICAL_FAILURES, start=1):
        print(f"\n[case {i}/{len(CLINICAL_FAILURES)}] {cf.drug_id:<16s} | "
              f"{cf.causal_off_target_gene} → {cf.causal_side_effects_display[:50]}")

        # ---- Resolve binding profile + (if catalog) leave-one-out SCM
        in_cat = cf.in_sider_catalog
        binding_profile = []
        edges_used = full_edges
        scm_provenance = ""
        molregno_resolved = None

        if in_cat:
            d = drugs_by_name.get(cf.drug_id.lower()) or drugs_by_name.get(cf.drug_search_name.lower())
            if d is None:
                print(f"          WARNING: claimed in catalog but not found")
                continue
            binding_profile = d["binding_profile"]
            molregno_resolved = d["molregno"]
            scm_provenance = f"leave-one-out from training (split={d['split']})"
            # Leave-one-out: retrain α(S|T) excluding this drug from training
            train_minus = [t for t in train_drugs if t["cid"] != d["cid"]]
            edges_used, _ = learn_edges(train_minus, se_vocab)
            print(f"          binding profile: {len(binding_profile)} targets (catalog); "
                  f"retrained SCM on {len(train_minus)} drugs")
        else:
            # Fetch from ChEMBL
            molregno = lookup_chembl_molregno(cf.drug_search_name)
            if molregno is None:
                print(f"          SKIP: not in ChEMBL")
                continue
            molregno_resolved = molregno
            conn = sqlite3.connect(CHEMBL_DB)
            binding_profile = query_binding_profile(conn, molregno)
            conn.close()
            edges_used = full_edges
            scm_provenance = "full training SCM (drug not in catalog)"
            print(f"          binding profile: {len(binding_profile)} targets (ChEMBL); "
                  f"using full training SCM")

        if not binding_profile:
            print(f"          SKIP: no usable binding profile")
            continue

        # Has the causal off-target?
        has_causal_target = any(
            t.get("uniprot") == cf.causal_off_target_uniprot for t in binding_profile
        )

        # ---- SCM scoring
        scm_ranked = top_k_predictions(
            binding_profile, edges_used, se_vocab, k=len(se_vocab),
        )
        scm_rank = best_causal_rank(scm_ranked, cf.causal_side_effects_umls)

        # ---- SCM per-target attribution: is causal off-target in top-3 for the predicted SE?
        from ..demo.scm_explainer import attribute_side_effect
        attribution_targets: list[str] = []
        attribution_pcts: list[float] = []
        scm_top_target_gene = ""
        if scm_rank:
            # Use the SCM's top-ranked causal SE for attribution analysis
            for se_id in scm_ranked[:scm_rank]:
                if se_id in cf.causal_side_effects_umls:
                    attribs = attribute_side_effect(
                        se_id, binding_profile, edges_used,
                        target_info, top_k_targets=5,
                    )
                    attribution_targets = [a.uniprot for a in attribs]
                    attribution_pcts = [a.contribution_pct for a in attribs]
                    if attribs:
                        scm_top_target_gene = attribs[0].gene_symbol
                    break
        causal_target_in_top3 = (
            cf.causal_off_target_uniprot in attribution_targets[:3]
            if attribution_targets else False
        )
        causal_target_rank_in_attribution = (
            attribution_targets.index(cf.causal_off_target_uniprot) + 1
            if cf.causal_off_target_uniprot in attribution_targets else None
        )

        # ---- RF-ECFP
        rf_rank: int | None = None
        if molregno_resolved is not None:
            smi = smiles_map_all.get(str(molregno_resolved)) or fetch_smiles(molregno_resolved)
            if smi:
                fp = compute_ecfp(smi)
                if fp is not None:
                    rf_ranked = rank_test_drug_rf(
                        fp, rf_models, se_vocab, base_rate_array,
                    )
                    rf_rank = best_causal_rank(rf_ranked, cf.causal_side_effects_umls)

        # ---- LLM-drug-blind
        llm_blind = rank_side_effects_llm_drug_blind(
            binding_profile, vocab_payload, client=client, top_k=TOP_K_RANK,
        )
        llm_blind_rank = best_causal_rank(
            llm_blind.ranked_side_effects, cf.causal_side_effects_umls,
        )

        # ---- LLM-with-name
        llm_name = rank_side_effects_llm_with_name(
            cf.drug_search_name, binding_profile, vocab_payload,
            client=client, top_k=TOP_K_RANK,
        )
        llm_name_rank = best_causal_rank(
            llm_name.ranked_side_effects, cf.causal_side_effects_umls,
        )

        elapsed = time.monotonic() - t_start
        eta = elapsed * (len(CLINICAL_FAILURES) - i) / max(i, 1)
        print(f"          ranks (lower = better): SCM={scm_rank}  RF-ECFP={rf_rank}  "
              f"LLM-blind={llm_blind_rank}  LLM-name={llm_name_rank}")
        print(f"          SCM attribution top-3 includes causal "
              f"{cf.causal_off_target_gene}? {causal_target_in_top3}"
              + (f"  (rank in attribution: {causal_target_rank_in_attribution})"
                 if causal_target_rank_in_attribution else ""))
        print(f"          ({elapsed/60:.1f}m / ETA {eta/60:.1f}m)")

        results.append({
            "drug_id": cf.drug_id,
            "withdrawal_year": cf.withdrawal_year,
            "withdrawal_reason": cf.withdrawal_reason,
            "causal_off_target_uniprot": cf.causal_off_target_uniprot,
            "causal_off_target_gene": cf.causal_off_target_gene,
            "causal_side_effects_display": cf.causal_side_effects_display,
            "causal_side_effects_umls": list(cf.causal_side_effects_umls),
            "in_sider_catalog": cf.in_sider_catalog,
            "scm_provenance": scm_provenance,
            "molregno": molregno_resolved,
            "n_binding_targets": len(binding_profile),
            "has_causal_off_target_in_binding": has_causal_target,
            "scm_rank": scm_rank,
            "rf_ecfp_rank": rf_rank,
            "llm_drug_blind_rank": llm_blind_rank,
            "llm_with_name_rank": llm_name_rank,
            "scm_top_target_gene_for_causal_se": scm_top_target_gene,
            "causal_target_in_attribution_top3": causal_target_in_top3,
            "causal_target_rank_in_attribution": causal_target_rank_in_attribution,
            "attribution_targets_top5": attribution_targets[:5],
            "attribution_pcts_top5": attribution_pcts[:5],
        })

    # ---- Aggregate metrics
    out_path = RESULTS / "sprint3_clinical_failures.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_cases": len(results),
            "per_case": results,
        }, f, indent=2)
    print(f"\nSaved per-case: {out_path}")

    # Summary
    def hits_at_k(records: list[dict], rank_key: str, k: int) -> int:
        return sum(1 for r in records if r[rank_key] is not None and r[rank_key] <= k)

    n = len(results)
    print("\n" + "=" * 78)
    print(f"Sprint 3 SUMMARY (n={n})")
    print("=" * 78)
    print(f"\n{'method':<22s} {'hit@3':<8s} {'hit@10':<8s} {'hit@20':<8s} {'hit@50':<8s}")
    for label, key in [("SCM", "scm_rank"),
                         ("RF-ECFP", "rf_ecfp_rank"),
                         ("LLM-drug-blind", "llm_drug_blind_rank"),
                         ("LLM-with-name", "llm_with_name_rank")]:
        h3 = hits_at_k(results, key, 3)
        h10 = hits_at_k(results, key, 10)
        h20 = hits_at_k(results, key, 20)
        h50 = hits_at_k(results, key, 50)
        print(f"{label:<22s} {h3}/{n:<7d} {h10}/{n:<7d} {h20}/{n:<7d} {h50}/{n:<7d}")

    attrib_hits = sum(1 for r in results if r["causal_target_in_attribution_top3"])
    attrib_avail = sum(1 for r in results if r["attribution_targets_top5"])
    n_has_target = sum(1 for r in results if r["has_causal_off_target_in_binding"])
    print(f"\nMechanism interpretability:")
    print(f"  Drugs where causal off-target IS in binding profile: {n_has_target}/{n}")
    print(f"  Drugs where SCM top-3 attribution includes causal target: "
          f"{attrib_hits}/{attrib_avail} (where attribution available)")

    # Save summary
    summary = {
        "n_cases": n,
        "hits": {
            label: {f"at_{k}": hits_at_k(results, key, k)
                    for k in (3, 5, 10, 20, 50)}
            for label, key in [("scm", "scm_rank"),
                                 ("rf_ecfp", "rf_ecfp_rank"),
                                 ("llm_drug_blind", "llm_drug_blind_rank"),
                                 ("llm_with_name", "llm_with_name_rank")]
        },
        "attribution": {
            "n_with_causal_target_in_binding": n_has_target,
            "n_attribution_top3_includes_causal_target": attrib_hits,
            "n_attribution_available": attrib_avail,
        },
    }
    with open(RESULTS / "sprint3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {RESULTS / 'sprint3_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
