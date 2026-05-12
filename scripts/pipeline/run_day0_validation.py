"""Day 0 validation: run paired pipelines on the n=122 safety-failure cohort.

Pipelines (both use the CURRENT production stack — prob-weighted SCM,
Phase 6.1 priors, Hybrid LLM Sonnet 4.6):

  A) Curated binding profile: name → ChEMBL bp (the historical baseline)
  B) SMILES input via TargetNet → predicted binding profile (Week 2 system)

Both pipelines use the SAME drug list, SAME SCM substrate, SAME LLM, SAME
prompt. The ONLY difference is how the binding profile is obtained.

Output: results/day0_validation_results.json
  {
    "n": int,
    "per_drug": [
      {
        "drug_search_name": ...,
        "severity": ...,
        "therapeutic_area": ...,
        "gt_umls": [...],
        "curated": { "hybrid_rank": int|None, "hybrid_top10": [...], "bp_size": int },
        "smiles":  { "hybrid_rank": int|None, "hybrid_top10": [...], "bp_size": int },
        "matched_in_curated": bool,    # cluster-aware hit at k=10
        "matched_in_smiles":  bool,
        ...
      }
    ]
  }
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import scripts.pipeline.run_phase_4_3 as p43
from scripts.pipeline.run_phase_4_3 import (
    _process_case, hit_at_k_clustered, collapse_top_k,
    load_curated_priors_for_override,
    SonnetClient, N_WORKERS,
)
from scripts.targetnet.predict import predict_binding_profile
from scripts.data.biologic_binding_profiles import get_biologic_binding

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"

TARGETNET_TOP_K = 20
TARGETNET_PROB_THRESHOLD = 0.5


def _get_smiles(mr):
    if mr is None:
        return None
    conn = sqlite3.connect(CHEMBL_DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT canonical_smiles FROM compound_structures WHERE molregno = ?",
        (int(mr),),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row and row[0] else None


# Two reconstruct_binding implementations selected by global flag
_USE_SMILES = False


def reconstruct_binding_dispatch(name, molregno=None):
    if _USE_SMILES:
        smi = _get_smiles(molregno)
        if smi:
            bp = predict_binding_profile(
                smi, top_k=TARGETNET_TOP_K,
                prob_threshold=TARGETNET_PROB_THRESHOLD,
            )
            if bp:
                return bp, molregno
        bp = get_biologic_binding(name) or []
        return bp, None
    else:
        # Original curated-bp path
        from scripts.pipeline.run_phase_4_3 import reconstruct_binding as _orig
        # NB: _orig was overwritten by the monkey patch in earlier modules
        # if any. We rebuild it from scratch here.
        from scripts.data.build_catalog import query_binding_profile
        from scripts.pipeline.run_sprint3_clinical_failures import lookup_chembl_molregno
        if molregno:
            try:
                conn = sqlite3.connect(CHEMBL_DB)
                bp = query_binding_profile(conn, molregno)
                conn.close()
                if bp:
                    return bp, molregno
            except Exception:
                pass
        mr2 = lookup_chembl_molregno(name)
        if mr2:
            try:
                conn = sqlite3.connect(CHEMBL_DB)
                bp = query_binding_profile(conn, mr2)
                conn.close()
                if bp:
                    return bp, mr2
            except Exception:
                pass
        bp = get_biologic_binding(name) or []
        return bp, None


# Monkey-patch
p43.reconstruct_binding = reconstruct_binding_dispatch


def _build_case(drug):
    """Translate cohort entry to the schema _process_case expects."""
    return {
        "name": drug["drug_search_name"],
        "molregno": drug.get("molregno"),
        "primary_target": drug.get("causal_off_target", ""),
        "gt_umls": drug["gt_umls"],
        "n_aes_in_vocab": len(drug["gt_umls"]),
    }


def run_pipeline(label: str, drugs: list[dict], context: dict) -> list[dict]:
    """Run all drugs through the patched _process_case once.

    Sets the global _USE_SMILES flag, then submits jobs via thread pool.
    """
    global _USE_SMILES
    _USE_SMILES = (label == "smiles")

    print(f"\n=== Pipeline '{label}' ({len(drugs)} drugs) ===", flush=True)
    t0 = time.monotonic()
    results = [None] * len(drugs)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {
            ex.submit(
                _process_case,
                _build_case(d),
                context["edges"], context["signed_edges"],
                context["target_action_n"], context["target_info"],
                context["vocab_payload"], context["se_vocab"], context["se_names"],
                context["client"], context["curated_priors"],
            ): i
            for i, d in enumerate(drugs)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"  drug {idx} ({drugs[idx]['drug_search_name']}) FAILED: {e}",
                      flush=True)
                r = {"name": drugs[idx]["drug_search_name"], "skipped": True,
                     "error": str(e)}
            results[idx] = r
            done += 1
            if done % 20 == 0:
                el = time.monotonic() - t0
                print(f"  {done}/{len(drugs)} ({el/60:.1f}m)", flush=True)
    print(f"=== '{label}' done in {(time.monotonic()-t0)/60:.1f}m ===", flush=True)
    return results


def main() -> None:
    cohort = json.load(open(RESULTS / "day0_validation_cohort.json"))
    cases = cohort["cases"]
    # Use ONLY the testable subset
    testable = [c for c in cases if c["smiles"] and c["gt_umls"]]
    print(f"Day 0 testable cohort: n={len(testable)}")

    # Shared engine context
    client = SonnetClient()
    edges = json.load(open(RESULTS / "scm_edges_blended_j.json"))
    signed = json.load(open(RESULTS / "scm_edges_signed.json"))
    signed_edges = signed.get("edges", {})
    target_action_n = signed.get("target_action_n_drugs", {})
    vocab_payload = json.load(open(RESULTS / "side_effect_vocab.json"))
    se_vocab = vocab_payload["umls_ids"]
    se_names = vocab_payload["display_names"]
    tv = json.load(open(RESULTS / "target_vocab.json"))
    target_info = {t["uniprot"]: t for t in tv["targets"]}
    curated_priors = load_curated_priors_for_override()

    context = dict(
        edges=edges, signed_edges=signed_edges,
        target_action_n=target_action_n, target_info=target_info,
        vocab_payload=vocab_payload, se_vocab=se_vocab, se_names=se_names,
        client=client, curated_priors=curated_priors,
    )

    # Run BOTH pipelines
    curated_results = run_pipeline("curated", testable, context)
    smiles_results = run_pipeline("smiles", testable, context)

    # Merge per-drug
    per_drug = []
    for d, c_r, s_r in zip(testable, curated_results, smiles_results):
        per_drug.append({
            "drug_search_name": d["drug_search_name"],
            "molregno": d["molregno"],
            "severity": d["severity"],
            "therapeutic_area": d["therapeutic_area"],
            "causal_off_target": d["causal_off_target"],
            "causal_se_text": d["causal_se_text"],
            "gt_umls": d["gt_umls"],
            "n_binding_targets_orig": d["n_binding_targets"],
            "stored_baselines": {
                "scm_rank": d.get("stored_scm_rank"),
                "hybrid_rank": d.get("stored_hybrid_rank"),
                "llm_drug_blind_rank": d.get("stored_llm_drug_blind_rank"),
                "llm_with_name_rank": d.get("stored_llm_with_name_rank"),
                "rf_ecfp_rank": d.get("stored_rf_ecfp_rank"),
            },
            "curated": {
                "scm_rank": c_r.get("scm_rank"),
                "hybrid_rank": c_r.get("hybrid_rank"),
                "hybrid_top10": c_r.get("hybrid_top10", []),
                "n_targets": c_r.get("n_targets", 0),
                "skipped": c_r.get("skipped", False),
                "error": c_r.get("error"),
            },
            "smiles": {
                "scm_rank": s_r.get("scm_rank"),
                "hybrid_rank": s_r.get("hybrid_rank"),
                "hybrid_top10": s_r.get("hybrid_top10", []),
                "n_targets": s_r.get("n_targets", 0),
                "skipped": s_r.get("skipped", False),
                "error": s_r.get("error"),
            },
        })

    out = RESULTS / "day0_validation_results.json"
    with open(out, "w") as f:
        json.dump({"n": len(per_drug), "per_drug": per_drug}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
