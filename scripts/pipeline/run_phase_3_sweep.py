"""Phase 3.3: Sweep override v2 scm_rank_threshold on onc/immuno + regression check.

Reuses saved Sprint K Hybrid output (no new LLM calls). Re-runs only the
SCM scoring + curated-prior override step with various thresholds.

Pre-registered in docs/PHASE_3_PRE_REGISTRATION.md.
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..baselines.llm_hybrid_reranker import load_action_types
from ..data.biologic_binding_profiles import get_biologic_binding
from ..data.build_catalog import query_binding_profile
from ..pipeline.run_sprint3_clinical_failures import lookup_chembl_molregno
from ..pipeline.run_sprint_f_eval import best_causal_rank
from ..scm.scoring import score_drug_side_effects_signed


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"

# Sweep values
THRESHOLDS = [3, 5, 7, 10, 15, 20]
STRONG_ALPHA = 0.85
HYBRID_RANK_THRESHOLD = 5
MAX_PROMOTIONS = 1


def apply_override_sweep(
    hybrid_top10: list[str],
    scm_ranked: list[str],
    bound_uniprots: set,
    curated_priors: dict,
    scm_rank_threshold: int,
) -> tuple[list[str], list[tuple]]:
    """Apply override v2 with given scm_rank_threshold.

    Returns (new_top10, promotions).
    """
    scm_rank_lookup = {ae: i + 1 for i, ae in enumerate(scm_ranked)}
    hybrid_rank_lookup = {ae: i + 1 for i, ae in enumerate(hybrid_top10)}

    promotions = []
    seen = set()
    for u in bound_uniprots:
        if u not in curated_priors:
            continue
        prior_row = curated_priors[u]
        for ae, alpha in prior_row.items():
            if alpha < STRONG_ALPHA:
                continue
            if ae in seen:
                continue
            scm_rank = scm_rank_lookup.get(ae)
            if scm_rank is None or scm_rank > scm_rank_threshold:
                continue
            hybrid_rank = hybrid_rank_lookup.get(ae)
            if hybrid_rank is not None and hybrid_rank <= HYBRID_RANK_THRESHOLD:
                continue
            seen.add(ae)
            promotions.append((ae, alpha, scm_rank))

    promotions.sort(key=lambda x: (x[2], -x[1]))
    promotions = promotions[:MAX_PROMOTIONS]
    promote_pairs = [(p[0], p[1]) for p in promotions]
    promote_set = {p[0] for p in promotions}

    if not promotions:
        return hybrid_top10, []

    promote_list = [p[0] for p in promotions]
    remaining = [r for r in hybrid_top10 if r not in promote_set]
    return promote_list + remaining, promote_pairs


def reconstruct_binding(name: str, drugs_by_name: dict) -> tuple[list[dict], int | None]:
    """Look up binding profile from catalog → ChEMBL → biologic fallback."""
    cat_drug = drugs_by_name.get((name or "").lower())
    if cat_drug is not None:
        return cat_drug["binding_profile"], cat_drug["molregno"]
    molregno = lookup_chembl_molregno(name)
    if molregno:
        conn = sqlite3.connect(CHEMBL_DB)
        bp = query_binding_profile(conn, molregno)
        conn.close()
        if bp:
            return bp, molregno
    bp = get_biologic_binding(name)
    if bp:
        return bp, None
    return [], None


def main():
    print("=" * 78)
    print("Phase 3.3: Override threshold sweep on onc/immuno (n=140) + regression")
    print("=" * 78)

    # Load Sprint K results
    main_records = json.load(open(RESULTS / "sprint_k_safety_sonnet.json"))["per_drug"]
    ood_records = json.load(open(RESULTS / "sprint_k_ood_safety_sonnet.json"))["per_drug"]
    all_records = []
    for r in main_records:
        r = dict(r)
        r["dataset"] = "main"
        all_records.append(r)
    for r in ood_records:
        r = dict(r)
        r["dataset"] = "ood"
        all_records.append(r)

    # Substrate
    edges = json.load(open(RESULTS / "scm_edges_blended_j.json"))
    signed = json.load(open(RESULTS / "scm_edges_signed.json"))
    signed_edges = signed.get("edges", {})
    target_action_n = signed.get("target_action_n_drugs", {})
    vocab = json.load(open(RESULTS / "side_effect_vocab.json"))
    se_vocab = vocab["umls_ids"]
    cat = json.load(open(RESULTS / "catalog.json"))
    drugs_by_name = {(d["drug_name"] or "").lower(): d for d in cat["drugs"]}
    curated_payload = json.load(open(RESULTS / "scm_edges_curated_priors.json"))
    curated_priors = curated_payload["priors"]

    # For each record, recompute scm_ranked (free, no LLM)
    print(f"\n[1] Recomputing SCM ranking for {len(all_records)} cases...")
    scm_ranked_by_drug = {}
    bound_uniprots_by_drug = {}
    skipped = 0
    for r in all_records:
        if r.get("skipped"):
            skipped += 1
            continue
        name = r["drug_search_name"]
        bp, molregno = reconstruct_binding(name, drugs_by_name)
        if not bp:
            r["recon_fail"] = True
            skipped += 1
            continue
        action_types = {}
        if molregno is not None:
            try:
                action_types = load_action_types(molregno)
            except Exception:
                pass
        scored = score_drug_side_effects_signed(
            bp, edges, signed_edges, action_types, target_action_n, se_vocab,
            min_drugs_for_signed=3, affinity_mode="log_sigmoid",
        )
        scm_ranked = [s for s, _ in scored]
        scm_ranked_by_drug[r["drug_id"]] = scm_ranked
        bound_uniprots_by_drug[r["drug_id"]] = {
            t.get("uniprot") for t in bp if t.get("uniprot")
        }
    print(f"  reconstructed: {len(scm_ranked_by_drug)}/{len(all_records)} "
          f"(skipped: {skipped})")

    # For each threshold, compute hit-rates on (a) onc/immuno, (b) full
    def stats_for_threshold(thresh: int):
        new_hybrids_by_drug = {}
        new_promotions_by_drug = {}
        for r in all_records:
            if r["drug_id"] not in scm_ranked_by_drug:
                continue
            scm_ranked = scm_ranked_by_drug[r["drug_id"]]
            bound = bound_uniprots_by_drug[r["drug_id"]]
            saved_top10 = r.get("hybrid_top10") or []
            # The saved hybrid_top10 reflects the Sprint K override that
            # was ALREADY applied (with threshold=3). To get a clean
            # comparison, we need the pre-override hybrid output.
            # That's `hybrid_rank_pre_override` for the rank-level test
            # but we don't have the full list pre-override.
            #
            # The cleanest reconstruction: assume "if threshold=3 promoted
            # 1 AE, prepend it; otherwise saved == pre-override". Since
            # max_promotions=1 in v2, removing the promoted AE from
            # saved_top10 gives the pre-override top-K (approximately).
            n_prom = r.get("n_promotions", 0)
            if n_prom > 0 and r.get("promotions"):
                promoted_umls = {p["umls"] if isinstance(p, dict) else p[0]
                                  for p in r["promotions"]}
                pre_override_top10 = [x for x in saved_top10
                                       if x not in promoted_umls]
            else:
                pre_override_top10 = saved_top10

            new_top10, new_proms = apply_override_sweep(
                pre_override_top10, scm_ranked, bound, curated_priors,
                scm_rank_threshold=thresh,
            )
            new_hybrids_by_drug[r["drug_id"]] = new_top10
            new_promotions_by_drug[r["drug_id"]] = new_proms
        return new_hybrids_by_drug, new_promotions_by_drug

    def hit_at_k(top10: list[str], causal: list[str], k: int) -> bool:
        causal_set = set(causal)
        for i, ae in enumerate(top10[:k], start=1):
            if ae in causal_set:
                return True
        return False

    # Compute & report
    print(f"\n{'thresh':<8s} {'OI hit@1':>10s} {'OI hit@3':>10s} {'OI hit@10':>11s} "
          f"{'FULL hit@10':>13s} {'firing rate':>13s}")
    print("-" * 78)

    summary = {}
    for thresh in THRESHOLDS:
        new_hybrids, new_proms = stats_for_threshold(thresh)

        # OI subset
        oi_drugs = [r for r in all_records
                     if r.get("therapeutic_area") in ("Oncology", "Immunology")
                     and r["drug_id"] in new_hybrids]
        oi_n = len(oi_drugs)
        oi_h1 = sum(1 for r in oi_drugs
                    if hit_at_k(new_hybrids[r["drug_id"]],
                                 r.get("causal_side_effects_umls") or [], 1))
        oi_h3 = sum(1 for r in oi_drugs
                    if hit_at_k(new_hybrids[r["drug_id"]],
                                 r.get("causal_side_effects_umls") or [], 3))
        oi_h10 = sum(1 for r in oi_drugs
                     if hit_at_k(new_hybrids[r["drug_id"]],
                                  r.get("causal_side_effects_umls") or [], 10))
        oi_firing = sum(1 for r in oi_drugs
                         if new_proms.get(r["drug_id"]))

        # Full Sprint K (main + OOD)
        full_drugs = [r for r in all_records if r["drug_id"] in new_hybrids]
        full_n = len(full_drugs)
        full_h10 = sum(1 for r in full_drugs
                       if hit_at_k(new_hybrids[r["drug_id"]],
                                    r.get("causal_side_effects_umls") or [], 10))

        print(f"{thresh:<8d} {oi_h1:>3d}/{oi_n} ({oi_h1/oi_n:>4.1%})  "
              f"{oi_h3:>3d}/{oi_n} ({oi_h3/oi_n:>4.1%})  "
              f"{oi_h10:>3d}/{oi_n} ({oi_h10/oi_n:>4.1%})   "
              f"{full_h10:>3d}/{full_n} ({full_h10/full_n:>5.1%})   "
              f"{oi_firing:>3d}/{oi_n} ({oi_firing/oi_n:>5.1%})")

        summary[thresh] = {
            "oi": {"n": oi_n, "h1": oi_h1, "h3": oi_h3, "h10": oi_h10,
                   "h1_rate": oi_h1/oi_n, "h3_rate": oi_h3/oi_n,
                   "h10_rate": oi_h10/oi_n,
                   "firing_rate": oi_firing/oi_n},
            "full": {"n": full_n, "h10": full_h10, "h10_rate": full_h10/full_n},
        }

    # Decision per pre-reg
    print("\n" + "=" * 78)
    print("Decision per pre-reg (smallest threshold satisfying criteria)")
    print("=" * 78)
    chosen = None
    decision_label = ""
    for thresh in THRESHOLDS:
        s = summary[thresh]
        oi_h3 = s["oi"]["h3_rate"]
        oi_h10 = s["oi"]["h10_rate"]
        firing = s["oi"]["firing_rate"]
        full_h10 = s["full"]["h10_rate"]
        # No regression on full
        no_regression = full_h10 >= 0.86  # within Sprint K CI lower bound
        if oi_h3 >= 0.90 and oi_h10 >= 0.96 and no_regression and firing >= 0.35:
            chosen = thresh
            decision_label = "STRONG WIN"
            break
    if chosen is None:
        for thresh in THRESHOLDS:
            s = summary[thresh]
            if (s["oi"]["h3_rate"] >= 0.88 and s["oi"]["h10_rate"] >= 0.95
                    and s["full"]["h10_rate"] >= 0.86):
                chosen = thresh
                decision_label = "MODERATE WIN"
                break
    if chosen is None:
        chosen = 3  # baseline
        decision_label = "NULL (no improvement)"

    print(f"\nChosen threshold: {chosen} ({decision_label})")
    if chosen != 3:
        print(f"\nLift vs baseline (threshold=3):")
        base = summary[3]
        new = summary[chosen]
        print(f"  Onc/Immuno hit@1:  {base['oi']['h1_rate']:.1%} → {new['oi']['h1_rate']:.1%}")
        print(f"  Onc/Immuno hit@3:  {base['oi']['h3_rate']:.1%} → {new['oi']['h3_rate']:.1%}")
        print(f"  Onc/Immuno hit@10: {base['oi']['h10_rate']:.1%} → {new['oi']['h10_rate']:.1%}")
        print(f"  Full Sprint K h10: {base['full']['h10_rate']:.1%} → {new['full']['h10_rate']:.1%}")
        print(f"  Firing rate (OI): {base['oi']['firing_rate']:.1%} → {new['oi']['firing_rate']:.1%}")

    out = {
        "thresholds": summary,
        "decision": {"chosen_threshold": chosen, "label": decision_label},
    }
    out_path = RESULTS / "phase_3_sweep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
