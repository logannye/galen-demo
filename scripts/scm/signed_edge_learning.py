"""Sprint 8B: Signed α(S|T, action) edge learning.

The Sprint 8A α(S|T) decomposition pooled all drugs binding T regardless
of mechanism (agonist/antagonist/inhibitor/activator). For receptors,
ion channels, and nuclear receptors where AGONISM vs ANTAGONISM flips
the safety profile (β-blocker → bradycardia vs β-agonist → tachycardia;
ACE-i → hyperkalemia vs ACE substrate → none), the pooled α dilutes
the signal.

This script re-decomposes α from the training drugs using DGIdb +
DrugCentral action class labels per (drug, target). For each (T, action),
we compute the conditional frequency separately:

  α_signed(S | T, A) = (# drugs binding T with action A that manifest S) /
                       (# drugs binding T with action A)

We compute this for both SIDER labels (the original training source) and
OnSIDES labels (Sprint 8A's 7th source), and merge by max.

Drugs with unknown action on a target are pooled separately as the
"unknown" bucket. The scoring layer falls back to action-agnostic α
when the signed bucket is too sparse (< K=3 drugs).

Output: results/scm_edges_signed.json
  {
    "n_targets": int,
    "min_drugs_for_signed": K,
    "edges": {
      uniprot: {
        "inhibit": {umls: alpha, ...},
        "activate": {umls: alpha, ...},
        "modulator": {umls: alpha, ...},
        "binder": {umls: alpha, ...},
        "unknown": {umls: alpha, ...}
      }
    },
    "target_action_n_drugs": {uniprot: {action: int}}
  }
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"

MIN_DRUGS_PER_SIGNED_BUCKET = 3
SMOOTHING = 1.0
ACTION_CLASSES = ("inhibit", "activate", "modulator", "binder", "unknown")


def load_action_lookup() -> dict[tuple[str, str], str]:
    """Returns {(molregno, uniprot): action_class}.

    DGIdb wins on conflict (higher source-count typically).
    """
    out: dict[tuple[str, str], str] = {}

    # DrugCentral first (less authoritative, gets overwritten)
    dc_path = RESULTS / "drugcentral_action_types.json"
    if dc_path.exists():
        with open(dc_path) as f:
            dc = json.load(f)
        for molregno, target_actions in (dc.get("drug_target_actions", {}) or {}).items():
            for u, info in target_actions.items():
                ac = info.get("action_class", "unknown")
                if ac:
                    out[(str(molregno), u)] = ac

    # DGIdb wins on conflict
    dgidb_path = RESULTS / "dgidb_action_types.json"
    if dgidb_path.exists():
        with open(dgidb_path) as f:
            dg = json.load(f)
        for molregno, target_actions in (dg.get("drug_target_actions", {}) or {}).items():
            for u, info in target_actions.items():
                ac = info.get("action_class", "unknown")
                if ac and ac != "unknown":
                    out[(str(molregno), u)] = ac

    return out


def load_training_drugs() -> list[dict]:
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    return [d for d in cat["drugs"] if d["split"] == "train"]


def load_sider_se_per_drug(train_drugs: list[dict]) -> dict[str, set[str]]:
    """{drug_id: set of SIDER UMLS SEs in vocab}."""
    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    vocab = set(v["umls_ids"])
    out: dict[str, set[str]] = {}
    for d in train_drugs:
        key = str(d.get("molregno"))
        ses = set(d.get("side_effects_in_vocab", [])) & vocab
        out[key] = ses
    return out


def load_onsides_se_per_drug() -> dict[str, set[str]]:
    """Re-derive {molregno: set of OnSIDES SEs in vocab} from Sprint 8A.

    We reuse the OnSIDES decomposition's per-drug AE set. That's the same
    as the ingest's `drug_onsides_se` dict, which we'll regenerate.
    Instead of re-streaming OnSIDES, we leverage the saved per-target
    edges as a proxy: a drug that binds T contributes to count if
    α_onsides(S|T) > base_rate. But that's noisy. Better to re-derive.

    For Sprint 8B we don't re-stream; we just use SIDER for signed
    decomposition. OnSIDES contribution remains action-agnostic in the
    8B blend (it's still a 7th source via its action-agnostic α from 8A).
    """
    # Sprint 8B scope decision: signed-α comes from SIDER only;
    # OnSIDES stays action-agnostic in the blended substrate. This
    # keeps the signed-α derivation tractable and clean.
    return {}


def main() -> int:
    print("=" * 78)
    print("Sprint 8B: Signed α(S|T, action) edge learning from SIDER training")
    print("=" * 78)

    action_lookup = load_action_lookup()
    print(f"[setup] action lookup: {len(action_lookup):,} (drug, target) pairs")

    train_drugs = load_training_drugs()
    print(f"[setup] training drugs: {len(train_drugs)}")

    sider_ses = load_sider_se_per_drug(train_drugs)
    n_drugs_with_ses = sum(1 for v in sider_ses.values() if v)
    print(f"[setup] training drugs with ≥1 SIDER SE in vocab: {n_drugs_with_ses}")

    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    se_vocab = v["umls_ids"]

    # Per (target, action) → (drug count, per-SE count)
    target_action_n: dict[tuple[str, str], int] = defaultdict(int)
    target_action_se_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    n_action_resolved = 0
    n_total_bindings = 0
    for d in train_drugs:
        molregno = str(d.get("molregno"))
        ses = sider_ses.get(molregno, set())
        for t in d.get("binding_profile", []):
            u = t.get("uniprot")
            if not u:
                continue
            n_total_bindings += 1
            action = action_lookup.get((molregno, u), "unknown")
            if action != "unknown":
                n_action_resolved += 1
            target_action_n[(u, action)] += 1
            for s in ses:
                target_action_se_counts[(u, action)][s] += 1

    print(f"[decompose] total (drug, target) bindings: {n_total_bindings:,}")
    print(f"[decompose] action-resolved bindings: {n_action_resolved:,} "
          f"({100*n_action_resolved/max(1,n_total_bindings):.1f}%)")

    # Distribution
    print(f"[decompose] per-action drug counts (sum across targets):")
    sums = defaultdict(int)
    for (u, a), n in target_action_n.items():
        sums[a] += n
    for a in ACTION_CLASSES:
        print(f"  {a:<12s}: {sums[a]:,}")

    # Compute α per (target, action, SE) with Laplace smoothing
    edges_signed: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    n_targets_with_signed = 0
    for (u, a), n_u in target_action_n.items():
        counts = target_action_se_counts.get((u, a), {})
        alpha_row = {
            s: (counts.get(s, 0) + SMOOTHING) / (n_u + 2 * SMOOTHING)
            for s in se_vocab
        }
        edges_signed.setdefault(u, {})[a] = alpha_row
        if a != "unknown":
            n_targets_with_signed += 1

    # Count targets that have ≥3 drugs in at least one non-unknown action bucket
    n_meaningfully_signed = 0
    for u, by_action in edges_signed.items():
        for a, _ in by_action.items():
            if a == "unknown":
                continue
            if target_action_n.get((u, a), 0) >= MIN_DRUGS_PER_SIGNED_BUCKET:
                n_meaningfully_signed += 1
                break

    print(f"[decompose] targets with signed-α: {len(edges_signed)}")
    print(f"[decompose] targets with ≥{MIN_DRUGS_PER_SIGNED_BUCKET} drugs in a "
          f"non-unknown action bucket: {n_meaningfully_signed}")

    out = {
        "n_targets": len(edges_signed),
        "min_drugs_for_signed": MIN_DRUGS_PER_SIGNED_BUCKET,
        "smoothing": SMOOTHING,
        "edges": edges_signed,
        "target_action_n_drugs": {
            u: {a: target_action_n.get((u, a), 0) for a in ACTION_CLASSES}
            for u in edges_signed
        },
    }
    out_path = RESULTS / "scm_edges_signed.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
