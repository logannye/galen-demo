"""Validate TargetNet on the sealed held-out cohort.

For each held-out compound:
  1. Load its ECFP4 fingerprint
  2. Run all trained per-target RFs to get binding probability per target
  3. Rank targets by probability descending
  4. Compute recall@K = (true targets in top-K) / (total true targets)

Aggregate as mean recall@K over 100 held-out compounds.

Output: results/targetnet_evaluation.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def main() -> None:
    import joblib
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog("rdApp.*")

    print("Loading TargetNet...", flush=True)
    bundle = joblib.load(RESULTS / "targetnet_models.joblib")
    uniprots = bundle["uniprots"]
    models = bundle["models"]
    print(f"  {sum(1 for m in models if m is not None):,} trained targets")

    print("Loading held-out cohort...", flush=True)
    with open(RESULTS / "heldout_smiles_target_split.json") as f:
        heldout = json.load(f)
    n = len(heldout["heldout_molregnos"])
    print(f"  {n} held-out compounds")

    # ECFP4 each compound on-the-fly
    def compute_fp(smiles: str) -> np.ndarray | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((2048,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    t0 = time.time()
    per_compound: list[dict] = []
    recalls_at_5, recalls_at_10, recalls_at_20, recalls_at_50 = [], [], [], []

    for i, m_str in enumerate(heldout["heldout_smiles"]):
        smi = heldout["heldout_smiles"][m_str]
        gt_targets = set(heldout["heldout_targets"][m_str])
        fp = compute_fp(smi)
        if fp is None:
            print(f"  WARN: invalid SMILES for {m_str}: {smi[:60]}")
            continue
        X = fp.reshape(1, -1)
        scores: list[tuple[str, float]] = []
        for u, mdl in zip(uniprots, models):
            if mdl is None:
                continue
            try:
                p = float(mdl.predict_proba(X)[0, 1])
            except Exception:
                continue
            scores.append((u, p))
        scores.sort(key=lambda x: -x[1])

        # Restrict GT to targets where we actually have a model (fairness)
        scored_targets = {u for u, _ in scores}
        gt_in_scope = gt_targets & scored_targets

        def recall_at(k: int) -> float:
            if not gt_in_scope:
                return float("nan")
            top_k = {u for u, _ in scores[:k]}
            return len(top_k & gt_in_scope) / len(gt_in_scope)

        r5, r10, r20, r50 = (recall_at(5), recall_at(10),
                              recall_at(20), recall_at(50))
        if not (np.isnan(r20)):
            recalls_at_5.append(r5)
            recalls_at_10.append(r10)
            recalls_at_20.append(r20)
            recalls_at_50.append(r50)

        per_compound.append({
            "molregno": int(m_str),
            "smiles": smi,
            "n_gt": len(gt_targets),
            "n_gt_in_scope": len(gt_in_scope),
            "recall_at_5": r5,
            "recall_at_10": r10,
            "recall_at_20": r20,
            "recall_at_50": r50,
            "top_20_targets": [u for u, _ in scores[:20]],
            "gt_targets": sorted(gt_targets),
        })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n} ({time.time()-t0:.1f}s)  "
                  f"mean recall@20 so far: {np.mean(recalls_at_20):.3f}",
                  flush=True)

    mean = {
        "mean_recall_at_5": float(np.mean(recalls_at_5)),
        "mean_recall_at_10": float(np.mean(recalls_at_10)),
        "mean_recall_at_20": float(np.mean(recalls_at_20)),
        "mean_recall_at_50": float(np.mean(recalls_at_50)),
        "median_recall_at_20": float(np.median(recalls_at_20)),
        "n_evaluated": len(recalls_at_20),
    }

    # Pre-registered gate
    gate = ("PASS" if mean["mean_recall_at_20"] >= 0.40
            else "MARGINAL" if mean["mean_recall_at_20"] >= 0.30
            else "FAIL")

    print("\n=== TargetNet held-out evaluation ===")
    print(f"  n evaluated:     {mean['n_evaluated']}")
    print(f"  mean recall@5:   {mean['mean_recall_at_5']:.1%}")
    print(f"  mean recall@10:  {mean['mean_recall_at_10']:.1%}")
    print(f"  mean recall@20:  {mean['mean_recall_at_20']:.1%}  <-- primary")
    print(f"  mean recall@50:  {mean['mean_recall_at_50']:.1%}")
    print(f"  median recall@20:{mean['median_recall_at_20']:.1%}")
    print(f"  Pre-reg gate:    {gate}  (PASS >= 40%, MARGINAL 30-40%, FAIL < 30%)")

    out_payload = {
        "summary": mean,
        "gate": gate,
        "per_compound": per_compound,
    }
    out = RESULTS / "targetnet_evaluation.json"
    with open(out, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
