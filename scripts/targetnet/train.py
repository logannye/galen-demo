"""Train per-target Random Forest binding classifiers.

For each SCM-vocab target with >=10 known binders in ChEMBL:
  Positives: ECFP4 of every known binder (<=10uM, not in held-out)
  Negatives: 5x as many compounds, sampled at random from compounds
             with no measured activity on this target

Trains a RandomForestClassifier per target and serializes via joblib
(the scikit-learn convention for trained models).

Output: results/targetnet_models.joblib
  {
    "uniprots": [uniprot, ...],              # ordered, length T
    "feature_bits": 2048,
    "models": [fitted_clf_or_None, ...],     # parallel to uniprots
    "training_meta": [{n_pos, n_neg, oob_score, ...}, ...]
  }
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
NEG_RATIO = 5
N_TREES = 100
MAX_DEPTH = 12
MAX_POS_PER_TARGET = 5000  # cap; large targets train fast enough w/ subset
SEED = 42
MIN_POS_FOR_TRAIN = 10
N_JOBS_PER_RF = -1  # use all cores per RF (sequential targets)


def main() -> None:
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    with open(RESULTS / "chembl_binders.json") as f:
        data = json.load(f)
    binders = data["binders"]
    print(f"Targets to train: {len(binders)}")

    with open(RESULTS / "heldout_smiles_target_split.json") as f:
        heldout = json.load(f)
    heldout_set = set(heldout["heldout_molregnos"])
    print(f"Held-out compounds to exclude from training: {len(heldout_set)}")

    fps_data = np.load(RESULTS / "ecfp4_fps.npz")
    all_fps = fps_data["fps"]
    all_mols = fps_data["molregnos"]
    mol_to_idx = {int(m): i for i, m in enumerate(all_mols)}
    print(f"Loaded fingerprints: {all_fps.shape}")

    all_compound_pool = sorted(set(int(m) for m in all_mols))
    pool_set = set(all_compound_pool)
    rng = np.random.default_rng(SEED)

    uniprots = sorted(binders)
    models: list = []
    meta: list[dict] = []

    t0 = time.time()
    last_print = t0
    for ti, u in enumerate(uniprots):
        pos_set = set(binders[u]) - heldout_set
        if len(pos_set) < MIN_POS_FOR_TRAIN:
            models.append(None)
            meta.append({"uniprot": u, "n_pos": len(pos_set), "skipped": True})
            continue
        # Cap positives at MAX_POS_PER_TARGET for training speed
        pos_list = sorted(pos_set)
        if len(pos_list) > MAX_POS_PER_TARGET:
            sel = rng.choice(len(pos_list), size=MAX_POS_PER_TARGET, replace=False)
            pos_list = [pos_list[i] for i in sel]
            capped = True
        else:
            capped = False

        non_binders = list(pool_set - set(pos_list) - heldout_set)
        n_neg = min(len(non_binders), NEG_RATIO * len(pos_list))
        neg_idx = rng.choice(len(non_binders), size=n_neg, replace=False)
        neg_set = [non_binders[i] for i in neg_idx]

        pos_rows = [mol_to_idx[m] for m in pos_list if m in mol_to_idx]
        neg_rows = [mol_to_idx[m] for m in neg_set if m in mol_to_idx]
        if len(pos_rows) < MIN_POS_FOR_TRAIN:
            models.append(None)
            meta.append({"uniprot": u, "n_pos": len(pos_rows), "skipped": True})
            continue

        X_pos = all_fps[pos_rows]
        X_neg = all_fps[neg_rows]
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(len(pos_rows)), np.zeros(len(neg_rows))])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = RandomForestClassifier(
                n_estimators=N_TREES, max_depth=MAX_DEPTH,
                class_weight="balanced", random_state=SEED,
                n_jobs=N_JOBS_PER_RF, oob_score=True,
            )
            try:
                clf.fit(X, y)
            except Exception as ex:
                models.append(None)
                meta.append({"uniprot": u, "n_pos": len(pos_rows),
                             "skipped": True, "error": str(ex)})
                continue
        oob_auc = getattr(clf, "oob_score_", None)
        models.append(clf)
        meta.append({
            "uniprot": u,
            "n_pos": len(pos_rows),
            "n_neg": len(neg_rows),
            "oob_score": float(oob_auc) if oob_auc is not None else None,
            "capped": capped,
            "skipped": False,
        })
        if time.time() - last_print > 30:
            n_trained_so_far = sum(1 for m in meta if not m.get("skipped"))
            print(f"  {ti+1}/{len(uniprots)} processed, {n_trained_so_far} trained "
                  f"({time.time()-t0:.1f}s)", flush=True)
            last_print = time.time()

    n_trained = sum(1 for m in meta if not m.get("skipped"))
    print(f"\nDone in {time.time()-t0:.1f}s. {n_trained}/{len(uniprots)} targets trained.")

    out = RESULTS / "targetnet_models.joblib"
    joblib.dump({
        "uniprots": uniprots,
        "feature_bits": 2048,
        "models": models,
        "training_meta": meta,
    }, out, compress=3)
    print(f"Wrote {out}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
