"""Random Forest on ECFP fingerprints (chemistry-only baseline).

For each drug, compute Morgan/ECFP4 fingerprints (radius=2, 2048 bits) from
ChEMBL SMILES. Train per-side-effect RF classifier (or multi-output RF) on
training drugs. Predict on test drugs.

This is the standard chemistry SOTA baseline. Tests whether molecular
structure alone (without explicit binding information) is sufficient to
predict side effects. The SCM uses binding profile (a richer feature
than structure); we expect SCM to beat RF-ECFP, but the magnitude of the
gap is the SOTA claim.
"""
from __future__ import annotations

import json
import sqlite3
import warnings
from pathlib import Path

import numpy as np

# Heavy native dependencies (rdkit + scikit-learn) are only needed by
# the training/eval scripts. Import lazily so the module can be loaded
# in environments that don't have them installed (e.g. the cloud demo
# inherits this module transitively via binding_inference but never
# calls into anything that uses these libs).
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    from sklearn.ensemble import RandomForestClassifier
    RDLogger.DisableLog("rdApp.*")
    _RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    AllChem = None
    RandomForestClassifier = None
    _RDKIT_AVAILABLE = False


def _require_rdkit():
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "rdkit + scikit-learn are required for this code path. "
            "Install with: pip install rdkit scikit-learn"
        )

CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"
FP_RADIUS = 2
FP_BITS = 2048


def fetch_smiles_for_drugs(
    drugs: list[dict],
) -> dict[str, str]:
    """Fetch canonical SMILES for each drug via molregno."""
    molregnos = [int(d["molregno"]) for d in drugs]
    placeholders = ",".join("?" * len(molregnos))
    conn = sqlite3.connect(CHEMBL_DB)
    cur = conn.cursor()
    cur.execute(f"""
        SELECT molregno, canonical_smiles
        FROM compound_structures
        WHERE molregno IN ({placeholders})
    """, molregnos)
    out: dict[str, str] = {}
    for molregno, smi in cur:
        if smi:
            out[str(molregno)] = smi
    conn.close()
    return out


def compute_ecfp(smiles: str) -> np.ndarray | None:
    """Morgan/ECFP4 fingerprint (radius=2, 2048 bits) as numpy array."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    arr = np.zeros((FP_BITS,), dtype=np.float32)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_ecfp_matrix(
    drugs: list[dict], smiles_map: dict[str, str],
) -> tuple[np.ndarray, list[int]]:
    """Build ECFP matrix; return (X, idx) where idx maps row→drugs-list index."""
    X_rows = []
    idx = []
    for i, d in enumerate(drugs):
        smi = smiles_map.get(str(d["molregno"]))
        if not smi:
            continue
        fp = compute_ecfp(smi)
        if fp is None:
            continue
        X_rows.append(fp)
        idx.append(i)
    if not X_rows:
        return np.zeros((0, FP_BITS), dtype=np.float32), []
    return np.stack(X_rows, axis=0), idx


def train_rf_models(
    X_train: np.ndarray, Y_train: np.ndarray, *,
    n_estimators: int = 50, max_depth: int | None = 8,
    random_state: int = 42,
) -> dict[int, RandomForestClassifier]:
    """One RF per side effect (binary classification)."""
    models: dict[int, RandomForestClassifier] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(Y_train.shape[1]):
            y = Y_train[:, j]
            if y.sum() == 0 or y.sum() == len(y):
                models[j] = None
                continue
            try:
                m = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=1,  # parallelism at outer level
                )
                m.fit(X_train, y)
                models[j] = m
            except Exception:
                models[j] = None
    return models


def rank_test_drug_rf(
    test_x: np.ndarray, models: dict[int, RandomForestClassifier],
    side_effect_vocab: list[str], base_rate: np.ndarray,
) -> list[str]:
    scores = np.zeros(len(side_effect_vocab), dtype=np.float32)
    for j in range(len(side_effect_vocab)):
        m = models.get(j)
        if m is None:
            scores[j] = base_rate[j]
        else:
            try:
                scores[j] = float(m.predict_proba(test_x.reshape(1, -1))[0, 1])
            except Exception:
                scores[j] = base_rate[j]
    order = np.argsort(-scores)
    return [side_effect_vocab[i] for i in order]
