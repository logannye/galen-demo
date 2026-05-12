"""LogReg baseline on binary target-binding indicators.

For each side effect S, we train a logistic regression classifier:
  P(drug has S | binary target-binding vector)

The features are the SAME as the SCM uses (binary target indicators). The
model is a LINEAR aggregation. If the SCM's noisy-OR adds meaningful
non-linearity, the SCM should beat LogReg. If not, LogReg matches it.

Multi-output: we use one LogReg per side effect. With sklearn's
liblinear/L2 regularization, n=247 training samples per side effect is
sufficient.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

# scikit-learn is only required by the training functions in this
# module; the demo runtime imports this file transitively but never
# trains a classifier. Lazy-load so the module can be imported in
# environments without sklearn installed (e.g. the cloud demo).
try:
    from sklearn.linear_model import LogisticRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    LogisticRegression = None
    _SKLEARN_AVAILABLE = False


def _require_sklearn():
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for this code path. "
            "Install with: pip install scikit-learn"
        )


def build_target_feature_matrix(
    drugs: list[dict], target_list: list[str],
) -> np.ndarray:
    """Binary target-indicator matrix: rows=drugs, cols=targets."""
    target_idx = {t: i for i, t in enumerate(target_list)}
    X = np.zeros((len(drugs), len(target_list)), dtype=np.float32)
    for i, d in enumerate(drugs):
        for t in d["binding_profile"]:
            j = target_idx.get(t["uniprot"])
            if j is not None:
                X[i, j] = 1.0
    return X


def build_se_label_matrix(
    drugs: list[dict], side_effect_vocab: list[str],
) -> np.ndarray:
    """Binary side-effect matrix: rows=drugs, cols=side-effects."""
    se_idx = {s: i for i, s in enumerate(side_effect_vocab)}
    Y = np.zeros((len(drugs), len(side_effect_vocab)), dtype=np.int8)
    for i, d in enumerate(drugs):
        for s in d.get("side_effects_in_vocab", []):
            j = se_idx.get(s)
            if j is not None:
                Y[i, j] = 1
    return Y


def train_logreg_models(
    X_train: np.ndarray, Y_train: np.ndarray, *,
    C: float = 1.0,
) -> dict[int, "LogisticRegression"]:
    """Train one LogReg per side effect column.

    Skip side effects with constant labels (always-0 or always-1).
    Returns a dict {col_idx: fitted_classifier_or_None}.
    """
    _require_sklearn()
    models: dict[int, "LogisticRegression"] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j in range(Y_train.shape[1]):
            y = Y_train[:, j]
            if y.sum() == 0 or y.sum() == len(y):
                models[j] = None
                continue
            try:
                m = LogisticRegression(
                    C=C, max_iter=1000, solver="liblinear",
                )
                m.fit(X_train, y)
                models[j] = m
            except Exception:
                models[j] = None
    return models


def rank_test_drug_logreg(
    test_x: np.ndarray, models: dict[int, LogisticRegression],
    side_effect_vocab: list[str], base_rate: np.ndarray,
) -> list[str]:
    """Rank side effects for one test drug by P(S=1 | x)."""
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
