"""Decagon-style baseline: matrix factorization on drug-target-side-effect graph.

The Decagon paper (Zitnik et al., 2018) used a relational GCN on a
heterogeneous graph of drug-target-protein-disease-side-effect. For our
monotherapy side-effect prediction task we implement a faithful but
tractable variant:

  - Build a drug-target binding matrix M (drugs × targets, binary)
  - Build a drug-side-effect matrix S (drugs × side effects, binary)
  - Factorize both jointly via low-rank decomposition
  - Predict side effects for a new drug by:
      1. Embedding the new drug from its target-binding profile
      2. Computing similarity to learned side-effect embeddings

This is mathematically a "Decagon-spirit" non-deep model that captures
the same intuition (relational embeddings on a heterogeneous graph)
without requiring a full GCN training loop. For runtime: 100-dim
embeddings, SVD-based, no gradient descent. ~30 seconds.
"""
from __future__ import annotations

import numpy as np


def fit_decagon_style_embeddings(
    X_train: np.ndarray, Y_train: np.ndarray, *, dim: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 3-block decomposition: drug ≈ Wt @ target_embeddings,
    side_effect ≈ Ws.

    Concretely:
      - Compute target embeddings as singular vectors of X_train (drug-target)
      - Compute side-effect embeddings as singular vectors of Y_train (drug-SE)
      - To embed a new drug from targets: u_drug = (target_emb.T @ x_drug)
      - To score a side effect: sigmoid(u_drug · v_se)
    """
    # Drug-target SVD
    U_dt, S_dt, Vt_dt = np.linalg.svd(X_train, full_matrices=False)
    dim_t = min(dim, len(S_dt))
    target_emb = Vt_dt[:dim_t].T * np.sqrt(S_dt[:dim_t])

    # Drug-SE SVD: get side-effect embeddings via SVD of Y_train
    U_ds, S_ds, Vt_ds = np.linalg.svd(Y_train.astype(np.float32), full_matrices=False)
    dim_s = min(dim, len(S_ds))
    se_emb = Vt_ds[:dim_s].T * np.sqrt(S_ds[:dim_s])
    drug_se_emb_train = U_ds[:, :dim_s] * np.sqrt(S_ds[:dim_s])

    # Linear bridge: predict drug_se_emb from target_emb
    # Solve drug_se_emb_train ≈ X_train @ W with X_train: drug-target
    # W is target × dim_s; solve via least squares
    # (cap at lstsq, n_drug=247 << n_target=983 so under-determined; use ridge)
    lam = 1.0
    XtX = X_train.T @ X_train + lam * np.eye(X_train.shape[1], dtype=np.float32)
    XtY = X_train.T @ drug_se_emb_train.astype(np.float32)
    W_bridge = np.linalg.solve(XtX, XtY)  # (targets × dim_s)

    return target_emb, se_emb, W_bridge


def rank_test_drug_gcn(
    test_x: np.ndarray, W_bridge: np.ndarray, se_emb: np.ndarray,
    side_effect_vocab: list[str],
) -> list[str]:
    """Predict side effects via target-mediated drug embedding."""
    drug_emb = test_x @ W_bridge  # (dim_s,)
    scores = se_emb @ drug_emb     # (n_se,)
    order = np.argsort(-scores)
    return [side_effect_vocab[i] for i in order]
