"""Learn SCM target → side-effect structural parameters from the training set.

For each (target T, side-effect S) pair, the structural parameter is the
empirical conditional frequency:
  α(S | T) = (# training drugs binding T that manifest S) / (# training drugs binding T)

This is the maximum-likelihood Bernoulli parameter assuming each (T → S)
contribution is independent. In the noisy-OR scoring step the per-target
contributions are aggregated across the drug's polypharmacology profile.

Smoothing: Laplace add-1 smoothing on numerator and denominator to handle
sparse targets (few training drugs per rare target).

Outputs:
  results/scm_edges.json — {target_uniprot: {side_effect_umls: alpha}}
  results/scm_target_priors.json — base rate per target (# training drugs
    binding the target / total training drugs)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"


def load_training_drugs() -> list[dict]:
    with open(RESULTS / "catalog.json") as f:
        cat = json.load(f)
    return [d for d in cat["drugs"] if d["split"] == "train"]


def learn_edges(
    train_drugs: list[dict], side_effect_vocab: list[str],
    smoothing: float = 1.0,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Compute α(S | T) for all (T, S) pairs.

    Returns:
      edges: {uniprot: {se_umls: alpha}}
      target_n_drugs: {uniprot: number of training drugs binding T}
    """
    se_vocab_set = set(side_effect_vocab)

    target_n_drugs: dict[str, int] = defaultdict(int)
    target_se_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for d in train_drugs:
        bind_uniprots = {t["uniprot"] for t in d["binding_profile"]}
        ses = set(d["side_effects_in_vocab"]) & se_vocab_set
        for u in bind_uniprots:
            target_n_drugs[u] += 1
            for s in ses:
                target_se_counts[u][s] += 1

    n_train = len(train_drugs)
    edges: dict[str, dict[str, float]] = {}
    for u, n_u in target_n_drugs.items():
        counts = target_se_counts.get(u, {})
        # Laplace add-1 smoothing per side effect (binomial smoothing)
        edges[u] = {
            s: (counts.get(s, 0) + smoothing) / (n_u + 2 * smoothing)
            for s in side_effect_vocab
        }
    return edges, dict(target_n_drugs)


def main() -> int:
    print("=" * 78)
    print("SCM edge learning: target → side-effect α from training set")
    print("=" * 78)

    train_drugs = load_training_drugs()
    print(f"[1/3] loaded {len(train_drugs)} training drugs")

    with open(RESULTS / "side_effect_vocab.json") as f:
        vocab_payload = json.load(f)
    side_effect_vocab = vocab_payload["umls_ids"]
    print(f"[2/3] vocab size: {len(side_effect_vocab)}")

    edges, target_n_drugs = learn_edges(train_drugs, side_effect_vocab)
    print(f"[3/3] learned edges for {len(edges)} targets")

    # Summary stats
    edge_alphas: list[float] = []
    for u, d in edges.items():
        edge_alphas.extend(d.values())
    edge_alphas.sort()
    n = len(edge_alphas)
    print(f"  total (target, side-effect) pairs: {n}")
    print(f"  α distribution: min={edge_alphas[0]:.4f} "
          f"median={edge_alphas[n//2]:.4f} "
          f"p95={edge_alphas[int(n*0.95)]:.4f} "
          f"max={edge_alphas[-1]:.4f}")
    coverage = sum(1 for n_u in target_n_drugs.values() if n_u >= 3)
    print(f"  targets with ≥3 training drugs: {coverage}/{len(target_n_drugs)}")

    out_edges = RESULTS / "scm_edges.json"
    with open(out_edges, "w") as f:
        json.dump(edges, f)
    print(f"  saved: {out_edges}")

    out_priors = RESULTS / "scm_target_priors.json"
    with open(out_priors, "w") as f:
        json.dump({
            "n_train_drugs": len(train_drugs),
            "target_n_drugs": target_n_drugs,
        }, f, indent=2)
    print(f"  saved: {out_priors}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
