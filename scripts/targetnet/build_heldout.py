"""Build a held-out set of 100 chemically-diverse compounds for
target-prediction evaluation, per the sealed pre-registration.

Selection criteria (sealed in docs/SMILES_TARGET_PREREG.md):
  1. >=3 measured target activities at <=10uM in our ChEMBL extract
  2. Not in our SIDER training-drug catalog (results/catalog.json)
  3. Selected via Bemis-Murcko scaffold split to maximize chemical
     diversity vs the training set
  4. Seed: 42

Output: results/heldout_smiles_target_split.json
  {
    "heldout_molregnos": [int, ...],            # length 100
    "heldout_smiles": {str(molregno): smiles},
    "heldout_targets": {str(molregno): [uniprot, ...]},
    "n_train_compounds": int,                   # remaining training pool
    "seed": 42,
    "selection_protocol": "scaffold-split"
  }
"""
from __future__ import annotations

import json
import random
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
SEED = 42
N_HELDOUT = 100
MIN_TARGETS = 3


def murcko_scaffold(smiles: str) -> str | None:
    """Bemis-Murcko scaffold as canonical SMILES; None if invalid."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaff = MurckoScaffold.GetScaffoldForMol(mol)
    if scaff is None:
        return None
    return Chem.MolToSmiles(scaff)


def main() -> None:
    rng = random.Random(SEED)

    with open(RESULTS / "chembl_binders.json") as f:
        data = json.load(f)
    compounds = data["compounds"]
    binders = data["binders"]

    # Build per-compound target list
    cmpd_targets: dict[int, list[str]] = {int(m): [] for m in compounds}
    for u, mol_list in binders.items():
        for m in mol_list:
            cmpd_targets[m].append(u)

    # SIDER training drugs (avoid leakage)
    with open(RESULTS / "catalog.json") as f:
        catalog = json.load(f)
    sider_molregnos: set[int] = set()
    for d in catalog.get("drugs", []):
        try:
            sider_molregnos.add(int(d["molregno"]))
        except (KeyError, ValueError, TypeError):
            continue
    print(f"SIDER drugs to exclude: {len(sider_molregnos):,}")

    # Candidate pool: >=MIN_TARGETS activities, not in SIDER
    candidates: list[int] = [
        m for m, ts in cmpd_targets.items()
        if len(ts) >= MIN_TARGETS and m not in sider_molregnos
    ]
    print(f"Candidate pool: {len(candidates):,} compounds "
          f"(>= {MIN_TARGETS} targets, not in SIDER)")

    # Compute scaffolds for the candidates
    print("Computing Bemis-Murcko scaffolds...")
    cmpd_to_scaff: dict[int, str] = {}
    for i, m in enumerate(candidates):
        smi = compounds[str(m)]["smiles"]
        scaff = murcko_scaffold(smi)
        if scaff is not None:
            cmpd_to_scaff[m] = scaff
        if (i + 1) % 50_000 == 0:
            print(f"  {i+1:,}/{len(candidates):,} scaffolded")
    print(f"Scaffolded: {len(cmpd_to_scaff):,}/{len(candidates):,}")

    # Group by scaffold
    scaff_to_cmpds: dict[str, list[int]] = {}
    for m, s in cmpd_to_scaff.items():
        scaff_to_cmpds.setdefault(s, []).append(m)
    scaffolds = sorted(scaff_to_cmpds)
    print(f"Unique scaffolds in candidate pool: {len(scaffolds):,}")

    # Scaffold split: sort scaffolds descending by size, fold to
    # heldout/train alternately for chemical-diversity coverage. Then
    # sample N_HELDOUT from the heldout-scaffold compounds.
    scaff_sizes = [(s, len(scaff_to_cmpds[s])) for s in scaffolds]
    scaff_sizes.sort(key=lambda x: -x[1])

    # The standard scaffold-split convention: put the LEAST-populated
    # scaffolds into held-out (singletons / rares) -- those are the
    # most chemically novel vs the training distribution. We then take
    # N_HELDOUT compounds from those rare scaffolds.
    rare_scaffolds = [s for s, n in scaff_sizes if n == 1]
    print(f"Singleton scaffolds: {len(rare_scaffolds):,}")
    rng.shuffle(rare_scaffolds)
    chosen_scaffolds = rare_scaffolds[:N_HELDOUT * 2]  # over-sample, drop later if no targets
    heldout_candidates = [scaff_to_cmpds[s][0] for s in chosen_scaffolds]
    heldout = heldout_candidates[:N_HELDOUT]
    if len(heldout) < N_HELDOUT:
        # Should never happen with >10k candidates, but safeguard
        raise RuntimeError(
            f"Only {len(heldout)} held-out compounds available; "
            f"need {N_HELDOUT}.")

    # Confirm: held-out have >= MIN_TARGETS each (sanity)
    assert all(len(cmpd_targets[m]) >= MIN_TARGETS for m in heldout)

    payload = {
        "heldout_molregnos": sorted(heldout),
        "heldout_smiles": {str(m): compounds[str(m)]["smiles"] for m in heldout},
        "heldout_targets": {str(m): sorted(cmpd_targets[m]) for m in heldout},
        "n_train_compounds": len(candidates) - len(heldout),
        "n_heldout": len(heldout),
        "seed": SEED,
        "selection_protocol": "scaffold-split-singletons",
        "min_targets": MIN_TARGETS,
    }

    out = RESULTS / "heldout_smiles_target_split.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")
    print(f"Held-out compounds: {len(heldout)}")
    print(f"Median targets per held-out compound: "
          f"{sorted([len(cmpd_targets[m]) for m in heldout])[len(heldout)//2]}")


if __name__ == "__main__":
    main()
