"""TargetNet: SMILES -> polypharmacology binding-profile predictor.

Replaces the brittle Tanimoto-NN inference path. See
docs/SMILES_TARGET_PREREG.md for the sealed sprint protocol.

Modules:
  build_chembl_map  : maps SCM uniprot vocab -> ChEMBL tid
  extract_activities: pulls binders (<=10uM) for those targets
  build_fingerprints: ECFP4 (radius 2, 2048 bit) per molregno
  build_heldout     : scaffold-split 100 held-out compounds
  train             : per-target Random Forest training
  predict           : inference (smiles -> [(uniprot, score), ...])
  evaluate          : recall@20 on the held-out cohort
"""
