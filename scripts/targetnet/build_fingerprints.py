"""Compute ECFP4 (Morgan radius=2, 2048-bit) fingerprints for every
ChEMBL compound in our binder corpus.

Output: results/ecfp4_fps.npz
  fps: uint8 array shape (N, 2048)   -- packed bits as 0/1 bytes
  molregnos: int32 array shape (N,)  -- corresponds to fps rows

We use uint8 for storage (16 MB per million compounds at 2048 bits);
the trainer unpacks lazily.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
FP_BITS = 2048
FP_RADIUS = 2


def _init_worker():
    """Configure each worker's RDKit logging once."""
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def compute_fp_pair(args):
    """(molregno, smiles) -> (molregno, fp_uint8) or (molregno, None)."""
    molregno, smiles = args
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return molregno, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    arr = np.zeros((FP_BITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return molregno, arr


def main() -> None:
    from multiprocessing import Pool, cpu_count

    with open(RESULTS / "chembl_binders.json") as f:
        data = json.load(f)
    compounds = data["compounds"]
    molregnos = sorted(int(m) for m in compounds)
    print(f"{len(molregnos):,} compounds to fingerprint", flush=True)

    args = [(m, compounds[str(m)]["smiles"]) for m in molregnos]

    n_proc = max(1, cpu_count() - 2)
    print(f"Using {n_proc} worker processes", flush=True)

    fps = np.zeros((len(molregnos), FP_BITS), dtype=np.uint8)
    valid = np.zeros(len(molregnos), dtype=bool)
    mol_to_idx = {m: i for i, m in enumerate(molregnos)}

    t0 = time.time()
    n_done = 0
    n_invalid = 0
    with Pool(processes=n_proc, initializer=_init_worker) as pool:
        for molregno, fp in pool.imap_unordered(compute_fp_pair, args,
                                                  chunksize=500):
            n_done += 1
            if fp is None:
                n_invalid += 1
                continue
            idx = mol_to_idx[molregno]
            fps[idx] = fp
            valid[idx] = True
            if n_done % 50_000 == 0:
                print(f"  {n_done:,}/{len(molregnos):,} ({time.time()-t0:.1f}s, "
                      f"{n_invalid} invalid)", flush=True)

    print(f"Done in {time.time()-t0:.1f}s. {n_invalid} invalid SMILES dropped.",
          flush=True)

    fps_valid = fps[valid]
    mols_valid = np.array(molregnos, dtype=np.int32)[valid]
    print(f"Final: {len(fps_valid):,} compounds × {FP_BITS} bits", flush=True)

    out = RESULTS / "ecfp4_fps.npz"
    np.savez(out, fps=fps_valid, molregnos=mols_valid)
    print(f"Wrote {out}: {out.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
