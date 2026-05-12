"""Extract ChEMBL binders <=10uM for our 1115-target SCM vocab.

3-phase approach (index-aligned, single sequential scans):
  Phase 1: build assay_id -> tid lookup for our 1075 ChEMBL tids
  Phase 2: scan activities filtered only by standard_units/type/value/relation
           (uses idx_galen_activities_type_units); keep rows whose assay_id
           is in our lookup
  Phase 3: fetch canonical SMILES for the unique molregnos (batched)

This avoids the multi-way JOIN cost on a remote SSD that punishes random seeks.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"
AFFINITY_THRESHOLD_NM = 10000.0
STANDARD_TYPES = ("Ki", "Kd", "IC50", "EC50")
MIN_BINDERS = 10


def main() -> None:
    with open(RESULTS / "chembl_target_map.json") as f:
        m = json.load(f)
    tid_to_uniprot = m["tid_to_uniprot"]
    tids = sorted({int(t) for t in tid_to_uniprot})
    print(f"Querying activities for {len(tids)} ChEMBL tids...", flush=True)

    conn = sqlite3.connect(CHEMBL_DB)
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -524288")
    cur = conn.cursor()

    t0 = time.time()

    # --- Phase 1: assay_to_tid lookup (uses idx_assays_tid) ---
    print("Phase 1: building assay_id -> tid lookup...", flush=True)
    ph = ",".join("?" * len(tids))
    cur.execute(f"SELECT assay_id, tid FROM assays WHERE tid IN ({ph})", tids)
    assay_to_tid: dict[int, int] = {int(a): int(t) for a, t in cur.fetchall()}
    print(f"  {len(assay_to_tid):,} assays mapped ({time.time()-t0:.1f}s)",
          flush=True)

    # --- Phase 2: scan activities by type+units (uses galen index) ---
    print("Phase 2: scanning activities...", flush=True)
    type_ph = ",".join("?" * len(STANDARD_TYPES))
    q = f"""
        SELECT assay_id, molregno, standard_value, standard_type
        FROM activities
        WHERE standard_units = 'nM'
          AND standard_type IN ({type_ph})
          AND standard_value IS NOT NULL
          AND standard_value <= ?
          AND standard_relation = '='
    """
    cur.execute(q, list(STANDARD_TYPES) + [AFFINITY_THRESHOLD_NM])

    binders: dict[str, set[int]] = {u: set() for u in m["uniprot_to_tid"]}
    needed_mols: set[int] = set()
    n_kept = 0
    n_seen = 0
    last_print = time.time()
    for assay_id, molregno, _sv, _st in cur:
        n_seen += 1
        tid = assay_to_tid.get(int(assay_id))
        if tid is None:
            continue
        uniprot = tid_to_uniprot[str(tid)]
        binders[uniprot].add(int(molregno))
        needed_mols.add(int(molregno))
        n_kept += 1
        if time.time() - last_print > 30:
            print(f"  seen={n_seen:,}  kept={n_kept:,}  compounds={len(needed_mols):,}"
                  f"  ({time.time()-t0:.1f}s)", flush=True)
            last_print = time.time()
    print(f"  Phase 2 done: scanned={n_seen:,}  kept={n_kept:,}  "
          f"unique compounds={len(needed_mols):,}  ({time.time()-t0:.1f}s)",
          flush=True)

    # --- Phase 3: SMILES batched lookup ---
    print("Phase 3: fetching SMILES...", flush=True)
    compounds: dict[int, dict] = {}
    CHUNK = 10000
    mol_list = sorted(needed_mols)
    t_p3 = time.time()
    for i in range(0, len(mol_list), CHUNK):
        chunk = mol_list[i:i+CHUNK]
        ph = ",".join("?" * len(chunk))
        cur.execute(
            f"SELECT molregno, canonical_smiles FROM compound_structures "
            f"WHERE molregno IN ({ph}) AND canonical_smiles IS NOT NULL",
            chunk,
        )
        for molregno, smi in cur:
            compounds[int(molregno)] = {"smiles": smi, "n_targets": 0}
    print(f"  Phase 3 done: SMILES for {len(compounds):,}/{len(needed_mols):,} "
          f"compounds  ({time.time()-t_p3:.1f}s)", flush=True)

    conn.close()

    # Drop compounds with no SMILES
    for u in list(binders):
        binders[u] = {m for m in binders[u] if m in compounds}
    for u, mset in binders.items():
        for m_ in mset:
            compounds[m_]["n_targets"] += 1

    target_n = {u: len(s) for u, s in binders.items()}
    sufficient = {u: sorted(s) for u, s in binders.items() if len(s) >= MIN_BINDERS}
    insufficient = [u for u, n in target_n.items() if n < MIN_BINDERS]

    print(f"\nTargets with >= {MIN_BINDERS} binders: {len(sufficient)} / {len(binders)}")
    print(f"Insufficient targets dropped: {len(insufficient)}")
    print(f"Compounds with >=1 binding measurement: {len(compounds):,}")

    payload = {
        "compounds": {str(k): v for k, v in compounds.items()},
        "binders": sufficient,
        "insufficient_targets": insufficient,
        "stats": {
            "n_targets_trained": len(sufficient),
            "n_targets_dropped": len(insufficient),
            "n_compounds": len(compounds),
            "n_activity_rows": n_kept,
            "n_scanned": n_seen,
            "min_binders_required": MIN_BINDERS,
            "affinity_threshold_nm": AFFINITY_THRESHOLD_NM,
            "standard_types": list(STANDARD_TYPES),
            "extraction_seconds": round(time.time() - t0, 1),
        },
    }
    out = RESULTS / "chembl_binders.json"
    with open(out, "w") as f:
        json.dump(payload, f)
    print(f"\nWrote {out}: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
