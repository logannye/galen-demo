"""Map SCM uniprot vocab -> ChEMBL tid for activity extraction.

The SCM substrate is indexed by uniprot. ChEMBL activities are indexed
by ChEMBL tid. The bridge is target_components -> component_sequences,
filtered to single-protein targets.

Output: results/chembl_target_map.json
  {
    "uniprot_to_tid": {"P00533": 219, ...},
    "tid_to_uniprot": {"219": "P00533", ...},
    "covered_uniprots": [...],
    "uncovered_uniprots": [...],
    "n_covered": int,
    "n_total_scm_vocab": int
  }
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
CHEMBL_DB = "/Volumes/Databank/databases/chembl_36.db"


def main() -> None:
    with open(RESULTS / "target_vocab.json") as f:
        tv = json.load(f)
    scm_uniprots = [t["uniprot"] for t in tv["targets"]]
    print(f"SCM vocab: {len(scm_uniprots)} uniprots")

    conn = sqlite3.connect(CHEMBL_DB)
    cur = conn.cursor()

    placeholders = ",".join("?" * len(scm_uniprots))
    cur.execute(f"""
        SELECT cs.accession, td.tid, td.chembl_id, td.pref_name
        FROM component_sequences cs
        JOIN target_components tc USING(component_id)
        JOIN target_dictionary td USING(tid)
        WHERE cs.accession IN ({placeholders})
          AND td.target_type = 'SINGLE PROTEIN'
    """, scm_uniprots)

    uniprot_to_tid: dict[str, int] = {}
    tid_to_uniprot: dict[str, str] = {}
    rows = cur.fetchall()
    for uniprot, tid, chembl_id, pref_name in rows:
        # If a uniprot has multiple single-protein entries (different organisms),
        # keep the first; ChEMBL canonical target picking is non-trivial but
        # for our purposes any homo-sapiens single-protein record will do.
        if uniprot in uniprot_to_tid:
            continue
        uniprot_to_tid[uniprot] = int(tid)
        tid_to_uniprot[str(tid)] = uniprot

    conn.close()

    covered = sorted(uniprot_to_tid.keys())
    uncovered = sorted(set(scm_uniprots) - set(covered))

    payload = {
        "uniprot_to_tid": uniprot_to_tid,
        "tid_to_uniprot": tid_to_uniprot,
        "covered_uniprots": covered,
        "uncovered_uniprots": uncovered,
        "n_covered": len(covered),
        "n_total_scm_vocab": len(scm_uniprots),
        "coverage_pct": round(100.0 * len(covered) / len(scm_uniprots), 1),
    }

    out = RESULTS / "chembl_target_map.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")
    print(f"Coverage: {payload['n_covered']}/{payload['n_total_scm_vocab']} "
          f"= {payload['coverage_pct']}%")
    if uncovered:
        print(f"First 10 uncovered uniprots: {uncovered[:10]}")


if __name__ == "__main__":
    main()
