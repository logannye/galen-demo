"""Sprint F: HPO-based AE rollup matcher.

Build a UMLS-child → UMLS-parent hierarchy from HPO (Human Phenotype
Ontology). In evaluation, if a Hybrid prediction's UMLS is a parent of
a ground-truth UMLS, count it as a hit. This corrects for the recall
penalty from overly-specific benchmark AE codes.

Strategy:
  1. Parse hp.obo for terms with UMLS xrefs.
  2. For each HPO term, record (term_id, [umls_xref], [is_a parents]).
  3. Walk the HPO hierarchy: for each term T with UMLS U, accumulate
     ancestor terms' UMLS codes → U_ancestors.
  4. Save {umls_child: [umls_ancestor]} map.

Output: results/hpo_umls_rollup.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent.parent.parent
RESULTS = WORKSPACE / "results"
HPO_OBO = WORKSPACE / "data/raw/hpo/hp.obo"


def parse_hpo_obo(path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Parse hp.obo file.

    Returns:
      hpo_to_umls: {HPO_id: [UMLS_codes]}
      hpo_parents: {HPO_id: [parent HPO_ids]}
    """
    hpo_to_umls: dict[str, list[str]] = defaultdict(list)
    hpo_parents: dict[str, list[str]] = defaultdict(list)

    current_id: str | None = None
    in_term = False
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line == "[Term]":
                in_term = True
                current_id = None
                continue
            if line.startswith("["):
                in_term = False
                current_id = None
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                current_id = line[len("id: "):]
            elif line.startswith("xref: ") and current_id:
                xref = line[len("xref: "):]
                m = re.match(r"UMLS:(C\d+)", xref)
                if m:
                    hpo_to_umls[current_id].append(m.group(1))
            elif line.startswith("is_a: ") and current_id:
                parent_id = line[len("is_a: "):].split(" !")[0].strip()
                hpo_parents[current_id].append(parent_id)
            elif line.startswith("is_obsolete: true"):
                # don't include obsolete terms
                if current_id:
                    hpo_to_umls.pop(current_id, None)
                    hpo_parents.pop(current_id, None)
                current_id = None
    return dict(hpo_to_umls), dict(hpo_parents)


def build_umls_rollup(hpo_to_umls: dict[str, list[str]],
                       hpo_parents: dict[str, list[str]],
                       max_depth: int = 6) -> dict[str, list[str]]:
    """For each UMLS code, return its ancestor UMLS codes (via HPO hierarchy).

    Bounded depth to avoid runaway traversal.
    """
    # Inverse: UMLS → HPO terms
    umls_to_hpo: dict[str, list[str]] = defaultdict(list)
    for hpo_id, umls_list in hpo_to_umls.items():
        for u in umls_list:
            umls_to_hpo[u].append(hpo_id)

    rollup: dict[str, set[str]] = defaultdict(set)

    def ancestors_umls(hpo_id: str, depth: int = 0, seen: set | None = None) -> set[str]:
        if seen is None:
            seen = set()
        if hpo_id in seen or depth > max_depth:
            return set()
        seen.add(hpo_id)
        out: set[str] = set()
        for parent in hpo_parents.get(hpo_id, []):
            # add parent's UMLS codes
            for u in hpo_to_umls.get(parent, []):
                out.add(u)
            # recurse
            out |= ancestors_umls(parent, depth + 1, seen)
        return out

    for umls, hpo_ids in umls_to_hpo.items():
        for h in hpo_ids:
            anc = ancestors_umls(h)
            rollup[umls] |= anc
        # don't include self in ancestors
        rollup[umls].discard(umls)

    return {k: sorted(v) for k, v in rollup.items() if v}


def main() -> int:
    print("=" * 78)
    print("Sprint F: HPO → UMLS rollup builder")
    print("=" * 78)

    print(f"[parse] Reading {HPO_OBO}...")
    hpo_to_umls, hpo_parents = parse_hpo_obo(HPO_OBO)
    print(f"[parse] HPO terms with UMLS xref: {len(hpo_to_umls):,}")
    print(f"[parse] HPO terms with is_a parents: {len(hpo_parents):,}")

    print("[rollup] Computing UMLS-ancestor map...")
    rollup = build_umls_rollup(hpo_to_umls, hpo_parents)
    print(f"[rollup] UMLS codes with ≥1 ancestor: {len(rollup):,}")

    # Stats
    n_ancestors = [len(v) for v in rollup.values()]
    if n_ancestors:
        n_ancestors.sort()
        print(f"[stats] ancestors per UMLS: "
              f"min={n_ancestors[0]} median={n_ancestors[len(n_ancestors)//2]} "
              f"max={n_ancestors[-1]}")

    # Filter to UMLS codes likely to appear in our vocab
    with open(RESULTS / "side_effect_vocab.json") as f:
        v = json.load(f)
    vocab_set = set(v["umls_ids"])
    in_vocab_rollup = {
        k: [a for a in v_anc if a in vocab_set]
        for k, v_anc in rollup.items()
        if k in vocab_set
    }
    in_vocab_rollup = {k: v for k, v in in_vocab_rollup.items() if v}
    print(f"[stats] UMLS-in-vocab with ≥1 in-vocab ancestor: "
          f"{len(in_vocab_rollup):,}")

    out_path = RESULTS / "hpo_umls_rollup.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_umls_with_ancestors": len(rollup),
            "n_in_vocab_with_in_vocab_ancestors": len(in_vocab_rollup),
            "rollup": rollup,
        }, f, indent=2)
    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
