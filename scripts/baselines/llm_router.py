"""Sprint J: smart LLM router (novelty-aware).

Decides whether to use Sonnet (in-distribution) or Opus (OOD/novel)
based on a simple novelty detection heuristic.

Novelty signals (in priority order):
  1. Drug NOT in catalog AND resolved via biologic_binding_profiles
     → biologic (likely modern; route to Opus)
  2. Drug NOT in catalog AND resolved via ChEMBL lookup → novel small-mol
     → route to Opus
  3. Drug NOT in catalog at all → unknown → route to Opus
  4. Drug IN catalog → in-distribution → route to Sonnet

This is a deterministic, rule-based router with NO LLM in the loop.
"""
from __future__ import annotations


def detect_drug_novelty(
    drug_search_name: str,
    drugs_by_name: dict,
    molregno_resolved: str | None,
    biologic_recovery: bool,
) -> str:
    """Returns 'in_distribution' or 'novel'.

    drugs_by_name: catalog dict (drug_name.lower() → drug record)
    molregno_resolved: the ChEMBL molregno that was found (or None)
    biologic_recovery: True if binding profile came from biologic_binding_profiles
    """
    name_lc = drug_search_name.lower().strip()
    if name_lc in drugs_by_name:
        return "in_distribution"
    if biologic_recovery:
        return "novel"  # biologics are typically modern (post-cutoff)
    if molregno_resolved is None:
        return "novel"  # no catalog/ChEMBL match
    # ChEMBL hit but not in our 247-drug catalog → likely OOD
    return "novel"


def route_llm(novelty: str) -> str:
    """Returns 'sonnet' or 'opus' for the given novelty class."""
    if novelty == "in_distribution":
        return "sonnet"
    return "opus"
