"""LLM baselines for side-effect prediction.

Two arms:
  - rank_side_effects_llm_drug_blind: given ONLY binding profile (no drug
    name, no SMILES, no synonyms), rank side effects from the vocabulary.
    This is the critical test arm — it forces mechanism-based reasoning.
  - rank_side_effects_llm_with_name: given drug name + binding profile,
    rank side effects. This is the memorization upper bound.

The gap between the two quantifies how much the LLM relies on memorized
drug-side-effect associations vs how much it can reason from mechanism.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..llm import SonnetClient


@dataclass(frozen=True)
class LLMRanking:
    arm: str  # "drug_blind" or "with_name"
    ranked_side_effects: list[str]  # UMLS ids, ranked
    confidence: str
    rationale: str
    raw_response: str = ""


_DRUG_BLIND_PROMPT = """You are a careful clinical pharmacologist. You are \
given a drug's polypharmacology binding profile (target list with affinities) \
but you are NOT told the drug's identity. You must reason from the binding \
profile alone — mechanism-based reasoning, not memory of specific drugs.

CRITICAL CONSTRAINTS:
- You do NOT know which drug this is. Reason from mechanism only.
- Use biological knowledge of each target's role in physiology to predict \
which side effects are mechanistically plausible.
- A drug binding target T at affinity K might cause side effects S if T \
modulates a pathway whose perturbation manifests as S in patients at \
therapeutic doses.
- Aggregate across all binding targets — the side-effect spectrum reflects \
the FULL polypharmacology, not just the strongest binding.
- Return the TOP {top_k} side effects from the vocabulary, ranked from \
MOST LIKELY to LEAST LIKELY.

Required output: a JSON object with the following keys:
  - ranked_side_effects: list of {top_k} side-effect UMLS ids from the \
vocabulary, most plausible first
  - confidence: one of "high", "medium", "low"
  - rationale: 2-3 sentences citing which targets and which physiological \
systems drove the top predictions

Return ONLY the JSON object, no surrounding text.

BINDING PROFILE (target gene symbol, UniProt, affinity):
{binding_block}

CANDIDATE SIDE-EFFECT VOCABULARY (UMLS id → display name):
{vocab_block}

JSON output:"""


_WITH_NAME_PROMPT = """You are a careful clinical pharmacologist. You are \
given:
  - drug name: {drug_name}
  - the drug's polypharmacology binding profile

Predict the drug's side-effect spectrum. Use both your knowledge of the \
drug and mechanistic reasoning from its binding profile.

Return the TOP {top_k} side effects from the vocabulary, ranked from \
MOST LIKELY to LEAST LIKELY.

Required output: a JSON object with:
  - ranked_side_effects: list of {top_k} UMLS ids from the vocabulary
  - confidence: "high" | "medium" | "low"
  - rationale: 2-3 sentences

Return ONLY the JSON object.

BINDING PROFILE:
{binding_block}

CANDIDATE SIDE-EFFECT VOCABULARY:
{vocab_block}

JSON output:"""


def _format_binding_block(binding_profile: list[dict], max_targets: int = 30) -> str:
    """Format binding profile for prompt inclusion."""
    lines = []
    for t in binding_profile[:max_targets]:
        gene = t.get("gene_symbol") or t.get("uniprot", "?")
        uniprot = t.get("uniprot", "?")
        stype = t.get("standard_type", "?")
        sval = t.get("standard_value_nm", 0)
        target_name = t.get("target_pref_name", "")
        lines.append(
            f"  - {gene} ({uniprot}): {stype}={sval:.1f}nM "
            f"[{target_name}]"
        )
    return "\n".join(lines)


def _format_vocab_block(vocab_payload: dict, max_entries: int = 500) -> str:
    """Format side-effect vocabulary as bullet list."""
    umls_ids = vocab_payload["umls_ids"][:max_entries]
    names = vocab_payload["display_names"]
    return "\n".join(
        f"  - {u}: {names.get(u, u)}" for u in umls_ids
    )


def _parse(raw: str, arm: str, vocab_set: set[str]) -> LLMRanking:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return LLMRanking(arm=arm, ranked_side_effects=[],
                               confidence="insufficient",
                               rationale=f"parse fail: {raw[:200]}",
                               raw_response=raw)
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return LLMRanking(arm=arm, ranked_side_effects=[],
                               confidence="insufficient",
                               rationale=f"parse fail: {raw[:200]}",
                               raw_response=raw)

    raw_ranked = obj.get("ranked_side_effects", []) or []
    seen: set[str] = set()
    filtered: list[str] = []
    for s in raw_ranked:
        if not isinstance(s, str):
            continue
        s = s.strip()
        if s in vocab_set and s not in seen:
            filtered.append(s)
            seen.add(s)
    return LLMRanking(
        arm=arm,
        ranked_side_effects=filtered,
        confidence=obj.get("confidence", "low"),
        rationale=(obj.get("rationale", "") or "")[:500],
        raw_response=raw,
    )


def rank_side_effects_llm_drug_blind(
    binding_profile: list[dict],
    vocab_payload: dict,
    client: SonnetClient | None = None,
    *,
    top_k: int = 50,
    max_tokens: int = 2048,
) -> LLMRanking:
    if client is None:
        client = SonnetClient()
    vocab_set = set(vocab_payload["umls_ids"])
    prompt = _DRUG_BLIND_PROMPT.format(
        top_k=top_k,
        binding_block=_format_binding_block(binding_profile),
        vocab_block=_format_vocab_block(vocab_payload),
    )
    resp = client.complete(prompt, max_tokens=max_tokens, temperature=0.0)
    if not resp.succeeded:
        return LLMRanking(arm="drug_blind", ranked_side_effects=[],
                           confidence="insufficient",
                           rationale=f"LLM call failed: {resp.error}",
                           raw_response=resp.raw_text)
    return _parse(resp.raw_text, "drug_blind", vocab_set)


def rank_side_effects_llm_with_name(
    drug_name: str,
    binding_profile: list[dict],
    vocab_payload: dict,
    client: SonnetClient | None = None,
    *,
    top_k: int = 50,
    max_tokens: int = 2048,
) -> LLMRanking:
    if client is None:
        client = SonnetClient()
    vocab_set = set(vocab_payload["umls_ids"])
    prompt = _WITH_NAME_PROMPT.format(
        drug_name=drug_name, top_k=top_k,
        binding_block=_format_binding_block(binding_profile),
        vocab_block=_format_vocab_block(vocab_payload),
    )
    resp = client.complete(prompt, max_tokens=max_tokens, temperature=0.0)
    if not resp.succeeded:
        return LLMRanking(arm="with_name", ranked_side_effects=[],
                           confidence="insufficient",
                           rationale=f"LLM call failed: {resp.error}",
                           raw_response=resp.raw_text)
    return _parse(resp.raw_text, "with_name", vocab_set)
