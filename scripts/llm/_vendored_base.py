"""LLM client base class and response type.

Per V1_PRE_REGISTRATION.md §1, §3, §8:
- Two LLM families: Haiku 4.5 (Bedrock) and Deepseek-v4-flash (OpenAI-compatible).
- Single seed (seed=0), T=0.0 across all calls.
- Per-cell errors scored as 0 — clients return error in LLMResponse rather than raise.
- Locked prompt templates per task family (TF-0/TF-A/TF-B/TF-N) in §3.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class LLMResponse:
    """Result of one LLM completion call.

    Per pre-reg §2: any cell that errors is scored as 0. Clients return
    error in this dataclass rather than raising — orchestrator decides
    whether to retry, fail fast, or proceed.

    Fields:
    - raw_text: the LLM's output (or empty string on error)
    - model: actual model identifier used
    - input_tokens / output_tokens: from API usage tracking
    - latency_seconds: wall-clock time of the call
    - error: None if success; diagnostic string if failure
    - retries: number of retries before success (or final failure)
    - request_id: API-side request identifier when available (for debugging)
    """

    raw_text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    error: Optional[str] = None
    retries: int = 0
    request_id: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.raw_text)


class LLMClient(ABC):
    """Abstract LLM client. Subclasses wrap a specific provider/model.

    Each subclass declares `default_max_tokens` reflecting that model's needs:
    - BedrockClient (Haiku 4.5): 1024 (Haiku writes reasoning preamble before JSON)
    - DeepseekClient (v4-flash thinking mode): 4096 (thinking consumes ~500-1000 tokens
      before producing the JSON; smaller budget truncates with finish_reason='length')

    Per V1_DEVIATIONS.md D6: the orchestrator's EvalConfig.max_tokens=None means
    "use the client's default", which is the right call for cross-LLM comparisons
    where each model has different output-budget needs.
    """

    name: str  # short identifier, e.g., "haiku45_bedrock"
    model: str  # full model identifier
    default_max_tokens: int = 1024  # subclasses override

    @abstractmethod
    def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> LLMResponse:
        """Run one completion. Always returns LLMResponse — never raises on API errors.

        max_tokens=None uses the client's default_max_tokens. Pass an explicit int
        to override for a specific call.
        """
        ...
