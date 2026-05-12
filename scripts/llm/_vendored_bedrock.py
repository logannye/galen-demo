"""Bedrock client for Haiku 4.5 — primary LLM per V1_PRE_REGISTRATION.md.

Uses boto3's default credential chain (env vars → ~/.aws/credentials →
~/.aws/config → IAM role). No credentials embedded in code.

Model: us.anthropic.claude-haiku-4-5-20251001-v1:0 (cross-region inference profile).
Same as v3.2 / v4 used. Configured via env vars:
- AWS_DEFAULT_REGION (defaults to us-east-1)
- AWS_PROFILE (boto3 auto-discovers from ~/.aws/credentials profiles)

Anthropic Messages API format. Returns LLMResponse with usage and latency.
Retries on transient errors with exponential backoff (boto3 native retry).
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from ._vendored_base import LLMClient, LLMResponse


DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_REGION_ENV = "AWS_DEFAULT_REGION"
FALLBACK_REGION = "us-east-1"


class BedrockClient(LLMClient):
    """Haiku 4.5 via AWS Bedrock (Anthropic Messages API).

    Lazy boto3 client construction on first call. Default credential chain.
    Retries on throttling / transient 5xx errors via boto3 native retry config.

    default_max_tokens=1024: validated empirically (Haiku writes reasoning
    preamble before JSON; smaller budgets truncate the JSON mid-string).
    """

    name = "haiku45_bedrock"
    model = DEFAULT_MODEL
    default_max_tokens = 1024

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        region: Optional[str] = None,
        max_retries: int = 5,
    ):
        self.model = model
        self.region = region or os.environ.get(DEFAULT_REGION_ENV, FALLBACK_REGION)
        self.max_retries = max_retries
        self._client = None  # Lazy

    def _get_client(self):
        if self._client is not None:
            return self._client
        import boto3
        from botocore.config import Config

        config = Config(
            retries={
                "max_attempts": self.max_retries,
                "mode": "adaptive",  # backoff + concurrency adaptation
            },
            connect_timeout=10,
            read_timeout=60,
        )
        self._client = boto3.client(
            "bedrock-runtime", region_name=self.region, config=config
        )
        return self._client

    def complete(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> LLMResponse:
        """Call Bedrock Anthropic Messages API.

        max_tokens=None uses self.default_max_tokens (1024 for Haiku).

        Note: Bedrock Anthropic models do not currently expose a `seed` parameter,
        so determinism relies on temperature=0.0 (and the runtime's internal
        non-determinism budget per the v3.2 cross-LLM observation: ~1-3pp
        per-cell drift even at T=0.0). Single-seed comparison via cluster
        bootstrap controls for this.
        """
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        client = self._get_client()

        # Anthropic Messages API body
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        start = time.monotonic()
        try:
            response = client.invoke_model(
                modelId=self.model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            elapsed = time.monotonic() - start
        except Exception as e:
            elapsed = time.monotonic() - start
            return LLMResponse(
                raw_text="",
                model=self.model,
                latency_seconds=elapsed,
                error=f"{type(e).__name__}: {e}",
            )

        try:
            payload = json.loads(response["body"].read())
        except Exception as e:
            return LLMResponse(
                raw_text="",
                model=self.model,
                latency_seconds=elapsed,
                error=f"failed to parse response body: {e}",
            )

        # Extract text from content blocks
        content_blocks = payload.get("content", [])
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        ]
        raw_text = "".join(text_parts)

        usage = payload.get("usage", {})
        request_id = response.get("ResponseMetadata", {}).get("RequestId")

        return LLMResponse(
            raw_text=raw_text,
            model=self.model,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_seconds=elapsed,
            request_id=request_id,
            extra={"stop_reason": payload.get("stop_reason")},
        )


# W4: frontier LLM clients for cross-LLM panel extension.
# Per docs/V4_PHASE4_DESIGN.md W4: add Sonnet 4.6 + Opus 4.7 to V1's Haiku/
# Deepseek panel to test whether V1 H6 ("substrate effect is LLM-capability-
# mediated") holds at the frontier.

SONNET_4_6_MODEL = "us.anthropic.claude-sonnet-4-6"
OPUS_4_7_MODEL = "us.anthropic.claude-opus-4-7"


class SonnetClient(BedrockClient):
    """Claude Sonnet 4.6 via Bedrock cross-region inference profile.

    Mid-capability tier between Haiku 4.5 and Opus 4.7. Slightly larger
    max_tokens budget than Haiku since Sonnet writes a longer reasoning
    preamble before JSON.
    """

    name = "sonnet46_bedrock"
    model = SONNET_4_6_MODEL
    default_max_tokens = 2048

    def __init__(self, region=None, max_retries: int = 5):
        super().__init__(model=SONNET_4_6_MODEL, region=region, max_retries=max_retries)


class OpusClient(BedrockClient):
    """Claude Opus 4.7 via Bedrock cross-region inference profile.

    Frontier-capability tier. Larger max_tokens (Opus thinks more verbosely
    in its preamble; observed runs need ~1500 output tokens for clean JSON).

    Note: Opus 4.7 deprecates the `temperature` parameter — it cannot be set
    at the API level. This client omits temperature from the request body
    and relies on Opus's internal greedy/near-greedy decoding. Per V1's
    cross-LLM observation (~1-3pp drift per cell even at T=0.0), this is
    a small additional source of non-determinism that the cluster-bootstrap
    confidence interval already accounts for.
    """

    name = "opus47_bedrock"
    model = OPUS_4_7_MODEL
    default_max_tokens = 4096

    def __init__(self, region=None, max_retries: int = 5):
        super().__init__(model=OPUS_4_7_MODEL, region=region, max_retries=max_retries)

    def complete(self, prompt, max_tokens=None, temperature=0.0, seed=0):
        """Override BedrockClient.complete to omit `temperature` from body."""
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        client = self._get_client()
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        start = time.monotonic()
        try:
            response = client.invoke_model(
                modelId=self.model,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            elapsed = time.monotonic() - start
        except Exception as e:
            elapsed = time.monotonic() - start
            return LLMResponse(
                raw_text="", model=self.model,
                latency_seconds=elapsed,
                error=f"{type(e).__name__}: {e}",
            )

        try:
            payload = json.loads(response["body"].read())
        except Exception as e:
            return LLMResponse(
                raw_text="", model=self.model,
                latency_seconds=elapsed,
                error=f"failed to parse response body: {e}",
            )
        content_blocks = payload.get("content", [])
        text_parts = [
            block.get("text", "") for block in content_blocks
            if block.get("type") == "text"
        ]
        raw_text = "".join(text_parts)
        usage = payload.get("usage", {})
        request_id = response.get("ResponseMetadata", {}).get("RequestId")
        return LLMResponse(
            raw_text=raw_text, model=self.model,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_seconds=elapsed, request_id=request_id,
            extra={"stop_reason": payload.get("stop_reason")},
        )
