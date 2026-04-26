"""
The 5 models we benchmark, plus a thin OpenRouter wrapper.

Cost-spread preset chosen by the user (model IDs current as of OpenRouter 2026-04):
    frontier_a: openai/gpt-4o
    frontier_b: anthropic/claude-sonnet-4.5
    mid_a:      openai/gpt-4o-mini
    mid_b:      meta-llama/llama-3.3-70b-instruct
    cheap:      deepseek/deepseek-chat
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


MODELS: list[dict] = [
    {"id": "openai/gpt-4o",                          "label": "gpt-4o",          "tier": "frontier"},
    {"id": "anthropic/claude-sonnet-4.5",            "label": "claude-sonnet-4.5","tier": "frontier"},
    {"id": "openai/gpt-4o-mini",                     "label": "gpt-4o-mini",     "tier": "mid"},
    {"id": "meta-llama/llama-3.3-70b-instruct",      "label": "llama-3.3-70b",   "tier": "mid"},
    {"id": "deepseek/deepseek-chat",                 "label": "deepseek-chat",   "tier": "cheap"},
]


CASCADE_ORDER: list[str] = [
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4.5",
]


@dataclass
class CallResult:
    model_id:      str
    prompt_id:     str
    task:          str
    text:          str
    cost_usd:      float
    latency_ms:    float
    prompt_tokens: int
    output_tokens: int
    error:         Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class OpenRouter:
    """Minimal blocking OpenRouter client used by phase 1."""

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 45.0):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Export it or add to .env before "
                "running phase1_collect.py."
            )
        self.timeout = timeout

    def call(self, model_id: str, prompt: str, max_tokens: int = 1024,
             system: Optional[str] = None) -> dict:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self.URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                    "HTTP-Referer":  "https://x25.ai/benchmark",
                    "X-Title":       "X25 Benchmark Suite",
                },
                json={
                    "model":       model_id,
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": 0.0,
                    "usage":       {"include": True},
                },
            )
        latency_ms = (time.time() - start) * 1000.0
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]["message"]["content"] or ""
        usage  = data.get("usage", {}) or {}
        cost   = float(usage.get("cost", 0.0))

        return {
            "text":          choice,
            "cost_usd":      cost,
            "latency_ms":    latency_ms,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        }
