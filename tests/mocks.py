"""Mock clients. No network, deterministic, and they count their own calls
so tests can assert on the API-call pattern."""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Type

from pydantic import BaseModel

from agent.clients import Clients
from agent.telemetry import BudgetExceeded, RunMetrics


class MockLLM:
    def __init__(self, settings, metrics: RunMetrics, responses: Optional[Dict[str, Any]] = None,
                 latency: float = 0.0, fail_on: Optional[Sequence[str]] = None):
        self.settings = settings
        self.metrics = metrics
        self.responses = responses or {}
        self.latency = latency
        self.fail_on = set(fail_on or [])
        self.calls: List[str] = []
        self._lock = threading.Lock()

    def _kind(self, prompt: str, schema=None) -> str:
        if schema is not None:
            return schema.__name__
        if "Write a cover note" in prompt:
            return "cover_note"
        return "text"

    def _record(self, kind: str) -> None:
        with self._lock:
            self.calls.append(kind)

    def invoke(self, prompt: str) -> str:
        self.metrics.charge_llm()
        kind = self._kind(prompt)
        self._record(kind)
        if self.latency:
            time.sleep(self.latency)
        if kind in self.fail_on:
            from agent.clients import LLMUnavailable
            raise LLMUnavailable(f"mock failure for {kind}")
        return self.responses.get(kind, "mock text response")

    def structured(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        self.metrics.charge_llm()
        kind = self._kind(prompt, schema)
        self._record(kind)
        if self.latency:
            time.sleep(self.latency)
        if kind in self.fail_on:
            from agent.clients import LLMUnavailable
            raise LLMUnavailable(f"mock failure for {kind}")
        value = self.responses.get(kind)
        if value is None:
            return schema()
        if isinstance(value, schema):
            return value
        return schema.model_validate(value)


class MockSearch:
    def __init__(self, settings, metrics: RunMetrics, results: Optional[List[Dict]] = None,
                 latency: float = 0.0, fail: bool = False):
        self.settings = settings
        self.metrics = metrics
        self.results = results if results is not None else [
            {"title": "About", "url": "https://example.com/a", "content": "Company builds things."}
        ]
        self.latency = latency
        self.fail = fail
        self.calls: List[str] = []

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        self.metrics.charge_search()
        self.calls.append(query)
        if self.latency:
            time.sleep(self.latency)
        if self.fail:
            return []
        return list(self.results)


class MockEmbeddings:
    """Deterministic pseudo-embeddings derived from token hashes, so cosine
    similarity is stable and meaningful across runs."""

    DIM = 24

    def __init__(self, settings, metrics: RunMetrics, latency: float = 0.0, fail: bool = False):
        self.settings = settings
        self.metrics = metrics
        self.latency = latency
        self.fail = fail
        self.batch_calls = 0

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.DIM
        for token in (text or "").lower().split():
            vec[hash(token) % self.DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> List[Optional[List[float]]]:
        self.metrics.charge_embedding()
        self.batch_calls += 1
        if self.latency:
            time.sleep(self.latency)
        if self.fail:
            raise RuntimeError("mock embedding outage")
        return [self._vector(t) for t in texts]

    def embed_one(self, text: str):
        return self.embed_documents([text])[0]


def make_clients(settings, metrics, **kwargs) -> Clients:
    return Clients(
        llm=MockLLM(settings, metrics, responses=kwargs.get("llm_responses"),
                    latency=kwargs.get("llm_latency", 0.0), fail_on=kwargs.get("llm_fail_on")),
        search=MockSearch(settings, metrics, results=kwargs.get("search_results"),
                          latency=kwargs.get("search_latency", 0.0), fail=kwargs.get("search_fail", False)),
        embeddings=MockEmbeddings(settings, metrics, latency=kwargs.get("embed_latency", 0.0),
                                  fail=kwargs.get("embed_fail", False)),
    )
