"""Instrumented, budgeted wrappers around Gemini, Tavily and Adzuna.

Every outbound call goes through here so that call counting, budgeting and
timing are impossible to bypass. Clients are constructed from config only;
no key is ever logged or placed in an exception message.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from pydantic import BaseModel, ValidationError

from . import cache
from .config import Settings, has_credential, require_credential
from .telemetry import BudgetExceeded, RunMetrics

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMUnavailable(RuntimeError):
    """The LLM could not produce a usable answer. Callers degrade gracefully."""


# --------------------------------------------------------------------------
# Defensive structured-output parsing
# --------------------------------------------------------------------------

def parse_structured(raw: str, schema: Type[TModel]) -> TModel:
    """Coerce a raw model response into `schema`.

    Tries, in order: direct JSON, fenced JSON block, first balanced object.
    Raises LLMUnavailable rather than silently substituting a default --
    the old code defaulted a failed parse to `score = 5`, which looked like
    a real judgement but was a parser failure.
    """
    if raw is None:
        raise LLMUnavailable("empty model response")
    text = str(raw).strip()

    candidates: List[str] = [text]
    block = _JSON_BLOCK.search(text)
    if block:
        candidates.insert(0, block.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return schema.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError):
            continue
    raise LLMUnavailable(f"could not parse a {schema.__name__} from the model response")


def schema_instructions(schema: Type[BaseModel]) -> str:
    """Append an explicit JSON contract to a prompt as a portable fallback
    for models/versions where native structured output is unavailable."""
    return (
        "\n\nReturn ONLY a JSON object matching this schema. No prose, no "
        "markdown fences, no explanation.\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

class LLMClient:
    """Wraps a chat model with budgeting, counting and structured output."""

    def __init__(self, settings: Settings, metrics: RunMetrics, raw_client: Any = None):
        self.settings = settings
        self.metrics = metrics
        self._client = raw_client

    @property
    def client(self) -> Any:
        if self._client is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._client = ChatGoogleGenerativeAI(
                model=self.settings.chat_model,
                temperature=self.settings.chat_temperature,
                google_api_key=require_credential("GOOGLE_API_KEY"),
            )
        return self._client

    def invoke(self, prompt: str) -> str:
        """One text completion. Charges the budget before the network call."""
        self.metrics.charge_llm()
        try:
            response = self.client.invoke(prompt)
        except Exception as exc:
            logger.warning("LLM call failed: %s", type(exc).__name__)
            raise LLMUnavailable(f"LLM call failed ({type(exc).__name__})") from exc
        return getattr(response, "content", str(response))

    def structured(self, prompt: str, schema: Type[TModel]) -> TModel:
        """One structured completion.

        Prefers the provider's native structured output; falls back to a
        JSON contract in the prompt. Either way this is a SINGLE charged call.
        """
        self.metrics.charge_llm()
        try:
            bound = self.client.with_structured_output(schema)
            result = bound.invoke(prompt)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
        except BudgetExceeded:
            raise
        except Exception as exc:
            logger.debug("native structured output unavailable (%s); using JSON contract", type(exc).__name__)

        try:
            raw = self.client.invoke(prompt + schema_instructions(schema))
        except Exception as exc:
            raise LLMUnavailable(f"LLM call failed ({type(exc).__name__})") from exc
        return parse_structured(getattr(raw, "content", str(raw)), schema)


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

class SearchClient:
    """Tavily wrapper. Returns [] rather than raising on failure."""

    def __init__(self, settings: Settings, metrics: RunMetrics, raw_client: Any = None):
        self.settings = settings
        self.metrics = metrics
        self._client = raw_client

    @property
    def client(self) -> Any:
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=require_credential("TAVILY_API_KEY"))
        return self._client

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        self.metrics.charge_search()
        limit = max_results or self.settings.tavily_max_results
        try:
            response = self.client.search(query, max_results=limit)
        except Exception as exc:
            logger.warning("search failed: %s", type(exc).__name__)
            return []
        results = []
        for item in (response or {}).get("results", []) or []:
            results.append(
                {
                    "title": str(item.get("title", "") or ""),
                    "url": str(item.get("url", "") or ""),
                    "content": str(item.get("content", "") or "")[:800],
                }
            )
        return results


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

class EmbeddingClient:
    """Batched embeddings with a per-text cache.

    The previous implementation called embed_query() once per job inside a
    loop -- 50 jobs meant 50 sequential HTTP round trips. This batches and
    caches by content hash, so unchanged jobs are never re-embedded.
    """

    def __init__(self, settings: Settings, metrics: RunMetrics, raw_client: Any = None):
        self.settings = settings
        self.metrics = metrics
        self._client = raw_client

    @property
    def client(self) -> Any:
        if self._client is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self._client = GoogleGenerativeAIEmbeddings(
                model=self.settings.embedding_model,
                google_api_key=require_credential("GOOGLE_API_KEY"),
            )
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> List[Optional[List[float]]]:
        """Embed many texts using as few API calls as possible.

        Returns one vector per input, or None in that slot if it could not
        be embedded. Never raises -- ranking falls back to a lexical score.
        """
        from .identity import content_hash

        results: List[Optional[List[float]]] = [None] * len(texts)
        pending: List[int] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            key = cache.make_key("embedding", self.settings.embedding_model, content_hash(text))
            hit = cache.get(key) if self.settings.cache_enabled else None
            if hit is not None:
                results[i] = hit
                self.metrics.record_cache_hit()
            else:
                self.metrics.record_cache_miss()
                pending.append(i)

        batch_size = max(1, self.settings.embedding_batch_size)
        for start in range(0, len(pending), batch_size):
            chunk_idx = pending[start : start + batch_size]
            chunk = [texts[i] for i in chunk_idx]
            try:
                self.metrics.charge_embedding()
            except BudgetExceeded as exc:
                self.metrics.note(str(exc))
                break
            try:
                vectors = self.client.embed_documents(list(chunk))
            except Exception as exc:
                logger.warning("batch embedding failed: %s", type(exc).__name__)
                self.metrics.note(
                    "Embeddings unavailable; opportunity ranking fell back to keyword similarity."
                )
                break
            for idx, vector in zip(chunk_idx, vectors or []):
                results[idx] = vector
                if self.settings.cache_enabled:
                    cache.set(
                        cache.make_key("embedding", self.settings.embedding_model, content_hash(texts[idx])),
                        vector,
                        self.settings.embedding_ttl_s,
                    )
        return results

    def embed_one(self, text: str) -> Optional[List[float]]:
        return self.embed_documents([text])[0]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

class Clients:
    """Bundle passed to every pipeline node. Injectable for tests."""

    def __init__(self, llm: LLMClient, search: SearchClient, embeddings: EmbeddingClient):
        self.llm = llm
        self.search = search
        self.embeddings = embeddings

    @classmethod
    def build(cls, settings: Settings, metrics: RunMetrics) -> "Clients":
        return cls(
            llm=LLMClient(settings, metrics),
            search=SearchClient(settings, metrics),
            embeddings=EmbeddingClient(settings, metrics),
        )


def credential_status() -> Dict[str, bool]:
    """Which integrations are configured. Booleans only -- never values."""
    return {
        "GOOGLE_API_KEY": has_credential("GOOGLE_API_KEY"),
        "TAVILY_API_KEY": has_credential("TAVILY_API_KEY"),
        "ADZUNA_APP_ID": has_credential("ADZUNA_APP_ID"),
        "ADZUNA_APP_KEY": has_credential("ADZUNA_APP_KEY"),
    }
