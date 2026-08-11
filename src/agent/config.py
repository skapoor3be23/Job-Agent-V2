"""Central configuration. No values are hardcoded elsewhere in the package.

Resolution order for every setting: Streamlit secrets -> environment
variable -> default. Secrets are never logged or echoed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()

_MISSING = object()


def _lookup(key: str) -> Optional[str]:
    """Read a value from st.secrets, falling back to os.environ.

    Importing streamlit is deliberately lazy: this package must be usable
    from tests and CLI scripts where streamlit has no script context.
    """
    try:
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key)


def _get_str(key: str, default: str) -> str:
    val = _lookup(key)
    return val if val not in (None, "") else default


def _get_int(key: str, default: int) -> int:
    val = _lookup(key)
    try:
        return int(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _get_float(key: str, default: float) -> float:
    val = _lookup(key)
    try:
        return float(val) if val not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _get_bool(key: str, default: bool) -> bool:
    val = _lookup(key)
    if val in (None, ""):
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ScoreWeights:
    """Opportunity Score components. Must sum to 1.0."""

    semantic: float = 0.45
    skill_coverage: float = 0.25
    experience: float = 0.15
    role_relevance: float = 0.15

    def validate(self) -> None:
        total = self.semantic + self.skill_coverage + self.experience + self.role_relevance
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Score weights must sum to 1.0, got {total:.4f}")


@dataclass(frozen=True)
class Settings:
    # --- feature switch (Phase: backward compatibility) ---
    use_new_pipeline: bool = True
    allow_legacy_fallback: bool = True

    # --- models ---
    chat_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    chat_temperature: float = 0.3

    # --- per-run call budgets (graceful degradation, not crashes) ---
    max_llm_calls_per_run: int = 8
    max_embedding_calls_per_run: int = 3
    max_search_calls_per_run: int = 4

    # --- opportunity discovery ---
    discovery_enabled: bool = True
    discovery_max_jobs: int = 15
    discovery_max_queries: int = 4
    discovery_results_per_query: int = 10
    adzuna_country: str = "in"

    # --- embeddings ---
    embedding_batch_size: int = 32

    # --- caching ---
    cache_enabled: bool = True
    company_research_ttl_s: int = 60 * 60 * 24 * 7   # company facts age slowly
    job_analysis_ttl_s: int = 60 * 60 * 24
    embedding_ttl_s: int = 60 * 60 * 24 * 30

    # --- cover note ---
    cover_note_min_words: int = 120
    cover_note_max_words: int = 180

    # --- scoring ---
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    # --- misc ---
    tavily_max_results: int = 4
    log_level: str = "INFO"

    @classmethod
    def load(cls) -> "Settings":
        s = cls(
            use_new_pipeline=_get_bool("USE_NEW_PIPELINE", True),
            allow_legacy_fallback=_get_bool("ALLOW_LEGACY_FALLBACK", True),
            chat_model=_get_str("CHAT_MODEL", "gemini-2.5-flash"),
            embedding_model=_get_str("EMBEDDING_MODEL", "models/gemini-embedding-001"),
            chat_temperature=_get_float("CHAT_TEMPERATURE", 0.3),
            max_llm_calls_per_run=_get_int("MAX_LLM_CALLS_PER_RUN", 8),
            max_embedding_calls_per_run=_get_int("MAX_EMBEDDING_CALLS_PER_RUN", 3),
            max_search_calls_per_run=_get_int("MAX_SEARCH_CALLS_PER_RUN", 4),
            discovery_enabled=_get_bool("DISCOVERY_ENABLED", True),
            discovery_max_jobs=_get_int("DISCOVERY_MAX_JOBS", 15),
            discovery_max_queries=_get_int("DISCOVERY_MAX_QUERIES", 4),
            discovery_results_per_query=_get_int("DISCOVERY_RESULTS_PER_QUERY", 10),
            adzuna_country=_get_str("ADZUNA_COUNTRY", "in"),
            embedding_batch_size=_get_int("EMBEDDING_BATCH_SIZE", 32),
            cache_enabled=_get_bool("CACHE_ENABLED", True),
            cover_note_min_words=_get_int("COVER_NOTE_MIN_WORDS", 120),
            cover_note_max_words=_get_int("COVER_NOTE_MAX_WORDS", 180),
            tavily_max_results=_get_int("TAVILY_MAX_RESULTS", 4),
            log_level=_get_str("LOG_LEVEL", "INFO"),
            weights=ScoreWeights(
                semantic=_get_float("WEIGHT_SEMANTIC", 0.45),
                skill_coverage=_get_float("WEIGHT_SKILL_COVERAGE", 0.25),
                experience=_get_float("WEIGHT_EXPERIENCE", 0.15),
                role_relevance=_get_float("WEIGHT_ROLE_RELEVANCE", 0.15),
            ),
        )
        s.weights.validate()
        return s


class CredentialError(RuntimeError):
    """Raised when a required credential is absent. Never contains the value."""


def require_credential(key: str) -> str:
    val = _lookup(key)
    if not val:
        raise CredentialError(
            f"Missing required credential '{key}'. Add it to .env locally or to "
            f"Streamlit Cloud -> App settings -> Secrets. (Value is never logged.)"
        )
    return val


def has_credential(key: str) -> bool:
    return bool(_lookup(key))


def get_settings() -> Settings:
    return Settings.load()
