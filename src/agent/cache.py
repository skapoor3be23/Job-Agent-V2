"""Process-wide cache with TTL, safe under Streamlit reruns and threads.

Streamlit re-executes the script top-to-bottom on every interaction, so a
plain module-level dict would be rebuilt constantly in some deployment
modes. st.cache_resource pins a single store per session; outside Streamlit
(tests, CLI) it degrades to a module-level dict.

Keys are explicit and input-derived -- never positional -- so a change in
any relevant input produces a different key rather than a stale hit.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_local_store: Dict[str, Tuple[float, Any]] = {}
_lock = threading.RLock()


def _get_store() -> Dict[str, Tuple[float, Any]]:
    try:
        import streamlit as st

        @st.cache_resource(show_spinner=False)
        def _store() -> Dict[str, Tuple[float, Any]]:
            return {}

        return _store()
    except Exception:
        return _local_store


def make_key(namespace: str, *parts: Any) -> str:
    """Build an explicit, collision-resistant cache key."""
    joined = "|".join(str(p) for p in parts)
    return f"{namespace}::{joined}"


def get(key: str) -> Optional[Any]:
    store = _get_store()
    with _lock:
        entry = store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if expires_at and expires_at < time.time():
            store.pop(key, None)
            return None
        return value


def set(key: str, value: Any, ttl_s: Optional[int] = None) -> None:
    store = _get_store()
    expires_at = time.time() + ttl_s if ttl_s else 0.0
    with _lock:
        store[key] = (expires_at, value)


def get_or_compute(
    key: str,
    compute: Callable[[], T],
    ttl_s: Optional[int] = None,
    metrics: Optional[Any] = None,
    enabled: bool = True,
) -> T:
    """Return a cached value or compute and store it.

    Cache hits/misses are recorded on the run metrics so the Performance
    panel can show them.
    """
    if not enabled:
        if metrics:
            metrics.record_cache_miss()
        return compute()

    hit = get(key)
    if hit is not None:
        if metrics:
            metrics.record_cache_hit()
        logger.debug("cache hit: %s", key)
        return hit

    if metrics:
        metrics.record_cache_miss()
    value = compute()
    if value is not None:
        set(key, value, ttl_s)
    return value


def clear() -> None:
    store = _get_store()
    with _lock:
        store.clear()


def size() -> int:
    return len(_get_store())
