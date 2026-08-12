"""Disk-backed persistence for successfully computed candidate profiles.

The generic cache in cache.py is process-memory only: an st.cache_resource
singleton under Streamlit, or a plain module dict outside it. Neither
survives a Streamlit dev-server restart, an app redeploy, or the process
simply being restarted. A profile that was already computed successfully
for a given resume + model should not need a fresh (and possibly failing,
e.g. on a quota outage) LLM call just because the process restarted -- this
gives that one cache a durable, on-disk backing, scoped narrowly to
candidate profiles only.

Never invents or repairs data: this module only stores and retrieves
exactly what was already successfully computed.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from .schemas import CandidateProfile

logger = logging.getLogger(__name__)

# A bare module attribute (not a function default) so tests can redirect it
# with monkeypatch and have every call -- even ones that don't pass `path`
# explicitly -- pick up the override.
DEFAULT_PATH = "data/profile_cache.json"

_lock = threading.RLock()


def _read_all(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def load(key: str, ttl_s: Optional[int] = None, path: Optional[str] = None) -> Optional[CandidateProfile]:
    """A previously persisted profile for this key, or None.

    Degrades to None (never raises) on a missing, corrupt, or unreadable
    store -- a broken cache file must not break profiling.
    """
    resolved_path = path or DEFAULT_PATH
    with _lock:
        data = _read_all(resolved_path)

    entry = data.get(key)
    if not entry:
        return None

    saved_at = entry.get("_saved_at", 0)
    if ttl_s and (time.time() - saved_at) > ttl_s:
        return None

    try:
        return CandidateProfile.model_validate(entry.get("profile", {}))
    except Exception:
        logger.warning("discarding unreadable persisted profile for key %s", key)
        return None


def save(key: str, profile: CandidateProfile, path: Optional[str] = None) -> None:
    """Persist a successfully computed profile so it survives a restart.

    Best-effort: a write failure (permissions, disk full) is logged and
    swallowed rather than failing the run that just successfully profiled.
    """
    resolved_path = path or DEFAULT_PATH
    with _lock:
        data = _read_all(resolved_path)
        data[key] = {"_saved_at": time.time(), "profile": profile.model_dump(mode="json")}
        try:
            Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except OSError as exc:
            logger.warning("could not persist candidate profile: %s", exc)
