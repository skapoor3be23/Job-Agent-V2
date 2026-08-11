"""Deterministic job identity.

(company, title) is NOT sufficient: one company routinely posts several
distinct roles under an identical title, and the same role appears under
different URLs. job_id therefore folds in URL, location and a fingerprint
of the description.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional
from urllib.parse import urlparse, urlunparse

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")

JOB_ID_LENGTH = 16
_JD_FINGERPRINT_CHARS = 600


def normalize_text(value: Optional[str]) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Stable across
    cosmetic edits (casing, spacing, trailing punctuation)."""
    if not value:
        return ""
    text = str(value).lower()
    text = _NON_ALNUM.sub(" ", text)
    return _WS.sub(" ", text).strip()


def canonical_url(url: Optional[str]) -> str:
    """Drop scheme, query string and fragment so tracking params
    (?utm_source=...) don't create phantom distinct jobs."""
    if not url:
        return ""
    try:
        parsed = urlparse(str(url).strip())
        if not parsed.netloc:
            return normalize_text(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/")
        return urlunparse(("", netloc, path, "", "", "")).lstrip("/")
    except Exception:
        return normalize_text(url)


def make_job_id(
    company: Optional[str],
    title: Optional[str],
    job_url: Optional[str] = None,
    location: Optional[str] = None,
    jd_text: Optional[str] = None,
) -> str:
    """Stable 16-hex-char identifier for a job posting.

    Identical inputs always produce an identical id (safe as a cache key
    and a database key). Any material difference -- different URL, different
    location, different description -- produces a different id.
    """
    parts = [
        normalize_text(company),
        normalize_text(title),
        canonical_url(job_url),
        normalize_text(location),
        normalize_text(jd_text)[:_JD_FINGERPRINT_CHARS],
    ]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:JOB_ID_LENGTH]


def resume_hash(resume_text: Optional[str]) -> str:
    """Content hash of a resume, used to cache parsing/profiling/embedding."""
    return hashlib.sha256(normalize_text(resume_text).encode("utf-8")).hexdigest()[:16]


def content_hash(*parts: Optional[str]) -> str:
    payload = "|".join(normalize_text(p) for p in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
