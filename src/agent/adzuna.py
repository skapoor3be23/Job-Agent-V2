"""Adzuna job source, refactored from the original fetch_jobs.py script.

Changes: credentials are validated before the request (the original sent
app_id=None and got an opaque 401), results are returned as JobPosting
objects with a job_id, and multiple queries fan out concurrently.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence

from .config import CredentialError, Settings, has_credential, require_credential
from .identity import make_job_id
from .schemas import JobPosting
from .telemetry import RunMetrics

logger = logging.getLogger(__name__)

BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"
REQUEST_TIMEOUT_S = 20


class JobSourceUnavailable(RuntimeError):
    """The job source could not be reached or is not configured."""


def credentials_available() -> bool:
    return has_credential("ADZUNA_APP_ID") and has_credential("ADZUNA_APP_KEY")


def _to_posting(raw: Dict) -> Optional[JobPosting]:
    """Convert one Adzuna record, tolerating missing fields."""
    try:
        company = str((raw.get("company") or {}).get("display_name") or "").strip()
        title = str(raw.get("title") or "").strip()
        jd_text = str(raw.get("description") or "").strip()
        location = str((raw.get("location") or {}).get("display_name") or "").strip()
        url = str(raw.get("redirect_url") or "").strip()
        if not title and not company:
            return None
        return JobPosting(
            job_id=make_job_id(company, title, url, location, jd_text),
            company=company or "(company not listed)",
            title=title or "(title not listed)",
            jd_text=jd_text, location=location, url=url, source="adzuna",
        )
    except Exception:
        logger.debug("skipping malformed Adzuna record")
        return None


def fetch_one(query: str, settings: Settings, results_per_page: Optional[int] = None) -> List[JobPosting]:
    """Fetch one query. Returns [] on failure rather than raising."""
    import requests

    try:
        app_id = require_credential("ADZUNA_APP_ID")
        app_key = require_credential("ADZUNA_APP_KEY")
    except CredentialError as exc:
        raise JobSourceUnavailable(str(exc)) from exc

    url = BASE_URL.format(country=settings.adzuna_country)
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": results_per_page or settings.discovery_results_per_query,
        "what": query,
        "content-type": "application/json",
    }
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("Adzuna query failed (%s): %s", type(exc).__name__, query)
        return []

    postings = [_to_posting(r) for r in payload.get("results", []) or []]
    return [p for p in postings if p is not None]


def fetch_many(
    queries: Sequence[str],
    settings: Settings,
    metrics: Optional[RunMetrics] = None,
) -> List[JobPosting]:
    """Run several queries concurrently and deduplicate by job_id.

    These are plain HTTP calls with no LLM cost, so they parallelise freely.
    """
    if not queries:
        return []
    if not credentials_available():
        raise JobSourceUnavailable(
            "Adzuna credentials are not configured (ADZUNA_APP_ID / ADZUNA_APP_KEY). "
            "Opportunity Discovery needs them to retrieve live jobs."
        )

    # Results are gathered concurrently (as_completed just drives progress),
    # but assembled back into QUERY order rather than completion order, so
    # which jobs end up first/duplicated-out never depends on network timing.
    results_by_query: List[List[JobPosting]] = [[] for _ in queries]
    with ThreadPoolExecutor(max_workers=min(len(queries), 6)) as pool:
        future_to_index = {pool.submit(fetch_one, q, settings): i for i, q in enumerate(queries)}
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results_by_query[index] = future.result()
            except Exception:
                logger.warning("query thread failed: %s", queries[index])

    collected = [job for jobs in results_by_query for job in jobs]

    seen, unique = set(), []
    for job in collected:
        if job.job_id in seen:
            continue
        seen.add(job.job_id)
        unique.append(job)
    logger.info("fetched %d jobs, %d unique", len(collected), len(unique))
    return unique
