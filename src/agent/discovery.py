"""Opportunity Discovery: find and rank other jobs worth applying to.

Cost profile is deliberately low so this can run inside the parallel branch
without extending the critical path:

- 0 extra LLM calls (role families come from the cached candidate profile)
- 1 batched embedding call for all retrieved JDs plus the resume
- K concurrent Adzuna HTTP calls (no LLM)
- Ranking and explanations are fully deterministic (see ranking.py)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import cache
from .adzuna import JobSourceUnavailable, fetch_many
from .eligibility import EligibilityResult, analyze_eligibility
from .clients import Clients
from .config import Settings
from .identity import resume_hash
from .ranking import score_opportunities
from .roadmap import build_roadmap
from .schemas import CandidateProfile, DiscoveryResult, JobPosting, Opportunity
from .telemetry import BudgetExceeded, RunMetrics


logger = logging.getLogger(__name__)


@dataclass
class _DiscoveryPool:
    """Everything about a profile's discovery pool that costs a network or
    embedding call to produce, cached by profile/resume signature only --
    never by which job the caller wants excluded, since that differs on
    every "analyze this job instead" click and would defeat the cache.
    Excluding the currently-viewed job and applying the display cap happen
    AFTER retrieval (hit or miss), in discover_opportunities().
    """

    jobs_fetched: int = 0
    jobs: List[JobPosting] = field(default_factory=list)
    opportunities: List[Opportunity] = field(default_factory=list)
    degraded_note: Optional[str] = None


def _compute_discovery_pool(
    profile: CandidateProfile,
    resume_text: str,
    queries: List[str],
    clients: Clients,
    settings: Settings,
    metrics: RunMetrics,
) -> _DiscoveryPool:
    """The expensive part of discovery: fetch, deterministic eligibility,
    one batched embedding call, and deterministic scoring. Pulled out of
    discover_opportunities() so it can be cached as a unit."""

    jobs = fetch_many(queries, settings, metrics)
    fetched = len(jobs)

    jobs = [job for job in jobs if (job.jd_text or "").strip()]

    if not jobs:
        return _DiscoveryPool(jobs_fetched=fetched)

    eligibility_results: Dict[str, EligibilityResult] = {
        job.job_id: analyze_eligibility(job, profile)
        for job in jobs
    }

    resume_vector = None
    job_vectors: List[Optional[Sequence[float]]] = [None] * len(jobs)
    degraded_note: Optional[str] = None

    try:
        texts = [resume_text] + [job.jd_text for job in jobs]
        vectors = clients.embeddings.embed_documents(texts)
        resume_vector = vectors[0]
        job_vectors = list(vectors[1:])
    except BudgetExceeded as exc:
        degraded_note = (
            f"Opportunity ranking: {exc} "
            "Ranked on keyword similarity instead."
        )
    except Exception as exc:
        logger.warning(
            "embedding failed during discovery: %s",
            type(exc).__name__,
        )
        degraded_note = (
            "Embeddings unavailable; opportunities ranked "
            "on keyword similarity instead."
        )

    opportunities = score_opportunities(
        jobs=jobs,
        profile=profile,
        resume_text=resume_text,
        resume_vector=resume_vector,
        job_vectors=job_vectors,
        weights=settings.weights,
        eligibility_results=eligibility_results,
    )

    return _DiscoveryPool(
        jobs_fetched=fetched,
        jobs=jobs,
        opportunities=opportunities,
        degraded_note=degraded_note,
    )


def build_search_queries(
    profile: CandidateProfile,
    settings: Settings,
) -> List[str]:
    """Derive search queries from role families the profile already supports.

    No LLM call: role_families was produced by the cached candidate-profile
    call, so the search strategy is free.
    """

    queries: List[str] = []

    for family in profile.role_families:
        family = str(family).strip()

        if family and family.lower() not in [q.lower() for q in queries]:
            queries.append(family)

        if len(queries) >= settings.discovery_max_queries:
            break

    if not queries:
        # Fall back to the strongest skills so discovery still works when
        # the profile call was degraded.
        top = profile.primary_skills[:2]

        if top:
            suffix = (
                "intern"
                if profile.experience_level in ("student", "intern")
                else "engineer"
            )
            queries = [f"{skill} {suffix}" for skill in top]

    return queries[: settings.discovery_max_queries]


def discover_opportunities(
    profile: CandidateProfile,
    resume_text: str,
    clients: Clients,
    settings: Settings,
    metrics: RunMetrics,
    exclude_job_ids: Optional[Sequence[str]] = None,
) -> DiscoveryResult:
    """Discover, analyze eligibility, embed and rank job opportunities.

    This function never raises for expected external failures.
    It returns a DiscoveryResult carrying its own status.
    """

    # ------------------------------------------------------------
    # 1. Check whether discovery is enabled
    # ------------------------------------------------------------

    if not settings.discovery_enabled:
        return DiscoveryResult(
            status="disabled",
            reason="Opportunity Discovery is turned off in configuration.",
        )

    # ------------------------------------------------------------
    # 2. Make sure we have enough candidate information
    # ------------------------------------------------------------

    if not profile.all_skills() and not profile.role_families:
        return DiscoveryResult(
            status="skipped",
            reason="Not enough profile information to search for related roles.",
        )

    # ------------------------------------------------------------
    # 3. Build job-search queries
    # ------------------------------------------------------------

    queries = build_search_queries(profile, settings)

    if not queries:
        return DiscoveryResult(
            status="skipped",
            reason="Could not derive any search queries from the resume.",
        )

    # ------------------------------------------------------------
    # 4-8. Fetch, deterministic eligibility, one batched embedding call and
    # deterministic scoring -- cached as a unit, keyed by profile/resume
    # signature (never by exclude_job_ids, which is different on every
    # "analyze this job instead" click and would defeat the cache). A
    # second job opened from the same discovery list within the TTL costs
    # zero additional Adzuna requests and zero additional embedding calls.
    # ------------------------------------------------------------

    pool_key = cache.make_key(
        "discovery_pool",
        settings.embedding_model,
        settings.adzuna_country,
        settings.discovery_results_per_query,
        resume_hash(resume_text),
        "|".join(q.lower() for q in queries),
    )

    def compute_pool() -> _DiscoveryPool:
        return _compute_discovery_pool(
            profile, resume_text, queries, clients, settings, metrics,
        )

    try:
        pool = cache.get_or_compute(
            pool_key, compute_pool, ttl_s=settings.job_analysis_ttl_s,
            metrics=metrics, enabled=settings.cache_enabled,
        )

    except JobSourceUnavailable as exc:
        metrics.note(f"Opportunity Discovery: {exc}")

        return DiscoveryResult(
            queries_used=queries,
            status="unavailable",
            reason=str(exc),
        )

    except Exception as exc:
        logger.exception("job retrieval failed")

        metrics.note("Opportunity Discovery could not retrieve jobs.")

        return DiscoveryResult(
            queries_used=queries,
            status="unavailable",
            reason=f"Job retrieval failed ({type(exc).__name__}).",
        )

    if pool.degraded_note:
        metrics.note(pool.degraded_note)

    if not pool.jobs:
        return DiscoveryResult(
            queries_used=queries,
            jobs_fetched=pool.jobs_fetched,
            status="no_results",
            reason="No usable opportunities were returned for your profile.",
        )

    # ------------------------------------------------------------
    # 5. Remove the excluded (currently-analyzed) job. IMPORTANT: no cap
    # here beyond that -- the display cap is applied only after ranking,
    # so which jobs survive is decided by relevance/eligibility, never by
    # which network request happened to complete first.
    # ------------------------------------------------------------

    excluded = set(exclude_job_ids or [])

    surviving_jobs = [job for job in pool.jobs if job.job_id not in excluded]
    opportunities = [o for o in pool.opportunities if o.job.job_id not in excluded]

    # ------------------------------------------------------------
    # 9. Build skill-gap roadmap over the FULL analyzed set (not the
    # display-capped one), so its stats reflect every job actually
    # retrieved rather than whichever ones happened to be capped in.
    # ------------------------------------------------------------

    roadmap = build_roadmap(surviving_jobs, profile)

    # ------------------------------------------------------------
    # 10. Apply the display cap AFTER deterministic ranking. `opportunities`
    # is already sorted by (priority, -score, job_id) -- see ranking.py --
    # so this keeps the top-ranked jobs regardless of fetch/completion order.
    # ------------------------------------------------------------

    opportunities = opportunities[: settings.discovery_max_jobs]

    # ------------------------------------------------------------
    # 11. Return complete discovery result
    # ------------------------------------------------------------

    return DiscoveryResult(
        queries_used=queries,
        jobs_fetched=pool.jobs_fetched,
        jobs_deduplicated=len(surviving_jobs),
        opportunities=opportunities,
        roadmap=roadmap,
        status="ok",
    )