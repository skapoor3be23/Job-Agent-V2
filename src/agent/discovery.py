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
from typing import Dict, List, Optional, Sequence

from .adzuna import JobSourceUnavailable, fetch_many
from .eligibility import EligibilityResult, analyze_eligibility
from .clients import Clients
from .config import Settings
from .ranking import score_opportunities
from .roadmap import build_roadmap
from .schemas import CandidateProfile, DiscoveryResult
from .telemetry import BudgetExceeded, RunMetrics


logger = logging.getLogger(__name__)


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
    # 4. Retrieve jobs from Adzuna
    # ------------------------------------------------------------

    try:
        jobs = fetch_many(queries, settings, metrics)

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

    # ------------------------------------------------------------
    # 5. Remove excluded / unusable jobs
    # ------------------------------------------------------------

    fetched = len(jobs)

    excluded = set(exclude_job_ids or [])

    jobs = [
        job
        for job in jobs
        if job.job_id not in excluded
        and (job.jd_text or "").strip()
    ]

    jobs = jobs[: settings.discovery_max_jobs]

    if not jobs:
        return DiscoveryResult(
            queries_used=queries,
            jobs_fetched=fetched,
            status="no_results",
            reason="No usable opportunities were returned for your profile.",
        )

    # ------------------------------------------------------------
    # 6. Deterministic eligibility analysis
    # ------------------------------------------------------------
    #
    # Every job is analyzed before ranking.
    #
    # IMPORTANT:
    # Ineligible jobs are NOT removed.
    #
    # Their eligibility result is preserved so the application can explain
    # why a job is unsuitable instead of silently hiding it.
    #

    eligibility_results: Dict[str, EligibilityResult] = {
        job.job_id: analyze_eligibility(job, profile)
        for job in jobs
    }

    # ------------------------------------------------------------
    # 7. One batched embedding call for the resume + every JD
    # ------------------------------------------------------------

    resume_vector = None

    job_vectors = [None] * len(jobs)

    try:
        texts = [resume_text] + [job.jd_text for job in jobs]

        vectors = clients.embeddings.embed_documents(texts)

        resume_vector = vectors[0]
        job_vectors = list(vectors[1:])

    except BudgetExceeded as exc:
        metrics.note(
            f"Opportunity ranking: {exc} "
            "Ranked on keyword similarity instead."
        )

    except Exception as exc:
        logger.warning(
            "embedding failed during discovery: %s",
            type(exc).__name__,
        )

        metrics.note(
            "Embeddings unavailable; opportunities ranked "
            "on keyword similarity instead."
        )

    # ------------------------------------------------------------
    # 8. Deterministic opportunity scoring
    # ------------------------------------------------------------
    #
    # ranking.py will receive the eligibility results and attach them
    # to each Opportunity.
    #

    opportunities = score_opportunities(
    jobs=jobs,
    profile=profile,
    resume_text=resume_text,
    resume_vector=resume_vector,
    job_vectors=job_vectors,
    weights=settings.weights,
    eligibility_results=eligibility_results,
)
    # ------------------------------------------------------------
    # 9. Build skill-gap roadmap
    # ------------------------------------------------------------

    roadmap = build_roadmap(jobs, profile)

    # ------------------------------------------------------------
    # 10. Return complete discovery result
    # ------------------------------------------------------------

    return DiscoveryResult(
        queries_used=queries,
        jobs_fetched=fetched,
        jobs_deduplicated=len(jobs),
        opportunities=opportunities,
        roadmap=roadmap,
        status="ok",
    )