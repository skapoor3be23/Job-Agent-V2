"""Structured gap analysis: which JD requirements the resume actually meets."""
from __future__ import annotations

import logging

from . import cache
from .clients import Clients, LLMUnavailable
from .config import Settings
from .identity import content_hash
from .schemas import CandidateProfile, GapAnalysis
from .skills import coverage, extract_skills
from .telemetry import BudgetExceeded, RunMetrics

logger = logging.getLogger(__name__)

GAP_PROMPT = """You are a technical recruiter comparing a resume against one job description.

RULES:
- A requirement counts as "matched" only if the resume shows concrete evidence.
  Adjacent or plausible experience is NOT a match.
- resume_edits must reword content already in the resume. Never invent a
  project, metric, tool or responsibility the candidate does not have.
- Be blunt. An honest "weak" verdict is more useful than a flattering one.

JOB: {title} at {company}

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}
"""

MIN_JD_CHARS = 20


def analyze_gap(
    resume_text: str,
    profile: CandidateProfile,
    company: str,
    title: str,
    jd_text: str,
    clients: Clients,
    settings: Settings,
    metrics: RunMetrics,
) -> GapAnalysis:
    """Never raises. Falls back to deterministic skill coverage if the LLM
    is unavailable, so the run still yields a usable comparison."""
    if not jd_text or len(jd_text.strip()) < MIN_JD_CHARS:
        return GapAnalysis(
            status="skipped",
            reason="Job description was empty or too short to analyse.",
        )

    def deterministic() -> GapAnalysis:
        _, matched, missing = coverage(extract_skills(jd_text), profile.all_skills())
        return GapAnalysis(
            matched_requirements=matched,
            missing_requirements=missing,
            fit_verdict="strong" if len(matched) > len(missing) else "weak",
            fit_reason="Derived from keyword skill matching (model analysis unavailable).",
            status="unavailable",
            reason="Detailed gap analysis unavailable; showing keyword-based comparison.",
        )

    key = cache.make_key(
        "gap_analysis", settings.chat_model,
        content_hash(resume_text)[:12], content_hash(company, title, jd_text),
    )

    def compute() -> GapAnalysis:
        prompt = GAP_PROMPT.format(
            title=title or "(unspecified)", company=company or "(unspecified)",
            jd_text=jd_text[:8000], resume_text=resume_text[:10000],
        )
        result = clients.llm.structured(prompt, GapAnalysis)
        result.status = "ok"
        return result

    try:
        return cache.get_or_compute(
            key, compute, ttl_s=settings.job_analysis_ttl_s,
            metrics=metrics, enabled=settings.cache_enabled,
        )
    except BudgetExceeded as exc:
        metrics.note(f"Gap analysis: {exc}")
        return deterministic()
    except LLMUnavailable as exc:
        logger.warning("gap analysis failed: %s", exc)
        return deterministic()
    except Exception:
        logger.exception("gap analysis failed")
        return deterministic()
