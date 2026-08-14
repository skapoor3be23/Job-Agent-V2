"""Resume parsing and candidate profiling -- performed once per resume.

Previously three modules each defined their own extract_resume_text() and
re-parsed the PDF independently. Here parsing is cached by content hash and
the profile (one structured LLM call) is cached by the same hash, so a
second run on the same resume costs zero calls.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from . import cache, profile_store
from .clients import Clients, LLMUnavailable
from .config import Settings
from .identity import resume_hash
from .schemas import CandidateProfile, ResumeEvidence
from .skills import extract_skills, normalize_skill_list
from .telemetry import BudgetExceeded, RunMetrics

logger = logging.getLogger(__name__)

PROFILE_PROMPT = """You are building a factual profile of a candidate from their resume.

ABSOLUTE RULE: extract only what the resume actually states. Do not infer,
embellish, or add skills that would be typical for someone with this
background. If the resume does not mention it, it does not exist.

For role_families, list realistic job-title families this resume genuinely
supports, most relevant first (maximum 4). Do not list aspirational roles
the resume provides no evidence for.

RESUME:
{resume_text}
"""


def extract_pdf_text(file_or_path: Any) -> str:
    """Extract text from a PDF path or file-like object."""
    from pypdf import PdfReader

    reader = PdfReader(file_or_path)
    return "".join(page.extract_text() or "" for page in reader.pages)


def parse_resume(file_or_path: Any, metrics: Optional[RunMetrics] = None) -> str:
    """Parse a resume to text. Cached where the source is a stable path."""
    if isinstance(file_or_path, (str, bytes)):
        key = cache.make_key("resume_text", str(file_or_path))
        return cache.get_or_compute(
            key, lambda: extract_pdf_text(file_or_path), metrics=metrics
        )
    return extract_pdf_text(file_or_path)


def _fallback_profile(resume_text: str) -> CandidateProfile:
    """Deterministic profile used when the LLM is unavailable or budgeted out.

    Keeps the app functional (Phase: graceful degradation) with reduced
    richness rather than failing the whole run. Always marked is_degraded
    so downstream scoring knows this profile's characteristics (notably
    role_families) are unknown, not confirmed-absent.
    """
    found = extract_skills(resume_text)
    return CandidateProfile(
        primary_skills=found[:8],
        secondary_skills=found[8:16],
        role_families=[],
        experience_level="student",
        years_experience=0.0,
        is_degraded=True,
    )


def _backfill_evidence_skills(evidence: List[ResumeEvidence]) -> None:
    """Fill in each evidence item's own `skills` tag from its claim text.

    The model may tag an item's skills sparsely even when the claim text
    itself plainly names a skill. This re-scans each claim with the same
    deterministic vocabulary extract_skills() uses everywhere else and adds
    anything found -- never removing an existing tag (the model may have
    tagged a skill from broader resume context that isn't spelled out in
    this specific bullet) and never inventing one the claim text doesn't
    state. Mutates each item's `skills` list in place.
    """
    for ev in evidence:
        existing = normalize_skill_list(ev.skills)
        seen = {s.lower() for s in existing}
        merged = list(existing)
        for skill in extract_skills(ev.claim):
            if skill.lower() not in seen:
                seen.add(skill.lower())
                merged.append(skill)
        ev.skills = merged


def _reuse_persisted_or_fallback(
    key: str,
    resume_text: str,
    settings: Settings,
) -> CandidateProfile:
    """When the LLM call fails, prefer a profile already successfully
    computed for this exact resume + model (surviving a process restart,
    since the in-memory cache does not) over the reduced-richness fallback.

    Never invents anything: this only returns exactly what a previous,
    successful call already produced, or the deterministic fallback.
    """
    if settings.cache_enabled:
        persisted = profile_store.load(key, ttl_s=settings.job_analysis_ttl_s)
        if persisted is not None:
            return persisted
    return _fallback_profile(resume_text)


def build_candidate_profile(
    resume_text: str,
    clients: Clients,
    settings: Settings,
    metrics: RunMetrics,
) -> CandidateProfile:
    """One structured LLM call, cached by resume hash.

    This same call also produces role_families, so Opportunity Discovery's
    search strategy costs no additional round trip.
    """
    if not resume_text or len(resume_text.strip()) < 50:
        metrics.note("Resume text was empty or too short to profile.")
        return CandidateProfile()

    key = cache.make_key("candidate_profile", settings.chat_model, resume_hash(resume_text))

    def compute() -> CandidateProfile:
        prompt = PROFILE_PROMPT.format(resume_text=resume_text[:12000])
        return clients.llm.structured(prompt, CandidateProfile)

    try:
        profile = cache.get_or_compute(
            key, compute, ttl_s=settings.job_analysis_ttl_s,
            metrics=metrics, enabled=settings.cache_enabled,
        )
    except BudgetExceeded as exc:
        metrics.note(f"Candidate profile: {exc}. Used keyword extraction instead.")
        return _reuse_persisted_or_fallback(key, resume_text, settings)
    except LLMUnavailable as exc:
        logger.warning("profile generation failed: %s", exc)
        metrics.note("Candidate profile could not be generated by the model; used keyword extraction instead.")
        return _reuse_persisted_or_fallback(key, resume_text, settings)

    # A real, successfully computed profile is never degraded -- this is
    # never trusted from the model's own output, only set here.
    profile.is_degraded = False

    # Reconcile with deterministic extraction: the model may phrase a skill
    # differently, but it must never ADD a skill absent from the resume.
    resume_skills = {s.lower() for s in extract_skills(resume_text)}
    profile.primary_skills = normalize_skill_list(profile.primary_skills)
    profile.secondary_skills = normalize_skill_list(profile.secondary_skills)

    unverified = [
        s for s in profile.primary_skills
        if s.lower() not in resume_skills and s.lower() not in resume_text.lower()
    ]
    if unverified:
        logger.info("dropping %d unverifiable skills from profile", len(unverified))
        profile.primary_skills = [s for s in profile.primary_skills if s not in unverified]
        profile.secondary_skills = [s for s in profile.secondary_skills if s not in unverified]

    # Add anything the deterministic pass found that the model omitted.
    known = {s.lower() for s in profile.all_skills()}
    for skill in extract_skills(resume_text):
        if skill.lower() not in known:
            profile.secondary_skills.append(skill)

    _backfill_evidence_skills(profile.evidence)

    # Persist so a later process restart (or an LLM outage) can reuse this
    # real profile instead of falling back to reduced-richness extraction.
    if settings.cache_enabled:
        profile_store.save(key, profile)

    return profile
