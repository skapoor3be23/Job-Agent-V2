"""Merged critique + revision.

The original pipeline spent two sequential round trips here: one to score,
one to rewrite. The rewrite depends only on the critique the same model just
produced, so nothing is gained by a second call. This asks for the score,
the issues, a needs_revision flag and the corrected note in one structured
response -- removing one hop from the critical path without dropping the
revision step.

The deterministic validator (cover_note.validate_cover_note) runs AFTER this,
because a model asked to check its own output is not a reliable guard.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from .clients import Clients, LLMUnavailable
from .config import Settings
from .schemas import (
    CandidateProfile,
    CompanyResearch,
    CoverNote,
    CoverNoteCritique,
    GapAnalysis,
)
from .telemetry import BudgetExceeded, RunMetrics

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT = """Evaluate a cover note, then revise it if needed. One response, both tasks.

The note may ONLY contain claims traceable to the material below. Your most
important job is to catch claims that are not supported by it.

--- TARGET ROLE ---
Company: {company}
Job title: {title}

--- JOB DESCRIPTION ---
{jd_text}

--- CANDIDATE EVIDENCE (the only permitted claims about the candidate) ---
{candidate_block}

--- VERIFIED COMPANY FACTS (the only permitted claims about the company) ---
{company_block}

--- COVER NOTE UNDER REVIEW ---
{note}

Score each 0-100:
- grounding_score: are ALL claims traceable to the material above? Any
  company fact not in the allow-list, or any candidate claim not in the
  evidence, must push this below 50 and appear in ungrounded_claims.
- specificity_score: concrete details rather than vague assertions.
- jd_alignment_score: does it address what the job description actually asks for?
- writing_score: natural, direct, free of filler phrases, {min_words}-{max_words} words.

List every unsupported claim verbatim in ungrounded_claims.

Set needs_revision=true if grounding_score < 80, or any other score < 60,
or ungrounded_claims is non-empty. In that case put a fully corrected note in
corrected_note: remove every unsupported claim rather than substituting a
different one, and keep it addressed to {company} for the {title} role.

If the note is already sound, set needs_revision=false and leave
corrected_note empty.
"""


def _weighted_score(c: CoverNoteCritique) -> int:
    """Grounding is weighted heaviest: a fluent note built on invented facts
    is worse than an unpolished honest one."""
    return int(round(
        0.40 * c.grounding_score
        + 0.20 * c.specificity_score
        + 0.20 * c.jd_alignment_score
        + 0.20 * c.writing_score
    ))


def critique_and_revise(
    note_text: str,
    company: str,
    title: str,
    jd_text: str,
    profile: CandidateProfile,
    research: CompanyResearch,
    candidate_block: str,
    company_block: str,
    clients: Clients,
    settings: Settings,
    metrics: RunMetrics,
) -> tuple[str, int, List[str], bool]:
    """Return (final_text, score, issues, revised).

    Degrades to the original draft on any failure -- returning the safest
    valid draft beats failing the run.
    """
    prompt = CRITIQUE_PROMPT.format(
        company=company or "(unnamed)", title=title or "(unspecified)",
        jd_text=(jd_text or "")[:5000],
        candidate_block=candidate_block, company_block=company_block,
        note=note_text,
        min_words=settings.cover_note_min_words, max_words=settings.cover_note_max_words,
    )

    try:
        critique = clients.llm.structured(prompt, CoverNoteCritique)
    except BudgetExceeded as exc:
        metrics.note(f"Cover-note critique: {exc}. The unreviewed draft is shown.")
        return note_text, 0, ["Critique skipped (call budget reached)."], False
    except LLMUnavailable as exc:
        logger.warning("critique failed: %s", exc)
        metrics.note("Cover-note critique unavailable; the unreviewed draft is shown.")
        return note_text, 0, ["Critique unavailable."], False

    issues = list(critique.issues)
    if critique.ungrounded_claims:
        issues.append("Unsupported claims found: " + "; ".join(critique.ungrounded_claims))

    if critique.needs_revision and critique.corrected_note.strip():
        return critique.corrected_note.strip(), _weighted_score(critique), issues, True
    return note_text, _weighted_score(critique), issues, False
