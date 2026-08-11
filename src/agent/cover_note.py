"""Grounded cover-note generation + deterministic validation.

The generation prompt supplies a CLOSED allow-list of company facts. The
original prompt's unconditional instruction to "reference specific company
facts" is deleted: that instruction is what produced invented funding rounds
whenever research came back empty.

Validation afterwards is deterministic. Cross-company contamination is
detected by string matching, not by asking another model -- a judge model
that hallucinates cannot be trusted to catch hallucination.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence

from .config import Settings
from .identity import normalize_text
from .schemas import (
    CandidateProfile,
    CompanyResearch,
    CoverNote,
    GapAnalysis,
    ValidationResult,
)
from .telemetry import RunMetrics

logger = logging.getLogger(__name__)

BANNED_PHRASES = [
    "i am passionate about",
    "i'm passionate about",
    "passionate about",
    "quick learner",
    "fast learner",
    "team player",
    "perfect fit",
    "ideal candidate",
    "excited to bring my skills",
    "i believe i would be",
    "hit the ground running",
    "think outside the box",
    "detail-oriented",
    "self-motivated",
    "hard worker",
    "go-getter",
    "wealth of experience",
]

# Words that signal an unsourced claim about the employer.
_COMPANY_CLAIM_MARKERS = re.compile(
    r"\b(funding|funded|series [a-e]\b|valuation|raised|ipo|unicorn|acquisition|acquired|"
    r"market leader|leading provider|industry leader|fastest[- ]growing|award[- ]winning|"
    r"mission to|your culture|renowned|pioneering|cutting[- ]edge platform)\b",
    re.I,
)

_LEGAL_SUFFIX = re.compile(
    r"\b(private limited|pvt\.? ?ltd\.?|limited|ltd\.?|inc\.?|llc|llp|corp\.?|corporation|"
    r"technologies|technology|solutions|systems|labs|group|holdings|co\.?)\b",
    re.I,
)

GENERATION_PROMPT = """Write a cover note for one specific job application.

=== THE ONLY FACTS YOU MAY USE ===

You may state ONLY what appears below. You have no other knowledge of this
company or this candidate. Anything not listed here does not exist for the
purposes of this note.

--- TARGET ROLE (this and no other) ---
Company: {company}
Job title: {title}

--- JOB DESCRIPTION ---
{jd_text}

--- CANDIDATE EVIDENCE (from their resume; the only claims you may make about them) ---
{candidate_block}

--- VERIFIED COMPANY FACTS (allow-list) ---
{company_block}

=== RULES ===

1. COMPANY FACTS. You may reference a company fact ONLY if it appears in the
   allow-list above. {company_rule}
2. CANDIDATE FACTS. Every claim about the candidate must trace to the
   candidate evidence above. Never invent a project, internship, metric,
   tool, certification or responsibility.
3. NO OVERCLAIMING. If the job description asks for something the candidate
   has not demonstrated, do not imply they have it. Silence is acceptable.
4. INTEREST. Ground stated interest in something concrete from the JOB
   DESCRIPTION (the actual work) rather than praise of the company.
5. LENGTH. {min_words}-{max_words} words.
6. TONE. Direct and specific, as a real candidate would write. Ban these
   phrases entirely: "passionate about", "quick learner", "team player",
   "perfect fit", "excited to bring my skills", "hit the ground running".
7. Address {company} for the {title} role and no other company or role.

Return only the cover note text. No subject line, no salutation placeholders
like [Hiring Manager], no commentary.
"""

NO_FACTS_RULE = (
    "No verified company facts are available, so you must make NO factual claims "
    "about the company at all -- not about its products, size, funding, customers, "
    "mission, culture, technology or achievements. Refer to it by name only, and "
    "build the note from the job description and the candidate's evidence."
)

WITH_FACTS_RULE = (
    "Use at most two of the listed facts, and only where genuinely relevant. "
    "Do not extend, embellish or generalise beyond what a listed fact states."
)


def _candidate_block(profile: CandidateProfile, resume_text: str) -> str:
    lines: List[str] = []
    if profile.primary_skills:
        lines.append("Demonstrated skills: " + ", ".join(profile.primary_skills))
    if profile.secondary_skills:
        lines.append("Secondary skills: " + ", ".join(profile.secondary_skills))
    for ev in profile.evidence[:8]:
        skills = f" [{', '.join(ev.skills)}]" if ev.skills else ""
        lines.append(f"- ({ev.kind}) {ev.claim}{skills}")
    if profile.education:
        lines.append("Education: " + "; ".join(profile.education))
    if not lines:
        lines.append(resume_text[:3000])
    return "\n".join(lines)


def _company_block(research: CompanyResearch) -> str:
    facts = research.usable_facts()
    if not facts:
        return "(none available -- see rule 1)"
    return "\n".join(f"- {f.fact}  [source: {f.source_title or f.source_url}]" for f in facts)


def company_name_tokens(company: str) -> List[str]:
    """Distinctive tokens of a company name, ignoring legal suffixes.

    "Acme Technologies Pvt Ltd" -> ["acme"], so a note for Acme is not
    flagged merely because another company also ends in "Technologies".
    """
    cleaned = _LEGAL_SUFFIX.sub(" ", company or "")
    tokens = [t for t in re.findall(r"[a-z0-9]+", cleaned.lower()) if len(t) > 2]
    return tokens


def validate_cover_note(
    note: str,
    company: str,
    title: str,
    profile: CandidateProfile,
    research: CompanyResearch,
    settings: Settings,
    other_companies: Optional[Sequence[str]] = None,
) -> ValidationResult:
    """Deterministic checks. No LLM. Failures reject the note; warnings don't."""
    result = ValidationResult()
    text = note or ""
    lowered = text.lower()
    result.word_count = len(text.split())

    if not text.strip():
        result.passed = False
        result.failures.append("Cover note is empty.")
        return result

    # --- cross-company contamination (the critical check) ---
    own_tokens = set(company_name_tokens(company))
    for other in other_companies or []:
        if normalize_text(other) == normalize_text(company):
            continue
        foreign = [t for t in company_name_tokens(other) if t not in own_tokens]
        hits = [t for t in foreign if re.search(rf"\b{re.escape(t)}\b", lowered)]
        if hits:
            result.passed = False
            result.failures.append(
                f"Mentions another company from this session ('{other}') "
                f"in a note addressed to '{company}'."
            )

    # --- the note must actually name its own target ---
    if own_tokens and not any(re.search(rf"\b{re.escape(t)}\b", lowered) for t in own_tokens):
        result.warnings.append(f"Note never names the target company '{company}'.")

    if title:
        title_tokens = [t for t in re.findall(r"[a-z]+", title.lower()) if len(t) > 3]
        if title_tokens and not any(t in lowered for t in title_tokens):
            result.warnings.append(f"Note never references the role '{title}'.")

    # --- unsourced company claims ---
    if not research.usable_facts():
        claims = _COMPANY_CLAIM_MARKERS.findall(text)
        if claims:
            result.passed = False
            result.failures.append(
                "Makes company-specific claims "
                f"({', '.join(sorted({c.lower() for c in claims}))}) "
                "although no verified company facts were available."
            )

    # --- banned filler ---
    found = [p for p in BANNED_PHRASES if p in lowered]
    if found:
        result.warnings.append("Generic filler phrases: " + ", ".join(found))

    # --- length ---
    if result.word_count < settings.cover_note_min_words:
        result.warnings.append(
            f"Short: {result.word_count} words (target {settings.cover_note_min_words}-{settings.cover_note_max_words})."
        )
    elif result.word_count > settings.cover_note_max_words * 1.4:
        result.warnings.append(
            f"Long: {result.word_count} words (target {settings.cover_note_min_words}-{settings.cover_note_max_words})."
        )

    return result


def build_generation_prompt(
    company: str,
    title: str,
    jd_text: str,
    profile: CandidateProfile,
    research: CompanyResearch,
    resume_text: str,
    settings: Settings,
) -> str:
    has_facts = bool(research.usable_facts())
    return GENERATION_PROMPT.format(
        company=company or "(unnamed company)",
        title=title or "(unspecified role)",
        jd_text=(jd_text or "(no job description supplied)")[:6000],
        candidate_block=_candidate_block(profile, resume_text),
        company_block=_company_block(research),
        company_rule=WITH_FACTS_RULE if has_facts else NO_FACTS_RULE,
        min_words=settings.cover_note_min_words,
        max_words=settings.cover_note_max_words,
    )
