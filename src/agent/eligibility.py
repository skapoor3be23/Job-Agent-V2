"""Deterministic job eligibility analysis.

This module decides whether a discovered job is realistically suitable
for the candidate before the opportunity is ranked.

No LLM calls are made here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from .schemas import CandidateProfile, JobPosting


@dataclass
class EligibilityResult:
    """Result of deterministic eligibility analysis."""

    status: str
    blockers: List[str]
    warnings: List[str]
    required_years: float
    detected_level: str
    explanation: str


# Strong seniority markers.
# These are checked against the JOB TITLE first, because a word such as
# "senior" appearing somewhere in the JD does not necessarily mean the
# role itself is senior.
SENIORITY_PATTERNS = {
    "intern": re.compile(
        r"\b(intern|internship|trainee|co[- ]?op)\b",
        re.IGNORECASE,
    ),
    "entry": re.compile(
        r"\b(entry[- ]level|graduate|new grad|fresher|junior)\b",
        re.IGNORECASE,
    ),
    "senior": re.compile(
        r"\b(senior|sr\.?|lead|staff|principal|manager|director|head|architect)\b",
        re.IGNORECASE,
    ),
}


# Explicit experience requirements such as:
# "3+ years"
# "3 years of experience"
# "minimum 3 years"
# "at least 3 years"
# "5 years in a similar role"
#
# A bare "<N> years" is NOT enough on its own: JDs routinely say things like
# "Amgen helped establish the biotechnology industry more than 40 years
# ago", which is company history, not a candidate requirement. Each pattern
# therefore requires an explicit qualifier word before the number OR an
# explicit "experience"/"role" phrase after it -- never a number floating
# with no experience-indicating context.
EXPERIENCE_PATTERNS = [
    re.compile(
        r"\b(?:minimum(?:\s+of)?|at\s+least|requires?|required|must\s+have)\s+"
        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*years?\s+"
        r"(?:of\s+)?(?:relevant\s+|related\s+|professional\s+|work\s+)?experience\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*years?\s+in\s+(?:a\s+)?(?:similar|related)\s+role\b",
        re.IGNORECASE,
    ),
]


def _extract_required_years(text: str) -> float:
    """Return the smallest explicit years-of-experience requirement.

    Only counts a number of years when the surrounding language clearly
    frames it as a requirement on the candidate -- never a bare "<N> years"
    that could just as easily be company history ("40 years ago"), product
    age, or anything else unrelated to eligibility.
    """

    if not text:
        return 0.0

    values = []

    for pattern in EXPERIENCE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                values.append(float(match.group(1)))
            except (TypeError, ValueError):
                continue

    return min(values) if values else 0.0


def _detect_job_level(job: JobPosting) -> str:
    """Infer seniority primarily from the job title."""

    title = (job.title or "").lower().strip()

    # Check more specific senior levels first.
    if re.search(
        r"\b(principal|staff|director|head|architect)\b",
        title,
        re.IGNORECASE,
    ):
        return "senior"

    if re.search(
        r"\b(manager|engineering manager|people manager)\b",
        title,
        re.IGNORECASE,
    ):
        return "manager"

    if re.search(
        r"\b(senior|sr\.?|lead)\b",
        title,
        re.IGNORECASE,
    ):
        return "senior"

    if re.search(
        r"\b(intern|internship|trainee|co[- ]?op)\b",
        title,
        re.IGNORECASE,
    ):
        return "intern"

    if re.search(
        r"\b(entry[- ]level|graduate|new grad|fresher|junior)\b",
        title,
        re.IGNORECASE,
    ):
        return "entry"

    return "unspecified"


def analyze_eligibility(
    job: JobPosting,
    profile: CandidateProfile,
) -> EligibilityResult:
    """Classify a job as eligible, likely eligible, uncertain, or ineligible."""

    jd = job.jd_text or ""
    title = job.title or ""

    required_years = _extract_required_years(jd)
    detected_level = _detect_job_level(job)

    blockers: List[str] = []
    warnings: List[str] = []

    candidate_years = profile.years_experience
    candidate_level = profile.experience_level

    # ------------------------------------------------------------
    # 1. Seniority / role-level blocker
    # ------------------------------------------------------------

    if detected_level in {"senior", "manager"}:
        if candidate_level in {"student", "intern", "entry"}:
            blockers.append(
                f"Role appears to be {detected_level}-level, while the "
                f"candidate profile is {candidate_level}-level."
            )

    # ------------------------------------------------------------
    # 2. Explicit years-of-experience requirement
    # ------------------------------------------------------------

    if required_years > 0 and candidate_years < required_years:
        shortfall = required_years - candidate_years

        if shortfall >= 3:
            blockers.append(
                f"Job asks for about {required_years:.0f}+ years of experience; "
                f"the candidate profile shows about {candidate_years:.0f}."
            )
        else:
            warnings.append(
                f"Job asks for about {required_years:.0f}+ years of experience; "
                f"the candidate profile shows about {candidate_years:.0f}."
            )

    # ------------------------------------------------------------
    # 3. Positive signals
    # ------------------------------------------------------------

    junior_signal = bool(
        re.search(
            r"\b(intern|internship|trainee|co[- ]?op|entry[- ]level|graduate|"
            r"new grad|fresher|junior)\b",
            title,
            re.IGNORECASE,
        )
    )

    # Blockers and warnings are real, specific findings against this job;
    # they must always win over the generic positive-signal message below.
    # Checking the positive signal first (as this used to) let a job with a
    # junior-friendly title but a genuine blocker (e.g. an unmet years
    # requirement) report a reassuring explanation while still being marked
    # ineligible -- a direct contradiction between status and explanation.
    if blockers:
        explanation = blockers[0]
    elif warnings:
        explanation = warnings[0]
    elif candidate_level in {"student", "intern"} and junior_signal:
        explanation = (
            "The role is explicitly targeted toward interns, graduates, "
            "or early-career candidates."
        )
    else:
        explanation = (
            "No high-confidence eligibility blocker was detected from "
            "the available job description."
        )

    # ------------------------------------------------------------
    # 4. Final classification
    # ------------------------------------------------------------

    if blockers:
        status = "ineligible"
    elif warnings:
        status = "uncertain"
    elif detected_level in {"intern", "entry"}:
        status = "eligible"
    else:
        status = "likely_eligible"

    return EligibilityResult(
        status=status,
        blockers=blockers,
        warnings=warnings,
        required_years=required_years,
        detected_level=detected_level,
        explanation=explanation,
    )