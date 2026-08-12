"""Deterministic opportunity scoring, priority assignment and explanations.

No LLM assigns a score or writes an explanation here.

The ranking system combines:

    45% semantic similarity
    25% required-skill coverage
    15% experience / eligibility match
    15% role relevance

Eligibility itself is calculated by eligibility.py and passed into this
module. This keeps eligibility logic separate from ranking logic.

All headline numbers such as "8 of 10 requirements demonstrated" are
deterministic arithmetic and can therefore be verified.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ScoreWeights
from .eligibility import EligibilityResult, analyze_eligibility
from .schemas import (
    CandidateProfile,
    GapAnalysis,
    JobPosting,
    Opportunity,
    ScoreBreakdown,
)
from .skills import (
    coverage,
    extract_skills,
    hard_requirements,
    lexical_similarity,
    role_relevance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eligibility -> numeric score
# ---------------------------------------------------------------------------
#
# eligibility.py is the ONLY module that decides eligibility. This map is the
# single deterministic translation from its status to a 0..1 score used by
# the composite formula. The four levels are intentionally NOT collapsed:
#
#   eligible        1.00  -- strongest: explicit junior/entry signal matched
#   likely_eligible 0.80  -- reasonably strong, but no explicit confirming signal
#   uncertain       0.50  -- meaningfully lower confidence (a real warning exists)
#   ineligible      0.00  -- blocked; a hard blocker is attached
#
# A missing/unknown result is never assumed to be "likely eligible" -- it is
# always resolved by calling analyze_eligibility() below, never guessed.
ELIGIBILITY_SCORE_MAP: Dict[str, float] = {
    "eligible": 1.0,
    "likely_eligible": 0.8,
    "uncertain": 0.5,
    "ineligible": 0.0,
}


# ---------------------------------------------------------------------------
# Priority thresholds
# ---------------------------------------------------------------------------

STRONG_COVERAGE = 0.70
MODERATE_COVERAGE = 0.45
STRETCH_COVERAGE = 0.25

MIN_EVIDENCE_MATCHES = 2
ELIGIBILITY_FLOOR = 0.50


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

def cosine_similarity(
    a: Optional[Sequence[float]],
    b: Optional[Sequence[float]],
) -> Optional[float]:
    """Calculate cosine similarity between two vectors."""

    if not a or not b or len(a) != len(b):
        return None

    dot = sum(x * y for x, y in zip(a, b))

    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na == 0 or nb == 0:
        return None

    return dot / (na * nb)


def rank_normalize(
    values: Sequence[Optional[float]],
) -> List[float]:
    """Map raw similarities onto the range 0..1 by rank.

    Raw embedding similarities can cluster tightly. Rank normalization
    provides better separation between jobs while retaining the raw
    similarity value for transparency.
    """

    present = [
        (i, v)
        for i, v in enumerate(values)
        if v is not None
    ]

    out = [0.0] * len(values)

    if not present:
        return out

    if len(present) == 1:
        out[present[0][0]] = 1.0
        return out

    ordered = sorted(
        present,
        key=lambda p: p[1],
    )

    n = len(ordered)

    i = 0

    while i < n:
        j = i

        while (
            j + 1 < n
            and abs(
                ordered[j + 1][1] - ordered[i][1]
            ) < 1e-9
        ):
            j += 1

        avg_rank = (i + j) / 2.0
        norm = avg_rank / (n - 1)

        for k in range(i, j + 1):
            out[ordered[k][0]] = round(norm, 4)

        i = j + 1

    return out


# ---------------------------------------------------------------------------
# Resume evidence
# ---------------------------------------------------------------------------

def _matching_evidence(
    profile: CandidateProfile,
    matched_skills: Sequence[str],
) -> List[str]:
    """Return resume evidence items that demonstrate matched skills."""

    wanted = {
        s.lower()
        for s in matched_skills
    }

    hits: List[str] = []

    for ev in profile.evidence:
        if any(
            s.lower() in wanted
            for s in ev.skills
        ):
            hits.append(ev.claim)

    return hits[:3]


# ---------------------------------------------------------------------------
# Eligibility integration
# ---------------------------------------------------------------------------

def _resolve_eligibility(
    job: JobPosting,
    profile: CandidateProfile,
    eligibility_result: Optional[EligibilityResult],
) -> EligibilityResult:
    """Return a real EligibilityResult for this job, computing one if the
    caller didn't supply it.

    There is no "assume it's fine" fallback: a missing entry is resolved by
    calling the canonical eligibility engine, not by guessing a score.
    """

    if eligibility_result is not None:
        return eligibility_result
    return analyze_eligibility(job, profile)


def _eligibility_details(
    eligibility_result: EligibilityResult,
) -> Tuple[float, List[str]]:
    """Convert a resolved EligibilityResult into (score, blockers).

    Only "ineligible" produces a hard blocker for assign_priority(); the
    other three statuses differ only in score, per ELIGIBILITY_SCORE_MAP.
    """

    status = (eligibility_result.status or "").strip().lower()
    score = ELIGIBILITY_SCORE_MAP.get(status, 0.5)

    if status == "ineligible":
        # The actual blocker text is the reason, never the free-text
        # `explanation` field -- explanation can (correctly) prioritize a
        # different message and must not overwrite the specific blocker
        # shown to the user here.
        reason = (
            (eligibility_result.blockers[0] if eligibility_result.blockers else "")
            or eligibility_result.explanation
            or "The role does not meet the candidate's eligibility criteria."
        )
        return score, [reason]

    return score, []


# ---------------------------------------------------------------------------
# Priority assignment
# ---------------------------------------------------------------------------

def assign_priority(
    coverage_score: float,
    eligibility: float,
    blockers: Sequence[str],
    evidence_count: int,
    relevance: float,
) -> Tuple[str, str]:
    """Assign application priority using deterministic evidence.

    Eligibility blockers take precedence over semantic similarity.
    """

    # ------------------------------------------------------------
    # Hard eligibility blocker
    # ------------------------------------------------------------

    if blockers:

        if coverage_score >= STRONG_COVERAGE:
            return (
                "Stretch Opportunity",
                (
                    f"Skills line up well ({coverage_score:.0%} coverage), "
                    f"but there is an eligibility issue: {blockers[0]}"
                ),
            )

        return (
            "Low Priority",
            f"Blocked on eligibility: {blockers[0]}",
        )

    # ------------------------------------------------------------
    # Strong match
    # ------------------------------------------------------------

    if (
        coverage_score >= STRONG_COVERAGE
        and eligibility >= ELIGIBILITY_FLOOR
        and evidence_count >= MIN_EVIDENCE_MATCHES
    ):
        return (
            "Apply Now",
            (
                f"{coverage_score:.0%} of the stated technical "
                f"requirements are demonstrated in your resume, "
                f"with {evidence_count} supporting items and "
                f"no eligibility blocker."
            ),
        )

    # ------------------------------------------------------------
    # Strong skills but weak evidence
    # ------------------------------------------------------------

    if (
        coverage_score >= STRONG_COVERAGE
        and evidence_count < MIN_EVIDENCE_MATCHES
    ):
        return (
            "Worth Applying",
            (
                f"Skill overlap is high ({coverage_score:.0%}) "
                f"but your resume shows limited direct evidence "
                f"for it, so the application needs a strong "
                f"cover note."
            ),
        )

    # ------------------------------------------------------------
    # Moderate match
    # ------------------------------------------------------------

    if (
        coverage_score >= MODERATE_COVERAGE
        and eligibility >= ELIGIBILITY_FLOOR
    ):
        return (
            "Worth Applying",
            (
                f"{coverage_score:.0%} skill coverage with "
                f"real gaps remaining, but nothing that "
                f"disqualifies you."
            ),
        )

    # ------------------------------------------------------------
    # Stretch
    # ------------------------------------------------------------

    if (
        coverage_score >= STRETCH_COVERAGE
        and relevance >= 0.3
    ):
        return (
            "Stretch Opportunity",
            (
                f"Only {coverage_score:.0%} of requirements "
                f"are demonstrated, though the role type "
                f"is relevant to your profile."
            ),
        )

    # ------------------------------------------------------------
    # Low priority
    # ------------------------------------------------------------

    return (
        "Low Priority",
        (
            f"Weak alignment: {coverage_score:.0%} skill "
            f"coverage and low role relevance."
        ),
    )


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def _why_match(
    matched: Sequence[str],
    total_required: int,
    evidence: Sequence[str],
) -> str:
    """Explain why the candidate matches the role."""

    if not total_required:
        return (
            "The posting does not state specific technical "
            "requirements, so this is ranked on overall similarity."
        )

    parts = [
        (
            f"{len(matched)} of the {total_required} stated "
            "technical requirements are demonstrated in your resume"
        )
    ]

    if matched:
        parts.append(
            "covering " + ", ".join(matched[:5])
        )

    if evidence:
        parts.append(
            "Your strongest supporting evidence: "
            + "; ".join(evidence[:2])
        )

    return ". ".join(parts) + "."


def _why_apply(
    priority: str,
    missing: Sequence[str],
    relevance: float,
) -> str:
    """Explain whether applying is worthwhile."""

    if priority == "Apply Now":
        base = (
            "Your time is well spent here — "
            "the fit is already strong."
        )

    elif priority == "Worth Applying":
        base = (
            "Worth an application, but expect "
            "to address the gaps directly."
        )

    elif priority == "Stretch Opportunity":
        base = (
            "A stretch: apply only if you have spare "
            "capacity after stronger matches."
        )

    else:
        base = (
            "Low return on effort compared with "
            "your better-matched options."
        )

    if missing:
        base += (
            " Main gap"
            + ("s" if len(missing) > 1 else "")
            + ": "
            + ", ".join(missing[:3])
            + "."
        )

    return base


# ---------------------------------------------------------------------------
# Shared opportunity assembly (used for both discovered jobs and the single
# currently-analyzed job, so the two paths can never compute contradictory
# scores/eligibility for the same kind of job)
# ---------------------------------------------------------------------------

def _assemble_opportunity(
    job: JobPosting,
    weights: ScoreWeights,
    required: Sequence[str],
    cov: float,
    matched: Sequence[str],
    missing: Sequence[str],
    eligibility_result: EligibilityResult,
    relevance: float,
    evidence: Sequence[str],
    semantic_raw: float,
    semantic_normalized: float,
) -> Opportunity:
    eligibility, eligibility_blockers = _eligibility_details(eligibility_result)

    total = (
        weights.semantic * semantic_normalized
        + weights.skill_coverage * cov
        + weights.experience * eligibility
        + weights.role_relevance * relevance
    )
    total = max(0.0, min(1.0, total))

    breakdown = ScoreBreakdown(
        semantic_raw=round(semantic_raw, 4),
        semantic_normalized=round(semantic_normalized, 4),
        skill_coverage=round(cov, 4),
        experience_match=round(eligibility, 4),
        role_relevance=round(relevance, 4),
        total=int(round(total * 100)),
        formula=(
            f"{weights.semantic:.0%}x{semantic_normalized:.2f} + "
            f"{weights.skill_coverage:.0%}x{cov:.2f} + "
            f"{weights.experience:.0%}x{eligibility:.2f} + "
            f"{weights.role_relevance:.0%}x{relevance:.2f}"
        ),
    )

    priority, reason = assign_priority(
        coverage_score=cov,
        eligibility=eligibility,
        blockers=eligibility_blockers,
        evidence_count=len(evidence),
        relevance=relevance,
    )

    return Opportunity(
        job=job,
        score=breakdown,
        eligibility_status=eligibility_result.status,
        eligibility_explanation=eligibility_result.explanation,
        eligibility_warnings=list(eligibility_result.warnings),
        matched_skills=list(matched),
        missing_skills=list(missing),
        evidence=list(evidence),
        priority=priority,
        priority_reason=reason,
        why_match=_why_match(matched, len(hard_requirements(required)), evidence),
        why_apply=_why_apply(priority, missing, relevance),
        blockers=eligibility_blockers,
    )


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_opportunities(
    jobs: Sequence[JobPosting],
    profile: CandidateProfile,
    resume_text: str,
    resume_vector: Optional[Sequence[float]],
    job_vectors: Sequence[Optional[Sequence[float]]],
    weights: ScoreWeights,
    eligibility_results: Optional[Dict] = None,
) -> List[Opportunity]:
    """Score and explain every job.

    This function is pure and deterministic.

    eligibility_results is produced by eligibility.py and keyed by
    job_id. Ranking does not independently invent eligibility decisions.
    """

    candidate_skills = profile.all_skills()

    eligibility_results = eligibility_results or {}

    # ------------------------------------------------------------
    # 1. Calculate semantic similarity for every job
    # ------------------------------------------------------------

    raw_similarities: List[Optional[float]] = []

    for i, job in enumerate(jobs):

        vector = (
            job_vectors[i]
            if i < len(job_vectors)
            else None
        )

        sim = cosine_similarity(
            resume_vector,
            vector,
        )

        if sim is None:
            # Deterministic fallback if embeddings unavailable.
            sim = lexical_similarity(
                resume_text,
                job.jd_text,
            )

        raw_similarities.append(sim)

    # ------------------------------------------------------------
    # 2. Normalize semantic similarities by rank
    # ------------------------------------------------------------

    normalized = rank_normalize(
        raw_similarities
    )

    opportunities: List[Opportunity] = []

    # ------------------------------------------------------------
    # 3. Analyze every job
    # ------------------------------------------------------------

    for i, job in enumerate(jobs):

        # --------------------------------------------------------
        # Skill analysis
        # --------------------------------------------------------

        required = extract_skills(
            job.jd_text
        )

        cov, matched, missing = coverage(
            required,
            candidate_skills,
        )

        # --------------------------------------------------------
        # Eligibility analysis (eligibility.py is the sole source of
        # truth; a missing entry is resolved, never assumed)
        # --------------------------------------------------------

        eligibility_result = _resolve_eligibility(
            job, profile, eligibility_results.get(job.job_id)
        )

        # --------------------------------------------------------
        # Role relevance
        # --------------------------------------------------------

        relevance = role_relevance(
            job.title,
            profile.role_families,
        )

        # --------------------------------------------------------
        # Resume evidence
        # --------------------------------------------------------

        evidence = _matching_evidence(
            profile,
            matched,
        )

        # --------------------------------------------------------
        # Score, priority and Opportunity assembly (shared with the
        # single-job path in evaluate_current_job)
        # --------------------------------------------------------

        opportunities.append(
            _assemble_opportunity(
                job=job,
                weights=weights,
                required=required,
                cov=cov,
                matched=matched,
                missing=missing,
                eligibility_result=eligibility_result,
                relevance=relevance,
                evidence=evidence,
                semantic_raw=raw_similarities[i] or 0.0,
                semantic_normalized=normalized[i],
            )
        )

    # ------------------------------------------------------------
    # 4. Deterministic ordering
    # ------------------------------------------------------------

    priority_rank = {
        "Apply Now": 0,
        "Worth Applying": 1,
        "Stretch Opportunity": 2,
        "Low Priority": 3,
    }

    # job_id is the tertiary key so equal priority+score jobs order the same
    # way regardless of fetch/completion order (never popularity, company
    # name or title keywords -- just a stable, arbitrary-but-fixed string).
    opportunities.sort(
        key=lambda o: (
            priority_rank[o.priority],
            -o.score.total,
            o.job.job_id,
        )
    )

    return opportunities


# ---------------------------------------------------------------------------
# Single-job evaluation (the currently analyzed job, outside discovery)
# ---------------------------------------------------------------------------

def evaluate_current_job(
    job: JobPosting,
    profile: CandidateProfile,
    gap: GapAnalysis,
    weights: ScoreWeights,
) -> Opportunity:
    """Score the one job the user pasted in, using the same eligibility
    engine, coverage function and Opportunity assembly as discovered jobs.

    There is no comparison pool to rank-normalize a semantic similarity
    against here, so the "semantic" component is replaced by the evidence
    ratio from the LLM-verified gap analysis (matched / (matched + missing)).
    This mirrors what the previous PipelineResult.match_score() did, but
    now shares one scoring implementation with ranking.py instead of a
    second, independently-written formula.
    """

    required = extract_skills(job.jd_text)
    candidate_skills = profile.all_skills()
    cov, matched, missing = coverage(required, candidate_skills)

    eligibility_result = analyze_eligibility(job, profile)
    relevance = role_relevance(job.title, profile.role_families)
    evidence = _matching_evidence(profile, matched)

    gap_matched = len(gap.matched_requirements)
    gap_total = gap_matched + len(gap.missing_requirements)
    evidence_ratio = (gap_matched / gap_total) if gap_total else cov

    return _assemble_opportunity(
        job=job,
        weights=weights,
        required=required,
        cov=cov,
        matched=matched,
        missing=missing,
        eligibility_result=eligibility_result,
        relevance=relevance,
        evidence=evidence,
        semantic_raw=evidence_ratio,
        semantic_normalized=evidence_ratio,
    )