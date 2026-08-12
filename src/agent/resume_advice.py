"""Deterministic, non-fabricating resume-improvement suggestions for the
currently analyzed job.

Every suggestion traces to profile.all_skills(), gap.resume_edits (already
LLM-verified to reword only existing resume content), or the JD text itself.
Nothing here invents a skill, project, certification or achievement the
candidate does not have. No LLM call is made in this module -- it reuses
data the pipeline already produced.
"""
from __future__ import annotations

from typing import List, Optional

from .config import ScoreWeights
from .schemas import (
    CandidateProfile,
    GapAnalysis,
    ResumeActionItem,
    ResumeRecommendations,
    ResumeSkillGap,
)
from .skills import coverage, extract_skills, hard_requirements

MIN_JD_CHARS = 20


def _find_evidence_for_skill(profile: CandidateProfile, skill: str) -> Optional[str]:
    """The candidate's own resume text demonstrating `skill`, if any."""
    skill_lower = skill.lower()
    for ev in profile.evidence:
        if skill_lower in {s.lower() for s in ev.skills}:
            return ev.claim
    return None


def _build_detailed_actions(
    profile: CandidateProfile,
    gap: GapAnalysis,
    emphasize: List[str],
    gaps: List[ResumeSkillGap],
) -> List[ResumeActionItem]:
    """What to change / suggested wording / why / score impact, per item.

    suggested_wording is always either verbatim resume text (for emphasize
    items) or an already-reword-only-constrained gap.resume_edits string --
    never synthesized. A missing skill never gets suggested wording, since
    any wording claiming it would be a fabrication.
    """
    actions: List[ResumeActionItem] = []

    for skill in emphasize:
        evidence_claim = _find_evidence_for_skill(profile, skill)
        if evidence_claim:
            actions.append(ResumeActionItem(
                what_to_change=f"Make {skill} more prominent",
                suggested_wording=evidence_claim,
                why=(
                    f"This job asks for {skill}. Your resume already demonstrates it in the "
                    f"line above, but {skill} is currently a secondary skill -- move this line "
                    f"into your summary or the top of your experience section so it isn't missed."
                ),
            ))
        else:
            actions.append(ResumeActionItem(
                what_to_change=f"Make {skill} more prominent",
                suggested_wording="",
                why=(
                    f"{skill} is already listed on your resume but isn't tied to a specific "
                    f"example. Add one concrete sentence describing a project or task where you "
                    f"used it, using only real details."
                ),
            ))

    for g in gaps:
        if g.estimated_score_gain <= 0:
            continue
        actions.append(ResumeActionItem(
            what_to_change=f"Missing: {g.skill}",
            suggested_wording="",
            why=(
                f"{g.skill} is not demonstrated anywhere in your resume. Do not add it unless "
                f"genuinely true -- if you have real, unlisted experience with it, describe that "
                f"specific project rather than just naming the skill."
            ),
            estimated_score_gain=g.estimated_score_gain,
        ))

    for edit in gap.resume_edits:
        actions.append(ResumeActionItem(
            what_to_change="Reword existing content",
            suggested_wording=edit,
            why=(
                "Rewords content already on your resume to match this job description's "
                "language more directly -- no new claims."
            ),
        ))

    return actions


def build_resume_recommendations(
    jd_text: str,
    profile: CandidateProfile,
    gap: GapAnalysis,
    current_score: int,
    weights: ScoreWeights,
) -> ResumeRecommendations:
    """Never invents a skill. Score estimates are point-in-time and describe
    the deterministic scoring model only, not a real-world guarantee."""
    if not jd_text or len(jd_text.strip()) < MIN_JD_CHARS:
        return ResumeRecommendations(
            status="skipped",
            note="No job description was supplied, so resume suggestions could not be generated.",
        )

    required = extract_skills(jd_text)
    candidate_skills = profile.all_skills()

    if not hard_requirements(required):
        return ResumeRecommendations(
            phrasing_suggestions=list(gap.resume_edits),
            unscored_gaps=list(gap.missing_requirements),
            coverage_now=1.0,
            status="insufficient_data",
            note=(
                "No specific technical requirements were detected in this job description, "
                "so skill-gap estimates aren't available. Any phrasing suggestions below come "
                "from the fuller job-description analysis instead."
            ),
        )

    cov, matched, missing = coverage(required, candidate_skills)

    # (a) Already present, but underrepresented: a real, verified skill
    # (already passed the anti-hallucination check in resume.py) that
    # matches this JD but only lives in secondary_skills.
    secondary_lower = {s.lower() for s in profile.secondary_skills}
    emphasize = [s for s in matched if s.lower() in secondary_lower]

    # (b)/(c) Missing skills, each with an honest marginal score estimate:
    # the exact point change the CURRENT deterministic formula would produce
    # if this single skill, and only this one, were genuinely demonstrated.
    # Every other score component is held fixed, isolating this skill's true
    # contribution -- not a guess, and not a real-world promise.
    gaps: List[ResumeSkillGap] = []
    for skill in missing:
        boosted_cov, _, _ = coverage(required, candidate_skills + [skill])
        delta = round((boosted_cov - cov) * weights.skill_coverage * 100)
        gaps.append(ResumeSkillGap(skill=skill, estimated_score_gain=max(0, delta)))
    gaps.sort(key=lambda g: (-g.estimated_score_gain, g.skill))

    # Gaps the LLM-based gap analysis surfaced that fall outside the
    # deterministic skill vocabulary: real signal, but there is no reliable
    # formula to price their score impact, so none is invented.
    covered_terms = {s.lower() for s in matched} | {s.lower() for s in missing}
    unscored = [
        item for item in gap.missing_requirements
        if not any(term in item.lower() for term in covered_terms)
    ]

    note = (
        f"Estimates assume the current deterministic scoring model for this specific job "
        f"description (current match score: {current_score}/100). They describe how the "
        f"match score would change if a skill were genuinely demonstrated -- not a guarantee "
        f"of a higher score or a real interview outcome. Never add a skill, project or "
        f"achievement you cannot honestly back up."
    )

    return ResumeRecommendations(
        emphasize=emphasize,
        phrasing_suggestions=list(gap.resume_edits),
        missing_skills=gaps,
        unscored_gaps=unscored,
        detailed_actions=_build_detailed_actions(profile, gap, emphasize, gaps),
        coverage_now=cov,
        status="ok",
        note=note,
    )
