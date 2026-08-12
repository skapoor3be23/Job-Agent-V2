"""evaluate_current_job(): a degraded (fallback) profile must not have
role_relevance scored as a confirmed 0% match just because role_families
was never computed. Normal (non-degraded) profile scoring must be
byte-for-byte identical to before this fix.

Regression context: the same resume/JD previously scored ~68, then scored
50 on a run where the Gemini profile call failed and the deterministic
fallback profile (role_families=[], evidence=[]) was used instead --
role_relevance was silently scored as a confirmed 0.0 for a candidate whose
actual relevance was simply unknown, not zero.
"""
from agent.config import ScoreWeights
from agent.ranking import evaluate_current_job
from agent.schemas import CandidateProfile, GapAnalysis, JobPosting

WEIGHTS = ScoreWeights()

# Partial coverage on purpose: cov=0.5, evidence_ratio=0.5, eligibility=1.0.
# This makes "confirmed zero" vs "excluded" vs "fully known" land on three
# clearly distinct totals instead of coincidentally overlapping at 100.
JD_TEXT = "Data Analyst Intern requiring Python, SQL, Docker and AWS."


def _job():
    return JobPosting(job_id="x", company="Acme", title="Data Analyst Intern", jd_text=JD_TEXT)


def _gap():
    return GapAnalysis(
        matched_requirements=["Python", "SQL"], missing_requirements=["Docker", "AWS"],
        status="unavailable", reason="Detailed gap analysis unavailable; showing keyword-based comparison.",
    )


def test_degraded_profile_excludes_role_relevance_weight_not_just_its_value():
    profile = CandidateProfile(
        primary_skills=["Python", "SQL"], role_families=[],
        experience_level="student", years_experience=0.0, is_degraded=True,
    )
    opp = evaluate_current_job(_job(), profile, _gap(), WEIGHTS)
    assert opp.score.role_relevance == 0.0
    assert "0%x" in opp.score.formula  # the WEIGHT is 0%, not just the value


def test_degraded_profile_score_matches_the_reweighted_formula_exactly():
    profile = CandidateProfile(
        primary_skills=["Python", "SQL"], role_families=[],
        experience_level="student", years_experience=0.0, is_degraded=True,
    )
    opp = evaluate_current_job(_job(), profile, _gap(), WEIGHTS)

    cov, eligibility, evidence_ratio = 0.5, 1.0, 0.5
    remaining = WEIGHTS.semantic + WEIGHTS.skill_coverage + WEIGHTS.experience
    scale = 1.0 / remaining
    expected_total = int(round((
        WEIGHTS.semantic * scale * evidence_ratio
        + WEIGHTS.skill_coverage * scale * cov
        + WEIGHTS.experience * scale * eligibility
    ) * 100))

    assert opp.score.total == expected_total


def test_degraded_profile_score_is_between_confirmed_zero_and_fully_known_relevance():
    """The whole point of the fix: excluding an unknown signal must land
    strictly between treating it as a confirmed 0% match (the reported
    broken behavior) and treating it as if it were fully known to be
    relevant -- never equal to either extreme."""
    profile = CandidateProfile(
        primary_skills=["Python", "SQL"], role_families=[],
        experience_level="student", years_experience=0.0, is_degraded=True,
    )
    opp = evaluate_current_job(_job(), profile, _gap(), WEIGHTS)

    cov, eligibility, evidence_ratio = 0.5, 1.0, 0.5

    old_confirmed_zero_total = int(round((
        WEIGHTS.semantic * evidence_ratio
        + WEIGHTS.skill_coverage * cov
        + WEIGHTS.experience * eligibility
        + WEIGHTS.role_relevance * 0.0
    ) * 100))

    fully_known_relevant_total = int(round((
        WEIGHTS.semantic * evidence_ratio
        + WEIGHTS.skill_coverage * cov
        + WEIGHTS.experience * eligibility
        + WEIGHTS.role_relevance * 1.0
    ) * 100))

    assert old_confirmed_zero_total < opp.score.total < fully_known_relevant_total


def test_normal_profile_scoring_is_unchanged():
    """A non-degraded profile must score EXACTLY as evaluate_current_job
    computed it before this fix -- the reweighting must never apply outside
    the degraded path."""
    from agent.skills import role_relevance

    profile = CandidateProfile(
        primary_skills=["Python", "SQL"], role_families=["Data Analyst Intern"],
        experience_level="student", years_experience=0.0, is_degraded=False,
    )
    opp = evaluate_current_job(_job(), profile, _gap(), WEIGHTS)

    cov, eligibility, evidence_ratio = 0.5, 1.0, 0.5
    relevance = role_relevance("Data Analyst Intern", ["Data Analyst Intern"])
    expected_total = int(round((
        WEIGHTS.semantic * evidence_ratio
        + WEIGHTS.skill_coverage * cov
        + WEIGHTS.experience * eligibility
        + WEIGHTS.role_relevance * relevance
    ) * 100))

    assert opp.score.total == expected_total
    assert opp.score.role_relevance == relevance
    assert f"{WEIGHTS.role_relevance:.0%}x" in opp.score.formula


def test_normal_profile_with_no_role_family_match_still_uses_unscaled_weights():
    """A REAL (non-degraded) profile with role_families that simply don't
    match this job is a confirmed 0% relevance -- that case is genuinely
    different from "unknown" and must keep the original weights, unlike
    the degraded path."""
    profile = CandidateProfile(
        primary_skills=["Python", "SQL"], role_families=["Marketing Coordinator"],
        experience_level="student", years_experience=0.0, is_degraded=False,
    )
    opp = evaluate_current_job(_job(), profile, _gap(), WEIGHTS)

    cov, eligibility, evidence_ratio = 0.5, 1.0, 0.5
    expected_total = int(round((
        WEIGHTS.semantic * evidence_ratio
        + WEIGHTS.skill_coverage * cov
        + WEIGHTS.experience * eligibility
        + WEIGHTS.role_relevance * 0.0
    ) * 100))

    assert opp.score.total == expected_total
    assert f"{WEIGHTS.role_relevance:.0%}x" in opp.score.formula
