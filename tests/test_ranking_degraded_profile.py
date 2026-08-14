"""evaluate_current_job(): a degraded (fallback) profile must not have
role_relevance scored as a confirmed 0% match just because role_families
was never computed. Normal (non-degraded) profile scoring must be
byte-for-byte identical to before this fix.

Regression context: the same resume/JD previously scored ~68, then scored
50 on a run where the Gemini profile call failed and the deterministic
fallback profile (role_families=[], evidence=[]) was used instead --
role_relevance was silently scored as a confirmed 0.0 for a candidate whose
actual relevance was simply unknown, not zero.

NOTE: evaluate_current_job() no longer takes a GapAnalysis. Its semantic
component now comes from the same embedding-cosine-similarity signal
score_opportunities() uses (see tests/test_scoring_consistency.py for why),
so these tests inject an explicit resume/job vector pair instead of a gap
analysis result.
"""
from agent.config import ScoreWeights
from agent.ranking import cosine_similarity, evaluate_current_job
from agent.schemas import CandidateProfile, JobPosting

WEIGHTS = ScoreWeights()

# An arbitrary, non-trivial vector pair (cosine similarity strictly between
# 0 and 1) so "confirmed zero" vs "excluded" vs "fully known" land on three
# clearly distinct totals instead of coincidentally overlapping.
RESUME_VECTOR = [1.0, 0.0]
JOB_VECTOR = [1.0, 1.0]
SEMANTIC = max(0.0, min(1.0, cosine_similarity(RESUME_VECTOR, JOB_VECTOR)))

JD_TEXT = "Data Analyst Intern requiring Python, SQL, Docker and AWS."


def _job():
    return JobPosting(job_id="x", company="Acme", title="Data Analyst Intern", jd_text=JD_TEXT)


def _evaluate(profile):
    return evaluate_current_job(
        _job(), profile, WEIGHTS,
        resume_vector=RESUME_VECTOR, job_vector=JOB_VECTOR,
    )


def test_degraded_profile_excludes_role_relevance_weight_not_just_its_value():
    profile = CandidateProfile(
        primary_skills=["Python", "SQL"], role_families=[],
        experience_level="student", years_experience=0.0, is_degraded=True,
    )
    opp = _evaluate(profile)
    assert opp.score.role_relevance == 0.0
    assert "0%x" in opp.score.formula  # the WEIGHT is 0%, not just the value


def test_degraded_profile_score_matches_the_reweighted_formula_exactly():
    profile = CandidateProfile(
        primary_skills=["Python", "SQL"], role_families=[],
        experience_level="student", years_experience=0.0, is_degraded=True,
    )
    opp = _evaluate(profile)

    cov, eligibility = 0.5, 1.0
    remaining = WEIGHTS.semantic + WEIGHTS.skill_coverage + WEIGHTS.experience
    scale = 1.0 / remaining
    expected_total = int(round((
        WEIGHTS.semantic * scale * SEMANTIC
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
    opp = _evaluate(profile)

    cov, eligibility = 0.5, 1.0

    old_confirmed_zero_total = int(round((
        WEIGHTS.semantic * SEMANTIC
        + WEIGHTS.skill_coverage * cov
        + WEIGHTS.experience * eligibility
        + WEIGHTS.role_relevance * 0.0
    ) * 100))

    fully_known_relevant_total = int(round((
        WEIGHTS.semantic * SEMANTIC
        + WEIGHTS.skill_coverage * cov
        + WEIGHTS.experience * eligibility
        + WEIGHTS.role_relevance * 1.0
    ) * 100))

    assert old_confirmed_zero_total < opp.score.total < fully_known_relevant_total


def test_normal_profile_scoring_is_unchanged():
    """A non-degraded profile must score EXACTLY as evaluate_current_job
    computes it -- the reweighting must never apply outside the degraded
    path."""
    from agent.skills import role_relevance

    profile = CandidateProfile(
        primary_skills=["Python", "SQL"], role_families=["Data Analyst Intern"],
        experience_level="student", years_experience=0.0, is_degraded=False,
    )
    opp = _evaluate(profile)

    cov, eligibility = 0.5, 1.0
    relevance = role_relevance("Data Analyst Intern", ["Data Analyst Intern"])
    expected_total = int(round((
        WEIGHTS.semantic * SEMANTIC
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
    opp = _evaluate(profile)

    cov, eligibility = 0.5, 1.0
    expected_total = int(round((
        WEIGHTS.semantic * SEMANTIC
        + WEIGHTS.skill_coverage * cov
        + WEIGHTS.experience * eligibility
        + WEIGHTS.role_relevance * 0.0
    ) * 100))

    assert opp.score.total == expected_total
    assert f"{WEIGHTS.role_relevance:.0%}x" in opp.score.formula
