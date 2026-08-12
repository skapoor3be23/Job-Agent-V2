"""Direct unit coverage for the canonical eligibility engine.

Before this test file, analyze_eligibility() had zero direct tests -- it was
only exercised indirectly through discovery/ranking. These pin down each
status branch so eligible/likely_eligible/uncertain/ineligible cannot
silently swap meaning.
"""
from agent.eligibility import analyze_eligibility
from agent.schemas import CandidateProfile, JobPosting

STUDENT = CandidateProfile(experience_level="student", years_experience=0.0)
MID = CandidateProfile(experience_level="mid", years_experience=4.0)


def job(title, jd_text=""):
    return JobPosting(job_id="x", company="Acme", title=title, jd_text=jd_text)


def test_intern_title_with_student_candidate_is_eligible():
    result = analyze_eligibility(job("Machine Learning Intern", "Python and PyTorch."), STUDENT)
    assert result.status == "eligible"
    assert not result.blockers


def test_unspecified_role_with_no_signal_is_likely_eligible():
    result = analyze_eligibility(job("Software Engineer", "Python and SQL."), MID)
    assert result.status == "likely_eligible"
    assert not result.blockers


def test_small_experience_shortfall_is_uncertain_not_blocked():
    result = analyze_eligibility(job("Data Analyst", "Requires 2 years of experience."), STUDENT)
    assert result.status == "uncertain"
    assert not result.blockers
    assert result.warnings


def test_large_experience_shortfall_is_ineligible():
    result = analyze_eligibility(job("Backend Engineer", "Requires 6+ years of experience."), STUDENT)
    assert result.status == "ineligible"
    assert result.blockers


def test_senior_title_with_junior_candidate_is_ineligible():
    result = analyze_eligibility(job("Senior Backend Engineer", "Build APIs."), STUDENT)
    assert result.status == "ineligible"
    assert result.blockers


def test_senior_title_with_experienced_candidate_is_not_blocked():
    result = analyze_eligibility(job("Senior Backend Engineer", "Build APIs."), MID)
    assert result.status != "ineligible"
    assert not result.blockers


def test_missing_fields_do_not_crash():
    result = analyze_eligibility(JobPosting(job_id="x"), CandidateProfile())
    assert result.status in {"eligible", "likely_eligible", "uncertain", "ineligible"}


# ---------------------------------------------------------------------------
# Regression: explanation must never contradict status.
#
# Root cause: the "explicitly targeted toward interns..." positive message
# was chosen BEFORE checking blockers/warnings, so a junior-friendly title
# with a genuine blocker (e.g. an unmet years requirement) produced an
# "ineligible"/"uncertain" status paired with a reassuring, unrelated
# explanation -- exactly the contradiction visible on the dashboard as
# "student, 0 yrs" + "explicitly targeted toward interns..." + a
# "Stretch Opportunity" priority whose stated blocker was that same
# positive-sounding sentence.
# ---------------------------------------------------------------------------

def test_blocker_explanation_is_never_overridden_by_the_positive_junior_message():
    result = analyze_eligibility(
        job("Junior Backend Developer", "Junior Backend Developer requiring 8+ years of experience."),
        STUDENT,
    )
    assert result.status == "ineligible"
    assert result.blockers
    assert result.explanation == result.blockers[0]
    assert "explicitly targeted" not in result.explanation


def test_warning_explanation_is_never_overridden_by_the_positive_junior_message():
    result = analyze_eligibility(
        job("Junior Backend Developer", "Junior Backend Developer requiring 2 years of experience."),
        STUDENT,
    )
    assert result.status == "uncertain"
    assert result.warnings
    assert result.explanation == result.warnings[0]
    assert "explicitly targeted" not in result.explanation


def test_positive_message_still_shown_when_nothing_is_actually_wrong():
    result = analyze_eligibility(job("Software Engineering Intern", "Python and Git."), STUDENT)
    assert result.status == "eligible"
    assert not result.blockers and not result.warnings
    assert "explicitly targeted" in result.explanation


# ---------------------------------------------------------------------------
# Regression: company-history years must never be read as an experience
# requirement.
#
# Real failure case: an actual Amgen posting states "Amgen helped establish
# the biotechnology industry more than 40 years ago." The old pattern
# `(?:minimum|at least|required|must have)?\s*(\d+)\s*\+?\s*years?` had an
# OPTIONAL qualifier, so a bare "40 years" matched regardless of context,
# and a student candidate was marked ineligible against a fabricated
# 40-year requirement that was never actually in the posting.
# ---------------------------------------------------------------------------

AMGEN_JD = (
    "Amgen helped establish the biotechnology industry more than 40 years ago, "
    "and remains on the cutting edge of innovation as we reach millions of "
    "patients each year."
)


def test_company_history_years_are_not_treated_as_a_requirement():
    result = analyze_eligibility(job("Associate Scientist", AMGEN_JD), STUDENT)
    assert result.required_years == 0.0
    assert result.status != "ineligible"
    assert not result.blockers


def test_real_requirement_still_detected_alongside_company_history():
    """The fix must not become so strict that it stops catching genuine
    requirements sitting in the same job description."""
    jd = AMGEN_JD + " This role requires a minimum of 5 years of relevant industry experience."
    result = analyze_eligibility(job("Associate Scientist", jd), STUDENT)
    assert result.required_years == 5.0
    assert result.status == "ineligible"
    assert result.blockers


def test_years_old_is_not_a_requirement():
    result = analyze_eligibility(
        job("Data Analyst", "Our flagship analytics product is over 10 years old and trusted company-wide."),
        STUDENT,
    )
    assert result.required_years == 0.0
    assert result.status != "ineligible"


def test_bare_years_with_no_experience_context_is_ignored():
    result = analyze_eligibility(
        job("Research Assistant", "Our lab has published for over 25 years in this field."),
        STUDENT,
    )
    assert result.required_years == 0.0
