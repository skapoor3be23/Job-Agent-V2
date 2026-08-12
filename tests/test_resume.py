"""Candidate profiling: ResumeEvidence.skills backfill.

resume.py's reconciliation pass already prunes/backfills primary_skills and
secondary_skills against the actual resume text; nothing previously did the
same for each evidence item's own `skills` tag -- which resume_advice.py
depends on to find safe, paste-ready wording. An LLM call that returns
evidence with sparse or empty `skills` (a routine LLM behavior when the
prompt doesn't emphasize the field) produced silent downstream gaps even
though the claim text itself demonstrably named a skill.
"""
from agent import cache
from agent.config import Settings
from agent.resume import build_candidate_profile
from agent.schemas import CandidateProfile, ResumeEvidence
from agent.telemetry import new_run
from mocks import make_clients

RESUME_TEXT = (
    "Kritagya. Mechatronics undergraduate. Built an EKF-based mobile robot in ROS 2 "
    "using Python. Computer vision pipeline using OpenCV and PyTorch. SQL data "
    "analysis with Pandas."
)


def _profile_with_evidence(evidence):
    return CandidateProfile(
        primary_skills=["Python", "PyTorch", "SQL"], secondary_skills=[],
        role_families=["Machine Learning Intern"],
        experience_level="student", years_experience=0.0,
        evidence=evidence,
    )


def _build(profile):
    settings = Settings(cache_enabled=False)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics, llm_responses={"CandidateProfile": profile})
    return build_candidate_profile(RESUME_TEXT, clients, settings, metrics)


def test_empty_evidence_skills_are_backfilled_from_claim_text():
    profile = _profile_with_evidence([
        ResumeEvidence(claim="Built an EKF-based mobile robot in ROS 2 using Python", skills=[], kind="project"),
    ])
    result = _build(profile)
    tags = {s.lower() for s in result.evidence[0].skills}
    assert "python" in tags
    assert "ros" in tags


def test_backfill_never_invents_a_skill_the_claim_does_not_state():
    profile = _profile_with_evidence([
        ResumeEvidence(claim="Led a small team through a semester-long capstone project", skills=[], kind="project"),
    ])
    result = _build(profile)
    assert result.evidence[0].skills == []


def test_backfill_preserves_existing_llm_tags_not_literally_in_the_claim_text():
    """The model may legitimately tag a skill from broader resume context
    that isn't spelled out in this specific bullet -- backfill must add to
    that, never remove it."""
    profile = _profile_with_evidence([
        ResumeEvidence(claim="Built an EKF-based mobile robot in ROS 2", skills=["Python"], kind="project"),
    ])
    result = _build(profile)
    tags = {s.lower() for s in result.evidence[0].skills}
    assert "python" in tags  # preserved, even though "Python" isn't in this claim's text
    assert "ros" in tags     # backfilled, since "ROS 2" is in the claim text


def test_backfill_deduplicates_case_insensitively():
    profile = _profile_with_evidence([
        ResumeEvidence(claim="Built an EKF-based mobile robot in ROS 2 using Python", skills=["python"], kind="project"),
    ])
    result = _build(profile)
    lowered = [s.lower() for s in result.evidence[0].skills]
    assert lowered.count("python") == 1


# ---------------------------------------------------------------------------
# Degraded-profile marking and disk persistence across a "process restart".
# ---------------------------------------------------------------------------

def test_successful_profile_is_not_marked_degraded():
    profile = _profile_with_evidence([])
    result = _build(profile)
    assert result.is_degraded is False


def test_successful_profile_overrides_a_model_provided_degraded_flag():
    """Only _fallback_profile may set is_degraded=True -- a real,
    successfully computed profile must never trust the model's own
    opinion on this control field."""
    profile = _profile_with_evidence([])
    profile.is_degraded = True  # simulate a model that (incorrectly) set this
    result = _build(profile)
    assert result.is_degraded is False


def test_llm_failure_without_a_persisted_profile_falls_back_and_is_degraded():
    settings = Settings(cache_enabled=True)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics, llm_fail_on=["CandidateProfile"])
    result = build_candidate_profile(RESUME_TEXT, clients, settings, metrics)
    assert result.is_degraded is True
    assert result.role_families == []


def test_llm_failure_reuses_a_previously_persisted_profile_not_the_fallback():
    """A profile successfully computed once must survive an in-memory
    cache wipe (simulating a process restart) when the LLM then fails on a
    later call for the exact same resume + model."""
    real_profile = _profile_with_evidence([
        ResumeEvidence(claim="Built an EKF-based mobile robot in ROS 2 using Python", skills=["Python"], kind="project"),
    ])
    settings = Settings(cache_enabled=True)

    metrics1 = new_run(settings)
    clients1 = make_clients(settings, metrics1, llm_responses={"CandidateProfile": real_profile})
    first = build_candidate_profile(RESUME_TEXT, clients1, settings, metrics1)
    assert first.is_degraded is False
    assert first.role_families == ["Machine Learning Intern"]

    # Simulate a process restart: the in-memory cache is gone, but the
    # on-disk persisted profile is not.
    cache.clear()

    metrics2 = new_run(settings)
    clients2 = make_clients(settings, metrics2, llm_fail_on=["CandidateProfile"])
    second = build_candidate_profile(RESUME_TEXT, clients2, settings, metrics2)

    assert second.is_degraded is False
    assert second.role_families == ["Machine Learning Intern"]
    assert second.evidence and second.evidence[0].claim == real_profile.evidence[0].claim


def test_persistence_respects_cache_disabled():
    """cache_enabled=False must skip both the read and the write of the
    on-disk profile store, matching the existing in-memory cache contract."""
    real_profile = _profile_with_evidence([])
    settings = Settings(cache_enabled=False)

    metrics1 = new_run(settings)
    clients1 = make_clients(settings, metrics1, llm_responses={"CandidateProfile": real_profile})
    build_candidate_profile(RESUME_TEXT, clients1, settings, metrics1)

    metrics2 = new_run(settings)
    clients2 = make_clients(settings, metrics2, llm_fail_on=["CandidateProfile"])
    second = build_candidate_profile(RESUME_TEXT, clients2, settings, metrics2)

    assert second.is_degraded is True  # nothing was persisted to fall back on
