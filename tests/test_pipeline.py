"""Pipeline behaviour: parallelism, degradation, budgets, caching."""
import time

import pytest
from agent import cache
from agent.config import Settings
from agent.graph import run_pipeline
from agent.schemas import CandidateProfile, GapAnalysis, ResumeEvidence, _FactExtraction
from agent.telemetry import new_run
from mocks import make_clients

RESUME = (
    "Kritagya. Mechatronics undergraduate. Projects: built an EKF-based mobile robot in ROS 2 "
    "with Python. Computer vision pipeline using OpenCV and PyTorch. SQL data analysis with Pandas."
)
JD = (
    "Machine Learning Intern. You will build NLP pipelines in Python using PyTorch. "
    "Familiarity with Docker and AWS is a plus. Collaboration with the team is important."
)

PROFILE = CandidateProfile(
    primary_skills=["Python", "PyTorch", "Computer Vision", "ROS"],
    secondary_skills=["SQL", "Pandas"],
    role_families=["Machine Learning Intern", "Computer Vision Intern"],
    experience_level="student",
    years_experience=0.0,
    evidence=[
        ResumeEvidence(claim="Built an EKF-based mobile robot in ROS 2", skills=["ROS", "Python"], kind="project"),
        ResumeEvidence(claim="Computer vision pipeline with OpenCV", skills=["Computer Vision", "PyTorch"], kind="project"),
    ],
)

RESPONSES = {
    "CandidateProfile": PROFILE,
    "GapAnalysis": GapAnalysis(
        matched_requirements=["Python", "PyTorch"],
        missing_requirements=["Docker", "AWS"],
        fit_verdict="moderate", fit_reason="Core ML skills present, infra gaps.",
    ),
    "_FactExtraction": _FactExtraction(
        entity_confirmed=True, facts=[
            {"fact": "Builds logistics software", "source_url": "https://example.com/a",
             "source_title": "About", "confidence": "high"}
        ], summary="A logistics software company."),
    "cover_note": (
        "I am applying for the Machine Learning Intern role at Northwind. The posting describes "
        "building NLP pipelines in Python with PyTorch. I built an EKF-based mobile robot in ROS 2 "
        "using Python, and a computer vision pipeline with OpenCV. " + "detail " * 110
    ),
}


def _run(settings=None, **client_kwargs):
    settings = settings or Settings(discovery_enabled=False)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics, llm_responses=RESPONSES, **client_kwargs)
    result = run_pipeline(
        resume_text=RESUME, company="Northwind", title="Machine Learning Intern",
        jd_text=JD, job_url="https://jobs.example.com/1",
        settings=settings, clients=clients, metrics=metrics,
    )
    return result, clients, metrics


def test_normal_execution_produces_all_artifacts():
    result, _, metrics = _run()
    assert result.job_id
    assert result.gap.status == "ok"
    assert result.research.status == "ok"
    assert result.cover_note.status == "ok"
    assert result.cover_note.job_id == result.job_id
    assert metrics.total_seconds() > 0


def test_llm_call_count_is_four_on_a_cold_run():
    """profile + gap + fact-extraction + cover-note + critique = 5.
    Discovery adds none. With a cached profile it drops to 4."""
    result, clients, metrics = _run()
    assert metrics.llm_calls == 5, clients.llm.calls
    assert clients.llm.calls.count("cover_note") == 1
    assert clients.llm.calls.count("CoverNoteCritique") == 1


def test_branches_execute_concurrently():
    """Gap analysis and company research each take 0.5s of mocked latency.
    Serial would be >=1.0s for those two alone."""
    settings = Settings(discovery_enabled=False)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics, llm_responses=RESPONSES, llm_latency=0.5)
    start = time.perf_counter()
    run_pipeline(
        resume_text=RESUME, company="Northwind", title="ML Intern", jd_text=JD,
        settings=settings, clients=clients, metrics=metrics,
    )
    elapsed = time.perf_counter() - start
    # profile(0.5) + max(gap, research+extract)(0.5) + cover(0.5) + critique(0.5) = ~2.0s
    # fully serial would be 5 x 0.5 = 2.5s
    assert elapsed < 2.35, f"branches did not overlap: {elapsed:.2f}s"


def test_failed_company_research_still_produces_analysis():
    result, _, _ = _run(search_fail=True)
    assert result.research.status == "no_results"
    assert result.research.usable_facts() == []
    assert result.gap.status == "ok"          # unaffected branch
    assert result.cover_note.status in ("ok", "rejected")


def test_failed_gap_branch_degrades_to_keyword_comparison():
    result, _, _ = _run(llm_fail_on=["GapAnalysis"])
    assert result.gap.status == "unavailable"
    assert result.gap.missing_requirements       # deterministic fallback still ran
    assert result.cover_note.status in ("ok", "rejected")


def test_budget_exhaustion_does_not_crash():
    settings = Settings(discovery_enabled=False, max_llm_calls_per_run=2)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics, llm_responses=RESPONSES)
    result = run_pipeline(
        resume_text=RESUME, company="Northwind", title="ML Intern", jd_text=JD,
        settings=settings, clients=clients, metrics=metrics,
    )
    assert metrics.llm_calls <= 2
    assert result.notes, "user must be told what was skipped"


def test_cache_miss_then_hit_on_rerun():
    settings = Settings(discovery_enabled=False)
    metrics1 = new_run(settings)
    clients1 = make_clients(settings, metrics1, llm_responses=RESPONSES)
    run_pipeline(resume_text=RESUME, company="Northwind", title="ML Intern", jd_text=JD,
                 settings=settings, clients=clients1, metrics=metrics1)
    first_calls = metrics1.llm_calls

    metrics2 = new_run(settings)
    clients2 = make_clients(settings, metrics2, llm_responses=RESPONSES)
    run_pipeline(resume_text=RESUME, company="Northwind", title="ML Intern", jd_text=JD,
                 settings=settings, clients=clients2, metrics=metrics2)
    assert metrics2.cache_hits > 0
    assert metrics2.llm_calls < first_calls, "re-running the same job should cost fewer calls"


def test_empty_job_description_skips_cover_note_without_crashing():
    settings = Settings(discovery_enabled=False)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics, llm_responses=RESPONSES)
    result = run_pipeline(resume_text=RESUME, company="Northwind", title="ML Intern",
                          jd_text="", settings=settings, clients=clients, metrics=metrics)
    assert result.gap.status == "skipped"
    assert result.cover_note.status == "skipped"
    assert result.cover_note.reason


def test_current_job_fit_uses_the_canonical_eligibility_engine():
    """PipelineResult.fit is the single, backend-computed scoring/eligibility
    result for the analyzed job -- match_score() must delegate to it rather
    than recomputing anything independently."""
    result, _, _ = _run()
    assert result.fit is not None
    assert result.fit.eligibility_status == "eligible"  # intern title + student profile
    assert not result.fit.blockers
    assert result.match_score() == result.fit.score.total


def test_current_job_fit_blocks_a_senior_role_for_a_junior_candidate():
    settings = Settings(discovery_enabled=False)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics, llm_responses=RESPONSES)
    result = run_pipeline(
        resume_text=RESUME, company="Northwind", title="Senior Backend Engineer",
        jd_text="Senior Backend Engineer. Requires 8+ years of experience.",
        settings=settings, clients=clients, metrics=metrics,
    )
    assert result.fit.eligibility_status == "ineligible"
    assert result.fit.blockers
    assert result.fit.priority in ("Low Priority", "Stretch Opportunity")


def test_cover_note_unavailable_reason_is_specific_not_generic():
    """The Cover Note section must show the actual failure reason (as
    LLMUnavailable reports it), never a hardcoded dead-end message that
    hides why generation failed."""
    settings = Settings(discovery_enabled=False)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics, llm_responses=RESPONSES, llm_fail_on=["cover_note"])
    result = run_pipeline(
        resume_text=RESUME, company="Northwind", title="ML Intern", jd_text=JD,
        settings=settings, clients=clients, metrics=metrics,
    )
    assert result.cover_note.status == "unavailable"
    assert result.cover_note.reason == "mock failure for cover_note"
    assert result.cover_note.reason != "The cover note could not be generated."


def test_resume_recommendations_reflect_the_gap_analysis():
    """resume_recommendations must be populated from the same gap analysis
    shown elsewhere in the run, with no fabricated skills."""
    result, _, _ = _run()
    rec = result.resume_recommendations
    assert rec.status == "ok"
    assert any(g.skill == "Docker" for g in rec.missing_skills)
    allowed = {s.lower() for s in result.profile.all_skills()} | {"docker", "aws", "machine learning", "nlp"}
    for g in rec.missing_skills:
        assert g.skill.lower() in allowed
    for skill in rec.emphasize:
        assert skill.lower() in {s.lower() for s in result.profile.all_skills()}
