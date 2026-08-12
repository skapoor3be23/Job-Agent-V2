"""Orchestration-level tests for discover_opportunities(): the display cap
must be applied AFTER deterministic ranking, never based on which jobs
happened to be fetched/listed first."""
from agent import discovery
from agent.config import Settings
from agent.identity import make_job_id
from agent.schemas import CandidateProfile, JobPosting
from agent.telemetry import new_run
from mocks import make_clients

PROFILE = CandidateProfile(
    primary_skills=["Python", "PyTorch", "NLP", "SQL"],
    role_families=["Machine Learning Intern"],
    experience_level="student",
    years_experience=0.0,
)


def make_job(company, title, jd, url):
    return JobPosting(
        job_id=make_job_id(company, title, url, "", jd),
        company=company, title=title, jd_text=jd, url=url,
    )


def test_cap_keeps_top_ranked_jobs_regardless_of_fetch_order(monkeypatch):
    """More jobs than the cap are returned, with a clear best/worst split.
    The weak/ineligible jobs are listed FIRST -- an arrival-order cap would
    wrongly keep them. The top-ranked jobs must survive instead."""
    strong_jobs = [
        make_job(
            f"Strong{i}", "Machine Learning Intern",
            "Machine Learning Intern. Python, PyTorch, NLP and SQL required.",
            f"https://x.com/strong{i}",
        )
        for i in range(3)
    ]
    weak_jobs = [
        make_job(
            f"Weak{i}", "Senior Backend Engineer",
            "Senior Backend Engineer. Java, Kubernetes required. 8+ years experience.",
            f"https://x.com/weak{i}",
        )
        for i in range(3)
    ]
    all_jobs = weak_jobs + strong_jobs  # weak (worse) jobs arrive first

    monkeypatch.setattr(discovery, "fetch_many", lambda queries, settings, metrics: all_jobs)

    settings = Settings(discovery_enabled=True, discovery_max_jobs=3, cache_enabled=False)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics)

    result = discovery.discover_opportunities(PROFILE, "Python PyTorch NLP SQL", clients, settings, metrics)

    assert result.status == "ok"
    assert len(result.opportunities) == 3
    surviving_companies = {o.job.company for o in result.opportunities}
    assert surviving_companies == {"Strong0", "Strong1", "Strong2"}


def test_roadmap_sees_the_full_analyzed_set_not_just_the_display_cap(monkeypatch):
    jobs = [
        make_job(f"C{i}", "ML Intern", "Python, Docker and AWS required.", f"https://x.com/{i}")
        for i in range(6)
    ]
    monkeypatch.setattr(discovery, "fetch_many", lambda queries, settings, metrics: jobs)

    settings = Settings(discovery_enabled=True, discovery_max_jobs=2, cache_enabled=False)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics)

    result = discovery.discover_opportunities(PROFILE, "Python", clients, settings, metrics)

    assert len(result.opportunities) == 2            # display cap applied
    assert result.roadmap.total_jobs_analyzed == 6    # roadmap saw every retrieved job
