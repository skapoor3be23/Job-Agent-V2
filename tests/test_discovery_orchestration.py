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


def test_second_discovery_call_for_the_same_profile_skips_the_network_fetch(monkeypatch):
    """Performance regression: opening a second job from the same discovery
    list (or re-running discovery for the same profile) must not re-fetch
    from Adzuna or re-run the batched embedding call -- both are cached as
    a unit, keyed by profile/resume signature, independent of which job is
    excluded on a given call."""
    jobs = [
        make_job(
            f"C{i}", "Machine Learning Intern",
            "Machine Learning Intern. Python, PyTorch, NLP and SQL required.",
            f"https://x.com/{i}",
        )
        for i in range(5)
    ]
    fetch_calls = {"n": 0}

    def fake_fetch_many(queries, settings, metrics):
        fetch_calls["n"] += 1
        return jobs

    monkeypatch.setattr(discovery, "fetch_many", fake_fetch_many)

    settings = Settings(discovery_enabled=True, cache_enabled=True)

    metrics1 = new_run(settings)
    clients1 = make_clients(settings, metrics1)
    result1 = discovery.discover_opportunities(
        PROFILE, "Python PyTorch NLP SQL", clients1, settings, metrics1,
        exclude_job_ids=[jobs[0].job_id],
    )

    metrics2 = new_run(settings)
    clients2 = make_clients(settings, metrics2)
    result2 = discovery.discover_opportunities(
        PROFILE, "Python PyTorch NLP SQL", clients2, settings, metrics2,
        exclude_job_ids=[jobs[1].job_id],  # a DIFFERENT excluded job
    )

    assert fetch_calls["n"] == 1, "second call re-fetched from Adzuna instead of reusing the cached pool"
    assert clients2.embeddings.batch_calls == 0, "second call re-ran the embedding batch instead of reusing it"

    # Exclusion is still applied correctly per-call despite the shared cache.
    assert jobs[0].job_id not in {o.job.job_id for o in result1.opportunities}
    assert jobs[1].job_id not in {o.job.job_id for o in result2.opportunities}
    assert jobs[1].job_id in {o.job.job_id for o in result1.opportunities}
    assert jobs[0].job_id in {o.job.job_id for o in result2.opportunities}


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
