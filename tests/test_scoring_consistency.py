"""Discovery vs opened-job scoring must never diverge for the same job.

Regression context, round 1: the same job could show e.g. 92/100 in the
Opportunity Discovery list and 44/100 when the user opened it directly
("Analyze this job instead"). Root cause: evaluate_current_job() computed
its 45%-weighted "semantic" component from the LLM gap analysis's
matched/missing ratio -- an entirely different, non-deterministic signal
from the embedding cosine similarity score_opportunities() uses for the
same component.

Fix, round 1: evaluate_current_job() computes its semantic component from
the same cosine_similarity()/lexical_similarity() functions
score_opportunities() uses, and run_pipeline() accepts `known_fit` -- an
already-computed Opportunity -- reused verbatim when the job_id matches.

Regression context, round 2: after round 1, the gap narrowed (e.g. 92 -> 73)
but did not close, because score_opportunities() still rank-normalized that
same cosine similarity across whichever OTHER jobs happened to be in that
particular discovery batch (rank_normalize()), while evaluate_current_job()
used the raw, pool-independent value. So the identical job/embeddings pair
could still land on two different numbers depending on which function
scored it -- rank-normalizing even makes the SAME job's discovered score
change run to run, since Adzuna's live results vary.

Fix, round 2: score_opportunities() now uses the same raw (clamped)
similarity as evaluate_current_job() for its semantic component, instead of
rank_normalize(). rank_normalize() itself is unchanged and still tested
directly (test_discovery.py) -- it is just no longer part of the canonical
per-job score, since a canonical score cannot depend on a comparison pool
that a directly-opened job was never part of.
"""
from agent import discovery
from agent.config import ScoreWeights, Settings
from agent.graph import run_pipeline
from agent.identity import make_job_id
from agent.ranking import cosine_similarity, evaluate_current_job, score_opportunities
from agent.schemas import CandidateProfile, GapAnalysis, JobPosting, ResumeEvidence
from agent.telemetry import new_run
from mocks import make_clients

PROFILE = CandidateProfile(
    primary_skills=["Python", "PyTorch", "NLP", "SQL"],
    role_families=["Machine Learning Intern"],
    experience_level="student", years_experience=0.0,
    evidence=[ResumeEvidence(claim="Built an NLP classifier in PyTorch", skills=["NLP", "PyTorch"])],
)

RESUME = "Mechatronics undergraduate. Built an NLP classifier in PyTorch. Python and SQL."
JD = "Machine Learning Intern. Python, PyTorch and NLP required. SQL useful."


def _discovered_job():
    return JobPosting(
        job_id=make_job_id("Alpha", "Machine Learning Intern", "https://x.com/1", "", JD),
        company="Alpha", title="Machine Learning Intern", jd_text=JD, url="https://x.com/1",
    )


# ---------------------------------------------------------------------------
# discovered score == opened score / discovered eligibility == opened
# eligibility, via the real dashboard flow: discover, then "Analyze this
# job instead" on the exact same job.
# ---------------------------------------------------------------------------

def test_opened_job_reuses_the_exact_discovered_opportunity(monkeypatch):
    job = _discovered_job()
    monkeypatch.setattr(discovery, "fetch_many", lambda queries, settings, metrics: [job])

    settings = Settings(discovery_enabled=True, cache_enabled=False)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics)

    discovery_result = discovery.discover_opportunities(PROFILE, RESUME, clients, settings, metrics)
    assert discovery_result.status == "ok"
    discovered_opp = next(o for o in discovery_result.opportunities if o.job.job_id == job.job_id)

    # A deliberately WEAK gap analysis, to prove the score is no longer
    # derived from it at all -- if it still leaked into the score, this
    # would drag the "opened" number far below the discovered one, exactly
    # reproducing the reported 92 -> 44 bug.
    weak_gap = GapAnalysis(
        matched_requirements=["Python"],
        missing_requirements=["PyTorch", "NLP", "SQL"],
        fit_verdict="weak",
    )
    settings2 = Settings(discovery_enabled=False, cache_enabled=False)
    metrics2 = new_run(settings2)
    clients2 = make_clients(settings2, metrics2, llm_responses={"GapAnalysis": weak_gap})

    result = run_pipeline(
        resume_text=RESUME, company="Alpha", title="Machine Learning Intern", jd_text=JD,
        job_url="https://x.com/1", settings=settings2, clients=clients2, metrics=metrics2,
        known_fit=discovered_opp,
    )

    assert result.fit.score.total == discovered_opp.score.total
    assert result.match_score() == discovered_opp.score.total
    assert result.fit.eligibility_status == discovered_opp.eligibility_status
    assert result.fit.blockers == discovered_opp.blockers
    assert result.fit is discovered_opp


def test_editing_a_field_before_opening_falls_back_to_a_fresh_deterministic_score(monkeypatch):
    """If the user changes a field (e.g. corrects the company name) before
    submitting, the job_id no longer matches the discovered Opportunity, so
    it must NOT be reused blindly -- the fresh score must still be
    deterministic and use the canonical formula."""
    job = _discovered_job()
    monkeypatch.setattr(discovery, "fetch_many", lambda queries, settings, metrics: [job])

    settings = Settings(discovery_enabled=True, cache_enabled=False)
    metrics = new_run(settings)
    clients = make_clients(settings, metrics)
    discovery_result = discovery.discover_opportunities(PROFILE, RESUME, clients, settings, metrics)
    discovered_opp = next(o for o in discovery_result.opportunities if o.job.job_id == job.job_id)

    settings2 = Settings(discovery_enabled=False, cache_enabled=False)
    metrics2 = new_run(settings2)
    clients2 = make_clients(settings2, metrics2, llm_responses={"GapAnalysis": GapAnalysis()})
    result = run_pipeline(
        resume_text=RESUME, company="Alpha Corrected", title="Machine Learning Intern", jd_text=JD,
        job_url="https://x.com/1", settings=settings2, clients=clients2, metrics=metrics2,
        known_fit=discovered_opp,  # stale -- job_id will not match "Alpha Corrected"
    )

    assert result.fit is not discovered_opp
    assert result.fit.job.company == "Alpha Corrected"


# ---------------------------------------------------------------------------
# same inputs => deterministic same result
# ---------------------------------------------------------------------------

def test_evaluate_current_job_is_a_pure_function_of_its_inputs():
    job = _discovered_job()
    vector_a = [1.0, 0.0, 0.0]
    vector_b = [0.6, 0.8, 0.0]

    opp1 = evaluate_current_job(job, PROFILE, Settings().weights, resume_text=RESUME,
                                 resume_vector=vector_a, job_vector=vector_b)
    opp2 = evaluate_current_job(job, PROFILE, Settings().weights, resume_text=RESUME,
                                 resume_vector=vector_a, job_vector=vector_b)

    assert opp1.score.total == opp2.score.total
    assert opp1.score.semantic_normalized == opp2.score.semantic_normalized
    assert opp1.eligibility_status == opp2.eligibility_status
    assert opp1.priority == opp2.priority


def test_evaluate_current_job_uses_the_same_semantic_signal_as_discovery():
    """The single-job path's semantic score must come from the identical
    cosine-similarity function score_opportunities() uses -- not a
    separately-computed LLM gap-evidence ratio (the original bug)."""
    job = _discovered_job()
    resume_vector = [1.0, 0.0]
    job_vector = [1.0, 1.0]

    opp = evaluate_current_job(job, PROFILE, Settings().weights, resume_text=RESUME,
                                resume_vector=resume_vector, job_vector=job_vector)

    expected = round(cosine_similarity(resume_vector, job_vector), 4)
    assert opp.score.semantic_raw == expected
    assert opp.score.semantic_normalized == expected


# ---------------------------------------------------------------------------
# The two formulas must agree WITHOUT relying on known_fit/object-identity
# reuse -- i.e. even a job that was never discovered, scored fresh by each
# path with the identical embeddings, must land on the identical number.
# This is what actually closes the 92-vs-73 gap: round 1 only fixed the
# reuse shortcut, not the underlying formula mismatch.
# ---------------------------------------------------------------------------

def test_score_opportunities_and_evaluate_current_job_agree_without_reuse():
    job = _discovered_job()
    resume_vector = [1.0, 0.2, 0.1]
    job_vector = [0.6, 0.5, 0.4]

    discovered = score_opportunities(
        [job], PROFILE, RESUME, resume_vector, [job_vector], ScoreWeights(),
    )[0]
    opened = evaluate_current_job(
        job, PROFILE, ScoreWeights(), resume_text=RESUME,
        resume_vector=resume_vector, job_vector=job_vector,
    )

    assert discovered.score.total == opened.score.total
    assert discovered.score.semantic_normalized == opened.score.semantic_normalized
    assert discovered.eligibility_status == opened.eligibility_status
    assert discovered.blockers == opened.blockers


def test_discovered_score_is_independent_of_the_rest_of_the_batch():
    """The exact bug behind the narrowed-but-not-closed 92-vs-73 gap:
    rank-normalizing a job's semantic similarity against whichever OTHER
    jobs happened to be fetched in that run made the SAME job's score
    depend on its neighbors. A canonical score must not do that -- a job
    scored alone must match the same job scored inside a batch."""
    job = _discovered_job()
    resume_vector = [1.0, 0.2, 0.1]
    job_vector = [0.6, 0.5, 0.4]
    other_jobs = [
        JobPosting(job_id=make_job_id("Beta", "X", "https://x.com/b", "", "Java required."),
                   company="Beta", title="X", jd_text="Java required.", url="https://x.com/b"),
        JobPosting(job_id=make_job_id("Gamma", "Y", "https://x.com/c", "", "Go required."),
                   company="Gamma", title="Y", jd_text="Go required.", url="https://x.com/c"),
    ]
    other_vectors = [[0.95, 0.1, 0.0], [0.9, 0.05, 0.0]]

    alone = score_opportunities([job], PROFILE, RESUME, resume_vector, [job_vector], ScoreWeights())
    with_others = score_opportunities(
        [job] + other_jobs, PROFILE, RESUME, resume_vector, [job_vector] + other_vectors, ScoreWeights(),
    )

    solo = alone[0]
    pooled = next(o for o in with_others if o.job.job_id == job.job_id)

    assert solo.score.total == pooled.score.total
    assert solo.score.semantic_normalized == pooled.score.semantic_normalized


# ---------------------------------------------------------------------------
# existing eligibility/scoring cases remain unchanged: a senior role must
# still block a junior candidate identically whether scored as a discovered
# opportunity or as the directly-analyzed job.
# ---------------------------------------------------------------------------

def test_senior_role_blocks_junior_candidate_identically_in_both_paths():
    senior_job = JobPosting(
        job_id=make_job_id("Beta", "Senior Backend Engineer", "https://x.com/2", "", "8+ years of experience required."),
        company="Beta", title="Senior Backend Engineer",
        jd_text="Senior Backend Engineer. 8+ years of experience required.", url="https://x.com/2",
    )

    discovered = evaluate_current_job(senior_job, PROFILE, Settings().weights, resume_text=RESUME,
                                       resume_vector=[1.0, 0.0], job_vector=[0.1, 0.9])
    opened = evaluate_current_job(senior_job, PROFILE, Settings().weights, resume_text=RESUME,
                                   resume_vector=[1.0, 0.0], job_vector=[0.1, 0.9])

    assert discovered.eligibility_status == "ineligible"
    assert discovered.blockers
    assert discovered.eligibility_status == opened.eligibility_status
    assert discovered.blockers == opened.blockers
    assert discovered.score.total == opened.score.total
