"""Cover-note factuality: the requirement that a note for Company B never
carries Company A's information."""
import pytest
from agent.config import Settings
from agent.cover_note import build_generation_prompt, validate_cover_note, NO_FACTS_RULE
from agent.graph import run_pipeline
from agent.schemas import CandidateProfile, CompanyResearch, VerifiedFact, _FactExtraction, GapAnalysis
from agent.telemetry import new_run
from mocks import make_clients

PROFILE = CandidateProfile(primary_skills=["Python", "PyTorch"], role_families=["ML Intern"])
FILLER = "word " * 130
JD = "Machine Learning Intern building NLP pipelines in Python and PyTorch."

NO_RESEARCH = CompanyResearch(company="Zylo", status="no_results")
GOOD_RESEARCH = CompanyResearch(
    company="Zylo", entity_confirmed=True, status="ok",
    verified_facts=[VerifiedFact(fact="Builds logistics software", source_url="https://z.com", source_title="About")],
)


def test_correct_company_note_passes():
    note = f"Applying to Zylo for the Machine Learning Intern role. {FILLER}"
    r = validate_cover_note(note, "Zylo", "Machine Learning Intern", PROFILE, GOOD_RESEARCH, Settings())
    assert r.passed


def test_note_mentioning_another_session_company_is_rejected():
    note = f"Applying to Zylo. My interest began at Northwind. {FILLER}"
    r = validate_cover_note(note, "Zylo", "ML Intern", PROFILE, GOOD_RESEARCH, Settings(),
                            other_companies=["Northwind Systems", "Zylo"])
    assert not r.passed
    assert any("Northwind" in f for f in r.failures)


def test_shared_legal_suffix_does_not_false_positive():
    note = f"Applying to Zylo Technologies for the ML Intern role. {FILLER}"
    r = validate_cover_note(note, "Zylo Technologies", "ML Intern", PROFILE, GOOD_RESEARCH, Settings(),
                            other_companies=["Northwind Technologies"])
    assert r.passed, r.failures


def test_unsourced_company_claims_rejected_when_no_research():
    note = f"Zylo recently raised a Series B and is the market leader in logistics. {FILLER}"
    r = validate_cover_note(note, "Zylo", "ML Intern", PROFILE, NO_RESEARCH, Settings())
    assert not r.passed


def test_missing_research_switches_prompt_to_no_claims_rule():
    prompt = build_generation_prompt("Zylo", "ML Intern", JD, PROFILE, NO_RESEARCH, "resume", Settings())
    assert NO_FACTS_RULE in prompt
    assert "(none available" in prompt


def test_entity_unconfirmed_research_releases_no_facts():
    """A search that returned a DIFFERENT company must contribute nothing."""
    research = CompanyResearch(
        company="Zylo", entity_confirmed=False, status="entity_unconfirmed",
        verified_facts=[VerifiedFact(fact="Raised $50M", source_url="https://other.com")],
    )
    assert research.usable_facts() == []
    prompt = build_generation_prompt("Zylo", "ML Intern", JD, PROFILE, research, "resume", Settings())
    assert "Raised $50M" not in prompt
    assert NO_FACTS_RULE in prompt


def test_note_never_names_target_company_is_warned():
    r = validate_cover_note(f"I would like this role. {FILLER}", "Zylo", "ML Intern",
                            PROFILE, GOOD_RESEARCH, Settings())
    assert any("never names" in w for w in r.warnings)


def test_two_companies_in_one_session_do_not_cross_contaminate():
    """Full pipeline: analyse Company A, then Company B, and confirm B's note
    is bound to B's job_id and free of A's name."""
    settings = Settings(discovery_enabled=False)
    responses_a = {
        "CandidateProfile": PROFILE,
        "GapAnalysis": GapAnalysis(matched_requirements=["Python"], missing_requirements=["Docker"]),
        "_FactExtraction": _FactExtraction(entity_confirmed=True, facts=[
            {"fact": "Alpha builds robots", "source_url": "https://a.com", "source_title": "A"}]),
        "cover_note": f"Applying to Alpha for the ML Intern role. {FILLER}",
    }
    m1 = new_run(settings)
    r1 = run_pipeline(resume_text="Python PyTorch projects", company="Alpha", title="ML Intern",
                      jd_text=JD, settings=settings, metrics=m1,
                      clients=make_clients(settings, m1, llm_responses=responses_a))

    responses_b = dict(responses_a)
    responses_b["cover_note"] = f"Applying to Beta for the ML Intern role. {FILLER}"
    m2 = new_run(settings)
    r2 = run_pipeline(resume_text="Python PyTorch projects", company="Beta", title="ML Intern",
                      jd_text=JD, settings=settings, metrics=m2,
                      clients=make_clients(settings, m2, llm_responses=responses_b),
                      session_companies=["Alpha", "Beta"])

    assert r1.job_id != r2.job_id
    assert r2.cover_note.job_id == r2.job_id
    assert "alpha" not in r2.cover_note.text.lower()
    assert r2.cover_note.status == "ok"


def test_leaked_previous_company_is_caught_by_the_pipeline():
    """If the model leaks Alpha into Beta's note, the run must reject it."""
    settings = Settings(discovery_enabled=False)
    responses = {
        "CandidateProfile": PROFILE,
        "GapAnalysis": GapAnalysis(matched_requirements=["Python"]),
        "_FactExtraction": _FactExtraction(entity_confirmed=False),
        "cover_note": f"Applying to Beta, much like my interest in Alpha. {FILLER}",
    }
    metrics = new_run(settings)
    result = run_pipeline(resume_text="Python", company="Beta", title="ML Intern", jd_text=JD,
                          settings=settings, metrics=metrics,
                          clients=make_clients(settings, metrics, llm_responses=responses),
                          session_companies=["Alpha", "Beta"])
    assert result.cover_note.status == "rejected"
    assert result.cover_note.validation.failures
