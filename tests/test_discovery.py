"""Opportunity Discovery: ranking, priority, roadmap, malformed input."""
import pytest
from agent.config import ScoreWeights, Settings
from agent.ranking import rank_normalize, score_opportunities, assign_priority
from agent.roadmap import build_roadmap
from agent.schemas import CandidateProfile, JobPosting, ResumeEvidence
from agent.identity import make_job_id

PROFILE = CandidateProfile(
    primary_skills=["Python", "PyTorch", "NLP", "SQL", "Pandas", "Machine Learning"],
    role_families=["Machine Learning Intern", "Data Science Intern"],
    experience_level="student", years_experience=0.0,
    evidence=[
        ResumeEvidence(claim="Built an NLP classifier", skills=["NLP", "PyTorch"]),
        ResumeEvidence(claim="SQL analysis of sales data", skills=["SQL", "Pandas"]),
    ],
)


def job(company, title, jd, url="https://x.com/1"):
    return JobPosting(job_id=make_job_id(company, title, url, "", jd),
                      company=company, title=title, jd_text=jd, url=url)


STRONG = job("Alpha", "Machine Learning Intern",
             "Machine Learning Intern. Python, PyTorch and NLP required. SQL useful.")
WEAK = job("Beta", "Frontend Developer",
           "Senior Frontend Developer. React, TypeScript, Node.js required. 6+ years experience.", "https://x.com/2")
MID = job("Gamma", "Data Science Intern",
          "Data Science Intern. Python and SQL needed, plus Docker and AWS.", "https://x.com/3")


def test_rank_normalize_spreads_tightly_clustered_values():
    raw = [0.81, 0.82, 0.83, 0.84]
    out = rank_normalize(raw)
    assert min(out) == 0.0 and max(out) == 1.0
    assert out == sorted(out)


def test_rank_normalize_handles_ties_and_missing():
    assert rank_normalize([None, None]) == [0.0, 0.0]
    assert rank_normalize([0.5]) == [1.0]
    tied = rank_normalize([0.5, 0.5, 0.9])
    assert tied[0] == tied[1]


def test_strong_match_outranks_weak_match():
    opps = score_opportunities([WEAK, STRONG, MID], PROFILE, "Python PyTorch NLP SQL",
                               None, [None, None, None], ScoreWeights())
    assert opps[0].job.company == "Alpha"
    assert opps[0].score.total > opps[-1].score.total


def test_senior_role_is_blocked_regardless_of_score():
    opps = score_opportunities([WEAK], PROFILE, "Python", None, [None], ScoreWeights())
    assert opps[0].blockers
    assert opps[0].priority in ("Low Priority", "Stretch Opportunity")


def test_explanations_are_derived_from_real_counts():
    opps = score_opportunities([STRONG], PROFILE, "Python PyTorch NLP", None, [None], ScoreWeights())
    o = opps[0]
    assert str(len(o.matched_skills)) in o.why_match
    assert set(o.matched_skills).issubset(set(PROFILE.all_skills()))
    assert not set(o.matched_skills) & set(o.missing_skills)


def test_score_breakdown_is_explainable():
    o = score_opportunities([MID], PROFILE, "Python SQL", None, [None], ScoreWeights())[0]
    assert 0 <= o.score.total <= 100
    assert "+" in o.score.formula
    assert o.score.skill_coverage >= 0


def test_priority_needs_evidence_not_just_coverage():
    p, _ = assign_priority(coverage_score=0.9, eligibility=1.0, blockers=[], evidence_count=0, relevance=1.0)
    assert p == "Worth Applying"
    p, _ = assign_priority(coverage_score=0.9, eligibility=1.0, blockers=[], evidence_count=3, relevance=1.0)
    assert p == "Apply Now"


def test_missing_fields_and_missing_url_do_not_crash():
    broken = JobPosting(job_id="x", company="", title="", jd_text="", url="")
    opps = score_opportunities([broken], PROFILE, "Python", None, [None], ScoreWeights())
    assert len(opps) == 1
    assert opps[0].job.url == ""


def test_roadmap_counts_come_from_real_jobs():
    jobs = [
        job("A", "ML Intern", "Python, Docker and AWS required.", "https://x.com/a"),
        job("B", "ML Intern", "Python, Docker required.", "https://x.com/b"),
        job("C", "DS Intern", "Python, Docker, AWS and FastAPI.", "https://x.com/c"),
        job("D", "AI Intern", "Python and Docker.", "https://x.com/d"),
    ]
    rm = build_roadmap(jobs, PROFILE)
    assert rm.status == "ok"
    assert rm.total_jobs_analyzed == 4
    docker = next(i for i in rm.items if i.skill == "Docker")
    assert docker.appears_in_jobs == 4          # actually present in all four
    assert docker.unlocks_jobs <= 4
    headline = rm.headline()
    if headline:
        assert str(rm.combo_unlocks) in headline


def test_roadmap_refuses_to_report_on_thin_data():
    rm = build_roadmap([STRONG], PROFILE)
    assert rm.status == "insufficient_data"
    assert rm.headline() == ""


def test_duplicate_jobs_collapse_by_job_id():
    a = job("Alpha", "ML Intern", "Python", "https://x.com/1")
    b = job("Alpha", "ML Intern", "Python", "https://x.com/1")
    assert a.job_id == b.job_id
    seen = {j.job_id: j for j in [a, b]}
    assert len(seen) == 1
