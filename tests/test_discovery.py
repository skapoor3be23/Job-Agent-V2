"""Opportunity Discovery: ranking, priority, roadmap, malformed input."""
import pytest
from agent.config import ScoreWeights, Settings
from agent.ranking import ELIGIBILITY_SCORE_MAP, rank_normalize, score_opportunities, assign_priority
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


# ---------------------------------------------------------------------------
# Eligibility -> Opportunity (single source of truth, no dead fields)
# ---------------------------------------------------------------------------

def test_eligibility_score_map_strictly_orders_confidence():
    assert ELIGIBILITY_SCORE_MAP["eligible"] > ELIGIBILITY_SCORE_MAP["likely_eligible"]
    assert ELIGIBILITY_SCORE_MAP["likely_eligible"] > ELIGIBILITY_SCORE_MAP["uncertain"]
    assert ELIGIBILITY_SCORE_MAP["uncertain"] > ELIGIBILITY_SCORE_MAP["ineligible"]


def test_opportunity_carries_the_full_eligibility_result():
    """The EligibilityResult computed for a job must reach the Opportunity,
    not be discarded down to a bare float + blocker string."""
    opps = score_opportunities([WEAK], PROFILE, "Python", None, [None], ScoreWeights())
    o = opps[0]
    assert o.eligibility_status == "ineligible"
    assert o.eligibility_explanation
    assert o.blockers


def test_likely_eligible_scores_higher_than_uncertain_end_to_end():
    """likely_eligible (no confirming signal, but nothing wrong either) and
    uncertain (an explicit but small experience-shortfall warning) must not
    collapse to the same score."""
    likely = job("Zeta", "Data Scientist", "Python and SQL needed.", "https://x.com/likely")
    uncertain = job(
        "Eta", "Data Scientist", "Python and SQL needed. Requires 2 years of experience.",
        "https://x.com/uncertain",
    )
    opps = score_opportunities(
        [likely, uncertain], PROFILE, "Python SQL", None, [None, None], ScoreWeights()
    )
    by_id = {o.job.job_id: o for o in opps}
    likely_opp = by_id[likely.job_id]
    uncertain_opp = by_id[uncertain.job_id]

    assert likely_opp.eligibility_status == "likely_eligible"
    assert uncertain_opp.eligibility_status == "uncertain"
    assert likely_opp.score.experience_match > uncertain_opp.score.experience_match
    assert not likely_opp.blockers and not uncertain_opp.blockers


def test_missing_eligibility_result_is_resolved_not_assumed_safe():
    """score_opportunities must never fall back to an optimistic constant
    when eligibility_results doesn't cover a job -- it must actually run
    the canonical eligibility engine."""
    opps = score_opportunities([WEAK], PROFILE, "Python", None, [None], ScoreWeights())
    assert opps[0].blockers
    assert opps[0].priority in ("Low Priority", "Stretch Opportunity")


# ---------------------------------------------------------------------------
# Deterministic ordering (no network-completion-order / arbitrary bias)
# ---------------------------------------------------------------------------

def test_stretch_opportunity_reason_never_contains_the_positive_eligibility_message():
    """A junior-titled job with a genuine blocker (large years shortfall)
    must report the real blocker in Opportunity.blockers/priority_reason,
    never the positive 'explicitly targeted' explanation eligibility.py
    produces when nothing is actually wrong."""
    strong_but_blocked = job(
        "Gamma", "Junior Backend Developer",
        "Junior Backend Developer. Python, PyTorch, NLP, SQL required. 8+ years of experience required.",
        "https://x.com/blocked",
    )
    opps = score_opportunities([strong_but_blocked], PROFILE, "Python PyTorch NLP SQL", None, [None], ScoreWeights())
    o = opps[0]
    assert o.eligibility_status == "ineligible"
    assert o.blockers
    assert "years of experience" in o.blockers[0]
    assert "explicitly targeted" not in o.blockers[0]
    assert "explicitly targeted" not in o.priority_reason
    assert o.priority == "Stretch Opportunity"


# ---------------------------------------------------------------------------
# Evidence-to-requirement mapping: one resume bullet, many requirements
# ---------------------------------------------------------------------------

def test_one_evidence_item_supports_multiple_requirements():
    """A single strong resume bullet that genuinely demonstrates several JD
    requirements must be credited for all of them -- not only the one skill
    an earlier LLM call happened to tag it with. Nothing here is invented:
    the extra credit comes only from re-scanning the bullet's own claim
    text with the same deterministic skill vocabulary used everywhere else.
    """
    profile = CandidateProfile(
        primary_skills=["Python", "PyTorch", "NLP", "Computer Vision"],
        role_families=["Machine Learning Intern"],
        experience_level="student", years_experience=0.0,
        evidence=[
            ResumeEvidence(
                claim="Built an NLP and computer vision pipeline in PyTorch and Python",
                skills=["NLP"],  # deliberately narrow tagging from profile-build time
            ),
        ],
    )
    strong_jd = job(
        "Delta", "Machine Learning Intern",
        "Machine Learning Intern. Python, PyTorch, NLP and Computer Vision required.",
        "https://x.com/delta",
    )

    opp = score_opportunities(
        [strong_jd], profile, "Python PyTorch NLP Computer Vision", None, [None], ScoreWeights(),
    )[0]

    assert set(opp.matched_skills) >= {"Python", "PyTorch", "NLP", "Computer Vision"}
    # Only one evidence item exists on the whole profile, yet it is credited
    # as supporting the match.
    assert len(opp.evidence) == 1
    assert "computer vision pipeline" in opp.evidence[0].lower()

    from agent.ranking import _evidence_matched_skills

    hits = _evidence_matched_skills(profile.evidence[0], opp.matched_skills)
    assert len(hits) >= 3  # one bullet backing Python, PyTorch, NLP and Computer Vision at once
    assert {"Python", "PyTorch", "NLP"}.issubset(set(hits))


def test_evidence_ordering_favors_the_item_supporting_the_most_requirements():
    """When several evidence items compete for the display cap, the one
    demonstrating the most matched requirements must surface first."""
    profile = CandidateProfile(
        primary_skills=["Python", "PyTorch", "NLP", "SQL"],
        role_families=["Machine Learning Intern"],
        experience_level="student", years_experience=0.0,
        evidence=[
            ResumeEvidence(claim="Used SQL for a class project", skills=["SQL"]),
            ResumeEvidence(
                claim="Built an NLP pipeline in PyTorch using Python end to end",
                skills=["NLP"],
            ),
        ],
    )
    strong_jd = job(
        "Epsilon", "Machine Learning Intern",
        "Machine Learning Intern. Python, PyTorch, NLP and SQL required.",
        "https://x.com/epsilon",
    )

    opp = score_opportunities(
        [strong_jd], profile, "Python PyTorch NLP SQL", None, [None], ScoreWeights(),
    )[0]

    assert opp.evidence[0].startswith("Built an NLP pipeline")


def test_tiebreak_by_job_id_is_stable_regardless_of_input_order():
    a = job("Same", "ML Intern", "Python PyTorch NLP required.", "https://x.com/aaa")
    b = job("Same", "ML Intern", "Python PyTorch NLP required.", "https://x.com/bbb")
    assert a.job_id != b.job_id

    forward = score_opportunities([a, b], PROFILE, "Python PyTorch NLP", None, [None, None], ScoreWeights())
    backward = score_opportunities([b, a], PROFILE, "Python PyTorch NLP", None, [None, None], ScoreWeights())

    assert [o.job.job_id for o in forward] == [o.job.job_id for o in backward]
    assert [o.job.job_id for o in forward] == sorted([a.job_id, b.job_id])
