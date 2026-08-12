"""Resume-change recommendations: deterministic, no fabrication, honest
score estimates that trace to the same coverage() formula used everywhere
else in the app."""
from agent.config import ScoreWeights
from agent.resume_advice import build_resume_recommendations
from agent.schemas import CandidateProfile, GapAnalysis
from agent.skills import coverage, extract_skills

WEIGHTS = ScoreWeights()


def test_empty_jd_is_skipped():
    rec = build_resume_recommendations("", CandidateProfile(), GapAnalysis(), current_score=0, weights=WEIGHTS)
    assert rec.status == "skipped"


def test_no_technical_requirements_is_insufficient_data():
    jd = "We value strong communication and teamwork in a collaborative environment."
    profile = CandidateProfile(primary_skills=["Python"])
    gap = GapAnalysis(resume_edits=["Reword 'helped with' to 'implemented'"], status="ok")
    rec = build_resume_recommendations(jd, profile, gap, current_score=60, weights=WEIGHTS)
    assert rec.status == "insufficient_data"
    assert rec.phrasing_suggestions == gap.resume_edits
    assert rec.missing_skills == []


def test_emphasize_lists_only_secondary_skills_matching_jd():
    jd = "Backend role requiring Python and SQL."
    profile = CandidateProfile(primary_skills=["Python"], secondary_skills=["SQL"])
    gap = GapAnalysis(matched_requirements=["Python", "SQL"], status="ok")
    rec = build_resume_recommendations(jd, profile, gap, current_score=70, weights=WEIGHTS)
    assert rec.emphasize == ["SQL"]
    assert "Python" not in rec.emphasize  # already primary -- nothing to elevate


def test_missing_skill_score_estimate_matches_formula():
    jd = "Machine Learning Intern requiring Python and Docker experience."
    profile = CandidateProfile(primary_skills=["Python"])
    gap = GapAnalysis(matched_requirements=["Python"], missing_requirements=["Docker"], status="ok")
    rec = build_resume_recommendations(jd, profile, gap, current_score=50, weights=WEIGHTS)

    required = extract_skills(jd)
    cov, _, missing = coverage(required, profile.all_skills())
    assert "Docker" in missing
    boosted_cov, _, _ = coverage(required, profile.all_skills() + ["Docker"])
    expected_gain = round((boosted_cov - cov) * WEIGHTS.skill_coverage * 100)

    docker_gap = next(g for g in rec.missing_skills if g.skill == "Docker")
    assert docker_gap.estimated_score_gain == expected_gain
    assert rec.status == "ok"


def test_never_fabricates_a_skill():
    jd = "Requires Python, Docker, Kubernetes and AWS."
    profile = CandidateProfile(primary_skills=["Python"], secondary_skills=["AWS"])
    gap = GapAnalysis(matched_requirements=["Python", "AWS"], missing_requirements=["Docker", "Kubernetes"], status="ok")
    rec = build_resume_recommendations(jd, profile, gap, current_score=40, weights=WEIGHTS)

    allowed = {s.lower() for s in profile.all_skills()} | {s.lower() for s in extract_skills(jd)}
    for skill in rec.emphasize:
        assert skill.lower() in allowed
    for gap_item in rec.missing_skills:
        assert gap_item.skill.lower() in allowed


def test_unscored_gaps_surfaces_llm_only_findings():
    jd = "Requires Python and strong ownership of ambiguous, cross-team roadmaps."
    profile = CandidateProfile(primary_skills=["Python"])
    gap = GapAnalysis(
        matched_requirements=["Python"],
        missing_requirements=["Experience owning ambiguous cross-team roadmaps"],
        status="ok",
    )
    rec = build_resume_recommendations(jd, profile, gap, current_score=55, weights=WEIGHTS)
    assert rec.status == "ok"
    assert rec.missing_skills == []
    assert rec.unscored_gaps == ["Experience owning ambiguous cross-team roadmaps"]


def test_note_frames_estimates_as_model_based_not_guaranteed():
    jd = "Requires Python and Docker."
    profile = CandidateProfile(primary_skills=["Python"])
    gap = GapAnalysis(matched_requirements=["Python"], missing_requirements=["Docker"], status="ok")
    rec = build_resume_recommendations(jd, profile, gap, current_score=33, weights=WEIGHTS)
    assert "33" in rec.note
    assert "not a guarantee" in rec.note.lower()
