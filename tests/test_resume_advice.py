"""Resume-change recommendations: deterministic, no fabrication, honest
score estimates that trace to the same coverage() formula used everywhere
else in the app."""
from agent.config import ScoreWeights
from agent.resume_advice import build_resume_recommendations
from agent.schemas import CandidateProfile, GapAnalysis, ResumeEvidence
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


# ---------------------------------------------------------------------------
# detailed_actions: concrete, directly usable wording -- never fabricated.
# ---------------------------------------------------------------------------

def test_emphasize_action_uses_verbatim_evidence_as_suggested_wording():
    jd = "Backend role requiring Python and SQL."
    profile = CandidateProfile(
        primary_skills=["Python"], secondary_skills=["SQL"],
        evidence=[ResumeEvidence(claim="Wrote SQL queries to analyze sales data", skills=["SQL"], kind="project")],
    )
    gap = GapAnalysis(matched_requirements=["Python", "SQL"], status="ok")
    rec = build_resume_recommendations(jd, profile, gap, current_score=70, weights=WEIGHTS)

    action = next(a for a in rec.detailed_actions if a.what_to_change == "Make SQL more prominent")
    assert action.suggested_wording == "Wrote SQL queries to analyze sales data"
    assert action.estimated_score_gain is None


def test_emphasize_action_without_evidence_has_no_suggested_wording():
    """No matching evidence item exists for SQL -- suggested_wording must
    stay empty rather than inventing a plausible-sounding bullet."""
    jd = "Backend role requiring Python and SQL."
    profile = CandidateProfile(primary_skills=["Python"], secondary_skills=["SQL"], evidence=[])
    gap = GapAnalysis(matched_requirements=["Python", "SQL"], status="ok")
    rec = build_resume_recommendations(jd, profile, gap, current_score=70, weights=WEIGHTS)

    action = next(a for a in rec.detailed_actions if a.what_to_change == "Make SQL more prominent")
    assert action.suggested_wording == ""


def test_missing_skill_action_never_has_suggested_wording():
    """A skill the candidate does not have must never get pasteable
    wording -- any such wording would itself be a fabricated claim."""
    jd = "Machine Learning Intern requiring Python and Docker experience."
    profile = CandidateProfile(primary_skills=["Python"])
    gap = GapAnalysis(matched_requirements=["Python"], missing_requirements=["Docker"], status="ok")
    rec = build_resume_recommendations(jd, profile, gap, current_score=50, weights=WEIGHTS)

    action = next(a for a in rec.detailed_actions if "Docker" in a.what_to_change)
    assert action.suggested_wording == ""
    assert action.estimated_score_gain == next(g.estimated_score_gain for g in rec.missing_skills if g.skill == "Docker")


def test_phrasing_action_uses_the_resume_edit_verbatim():
    jd = "Backend role requiring Python."
    profile = CandidateProfile(primary_skills=["Python"])
    gap = GapAnalysis(
        matched_requirements=["Python"],
        resume_edits=["Reword 'helped with' to 'implemented' in the API project bullet"],
        status="ok",
    )
    rec = build_resume_recommendations(jd, profile, gap, current_score=80, weights=WEIGHTS)

    action = next(a for a in rec.detailed_actions if a.what_to_change == "Reword existing content")
    assert action.suggested_wording == "Reword 'helped with' to 'implemented' in the API project bullet"
    assert action.estimated_score_gain is None


def test_detailed_actions_never_invent_a_skill_not_in_profile_or_jd():
    jd = "Requires Python, Docker, Kubernetes and AWS."
    profile = CandidateProfile(
        primary_skills=["Python"], secondary_skills=["AWS"],
        evidence=[ResumeEvidence(claim="Deployed services on AWS EC2", skills=["AWS"], kind="project")],
    )
    gap = GapAnalysis(
        matched_requirements=["Python", "AWS"], missing_requirements=["Docker", "Kubernetes"],
        resume_edits=["Reword 'used AWS' to 'deployed production services on AWS'"], status="ok",
    )
    rec = build_resume_recommendations(jd, profile, gap, current_score=40, weights=WEIGHTS)

    allowed_terms = (
        {s.lower() for s in profile.all_skills()}
        | {s.lower() for s in extract_skills(jd)}
        | {ev.claim.lower() for ev in profile.evidence}
        | {e.lower() for e in gap.resume_edits}
    )
    for action in rec.detailed_actions:
        if action.suggested_wording:
            assert action.suggested_wording.lower() in allowed_terms or any(
                term in action.suggested_wording.lower() for term in allowed_terms
            )
