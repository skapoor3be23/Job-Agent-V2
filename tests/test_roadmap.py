"""Skill-gap roadmap: the combo headline must never ask for more skills
than are mathematically necessary.

Root cause of the reported bug: the combo search compared candidate combos
only against an initial value of 0, never against the best SINGLE skill's
own unlock count. A combo that tied (or lost to) its best member still
"won" and was reported as the headline -- e.g. "Adding IoT + TensorFlow
could improve eligibility for 1 of 36 jobs" while the table showed IoT
unlocking 1 job alone and TensorFlow unlocking 0. TensorFlow contributed
nothing; the headline should have named IoT alone.
"""
from agent.identity import make_job_id
from agent.roadmap import build_roadmap
from agent.schemas import CandidateProfile, JobPosting

PROFILE = CandidateProfile(primary_skills=["Python"], experience_level="student", years_experience=0.0)


def job(company, title, jd, url):
    return JobPosting(job_id=make_job_id(company, title, url, "", jd), company=company, title=title, jd_text=jd, url=url)


def test_headline_never_pairs_a_useful_skill_with_a_redundant_one():
    """IoT alone unlocks a job; Python is otherwise fully covered. A second
    job needs FIVE unrelated missing skills at once, so no single one of
    them, nor any pair, unlocks it -- each is individually useless here.
    The combo search must not pair IoT with one of these useless skills
    just because the pairing's total (still just IoT's own contribution)
    beats the initial value of 0."""
    unlockable = job("Alpha", "Backend Engineer", "Backend Engineer requiring Python and IoT experience.", "https://x.com/a")
    unreachable = job(
        "Beta", "Data Engineer",
        "Requires Python, TensorFlow, Kubernetes, Rust, golang and Java experience.",
        "https://x.com/b",
    )
    filler_1 = job("Gamma", "Backend Engineer", "Backend Engineer requiring Python experience.", "https://x.com/c")
    filler_2 = job("Delta", "Backend Engineer", "Backend Engineer requiring Python experience.", "https://x.com/d")

    rm = build_roadmap([unlockable, unreachable, filler_1, filler_2], PROFILE)

    assert rm.status == "ok"
    iot_item = next(i for i in rm.items if i.skill == "IoT")
    assert iot_item.unlocks_jobs == 1

    # Every OTHER listed skill must show 0 individual unlocks (unreachable
    # needs at least 5 of its 6 requirements to cross the threshold).
    other_items = [i for i in rm.items if i.skill != "IoT"]
    assert other_items
    for item in other_items:
        assert item.unlocks_jobs == 0

    # The headline must recommend IoT alone -- never IoT plus one of the
    # zero-contribution skills.
    assert rm.combo_skills == ["IoT"]
    assert rm.combo_unlocks == 1
    headline = rm.headline()
    assert "IoT" in headline
    assert "+" not in headline
    assert "1" in headline


def test_headline_reports_a_combo_only_when_it_beats_every_single_skill():
    """Spark and Hadoop unlock a job only TOGETHER -- neither alone crosses
    the threshold. This genuine synergy must still be reported as a combo,
    since it strictly beats any single skill's contribution (0)."""
    synergy = job(
        "Zeta", "Data Engineer", "Data Engineer requiring Python, Spark and Hadoop experience.", "https://x.com/z",
    )
    filler_1 = job("A1", "Backend Engineer", "Backend Engineer requiring Python experience.", "https://x.com/1")
    filler_2 = job("A2", "Backend Engineer", "Backend Engineer requiring Python experience.", "https://x.com/2")
    filler_3 = job("A3", "Backend Engineer", "Backend Engineer requiring Python experience.", "https://x.com/3")

    rm = build_roadmap([synergy, filler_1, filler_2, filler_3], PROFILE)

    assert rm.status == "ok"
    for item in rm.items:
        assert item.unlocks_jobs == 0  # neither Spark nor Hadoop unlocks anything alone

    assert set(rm.combo_skills) == {"Spark", "Hadoop"}
    assert rm.combo_unlocks == 1
    headline = rm.headline()
    assert "Spark" in headline and "Hadoop" in headline
    assert "+" in headline


def test_thin_data_still_returns_no_headline():
    rm = build_roadmap(
        [job("A", "ML Intern", "Python required.", "https://x.com/only")],
        PROFILE,
    )
    assert rm.status == "insufficient_data"
    assert rm.headline() == ""
