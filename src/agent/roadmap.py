"""Skill-gap roadmap: which missing skills unlock the most opportunities.

Pure counting over the jobs actually retrieved. Zero LLM calls, so the
headline number cannot be invented -- it is recomputed coverage.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import List, Sequence

from .schemas import CandidateProfile, JobPosting, SkillGapItem, SkillRoadmap
from .skills import coverage, extract_skills, hard_requirements

UNLOCK_THRESHOLD = 0.70   # coverage at which a job becomes a realistic target
MIN_JOBS_FOR_ROADMAP = 4
MAX_ITEMS = 6


def build_roadmap(
    jobs: Sequence[JobPosting],
    profile: CandidateProfile,
    combo_size: int = 2,
) -> SkillRoadmap:
    if len(jobs) < MIN_JOBS_FOR_ROADMAP:
        return SkillRoadmap(
            status="insufficient_data",
            total_jobs_analyzed=len(jobs),
        )

    candidate_skills = [s.lower() for s in profile.all_skills()]
    job_requirements = [hard_requirements(extract_skills(j.jd_text)) for j in jobs]

    frequency: Counter = Counter()
    for reqs in job_requirements:
        for skill in set(reqs):
            if skill.lower() not in candidate_skills:
                frequency[skill] += 1

    if not frequency:
        return SkillRoadmap(status="insufficient_data", total_jobs_analyzed=len(jobs))

    def unlocked_by(extra: Sequence[str]) -> int:
        """Jobs that move from below to at/above the coverage threshold."""
        boosted = candidate_skills + [s.lower() for s in extra]
        count = 0
        for reqs in job_requirements:
            before, _, _ = coverage(reqs, candidate_skills)
            after, _, _ = coverage(reqs, boosted)
            if before < UNLOCK_THRESHOLD <= after:
                count += 1
        return count

    items = [
        SkillGapItem(
            skill=skill,
            appears_in_jobs=count,
            total_jobs=len(jobs),
            unlocks_jobs=unlocked_by([skill]),
        )
        for skill, count in frequency.most_common(MAX_ITEMS)
    ]
    items.sort(key=lambda i: (-i.unlocks_jobs, -i.appears_in_jobs))

    best_combo: List[str] = []
    best_unlocks = 0
    pool = [i.skill for i in items[:4]]
    for combo in combinations(pool, min(combo_size, len(pool))) if len(pool) >= combo_size else []:
        unlocks = unlocked_by(list(combo))
        if unlocks > best_unlocks:
            best_unlocks, best_combo = unlocks, list(combo)

    return SkillRoadmap(
        items=items,
        combo_skills=best_combo,
        combo_unlocks=best_unlocks,
        total_jobs_analyzed=len(jobs),
        status="ok",
    )
