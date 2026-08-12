"""Structured models for every LLM output and every pipeline artifact.

Structured output is used instead of regex-parsing free text: the previous
implementation parsed `SCORE:` / `ISSUES:` with regex and silently fell back
to a made-up default (5 or 50) whenever the format drifted.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium"]
Priority = Literal["Apply Now", "Worth Applying", "Stretch Opportunity", "Low Priority"]


# --------------------------------------------------------------------------
# Candidate
# --------------------------------------------------------------------------

class ResumeEvidence(BaseModel):
    """A concrete, quotable item from the resume. Never invented."""

    claim: str = Field(description="What the candidate did, in one line, taken from the resume")
    skills: List[str] = Field(default_factory=list, description="Skills this item demonstrates")
    kind: Literal["project", "internship", "work", "education", "achievement", "certification"] = "project"


class CandidateProfile(BaseModel):
    """Built once per resume, cached by resume hash, reused everywhere."""

    primary_skills: List[str] = Field(default_factory=list, description="Strongest demonstrated technical skills")
    secondary_skills: List[str] = Field(default_factory=list, description="Skills present but less demonstrated")
    domains: List[str] = Field(default_factory=list, description="Domains actually worked in")
    education: List[str] = Field(default_factory=list)
    evidence: List[ResumeEvidence] = Field(default_factory=list)
    role_families: List[str] = Field(
        default_factory=list,
        description="Realistic job-title families supported by this resume, most relevant first",
    )
    experience_level: Literal["student", "intern", "entry", "mid", "senior"] = "student"
    years_experience: float = Field(default=0.0, ge=0.0, le=50.0)

    def all_skills(self) -> List[str]:
        seen, out = set(), []
        for s in list(self.primary_skills) + list(self.secondary_skills):
            k = s.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(s.strip())
        return out


# --------------------------------------------------------------------------
# Company research
# --------------------------------------------------------------------------

class VerifiedFact(BaseModel):
    fact: str = Field(description="A single verifiable statement about the company")
    source_url: str = Field(default="", description="URL the statement came from")
    source_title: str = Field(default="", description="Title of the source page")
    confidence: Confidence = "medium"


class CompanyResearch(BaseModel):
    company: str
    verified_facts: List[VerifiedFact] = Field(default_factory=list)
    entity_confirmed: bool = Field(
        default=False,
        description="True only if retrieved sources demonstrably concern THIS company",
    )
    summary: str = Field(default="", description="Neutral summary built only from verified_facts")
    status: Literal["ok", "no_results", "entity_unconfirmed", "unavailable", "skipped"] = "skipped"
    reason: str = ""

    def usable_facts(self) -> List[VerifiedFact]:
        """Facts the cover note is permitted to use. Empty unless the entity
        was positively confirmed -- silence beats a confident wrong fact."""
        return list(self.verified_facts) if self.entity_confirmed else []


class _FactExtraction(BaseModel):
    """Raw structured output from the fact-extraction call."""

    entity_confirmed: bool = Field(
        description="True ONLY if the snippets clearly concern the named company, not a similarly named one"
    )
    entity_reasoning: str = Field(default="", description="One line explaining the entity decision")
    facts: List[VerifiedFact] = Field(default_factory=list)
    summary: str = Field(default="")


# --------------------------------------------------------------------------
# Gap analysis
# --------------------------------------------------------------------------

class GapAnalysis(BaseModel):
    matched_requirements: List[str] = Field(default_factory=list, description="JD requirements the resume demonstrably meets")
    missing_requirements: List[str] = Field(default_factory=list, description="JD requirements not demonstrated in the resume")
    resume_edits: List[str] = Field(default_factory=list, description="Concrete rewordings using only existing resume content")
    fit_verdict: Literal["strong", "moderate", "weak"] = "moderate"
    fit_reason: str = ""
    status: Literal["ok", "skipped", "unavailable"] = "skipped"
    reason: str = ""


# --------------------------------------------------------------------------
# Cover note
# --------------------------------------------------------------------------

class CoverNoteCritique(BaseModel):
    """One structured call replaces the old critique + regenerate round trips."""

    grounding_score: int = Field(default=0, ge=0, le=100, description="Are all claims traceable to resume/JD/verified facts")
    specificity_score: int = Field(default=0, ge=0, le=100)
    jd_alignment_score: int = Field(default=0, ge=0, le=100)
    writing_score: int = Field(default=0, ge=0, le=100)
    issues: List[str] = Field(default_factory=list)
    ungrounded_claims: List[str] = Field(default_factory=list, description="Claims not supported by supplied material")
    needs_revision: bool = False
    corrected_note: str = Field(default="", description="Revised note; empty if needs_revision is false")


class ValidationResult(BaseModel):
    """Deterministic post-generation check. No LLM involved."""

    passed: bool = True
    failures: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    word_count: int = 0


class CoverNote(BaseModel):
    job_id: str
    company: str
    title: str
    text: str = ""
    score: int = Field(default=0, ge=0, le=100)
    critique_issues: List[str] = Field(default_factory=list)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    used_company_facts: List[str] = Field(default_factory=list)
    revised: bool = False
    status: Literal["ok", "rejected", "unavailable", "skipped"] = "skipped"
    reason: str = ""


# --------------------------------------------------------------------------
# Opportunities
# --------------------------------------------------------------------------

class JobPosting(BaseModel):
    job_id: str = ""
    company: str = ""
    title: str = ""
    jd_text: str = ""
    location: str = ""
    url: str = ""
    source: str = "adzuna"


class ScoreBreakdown(BaseModel):
    """Every component is shown to the user; the score is never a black box."""

    semantic_raw: float = Field(default=0.0, description="Raw cosine similarity, kept for transparency")
    semantic_normalized: float = 0.0
    skill_coverage: float = 0.0
    experience_match: float = 0.0
    role_relevance: float = 0.0
    total: int = Field(default=0, ge=0, le=100)
    formula: str = ""


class Opportunity(BaseModel):
    job: JobPosting
    score: ScoreBreakdown = Field(default_factory=ScoreBreakdown)

    # Deterministic eligibility analysis
    eligibility_status: Literal[
        "eligible",
        "likely_eligible",
        "uncertain",
        "ineligible",
    ] = "uncertain"

    eligibility_explanation: str = ""
    eligibility_warnings: List[str] = Field(default_factory=list)

    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(
        default_factory=list,
        description="Resume items supporting the match",
    )

    priority: Priority = "Low Priority"
    priority_reason: str = ""
    why_match: str = ""
    why_apply: str = ""
    blockers: List[str] = Field(default_factory=list)


class ResumeSkillGap(BaseModel):
    """One JD requirement the resume does not currently demonstrate.

    estimated_score_gain is the exact point change the deterministic scoring
    formula would produce if this single skill were genuinely demonstrated --
    computed the same way roadmap.py computes "unlocks_jobs", never guessed.
    """

    skill: str
    estimated_score_gain: int = Field(
        default=0,
        description="Points (0-100 scale) the match score would gain if this skill were genuinely added",
    )


class ResumeRecommendations(BaseModel):
    """Deterministic, non-fabricating suggestions for the current job only.

    Every entry traces to profile.all_skills(), gap.resume_edits, or the JD
    text itself -- nothing here is invented.
    """

    emphasize: List[str] = Field(
        default_factory=list,
        description="Skills the candidate genuinely has (secondary_skills) that match this JD but are underrepresented",
    )
    phrasing_suggestions: List[str] = Field(
        default_factory=list,
        description="Rewordings of existing resume content (from gap analysis), not new claims",
    )
    missing_skills: List[ResumeSkillGap] = Field(
        default_factory=list,
        description="JD requirements not demonstrated in the resume, with an estimated score impact if genuinely acquired",
    )
    unscored_gaps: List[str] = Field(
        default_factory=list,
        description="Additional gaps identified by the LLM gap analysis that fall outside the deterministic skill vocabulary -- score impact is not calculable for these",
    )
    coverage_now: float = 0.0
    status: Literal["ok", "insufficient_data", "skipped"] = "skipped"
    note: str = ""


class SkillGapItem(BaseModel):
    skill: str
    appears_in_jobs: int = 0
    total_jobs: int = 0
    unlocks_jobs: int = Field(default=0, description="Jobs that would cross the coverage threshold if acquired")


class SkillRoadmap(BaseModel):
    items: List[SkillGapItem] = Field(default_factory=list)
    combo_skills: List[str] = Field(default_factory=list)
    combo_unlocks: int = 0
    total_jobs_analyzed: int = 0
    status: Literal["ok", "insufficient_data", "skipped"] = "skipped"

    def headline(self) -> str:
        if self.status != "ok" or not self.combo_skills or self.combo_unlocks <= 0:
            return ""
        return (
            f"Adding {' + '.join(self.combo_skills)} could improve your eligibility "
            f"for {self.combo_unlocks} of the {self.total_jobs_analyzed} discovered opportunities."
        )


class DiscoveryResult(BaseModel):
    queries_used: List[str] = Field(default_factory=list)
    jobs_fetched: int = 0
    jobs_deduplicated: int = 0
    opportunities: List[Opportunity] = Field(default_factory=list)
    roadmap: SkillRoadmap = Field(default_factory=SkillRoadmap)
    status: Literal["ok", "no_results", "unavailable", "skipped", "disabled"] = "skipped"
    reason: str = ""
