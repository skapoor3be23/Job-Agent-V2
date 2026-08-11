"""The v2 pipeline: a LangGraph with one genuine parallel superstep.

Topology
--------
                       parse_resume / candidate_profile
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
   gap_analysis            company_research           opportunity_discovery
        |                           |                           |
        +---------------------------+---------------------------+
                                    | join
                               cover_note
                                    |
                        critique_and_revise (+ deterministic validation)
                                    |
                              final results

Two rules make the parallel superstep safe, both verified by tests:

1. Every node returns ONLY its delta. Returning the whole state (as the
   original nodes did) raises InvalidUpdateError under a parallel superstep,
   because two branches then both write `resume_text`.
2. Fields that more than one branch appends to carry an `operator.add`
   reducer so concurrent writes merge instead of colliding.
"""
from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph import END, StateGraph

from . import cache
from .clients import Clients, LLMUnavailable
from .company import research_company
from .config import Settings, get_settings
from .cover_note import (
    _candidate_block,
    _company_block,
    build_generation_prompt,
    validate_cover_note,
)
from .critique import critique_and_revise
from .discovery import discover_opportunities
from .gap import analyze_gap
from .identity import content_hash, make_job_id
from .resume import build_candidate_profile
from .schemas import (
    CandidateProfile,
    CompanyResearch,
    CoverNote,
    DiscoveryResult,
    GapAnalysis,
)
from .telemetry import BudgetExceeded, RunMetrics, new_run

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    # --- inputs (written once, before the parallel superstep) ---
    resume_text: str
    company: str
    title: str
    jd_text: str
    job_url: str
    location: str
    job_id: str
    session_companies: List[str]

    # --- runtime handles (not written by parallel branches) ---
    _clients: Any
    _settings: Any
    _metrics: Any

    # --- branch outputs (each written by exactly one branch) ---
    profile: Any
    gap: Any
    research: Any
    discovery: Any

    # --- join outputs ---
    cover_note: Any

    # --- appended by several branches: needs a reducer ---
    stage_notes: Annotated[List[str], operator.add]


# --------------------------------------------------------------------------
# Nodes. Each returns a DELTA dict only.
# --------------------------------------------------------------------------

def node_candidate_profile(state: PipelineState) -> Dict[str, Any]:
    clients, settings, metrics = state["_clients"], state["_settings"], state["_metrics"]
    with metrics.stage("Candidate Profile"):
        profile = build_candidate_profile(state["resume_text"], clients, settings, metrics)
    return {"profile": profile}


def node_gap_analysis(state: PipelineState) -> Dict[str, Any]:
    clients, settings, metrics = state["_clients"], state["_settings"], state["_metrics"]
    notes: List[str] = []
    with metrics.stage("Gap Analysis"):
        gap = analyze_gap(
            resume_text=state["resume_text"], profile=state["profile"],
            company=state.get("company", ""), title=state.get("title", ""),
            jd_text=state.get("jd_text", ""),
            clients=clients, settings=settings, metrics=metrics,
        )
    if gap.status != "ok" and gap.reason:
        notes.append(f"Gap analysis: {gap.reason}")
    return {"gap": gap, "stage_notes": notes}


def node_company_research(state: PipelineState) -> Dict[str, Any]:
    clients, settings, metrics = state["_clients"], state["_settings"], state["_metrics"]
    notes: List[str] = []
    with metrics.stage("Company Research"):
        research = research_company(
            company=state.get("company", ""), title=state.get("title", ""),
            clients=clients, settings=settings, metrics=metrics,
        )
    if research.status != "ok" and research.reason:
        notes.append(f"Company research: {research.reason}")
    return {"research": research, "stage_notes": notes}


def node_opportunity_discovery(state: PipelineState) -> Dict[str, Any]:
    clients, settings, metrics = state["_clients"], state["_settings"], state["_metrics"]
    notes: List[str] = []
    with metrics.stage("Opportunity Discovery"):
        discovery = discover_opportunities(
            profile=state["profile"], resume_text=state["resume_text"],
            clients=clients, settings=settings, metrics=metrics,
            exclude_job_ids=[state.get("job_id", "")],
        )
    if discovery.status != "ok" and discovery.reason:
        notes.append(f"Opportunity Discovery: {discovery.reason}")
    return {"discovery": discovery, "stage_notes": notes}


def node_cover_note(state: PipelineState) -> Dict[str, Any]:
    """Generate, then critique+revise (one call), then validate deterministically."""
    clients, settings, metrics = state["_clients"], state["_settings"], state["_metrics"]
    company, title = state.get("company", ""), state.get("title", "")
    job_id = state.get("job_id", "")
    profile: CandidateProfile = state["profile"]
    research: CompanyResearch = state.get("research") or CompanyResearch(company=company, status="skipped")
    jd_text = state.get("jd_text", "")
    notes: List[str] = []

    if not jd_text or len(jd_text.strip()) < 20:
        return {
            "cover_note": CoverNote(
                job_id=job_id, company=company, title=title, status="skipped",
                reason="No usable job description, so no cover note was generated.",
            ),
            "stage_notes": ["Cover note skipped: job description was empty or too short."],
        }

    candidate_block = _candidate_block(profile, state["resume_text"])
    company_block = _company_block(research)

    with metrics.stage("Cover Note"):
        prompt = build_generation_prompt(
            company=company, title=title, jd_text=jd_text, profile=profile,
            research=research, resume_text=state["resume_text"], settings=settings,
        )
        try:
            draft = clients.llm.invoke(prompt).strip()
        except BudgetExceeded as exc:
            metrics.note(f"Cover note: {exc}")
            return {
                "cover_note": CoverNote(
                    job_id=job_id, company=company, title=title,
                    status="unavailable", reason=str(exc),
                ),
                "stage_notes": [f"Cover note: {exc}"],
            }
        except LLMUnavailable as exc:
            return {
                "cover_note": CoverNote(
                    job_id=job_id, company=company, title=title, status="unavailable",
                    reason="The cover note could not be generated.",
                ),
                "stage_notes": [f"Cover note unavailable: {exc}"],
            }

    with metrics.stage("Critique / Revision"):
        final_text, score, issues, revised = critique_and_revise(
            note_text=draft, company=company, title=title, jd_text=jd_text,
            profile=profile, research=research,
            candidate_block=candidate_block, company_block=company_block,
            clients=clients, settings=settings, metrics=metrics,
        )

    with metrics.stage("Validation"):
        validation = validate_cover_note(
            note=final_text, company=company, title=title, profile=profile,
            research=research, settings=settings,
            other_companies=state.get("session_companies", []),
        )
        # If the revision failed validation but the draft passes, prefer the draft.
        if not validation.passed and revised:
            draft_validation = validate_cover_note(
                note=draft, company=company, title=title, profile=profile,
                research=research, settings=settings,
                other_companies=state.get("session_companies", []),
            )
            if draft_validation.passed:
                final_text, validation, revised = draft, draft_validation, False
                notes.append("The revised note failed validation; the original draft was kept.")

    status = "ok" if validation.passed else "rejected"
    if not validation.passed:
        notes.append("Cover note failed factual validation: " + "; ".join(validation.failures))

    return {
        "cover_note": CoverNote(
            job_id=job_id, company=company, title=title, text=final_text,
            score=score, critique_issues=issues, validation=validation,
            used_company_facts=[f.fact for f in research.usable_facts()],
            revised=revised, status=status,
        ),
        "stage_notes": notes,
    }


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

def build_graph(enable_discovery: bool = True):
    graph = StateGraph(PipelineState)
    graph.add_node("candidate_profile", node_candidate_profile)
    graph.add_node("gap_analysis", node_gap_analysis)
    graph.add_node("company_research", node_company_research)
    graph.add_node("cover_note", node_cover_note)

    graph.set_entry_point("candidate_profile")

    branches = ["gap_analysis", "company_research"]
    if enable_discovery:
        graph.add_node("opportunity_discovery", node_opportunity_discovery)
        branches.append("opportunity_discovery")

    # Fan out: all branches run in ONE superstep, on separate threads.
    for branch in branches:
        graph.add_edge("candidate_profile", branch)
        graph.add_edge(branch, "cover_note")

    graph.add_edge("cover_note", END)
    return graph.compile()


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

class PipelineResult:
    """Everything the dashboard needs, keyed to one job_id."""

    def __init__(
        self,
        job_id: str, company: str, title: str, job_url: str,
        profile: CandidateProfile, gap: GapAnalysis, research: CompanyResearch,
        discovery: DiscoveryResult, cover_note: CoverNote,
        metrics: RunMetrics, notes: Sequence[str],
    ):
        self.job_id = job_id
        self.company = company
        self.title = title
        self.job_url = job_url
        self.profile = profile
        self.gap = gap
        self.research = research
        self.discovery = discovery
        self.cover_note = cover_note
        self.metrics = metrics
        self.notes = list(notes)

    def match_score(self) -> int:
        """Unified 0-100 match score for the analysed job.

        Same formula as the opportunity score so the current job and the
        discovered jobs are directly comparable.
        """
        from .ranking import cosine_similarity
        from .skills import coverage, experience_match, extract_skills, role_relevance

        required = extract_skills(self.gap_jd_text or "")
        cov, _, _ = coverage(required, self.profile.all_skills())
        eligibility, _ = experience_match(
            self.gap_jd_text or "", self.profile.years_experience, self.profile.experience_level
        )
        relevance = role_relevance(self.title, self.profile.role_families)
        # Without a comparison pool there is no rank-normalisation, so the
        # semantic component is replaced by the evidence-backed match ratio.
        matched = len(self.gap.matched_requirements)
        total = matched + len(self.gap.missing_requirements)
        evidence_ratio = (matched / total) if total else cov
        score = 0.45 * evidence_ratio + 0.25 * cov + 0.15 * eligibility + 0.15 * relevance
        return int(round(max(0.0, min(1.0, score)) * 100))

    gap_jd_text: str = ""


def run_pipeline(
    resume_text: str,
    company: str,
    title: str,
    jd_text: str,
    job_url: str = "",
    location: str = "",
    session_companies: Optional[Sequence[str]] = None,
    settings: Optional[Settings] = None,
    clients: Optional[Clients] = None,
    metrics: Optional[RunMetrics] = None,
) -> PipelineResult:
    """Run the v2 pipeline for exactly one job.

    Never raises for recoverable conditions: each stage carries its own
    status and reason, and the UI reports what was skipped and why.
    """
    settings = settings or get_settings()
    metrics = metrics or new_run(settings)
    clients = clients or Clients.build(settings, metrics)

    job_id = make_job_id(company, title, job_url, location, jd_text)

    initial: PipelineState = {
        "resume_text": resume_text or "",
        "company": company or "",
        "title": title or "",
        "jd_text": jd_text or "",
        "job_url": job_url or "",
        "location": location or "",
        "job_id": job_id,
        "session_companies": list(session_companies or []),
        "_clients": clients,
        "_settings": settings,
        "_metrics": metrics,
        "stage_notes": [],
    }

    app = build_graph(enable_discovery=settings.discovery_enabled)
    final = app.invoke(initial)

    result = PipelineResult(
        job_id=job_id, company=company, title=title, job_url=job_url,
        profile=final.get("profile") or CandidateProfile(),
        gap=final.get("gap") or GapAnalysis(status="skipped"),
        research=final.get("research") or CompanyResearch(company=company, status="skipped"),
        discovery=final.get("discovery") or DiscoveryResult(status="skipped"),
        cover_note=final.get("cover_note") or CoverNote(job_id=job_id, company=company, title=title),
        metrics=metrics,
        notes=list(final.get("stage_notes", [])) + list(metrics.notes),
    )
    result.gap_jd_text = jd_text or ""
    return result
