"""The v2 pipeline: a LangGraph with one genuine parallel superstep, plus one
branch (company research) that runs on a plain background thread outside
the graph entirely.

Topology
--------
   candidate_profile                    company_research
   (LangGraph, from START)         (plain thread, from START --
        |                           no data dependency on profile)
        |                                    |
        +----------------+----------+        |
        |                |          |        |
   gap_analysis    fit_scoring  opportunity_discovery
        |                |          |        |
        +----------------+----------+--------+
                          | join (awaits the research future)
                     cover_note
                          |
              critique_and_revise (+ deterministic validation)
                          |
                    final results

Two rules make the LangGraph parallel superstep safe, both verified by tests:

1. Every node returns ONLY its delta. Returning the whole state (as the
   original nodes did) raises InvalidUpdateError under a parallel superstep,
   because two branches then both write `resume_text`.
2. Fields that more than one branch appends to carry an `operator.add`
   reducer so concurrent writes merge instead of colliding.

company_research is deliberately NOT a graph branch off START alongside
candidate_profile, even though it has no data dependency on `profile` (it
only needs `company`/`title`, both available immediately): LangGraph fires
a downstream node once per superstep in which ANY predecessor completes,
not once the union of all predecessors (across supersteps) is ready. A
branch finishing one superstep earlier than its siblings made cover_note
run -- and bill its two LLM calls -- twice. Running it on a plain thread
and awaiting the future only where the result is actually needed
(node_cover_note) sidesteps that entirely while still overlapping it with
candidate_profile's own LLM call.
"""
from __future__ import annotations

import logging
import operator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langgraph.graph import END, START, StateGraph

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
from .ranking import evaluate_current_job
from .resume import build_candidate_profile
from .resume_advice import build_resume_recommendations
from .schemas import (
    CandidateProfile,
    CompanyResearch,
    CoverNote,
    DiscoveryResult,
    GapAnalysis,
    JobPosting,
    Opportunity,
    ResumeRecommendations,
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
    known_fit: Any
    research_future: Any

    # --- runtime handles (not written by parallel branches) ---
    _clients: Any
    _settings: Any
    _metrics: Any

    # --- branch outputs (each written by exactly one branch) ---
    profile: Any
    gap: Any
    research: Any
    discovery: Any
    fit: Any

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


def node_fit_scoring(state: PipelineState) -> Dict[str, Any]:
    """Score the analyzed job itself -- the canonical match score/eligibility
    shown as "Match score" and reused by Opportunity Discovery's own listing
    for this exact job. Only depends on `profile`, so it runs inside the
    same parallel superstep as gap_analysis/company_research/discovery
    instead of serially after the whole graph (including cover-note
    generation and critique) the way a prior version of this computed it.
    """
    clients, settings, metrics = state["_clients"], state["_settings"], state["_metrics"]
    profile = state["profile"]
    job_id = state.get("job_id", "")
    known_fit = state.get("known_fit")

    if known_fit is not None and known_fit.job.job_id == job_id:
        # Same job Opportunity Discovery already scored -- reuse it rather
        # than paying for a second embedding call and risking a different
        # answer.
        return {"fit": known_fit}

    current_job = JobPosting(
        job_id=job_id, company=state.get("company", ""), title=state.get("title", ""),
        jd_text=state.get("jd_text", ""), location=state.get("location", ""),
        url=state.get("job_url", ""), source="analyzed",
    )

    with metrics.stage("Match Scoring"):
        resume_vector, job_vector = None, None
        try:
            vectors = clients.embeddings.embed_documents(
                [state.get("resume_text", "") or "", state.get("jd_text", "") or ""]
            )
            resume_vector, job_vector = vectors[0], vectors[1]
        except BudgetExceeded as exc:
            metrics.note(f"Match scoring: {exc} Falling back to keyword similarity.")
        except Exception as exc:
            logger.warning("embedding failed while scoring the current job: %s", type(exc).__name__)
            metrics.note("Embeddings unavailable; match score fell back to keyword similarity.")

        fit = evaluate_current_job(
            current_job, profile, settings.weights,
            resume_text=state.get("resume_text", "") or "",
            resume_vector=resume_vector, job_vector=job_vector,
        )

    return {"fit": fit}


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

    # Company research runs on its own background thread (see run_pipeline),
    # concurrently with candidate_profile and everything derived from it.
    # This is the one place its result is actually needed, so this is the
    # one place that blocks on it -- by the time the rest of the parallel
    # superstep finishes, it is normally already done.
    research_future: Optional[Future] = state.get("research_future")
    research_delta: Dict[str, Any] = research_future.result() if research_future is not None else {}
    research: CompanyResearch = research_delta.get("research") or CompanyResearch(company=company, status="skipped")

    jd_text = state.get("jd_text", "")
    notes: List[str] = list(research_delta.get("stage_notes", []))

    if not jd_text or len(jd_text.strip()) < 20:
        return {
            "cover_note": CoverNote(
                job_id=job_id, company=company, title=title, status="skipped",
                reason="No usable job description, so no cover note was generated.",
            ),
            "stage_notes": notes + ["Cover note skipped: job description was empty or too short."],
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
                    reason=str(exc),
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
    """company_research is intentionally NOT a node here -- see the module
    docstring for why it runs on a plain background thread (started in
    run_pipeline) instead of as a graph branch."""
    graph = StateGraph(PipelineState)
    graph.add_node("candidate_profile", node_candidate_profile)
    graph.add_node("gap_analysis", node_gap_analysis)
    graph.add_node("fit_scoring", node_fit_scoring)
    graph.add_node("cover_note", node_cover_note)

    graph.add_edge(START, "candidate_profile")

    # These DO need `profile` (skills/role_families/eligibility), so they
    # fan out from candidate_profile in one superstep, on separate threads.
    branches = ["gap_analysis", "fit_scoring"]
    if enable_discovery:
        graph.add_node("opportunity_discovery", node_opportunity_discovery)
        branches.append("opportunity_discovery")

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
        fit: Optional[Opportunity] = None,
        resume_recommendations: Optional[ResumeRecommendations] = None,
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
        self.fit = fit
        self.resume_recommendations = resume_recommendations or ResumeRecommendations()

    def match_score(self) -> int:
        """Unified 0-100 match score for the analysed job.

        Delegates to ranking.evaluate_current_job() (computed once in
        run_pipeline and stored as `fit`) so this and the dashboard never
        disagree with each other or with how discovered jobs are scored.
        """
        return self.fit.score.total if self.fit else 0

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
    known_fit: Optional[Opportunity] = None,
) -> PipelineResult:
    """Run the v2 pipeline for exactly one job.

    Never raises for recoverable conditions: each stage carries its own
    status and reason, and the UI reports what was skipped and why.

    known_fit: an Opportunity already computed for this exact job -- e.g.
    the one Opportunity Discovery produced for it. When its job_id matches
    the job being analyzed, it is reused verbatim instead of scoring the
    job a second time. This is the canonical scoring path: a job opened
    from the discovery list must show the identical score, eligibility and
    blockers it showed there, never an independently recomputed one.
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
        "known_fit": known_fit,
        "_clients": clients,
        "_settings": settings,
        "_metrics": metrics,
        "stage_notes": [],
    }

    # Company research has no data dependency on the candidate profile --
    # only `company`/`title`, both already in `initial` -- so it starts
    # immediately on its own thread instead of waiting for
    # candidate_profile's LLM call the way a graph branch would have to
    # (see the module docstring for why it isn't a graph branch). The pool
    # stays open for the whole invoke() call since node_cover_note awaits
    # this future from inside the graph; `with` guarantees it is resolved
    # before we read it again below.
    with ThreadPoolExecutor(max_workers=1) as research_pool:
        research_future: Future = research_pool.submit(node_company_research, initial)
        initial["research_future"] = research_future

        app = build_graph(enable_discovery=settings.discovery_enabled)
        final = app.invoke(initial)

    profile: CandidateProfile = final.get("profile") or CandidateProfile()
    gap: GapAnalysis = final.get("gap") or GapAnalysis(status="skipped")

    research_delta = research_future.result()
    research: CompanyResearch = research_delta.get("research") or CompanyResearch(company=company, status="skipped")

    # fit_scoring runs inside the graph's parallel superstep (see
    # node_fit_scoring) -- it only depends on `profile`, so scoring the
    # analyzed job overlaps with gap_analysis/company_research/discovery
    # instead of adding to the critical path after cover-note generation.
    # This fallback should not normally trigger (the node never raises),
    # but keeps `fit` from ever being None, matching the prior contract.
    fit: Opportunity = final.get("fit") or evaluate_current_job(
        JobPosting(
            job_id=job_id, company=company or "", title=title or "",
            jd_text=jd_text or "", location=location or "", url=job_url or "", source="analyzed",
        ),
        profile, settings.weights, resume_text=resume_text or "",
    )

    resume_recommendations = build_resume_recommendations(
        jd_text=jd_text or "", profile=profile, gap=gap,
        current_score=fit.score.total, weights=settings.weights,
    )

    result = PipelineResult(
        job_id=job_id, company=company, title=title, job_url=job_url,
        profile=profile,
        gap=gap,
        research=research,
        discovery=final.get("discovery") or DiscoveryResult(status="skipped"),
        cover_note=final.get("cover_note") or CoverNote(job_id=job_id, company=company, title=title),
        metrics=metrics,
        notes=list(final.get("stage_notes", [])) + list(metrics.notes),
        fit=fit,
        resume_recommendations=resume_recommendations,
    )
    result.gap_jd_text = jd_text or ""
    return result
