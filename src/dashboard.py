"""AI Job Opportunity Intelligence Agent — Streamlit dashboard.

Entry point:  streamlit run src/dashboard.py

Selects between the v2 pipeline (src/agent/) and the original pipeline
(src/live_pipeline.py) via the USE_NEW_PIPELINE setting, so the previously
working app remains one toggle away at all times.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import store
from agent.clients import credential_status
from agent.config import get_settings
from agent.graph import run_pipeline
from agent.resume import extract_pdf_text
from agent.schemas import DiscoveryResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Job Opportunity Intelligence Agent", layout="wide")

SETTINGS = get_settings()

PRIORITY_STYLE = {
    "Apply Now": "🟢",
    "Worth Applying": "🔵",
    "Stretch Opportunity": "🟡",
    "Low Priority": "⚪",
}


# --------------------------------------------------------------------------
# Legacy switch
# --------------------------------------------------------------------------

def _use_new_pipeline() -> bool:
    default = SETTINGS.use_new_pipeline
    with st.sidebar:
        st.subheader("Pipeline")
        choice = st.radio(
            "Which pipeline should run?",
            ["New (v2, parallel)", "Original (v1, sequential)"],
            index=0 if default else 1,
            help="The original pipeline is preserved unchanged as a fallback.",
        )
        st.caption(f"Default from config: USE_NEW_PIPELINE={default}")
    return choice.startswith("New")


def _render_sidebar_status() -> None:
    with st.sidebar:
        st.divider()
        st.subheader("Configuration")
        for key, present in credential_status().items():
            st.write(("✅ " if present else "❌ ") + key)
        st.caption("Only presence is shown; values are never displayed or logged.")
        st.divider()
        st.subheader("Budgets (per run)")
        st.write(f"LLM calls: {SETTINGS.max_llm_calls_per_run}")
        st.write(f"Embedding calls: {SETTINGS.max_embedding_calls_per_run}")
        st.write(f"Search calls: {SETTINGS.max_search_calls_per_run}")
        st.write(f"Discovery jobs: {SETTINGS.discovery_max_jobs}")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def section_candidate_profile(result) -> None:
    st.header("1. Candidate Profile")
    profile = result.profile
    if not profile.all_skills() and not profile.role_families:
        st.info("No profile could be extracted from this resume.")
        return
    left, right = st.columns(2)
    with left:
        st.markdown("**Strongest skills**")
        st.write(", ".join(profile.primary_skills) or "—")
        st.markdown("**Secondary skills**")
        st.write(", ".join(profile.secondary_skills) or "—")
        st.markdown("**Experience level**")
        st.write(f"{profile.experience_level} ({profile.years_experience:g} yrs)")
    with right:
        st.markdown("**Relevant role families**")
        for family in profile.role_families or ["—"]:
            st.write(f"- {family}")
        st.markdown("**Key resume evidence**")
        for ev in profile.evidence[:4]:
            st.write(f"- {ev.claim}")
        if not profile.evidence:
            st.caption("No individual evidence items were extracted.")


def section_current_job(result) -> None:
    st.header("2. Current Job Analysis")
    gap = result.gap
    cols = st.columns([2, 2, 1])
    cols[0].metric("Company", result.company or "—")
    cols[1].metric("Role", result.title or "—")
    cols[2].metric("Match score", f"{result.match_score()}/100")

    if gap.status != "ok":
        st.warning(f"Gap analysis: {gap.reason}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Strong matches**")
        for item in gap.matched_requirements or ["—"]:
            st.write(f"- {item}")
    with right:
        st.markdown("**Skill gaps**")
        for item in gap.missing_requirements or ["—"]:
            st.write(f"- {item}")

    if gap.fit_reason:
        st.markdown(f"**Verdict:** `{gap.fit_verdict}` — {gap.fit_reason}")
    if gap.resume_edits:
        with st.expander("Suggested resume edits"):
            for edit in gap.resume_edits:
                st.write(f"- {edit}")

    st.markdown("**Verified company information**")
    research = result.research
    facts = research.usable_facts()
    if facts:
        for fact in facts:
            label = fact.source_title or fact.source_url
            st.write(f"- {fact.fact}  \n  <small>source: [{label}]({fact.source_url}) · confidence: {fact.confidence}</small>",
                     unsafe_allow_html=True)
    else:
        st.info(
            f"No verified company facts were available ({research.status}). "
            f"{research.reason} The cover note below therefore makes no company-specific claims."
        )


def section_recommendation(result) -> None:
    st.header("3. Application Recommendation")
    from agent.ranking import assign_priority
    from agent.skills import coverage, experience_match, extract_skills, role_relevance

    jd = result.gap_jd_text
    required = extract_skills(jd)
    cov, matched, missing = coverage(required, result.profile.all_skills())
    eligibility, blockers = experience_match(jd, result.profile.years_experience, result.profile.experience_level)
    relevance = role_relevance(result.title, result.profile.role_families)
    evidence_count = len([e for e in result.profile.evidence if set(s.lower() for s in e.skills) & {m.lower() for m in matched}])
    priority, reason = assign_priority(cov, eligibility, blockers, evidence_count, relevance)

    st.subheader(f"{PRIORITY_STYLE.get(priority,'')} {priority}")
    st.write(reason)
    if blockers:
        st.error("Blockers: " + "; ".join(blockers))
    if missing:
        st.markdown("**Important gaps:** " + ", ".join(missing[:6]))
    return priority


def section_cover_note(result) -> None:
    st.header("4. Cover Note")
    note = result.cover_note
    st.caption(
        f"Generated specifically for **{note.company} — {note.title}** "
        f"(job_id `{note.job_id}`). This panel only ever shows the note bound to this job_id."
    )
    if note.status == "skipped":
        st.info(note.reason)
        return
    if note.status == "unavailable":
        st.warning(note.reason)
        return
    if note.status == "rejected":
        st.error("This note failed factual validation and is shown for inspection only:")
        for failure in note.validation.failures:
            st.write(f"- {failure}")

    st.text_area("Cover note", note.text, height=280, key=f"note_{note.job_id}")
    cols = st.columns(3)
    cols[0].metric("Quality score", f"{note.score}/100")
    cols[1].metric("Words", note.validation.word_count)
    cols[2].metric("Revised", "yes" if note.revised else "no")

    if note.used_company_facts:
        with st.expander("Company facts this note was permitted to use"):
            for fact in note.used_company_facts:
                st.write(f"- {fact}")
    else:
        st.caption("No verified company facts were available, so the note makes no company-specific claims.")
    if note.critique_issues:
        with st.expander("Critique issues"):
            for issue in note.critique_issues:
                st.write(f"- {issue}")
    if note.validation.warnings:
        with st.expander("Validation warnings"):
            for warning in note.validation.warnings:
                st.write(f"- {warning}")


def section_opportunities(result) -> None:
    st.header("5. Discover More Opportunities")
    discovery: DiscoveryResult = result.discovery

    if discovery.status != "ok":
        st.info(f"Opportunity Discovery unavailable ({discovery.status}). {discovery.reason}")
        return

    st.caption(
        f"Searched: {', '.join(discovery.queries_used)} · "
        f"{discovery.jobs_fetched} fetched · {discovery.jobs_deduplicated} unique and analysed"
    )

    for opp in discovery.opportunities:
        header = (
            f"{PRIORITY_STYLE.get(opp.priority,'')} {opp.score.total}/100 — "
            f"{opp.job.title} at {opp.job.company}"
        )
        with st.expander(header):
            st.markdown(f"**{opp.priority}** — {opp.priority_reason}")
            st.markdown("**Why this matches you**")
            st.write(opp.why_match)
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Strong matches**")
                for skill in opp.matched_skills or ["—"]:
                    st.write(f"- {skill}")
            with cols[1]:
                st.markdown("**Potential gaps**")
                for skill in opp.missing_skills or ["—"]:
                    st.write(f"- {skill}")
            if opp.evidence:
                st.markdown("**Your supporting evidence**")
                for item in opp.evidence:
                    st.write(f"- {item}")
            st.markdown("**Why you should consider applying**")
            st.write(opp.why_apply)
            with st.expander("Score breakdown"):
                st.code(
                    f"semantic (raw {opp.score.semantic_raw:.3f} -> normalised {opp.score.semantic_normalized:.2f})\n"
                    f"skill coverage   {opp.score.skill_coverage:.2f}\n"
                    f"experience match {opp.score.experience_match:.2f}\n"
                    f"role relevance   {opp.score.role_relevance:.2f}\n"
                    f"total = {opp.score.formula} = {opp.score.total}/100"
                )
            if opp.job.url:
                st.link_button("Open job listing", opp.job.url)
            else:
                st.caption("No listing URL was provided for this job.")
            if st.button("Run deep analysis on this job", key=f"deep_{opp.job.job_id}"):
                st.session_state["deep_analysis_request"] = {
                    "company": opp.job.company, "title": opp.job.title,
                    "jd_text": opp.job.jd_text, "job_url": opp.job.url,
                    "location": opp.job.location,
                }
                st.rerun()


def section_roadmap(result) -> None:
    st.header("6. Skill Gap Roadmap")
    roadmap = result.discovery.roadmap
    if roadmap.status != "ok":
        st.info("Not enough discovered opportunities to identify reliable skill trends.")
        return
    headline = roadmap.headline()
    if headline:
        st.success(headline)
    rows = [
        {
            "Skill": item.skill,
            "Appears in": f"{item.appears_in_jobs}/{item.total_jobs} jobs",
            "Would unlock": f"{item.unlocks_jobs} jobs",
        }
        for item in roadmap.items
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Counts are computed directly from the retrieved job descriptions, not estimated.")


def section_performance(result) -> None:
    st.header("7. Performance")
    data = result.metrics.as_dict()
    st.code(result.metrics.render_text())
    cols = st.columns(5)
    cols[0].metric("Gemini calls", data["llm_calls"])
    cols[1].metric("Search calls", data["search_calls"])
    cols[2].metric("Embedding calls", data["embedding_calls"])
    cols[3].metric("Cache hits", data["cache_hits"])
    cols[4].metric("Cache misses", data["cache_misses"])
    st.caption(
        f"Budget used — LLM {data['budget']['llm']} · "
        f"search {data['budget']['search']} · embeddings {data['budget']['embedding']}"
    )
    if result.notes:
        st.markdown("**What was skipped, and why**")
        for note in result.notes:
            st.write(f"- {note}")


def section_tracker(result, priority: str) -> None:
    st.header("8. Application Tracker")
    note = result.cover_note
    if st.button("Save this analysis to the tracker"):
        outcome = store.save_application(
            job_id=result.job_id, company=result.company, title=result.title,
            cover_note=note.text, match_score=float(result.match_score()),
            priority=priority, job_url=result.job_url,
        )
        st.success(f"{outcome['action'].capitalize()} application #{outcome['id']} (job_id `{result.job_id}`).")

    rows = store.list_applications()
    if not rows:
        st.info("No saved applications yet.")
        return

    frame = pd.DataFrame(rows)
    statuses = ["All"] + sorted(frame["status"].dropna().unique().tolist())
    chosen = st.selectbox("Filter by status", statuses)
    view = frame if chosen == "All" else frame[frame["status"] == chosen]
    display_cols = [c for c in ["id", "job_id", "company", "title", "match_score", "priority", "status", "date_applied"] if c in view.columns]
    st.dataframe(view[display_cols], use_container_width=True, hide_index=True)

    if not view.empty:
        app_id = st.selectbox("Application to update", view["id"].tolist())
        new_status = st.selectbox("New status", store.STATUSES)
        if st.button("Update status"):
            store.update_status(int(app_id), new_status)
            st.success(f"Updated #{app_id} to {new_status}.")
            st.rerun()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def render_new_dashboard() -> None:
    st.title("AI Job Opportunity Intelligence Agent")
    st.caption(
        "Understands your profile, evaluates fit, verifies company information, "
        "generates a grounded cover note, and finds better opportunities."
    )

    pending = st.session_state.pop("deep_analysis_request", None)

    with st.form("run_form"):
        cols = st.columns(2)
        with cols[0]:
            company = st.text_input("Company", value=(pending or {}).get("company", ""))
            title = st.text_input("Job title", value=(pending or {}).get("title", ""))
            job_url = st.text_input("Job listing URL (optional)", value=(pending or {}).get("job_url", ""))
        with cols[1]:
            resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])
            location = st.text_input("Location (optional)", value=(pending or {}).get("location", ""))
        jd_text = st.text_area("Job description", height=180, value=(pending or {}).get("jd_text", ""))
        submitted = st.form_submit_button("Run analysis")

    if submitted:
        if not resume_file:
            st.error("Upload a resume PDF first.")
            return
        if not company or not title:
            st.error("Company and job title are both required.")
            return
        try:
            resume_text = extract_pdf_text(resume_file)
        except Exception as exc:
            st.error(f"Could not read the resume PDF ({type(exc).__name__}). Try re-exporting it.")
            return
        if len(resume_text.strip()) < 50:
            st.error("No readable text was extracted from the resume. It may be a scanned image.")
            return

        seen = st.session_state.setdefault("session_companies", [])
        if company not in seen:
            seen.append(company)

        with st.status("Running pipeline...", expanded=False) as status:
            try:
                result = run_pipeline(
                    resume_text=resume_text, company=company, title=title, jd_text=jd_text,
                    job_url=job_url, location=location, session_companies=seen,
                    settings=SETTINGS,
                )
                status.update(label="Analysis complete", state="complete")
            except Exception as exc:
                status.update(label="Pipeline failed", state="error")
                logging.exception("pipeline failed")
                st.error(f"The analysis could not be completed ({type(exc).__name__}). Nothing was saved.")
                return

        # UI state is keyed by job_id so a previous job's results can never
        # be rendered under a new one.
        st.session_state["result_by_job_id"] = {result.job_id: result}
        st.session_state["current_job_id"] = result.job_id

    current_id = st.session_state.get("current_job_id")
    results = st.session_state.get("result_by_job_id", {})
    if not current_id or current_id not in results:
        st.info("Upload a resume and paste a job description to begin.")
        return

    result = results[current_id]
    section_candidate_profile(result)
    section_current_job(result)
    priority = section_recommendation(result)
    section_cover_note(result)
    section_opportunities(result)
    section_roadmap(result)
    section_performance(result)
    section_tracker(result, priority)


def main() -> None:
    use_new = _use_new_pipeline()
    _render_sidebar_status()
    if use_new:
        render_new_dashboard()
    else:
        st.title("Job Application Agent (original pipeline)")
        st.caption("Running the original v1 pipeline. Switch in the sidebar to use the new one.")
        from dashboard_legacy import render_legacy_dashboard
        render_legacy_dashboard()


main()
