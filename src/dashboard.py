"""AI Job Opportunity Intelligence Agent — Streamlit dashboard.

Entry point:  streamlit run src/dashboard.py

Presentation only. Every number shown here (match score, priority,
eligibility, resume-change estimates) comes from the deterministic backend
in agent/graph.py / agent/ranking.py / agent/resume_advice.py -- this file
never recomputes scoring, eligibility or skill matching on its own.
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

PRIORITY_COLOR = {
    "Apply Now": "#1a7f37",
    "Worth Applying": "#0969da",
    "Stretch Opportunity": "#9a6700",
    "Low Priority": "#6e7781",
}

CUSTOM_CSS = """
<style>
.stTextArea textarea { font-size: 1.05rem !important; line-height: 1.65 !important; }
.priority-badge {
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    font-size: 0.82rem; font-weight: 600; letter-spacing: 0.01em;
    white-space: nowrap;
}
</style>
"""


def _priority_badge(priority: str) -> None:
    color = PRIORITY_COLOR.get(priority, "#6e7781")
    st.markdown(
        f'<span class="priority-badge" style="background:{color}1a;color:{color};'
        f'border:1px solid {color}66;">{priority}</span>',
        unsafe_allow_html=True,
    )


def _compact_list(items, limit: int = 6) -> str:
    """Comma-joined preview capped at `limit`, with a '+N more' suffix."""
    items = [str(i) for i in items if i]
    if not items:
        return "—"
    shown = ", ".join(items[:limit])
    remaining = len(items) - limit
    return shown + (f" (+{remaining} more)" if remaining > 0 else "")


def _render_bulleted(items, limit: int = 5) -> None:
    """Bulleted list capped at `limit`, with a '+N more' caption instead of
    dumping every item -- keeps long lists (JD phrases, not short skill
    names) scannable without hiding that more exist."""
    items = [str(i) for i in items if i]
    if not items:
        st.write("—")
        return
    for item in items[:limit]:
        st.write(f"- {item}")
    remaining = len(items) - limit
    if remaining > 0:
        st.caption(f"+{remaining} more")


def _render_sidebar_status() -> None:
    with st.sidebar:
        st.subheader("Configuration")
        for key, present in credential_status().items():
            st.write(("✅ " if present else "❌ ") + key)
        st.caption("Only presence is shown; values are never displayed or logged.")


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def section_profile(result) -> None:
    profile = result.profile
    if not profile.all_skills() and not profile.role_families:
        st.info("No profile could be extracted from this resume.")
        return
    st.subheader("Candidate")
    cols = st.columns([2, 2, 1])
    with cols[0]:
        st.markdown("**Top skills**")
        st.write(_compact_list(profile.primary_skills, limit=6))
    with cols[1]:
        st.markdown("**Role fit**")
        st.write(_compact_list(profile.role_families, limit=4))
    with cols[2]:
        st.markdown("**Level**")
        st.write(f"{profile.experience_level} · {profile.years_experience:g} yrs")


def section_job_fit(result) -> None:
    st.subheader("Job Fit")
    fit = result.fit
    gap = result.gap

    cols = st.columns([1, 1, 3])
    cols[0].metric("Match score", f"{result.match_score()}/100")
    with cols[1]:
        st.write("")
        if fit:
            _priority_badge(fit.priority)
    with cols[2]:
        st.write("")
        if fit and fit.priority_reason:
            st.caption(fit.priority_reason)

    if fit and fit.eligibility_status == "ineligible":
        st.error("Eligibility: " + (fit.eligibility_explanation or "This role has a stated blocker."))
    elif fit and fit.eligibility_status == "uncertain" and fit.eligibility_warnings:
        st.warning("Eligibility: " + fit.eligibility_warnings[0])

    if gap.status != "ok" and gap.reason:
        st.caption(gap.reason)

    left, right = st.columns(2)
    with left:
        st.markdown("**You demonstrate**")
        _render_bulleted(gap.matched_requirements, limit=5)
    with right:
        st.markdown("**Gaps**")
        _render_bulleted(gap.missing_requirements, limit=5)


def section_resume_changes(result) -> None:
    st.subheader("Resume Changes to Make")
    rec = result.resume_recommendations

    if rec.status == "skipped":
        st.info(rec.note or "No job description was supplied, so resume suggestions aren't available.")
        return

    if rec.note:
        st.caption(rec.note)

    if rec.status == "insufficient_data":
        if rec.phrasing_suggestions:
            with st.expander("Reword existing content"):
                for item in rec.phrasing_suggestions:
                    st.write(f"- {item}")
        else:
            st.info("No specific suggestions could be derived for this job description.")
        return

    has_content = bool(rec.emphasize or rec.phrasing_suggestions or rec.missing_skills)
    if not has_content:
        st.success("No specific resume gaps were identified against this posting's requirements.")
        return

    # The 3-5 highest-impact changes, one line each. Everything else stays
    # available in the breakdown below rather than being repeated here.
    top_actions = [f"Emphasize **{skill}** — already on your resume, currently secondary." for skill in rec.emphasize[:2]]
    top_actions += [
        f"Consider **{g.skill}**, if genuinely true — an estimated +{g.estimated_score_gain} pts."
        for g in rec.missing_skills[:3] if g.estimated_score_gain > 0
    ]
    top_actions = top_actions[:5]

    if top_actions:
        st.markdown("**Top changes**")
        for action in top_actions:
            st.write(f"- {action}")

    with st.expander("Full breakdown"):
        if rec.emphasize:
            st.markdown("**Already present — emphasize more**")
            st.caption("Verified skills already in your resume that match this job but are currently secondary.")
            st.write(", ".join(rec.emphasize))

        if rec.phrasing_suggestions:
            st.markdown("**Reword existing content**")
            for item in rec.phrasing_suggestions:
                st.write(f"- {item}")

        if rec.missing_skills:
            st.markdown("**Missing from your resume — do not add unless genuinely true**")
            rows = [
                {"Skill": g.skill, "Estimated score impact": f"+{g.estimated_score_gain} pts"}
                for g in rec.missing_skills
            ]
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        if rec.unscored_gaps:
            st.markdown("**Other gaps identified (score impact not calculable)**")
            for item in rec.unscored_gaps:
                st.write(f"- {item}")


def section_cover_note(result) -> None:
    st.subheader("Cover Note")
    note = result.cover_note

    if note.status == "skipped":
        st.info(note.reason)
        return
    if note.status == "unavailable":
        st.warning(note.reason)
        return
    if note.status == "rejected":
        st.error("This note failed factual validation and needs review before use:")
        for failure in note.validation.failures:
            st.write(f"- {failure}")

    with st.container(border=True):
        st.text_area(
            "Cover note", note.text, height=320,
            key=f"note_{note.job_id}", label_visibility="collapsed",
        )
    st.caption(
        f"{note.validation.word_count} words · quality {note.score}/100"
        + (" · revised" if note.revised else "")
    )

    if note.critique_issues or note.validation.warnings or note.used_company_facts:
        with st.expander("Details"):
            if note.used_company_facts:
                st.markdown("**Company facts used**")
                for fact in note.used_company_facts:
                    st.write(f"- {fact}")
            if note.critique_issues:
                st.markdown("**Critique issues**")
                for issue in note.critique_issues:
                    st.write(f"- {issue}")
            if note.validation.warnings:
                st.markdown("**Validation warnings**")
                for warning in note.validation.warnings:
                    st.write(f"- {warning}")


def section_opportunities(result) -> None:
    st.subheader("Discover More Opportunities")
    discovery: DiscoveryResult = result.discovery

    if discovery.status != "ok":
        st.info(f"Not available ({discovery.status}). {discovery.reason}")
        return

    st.caption(
        f"{discovery.jobs_deduplicated} unique roles analyzed; ordered by priority "
        f"first and match score second, so a higher score can still appear "
        f"below a lower one — showing the top {len(discovery.opportunities)}."
    )

    ELIGIBILITY_FLAG = {"ineligible": "eligibility blocked", "uncertain": "eligibility uncertain"}

    for opp in discovery.opportunities:
        header = f"{opp.score.total}/100 · {opp.priority} — {opp.job.title} at {opp.job.company}"
        flag = ELIGIBILITY_FLAG.get(opp.eligibility_status)
        if flag:
            header += f" · {flag}"
        with st.expander(header):
            _priority_badge(opp.priority)
            st.caption(opp.priority_reason)
            st.write(opp.why_match)

            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Matches**")
                for skill in opp.matched_skills or ["—"]:
                    st.write(f"- {skill}")
            with cols[1]:
                st.markdown("**Gaps**")
                for skill in opp.missing_skills or ["—"]:
                    st.write(f"- {skill}")

            if opp.eligibility_status == "ineligible":
                st.error(opp.eligibility_explanation or "Eligibility blocker identified for this role.")
            elif opp.eligibility_status == "uncertain" and opp.eligibility_warnings:
                st.warning(opp.eligibility_warnings[0])

            if opp.job.url:
                st.link_button("Open job listing", opp.job.url)

            if st.button("Analyze this job instead", key=f"deep_{opp.job.job_id}"):
                st.session_state["deep_analysis_request"] = {
                    "company": opp.job.company, "title": opp.job.title,
                    "jd_text": opp.job.jd_text, "job_url": opp.job.url,
                    "location": opp.job.location,
                }
                st.rerun()

            with st.expander("Score breakdown"):
                st.code(
                    f"semantic (raw {opp.score.semantic_raw:.3f} -> normalised {opp.score.semantic_normalized:.2f})\n"
                    f"skill coverage   {opp.score.skill_coverage:.2f}\n"
                    f"eligibility      {opp.score.experience_match:.2f}\n"
                    f"role relevance   {opp.score.role_relevance:.2f}\n"
                    f"total = {opp.score.formula} = {opp.score.total}/100"
                )


def section_roadmap(result) -> None:
    roadmap = result.discovery.roadmap
    if roadmap.status != "ok":
        return
    st.subheader("Skill Gap Roadmap")
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
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def section_performance(result) -> None:
    with st.expander("Performance & diagnostics"):
        data = result.metrics.as_dict()
        st.code(result.metrics.render_text())
        cols = st.columns(5)
        cols[0].metric("Gemini calls", data["llm_calls"])
        cols[1].metric("Search calls", data["search_calls"])
        cols[2].metric("Embedding calls", data["embedding_calls"])
        cols[3].metric("Cache hits", data["cache_hits"])
        cols[4].metric("Cache misses", data["cache_misses"])
        if result.notes:
            st.markdown("**What was skipped, and why**")
            for note in result.notes:
                st.write(f"- {note}")


def section_tracker(result) -> None:
    st.subheader("Tracker")
    note = result.cover_note
    priority = result.fit.priority if result.fit else ""

    if st.button("Save this analysis"):
        outcome = store.save_application(
            job_id=result.job_id, company=result.company, title=result.title,
            cover_note=note.text, match_score=float(result.match_score()),
            priority=priority, job_url=result.job_url,
        )
        st.success(f"{outcome['action'].capitalize()} application #{outcome['id']}.")

    rows = store.list_applications()
    if not rows:
        st.caption("No saved applications yet.")
        return

    frame = pd.DataFrame(rows)
    with st.expander(f"Saved applications ({len(rows)})"):
        statuses = ["All"] + sorted(frame["status"].dropna().unique().tolist())
        chosen = st.selectbox("Filter by status", statuses)
        view = frame if chosen == "All" else frame[frame["status"] == chosen]
        display_cols = [c for c in ["id", "job_id", "company", "title", "match_score", "priority", "status", "date_applied"] if c in view.columns]
        st.dataframe(view[display_cols], width='stretch', hide_index=True)

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

def render_dashboard() -> None:
    st.title("AI Job Opportunity Intelligence Agent")
    st.caption("Upload a resume and a job posting to get a match score, resume changes, and a grounded cover note.")

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
        jd_text_input = st.text_area("Job description (paste text)", height=160, value=(pending or {}).get("jd_text", ""))
        jd_pdf_file = st.file_uploader("...or upload the job description as a PDF instead", type=["pdf"], key="jd_pdf_uploader")
        submitted = st.form_submit_button("Analyze")

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

        # The JD can come from pasted text or an uploaded PDF. If both are
        # given, the PDF wins -- it's the more deliberate input -- and the
        # user is told so explicitly rather than leaving it ambiguous which
        # one the analysis actually used.
        jd_text = jd_text_input
        if jd_pdf_file is not None:
            try:
                jd_pdf_text = extract_pdf_text(jd_pdf_file)
            except Exception as exc:
                st.error(f"Could not read the job description PDF ({type(exc).__name__}). Try re-exporting it or paste the text instead.")
                return
            if len(jd_pdf_text.strip()) < 20:
                st.error("No readable text was extracted from the job description PDF. It may be a scanned image — try pasting the text instead.")
                return
            jd_text = jd_pdf_text
            if jd_text_input.strip():
                st.info("Both a pasted job description and a PDF were provided — using the text extracted from the uploaded PDF.")

        seen = st.session_state.setdefault("session_companies", [])
        if company not in seen:
            seen.append(company)

        with st.status("Running analysis...", expanded=False) as status:
            try:
                result = run_pipeline(
                    resume_text=resume_text, company=company, title=title, jd_text=jd_text,
                    job_url=job_url, location=location, session_companies=seen,
                    settings=SETTINGS,
                )
                status.update(label="Analysis complete", state="complete")
            except Exception as exc:
                status.update(label="Analysis failed", state="error")
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
    section_profile(result)
    section_job_fit(result)
    section_resume_changes(result)
    section_cover_note(result)
    section_opportunities(result)
    section_roadmap(result)
    section_performance(result)
    section_tracker(result)


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _render_sidebar_status()
    render_dashboard()


main()
