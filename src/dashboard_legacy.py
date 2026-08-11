"""LEGACY dashboard, preserved verbatim as a fallback.

This is the original src/dashboard.py, wrapped in render_legacy_dashboard()
so the new dashboard can call it when USE_NEW_PIPELINE is false. Its
behaviour is unchanged on purpose: it is the known-good path while the v2
architecture is being validated.
"""
import sqlite3

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from live_pipeline import run_live_pipeline


def render_legacy_dashboard() -> None:

    st.title("Job Application Agent Dashboard")

    st.subheader("Run Pipeline Live")
    st.caption(
        "Upload your resume and paste a job description to run the full "
        "5-agent pipeline (gap analysis → company research → cover note → "
        "critique → rewrite) live. Takes 15-30 seconds."
    )

    jd_input_mode = st.radio("Job description input", ["Paste text", "Upload PDF"], horizontal=True)

    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        preview_resume_file = st.file_uploader("Resume (PDF) — for preview", type=["pdf"], key="preview_resume")
        if preview_resume_file:
            preview_reader = PdfReader(preview_resume_file)
            preview_resume_text = "".join(page.extract_text() or "" for page in preview_reader.pages)
            with st.expander("Preview extracted resume text (check it's readable before running)"):
                st.text(preview_resume_text[:1500] + ("..." if len(preview_resume_text) > 1500 else ""))

    with preview_col2:
        if jd_input_mode == "Upload PDF":
            preview_jd_file = st.file_uploader("Job description (PDF) — for preview", type=["pdf"], key="preview_jd")
            if preview_jd_file:
                preview_jd_reader = PdfReader(preview_jd_file)
                preview_jd_text = "".join(page.extract_text() or "" for page in preview_jd_reader.pages)
                with st.expander("Preview extracted JD text (check it's readable before running)"):
                    st.text(preview_jd_text[:1500] + ("..." if len(preview_jd_text) > 1500 else ""))

    st.caption(
        "Each run makes ~4-5 Gemini calls and 1 Tavily search call. "
        f"Runs so far this session: {st.session_state.get('run_count', 0)}."
    )

    with st.form("live_run_form"):
        col1, col2 = st.columns(2)
        with col1:
            live_company = st.text_input("Company")
            live_title = st.text_input("Job title")
        with col2:
            resume_file = st.file_uploader("Resume (PDF)", type=["pdf"])

        if jd_input_mode == "Paste text":
            live_jd = st.text_area("Job description", height=150)
            jd_file = None
        else:
            jd_file = st.file_uploader("Job description (PDF)", type=["pdf"], key="jd_pdf")
            live_jd = ""

        submitted = st.form_submit_button("Run pipeline")

    if submitted:
        if not resume_file:
            st.error("Upload a resume PDF first.")
        elif jd_input_mode == "Paste text" and (not live_jd or len(live_jd.strip()) < 20):
            st.error("Paste a job description (at least a few sentences).")
        elif jd_input_mode == "Upload PDF" and not jd_file:
            st.error("Upload a job description PDF.")
        elif not live_company or not live_title:
            st.error("Fill in company and job title.")
        else:
            try:
                reader = PdfReader(resume_file)
                resume_text = "".join(page.extract_text() or "" for page in reader.pages)

                if jd_input_mode == "Upload PDF":
                    jd_reader = PdfReader(jd_file)
                    live_jd = "".join(page.extract_text() or "" for page in jd_reader.pages)
                    if len(live_jd.strip()) < 20:
                        st.error("Could not extract readable text from the JD PDF.")
                        st.stop()

                with st.status("Running pipeline...", expanded=True) as status:
                    st.write("Extracting resume text... done.")
                    st.write("Running gap analysis, company research, cover note, critique, and rewrite...")
                    result = run_live_pipeline(resume_text, live_company, live_title, live_jd)
                    status.update(label="Pipeline complete", state="complete")

                st.session_state["run_count"] = st.session_state.get("run_count", 0) + 1

                st.subheader("Gap Analysis")
                st.write(result["gap_analysis"])

                st.subheader("Company Research")
                st.write(result["company_research"])

                st.subheader("Critique")
                st.write(f"Score: {result['critique_score']}/100 — {result['critique_issues']}")

                st.session_state["last_live_result"] = {
                    "company": live_company,
                    "title": live_title,
                    "cover_note": result["final_cover_note"],
                }
                st.caption("Cover note for this run is shown in the 'Cover Note' section below.")

            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

    if "last_live_result" in st.session_state:
        st.info(
            f"Last live result for **{st.session_state['last_live_result']['company']} — "
            f"{st.session_state['last_live_result']['title']}** is shown above and not yet "
            f"saved to your Applications table below."
        )
        if st.button("Save this result to Applications"):
            save_conn = sqlite3.connect("data/tracker.db")
            new_company = st.session_state["last_live_result"]["company"].strip()
            new_title = st.session_state["last_live_result"]["title"].strip()
            # Case-insensitive dedup check, not just exact match, to catch
            # "tech intern" vs "Tech Intern" vs " Tech Intern " style near-duplicates.
            existing = save_conn.execute(
                "SELECT id FROM applications WHERE LOWER(TRIM(company)) = LOWER(?) "
                "AND LOWER(TRIM(title)) = LOWER(?)",
                (new_company, new_title),
            ).fetchone()
            if existing:
                st.warning(
                    f"An application for '{new_company}' — '{new_title}' already exists "
                    f"(ID {existing[0]}). Not saving a duplicate. Update its status below instead, "
                    f"or change the company/title if this is genuinely a different role."
                )
            else:
                save_conn.execute(
                    "INSERT INTO applications (company, title, match_score, status, cover_note) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        new_company,
                        new_title,
                        None,  # live pipeline doesn't compute a match_score; only the old batch matcher does
                        "not_applied",
                        st.session_state["last_live_result"]["cover_note"],
                    ),
                )
                save_conn.commit()
                st.success("Saved to Applications table below.")
                del st.session_state["last_live_result"]
                st.rerun()
            save_conn.close()

    st.divider()

    conn = sqlite3.connect("data/tracker.db")
    df = pd.read_sql("SELECT * FROM applications", conn)

    st.subheader("Saved Applications (from tracker.db)")
    status_filter = st.selectbox("Filter by status", ["All"] + df["status"].unique().tolist())
    if status_filter != "All":
        df_display = df[df["status"] == status_filter]
    else:
        df_display = df

    st.dataframe(df_display[["id", "company", "title", "match_score", "status", "date_applied"]])

    st.subheader("Update Status")
    app_id = st.selectbox("Select application ID", df["id"].tolist())
    new_status = st.selectbox("New status", ["not_applied", "applied", "interview", "rejected", "ghosted"])
    if st.button("Update"):
        conn.execute("UPDATE applications SET status = ? WHERE id = ?", (new_status, app_id))
        conn.commit()
        st.success(f"Updated ID {app_id} to {new_status}")
        st.rerun()

    st.subheader("Cover Note")
    if "last_live_result" in st.session_state:
        st.caption(
            f"Showing the live result for **{st.session_state['last_live_result']['company']} — "
            f"{st.session_state['last_live_result']['title']}** (not yet saved). "
            f"Save it above, or select a different saved application below to view its note instead."
        )
        st.text_area("Live cover note", st.session_state["last_live_result"]["cover_note"], height=300)
    else:
        selected_row = df[df["id"] == app_id]
        if not selected_row.empty:
            st.caption(f"Showing the saved note for application ID {app_id}.")
            st.text_area("Saved cover note", selected_row.iloc[0]["cover_note"], height=300)

    st.subheader("Status Breakdown (saved applications only)")
    status_counts = df["status"].value_counts()
    st.bar_chart(status_counts)

    conn.close()
