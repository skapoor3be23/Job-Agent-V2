# Project Audit — Job Agent V2

Audit basis: the complete accessible checkout at the repository root, inspected on 12 August 2026. This document records evidence from the code only; it intentionally contains no credential values. The only file created during this audit is this report.

## Executive summary

Job Agent V2 is a Streamlit application that accepts a text-based PDF resume and a target job description, uses Gemini to construct a candidate profile and evaluate the job, retrieves and verifies company information with Tavily + Gemini, produces a guarded cover note, discovers Adzuna opportunities, ranks them deterministically, and persists selected applications in SQLite.

The v2 workflow is a LangGraph pipeline. After profiling, gap analysis, company research, and discovery are separate concurrent branches; cover-note generation then joins their results. Scoring, eligibility, explanations, roadmap counts, and cover-note validation are intentionally deterministic. The test suite was run: **39 passed, 1 failed**. The failure is reproducible and described in Known issues.

## Complete project structure

The application-owned files currently present are:

```text
Job-Agent-V2/
├── .env                         # local configuration/secrets file; values not inspected or reproduced
├── .env.example                 # public configuration template
├── .gitignore
├── README.md
├── requirements.txt
├── PROJECT_AUDIT.md              # this audit
├── data/
│   └── tracker.db                # SQLite application tracker
├── src/
│   ├── dashboard.py              # v2 Streamlit entry point
│   ├── dashboard_legacy.py       # v1 UI fallback (depends on missing live_pipeline)
│   └── agent/
│       ├── __init__.py
│       ├── adzuna.py
│       ├── cache.py
│       ├── clients.py
│       ├── company.py
│       ├── config.py
│       ├── cover_note.py
│       ├── critique.py
│       ├── discovery.py
│       ├── eligibility.py
│       ├── gap.py
│       ├── graph.py
│       ├── identity.py
│       ├── ranking.py
│       ├── resume.py
│       ├── roadmap.py
│       ├── schemas.py
│       ├── skills.py
│       ├── store.py
│       └── telemetry.py
└── tests/
    ├── conftest.py
    ├── mocks.py
    ├── test_cover_note.py
    ├── test_discovery.py
    ├── test_identity.py
    ├── test_pipeline.py
    └── test_store.py
```

Generated/local directories also exist: `.venv/`, `.pytest_cache/`, and Python `__pycache__/` directories. Their complete installed-package / bytecode contents are environment-generated and are not source artifacts. No `pyproject.toml`, lockfile, Docker configuration, CI configuration, `live_pipeline.py`, or `pipeline.py` exists in this checkout.

## Dependencies

`requirements.txt` is the only dependency manifest found:

| Dependency | Declared version | Used for |
|---|---:|---|
| streamlit | `>=1.32` | Dashboard, session state, secrets, resource cache |
| pandas | `>=2.0` | Tracker and roadmap tables |
| numpy | `>=1.24` | Declared but no direct import found in project source |
| pypdf | `>=4.0` | PDF text extraction |
| langchain-google-genai | `>=2.0` | Gemini chat and embeddings adapters |
| langgraph | `>=0.2` | v2 concurrent pipeline graph |
| pydantic | `>=2.6` | Structured LLM output and data models |
| tavily-python | `>=0.3` | Company web search |
| requests | `>=2.31` | Adzuna HTTP retrieval |
| python-dotenv | `>=1.0` | Local `.env` loading |

Python standard-library dependencies include `sqlite3`, `hashlib`, `threading`, `concurrent.futures`, `dataclasses`, `typing`, `re`, `json`, `logging`, and URL parsing.

## Architecture and end-to-end data flow

```text
PDF resume upload
  -> pypdf extraction (dashboard / resume)
  -> candidate profile (Gemini structured output, reconciled with skill vocabulary)
  -> [parallel LangGraph superstep]
       -> gap analysis (Gemini; deterministic skills fallback)
       -> company research (Tavily search -> Gemini entity/fact verification)
       -> discovery (Adzuna -> eligibility -> Gemini embeddings -> deterministic ranking/roadmap)
  -> cover-note Gemini generation using only resume/JD/verified-fact allow-lists
  -> Gemini critique/revision
  -> deterministic note validation
  -> Streamlit results, optional SQLite tracker persistence
```

Detailed requested path:

1. **Resume upload:** `dashboard.py` accepts a PDF only, calls `resume.extract_pdf_text`, and rejects unreadable or under-50-character text. `resume.parse_resume` is a reusable cached path/file parser but the v2 dashboard currently calls `extract_pdf_text` directly.
2. **Resume parsing → candidate profile:** `build_candidate_profile` makes one structured Gemini call for `CandidateProfile`; it caches by normalized resume hash, removes model skills not evidenced by the resume/vocabulary, and supplements omitted vocabulary skills. Budget/API failure falls back to vocabulary extraction with `student`, 0 years, and no role families.
3. **Job analysis / gap analysis:** `gap.analyze_gap` asks Gemini for evidenced matches, gaps, edits, and verdict; cache key includes model, resume, company/title/JD. Failure uses deterministic skill coverage.
4. **Eligibility:** discovery calls `eligibility.analyze_eligibility` for every retained Adzuna job before ranking. It keeps unsuitable jobs, exposing blockers instead of silently removing them.
5. **Opportunity discovery:** search queries come from profile role families (or up to two primary skills plus `intern`/`engineer`). Adzuna calls run concurrently, then deduplicate by deterministic `job_id`, drop empty descriptions/current job, and cap retained jobs.
6. **Ranking:** resume and retained JDs are embedded in one batch where available; ranking uses rank-normalized cosine similarity or lexical Jaccard fallback, skills, eligibility, and role relevance. Explanations derive from counts/evidence, not an LLM.
7. **Gap roadmap:** counts missing recognized technical skills across discovered JDs and reports individual skills and best two-skill combination that crosses 70% coverage.
8. **Company research:** Tavily returns snippets; Gemini must confirm the entity and extract only URL-backed facts. Unconfirmed or unsourced facts are discarded.
9. **Cover note:** Gemini receives target JD, candidate evidence, and only usable verified company facts. A critique/revision structured Gemini call follows. Deterministic checks reject empty notes, other session-company names, and company claims when no facts exist; they warn on missing company/title, filler, and length.
10. **Dashboard:** presents the profile, analysed job, recommendation, note, discoveries, roadmap, metrics, and tracker. Selecting a discovered job carries its fields to a fresh analysis request.

## Important module inventory

| Module | Purpose; inputs → outputs | Key functions/classes; internal dependencies |
|---|---|---|
| `dashboard.py` | v2 UI; PDF/form/session state → rendered analysis and tracker writes. | `render_new_dashboard`, eight `section_*` renderers; uses `graph`, `resume`, `ranking`, `skills`, `store`, `config`, `clients`. |
| `dashboard_legacy.py` | Preserved v1 UI; PDF/text form → legacy result/SQLite UI. | `render_legacy_dashboard`; imports missing `live_pipeline.run_live_pipeline`. |
| `agent/config.py` | Resolves config: Streamlit secrets → environment → defaults. | `Settings`, `ScoreWeights`, `get_settings`, credential helpers; used by all configurable services. |
| `agent/schemas.py` | Pydantic contracts exchanged between LLM/pipeline/UI. | `CandidateProfile`, `GapAnalysis`, `CompanyResearch`, `CoverNote`, `JobPosting`, `Opportunity`, roadmap/discovery models. |
| `agent/clients.py` | Budgeted/time-accounted wrappers for Gemini chat/embeddings and Tavily. | `LLMClient`, `SearchClient`, `EmbeddingClient`, `Clients`, robust `parse_structured`; depends on config/cache/telemetry. |
| `agent/telemetry.py` | Thread-safe per-run resource budgets, timings, counters, notes. | `RunMetrics`, `BudgetExceeded`, `new_run`; consumed by clients and UI. |
| `agent/cache.py` | In-memory TTL cache, backed by `st.cache_resource` in Streamlit. | `get_or_compute`, `make_key`; used by resume, gap, company, embeddings. |
| `agent/identity.py` | Stable content/job IDs and normalization. | `make_job_id`, `resume_hash`, `content_hash`; used by caches, Adzuna and SQLite. |
| `agent/resume.py` | PDF text parsing and factual candidate profile. | `extract_pdf_text`, `parse_resume`, `build_candidate_profile`; depends on pypdf, LLM client, skills/cache. |
| `agent/skills.py` | Curated canonical vocabulary and deterministic matching. | `extract_skills`, `coverage`, `experience_match`, `role_relevance`, `lexical_similarity`; used by profile fallback, gap, ranking, dashboard. |
| `agent/gap.py` | Structured resume/JD gap analysis with safe fallback. | `analyze_gap`; depends on LLM/cache/skills/schemas. |
| `agent/company.py` | Search/verify employer facts with provenance gate. | `research_company`; depends on Tavily wrapper, Gemini, cache, schemas. |
| `agent/cover_note.py` | Builds constrained prompt and validates resulting note without LLM. | `build_generation_prompt`, `validate_cover_note`; depends on schemas/config/identity. |
| `agent/critique.py` | Scores and optionally revises cover note in one structured LLM call. | `critique_and_revise`, `_weighted_score`; depends on clients/cover-note input material. |
| `agent/adzuna.py` | Adzuna adapter; raw records → normalized `JobPosting`s. | `fetch_one`, concurrent `fetch_many`, `_to_posting`; depends on requests/config/identity. |
| `agent/eligibility.py` | Deterministic suitability classification for discovered postings. | `EligibilityResult`, `analyze_eligibility`; depends on `CandidateProfile`/`JobPosting`. |
| `agent/ranking.py` | Pure opportunity scoring, priority, evidence and explanations. | `cosine_similarity`, `rank_normalize`, `assign_priority`, `score_opportunities`; depends on skills/config/schemas. |
| `agent/discovery.py` | Orchestrates query construction, retrieval, eligibility, embeddings, ranking, roadmap. | `build_search_queries`, `discover_opportunities`; depends on Adzuna, eligibility, clients, ranking, roadmap. |
| `agent/roadmap.py` | Computes most valuable missing skills from retrieved jobs. | `build_roadmap`; depends on skills/schemas only. |
| `agent/graph.py` | LangGraph orchestration and public `PipelineResult`. | node functions, `build_graph`, `run_pipeline`; integrates all analysis modules. |
| `agent/store.py` | SQLite schema creation/migration and application tracking. | `init_db`, `save_application`, list/get/update/count; depends on identity. |

## External services

| Service | Integration/use | Credential/config |
|---|---|---|
| Google Gemini | `ChatGoogleGenerativeAI` for profile, gap, fact extraction, note, critique; `GoogleGenerativeAIEmbeddings` for resume/JD vectors. | `GOOGLE_API_KEY`; `CHAT_MODEL`, `EMBEDDING_MODEL`, `CHAT_TEMPERATURE` |
| Tavily | Company overview/product search, up to configured result count. | `TAVILY_API_KEY`; `TAVILY_MAX_RESULTS` |
| Adzuna | Live job search at `api.adzuna.com`, country path and concurrent query retrieval. | `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`; `ADZUNA_COUNTRY` |
| Streamlit | Web UI, session state, secrets precedence, resource cache. | Streamlit Cloud secrets are supported; no additional variable required. |
| SQLite | Local `data/tracker.db` application tracker. | No credential. |

## Configuration and environment variables (names only)

Credentials: `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`.

Feature/pipeline: `USE_NEW_PIPELINE`, `ALLOW_LEGACY_FALLBACK`.

Gemini settings: `CHAT_MODEL`, `EMBEDDING_MODEL`, `CHAT_TEMPERATURE`, `EMBEDDING_BATCH_SIZE`.

Budgets: `MAX_LLM_CALLS_PER_RUN`, `MAX_EMBEDDING_CALLS_PER_RUN`, `MAX_SEARCH_CALLS_PER_RUN`.

Discovery: `DISCOVERY_ENABLED`, `DISCOVERY_MAX_JOBS`, `DISCOVERY_MAX_QUERIES`, `DISCOVERY_RESULTS_PER_QUERY`, `ADZUNA_COUNTRY`.

Scoring: `WEIGHT_SEMANTIC`, `WEIGHT_SKILL_COVERAGE`, `WEIGHT_EXPERIENCE`, `WEIGHT_ROLE_RELEVANCE` (validated to sum to 1.0).

Other: `CACHE_ENABLED`, `COVER_NOTE_MIN_WORDS`, `COVER_NOTE_MAX_WORDS`, `TAVILY_MAX_RESULTS`, `LOG_LEVEL`.

## Scoring, priority, eligibility, and retrieval logic

### Opportunity score

Default score, clamped to 0–100:

```text
100 × (0.45 × semantic_rank_normalized
     + 0.25 × technical_skill_coverage
     + 0.15 × eligibility_score
     + 0.15 × role_relevance)
```

Semantic similarity is cosine(resume embedding, JD embedding), then normalized by ascending rank within the current retrieved pool: lowest 0, highest 1, tied values receive average rank. If embeddings cannot be produced, word-token Jaccard similarity is substituted before ranking. Technical coverage ignores four soft skills (`Communication`, `Teamwork`, `Problem Solving`, `Excel`) and is 1.0 when no recognized technical requirements are present. Role relevance is the best token overlap of job title against profile role families.

For the **current manually analysed job**, `PipelineResult.match_score()` cannot rank-normalize without a pool, so it substitutes `matched_gap_requirements / (matched + missing)` (or coverage if denominator is zero) for semantic similarity, while retaining the other weights.

Priority does **not** use total score. `Apply Now` requires >=70% technical coverage, eligibility >=0.50, no blocker, and at least two supporting resume evidence items. High coverage with insufficient evidence is `Worth Applying`; lower coverage thresholds are 45% (`Worth Applying`) and 25% (`Stretch Opportunity`); blockers produce `Stretch Opportunity` only at >=70% coverage, otherwise `Low Priority`.

### Eligibility

`eligibility.py` classifies discovered jobs from title/JD markers and stated years:

- Senior/manager title + `student`/`intern`/`entry` candidate is a blocker.
- Explicit experience shortfall >=3 years is a blocker; a smaller shortfall is a warning.
- Intern/entry title with student/intern candidate is `eligible`; no blocker/warning with other titles is `likely_eligible`; warnings yield `uncertain`; blockers yield `ineligible`.

Ranking converts `eligible` to 1.0 and `ineligible` to 0 with a blocker; all other/non-present results become 0.5 without a blocker. This differs from `skills.experience_match`, used by the current-job score/UI recommendation, which checks broad JD junior/senior markers and has a more graduated score. The duplicate logic is a source of inconsistent outcomes.

### Discovery/retrieval

Role-family queries are unique, ordered, and capped; if none exist, two primary skills become `"<skill> intern"` for student/intern profiles, otherwise `"<skill> engineer"`. Adzuna requests one page per query concurrently (maximum six workers), then results are deduplicated by 16-character SHA-256-based job ID. Empty JDs and the current job ID are removed; the list is then capped. Ineligible jobs remain in the results for explanation. No other job boards or persistence of retrieved listings are implemented.

## Caching strategy

The cache is process/session memory only—no cache contents are written to `data/tracker.db`. Streamlit uses `st.cache_resource`; non-Streamlit execution falls back to a module dictionary. Entries have TTL and a re-entrant lock, with hit/miss metrics.

| Cached artifact | Key basis | TTL |
|---|---|---:|
| Stable-path PDF text (`parse_resume`) | file/path string | no expiration |
| Candidate profile | chat model + normalized resume hash | 24 hours (`job_analysis_ttl_s`) |
| Gap analysis | chat model + resume/company/title/JD hashes | 24 hours |
| Company research | chat model + company/title hash | 7 days |
| Each embedding vector | embedding model + content hash | 30 days |

The dashboard bypasses `parse_resume` and uses `extract_pdf_text` on the uploaded file, so uploads are not PDF-text cached by the implemented v2 UI. Cache keys do not include prompt-template versions or all configuration that can affect output (for example temperature), which can retain prior results until TTL after such a setting changes.

## LLM calls, budget, and resilience

Defaults are 8 LLM calls, 3 embedding calls, and 4 Tavily searches per run. All Gemini/Tavily/embedding calls go through wrappers that charge a thread-safe budget before invocation. Default cold v2 run is five Gemini calls: profile; gap; company fact extraction; cover note; critique/revision. Company research also makes one Tavily call. Discovery uses no additional LLM calls and aims for one embedding batch for resume + jobs; several batches are possible if `EMBEDDING_BATCH_SIZE` is exceeded.

The profile, gap, company, embeddings, and cover-note critique degrade rather than crash: keyword profile/gap fallbacks, no-facts company research, lexical ranking, unreviewed draft, and user-facing run notes. Structured LLM calls prefer provider-native structured output, then make a fallback call with JSON-schema instructions. The stated intent is one charged call; however the fallback occurs after a single budget charge and invokes the underlying client directly, so it is still one logical/budgeted operation but can be a second provider request after native structured invocation fails.

## Telemetry and performance

`RunMetrics` captures total time, each named stage's time/status/error detail, LLM/search/embedding call counts, cache hits/misses, resource budget use, and de-duplicated user-facing degradation notes. `RunMetrics` locks mutations for concurrent graph branches. The Performance dashboard renders a text timing table plus five counters and the three budgets. It is in-memory per run; there is no durable telemetry, tracing backend, analytics, or metrics export.

## Dashboard sections and data sources

| Section | Data source |
|---|---|
| Sidebar | `Settings` and boolean-only `credential_status()` |
| 1. Candidate Profile | `PipelineResult.profile` from `resume.build_candidate_profile` |
| 2. Current Job Analysis | `result.gap` from `gap`, company facts from `result.research` |
| 3. Application Recommendation | recomputes skills/experience/relevance from `result.gap_jd_text`, profile, title via `skills` + `ranking.assign_priority` |
| 4. Cover Note | `result.cover_note` from graph generation/critique/validation |
| 5. Discover More Opportunities | `result.discovery` from `discovery`/Adzuna/ranking |
| 6. Skill Gap Roadmap | `result.discovery.roadmap` from `roadmap.build_roadmap` |
| 7. Performance | `result.metrics` and graph/metrics notes |
| 8. Application Tracker | current result plus `store` reads/writes to `data/tracker.db` |

## Tests

Test infrastructure injects mocked LLM, Tavily, and embedding clients, so tests do not require network credentials. The checked suite contains 40 tests.

- `test_pipeline.py`: normal artifacts, expected cold-call count, parallel branches, company/gap failures, exhausted budget, cache reuse, empty JD behavior.
- `test_discovery.py`: rank normalization/ties, score ordering/explanations/formula, priority evidence threshold, malformed fields, roadmap counts/thin data, job-ID deduplication.
- `test_cover_note.py`: validation, factuality allow-list, entity gate, cross-company contamination, and full-pipeline note isolation.
- `test_identity.py`: ID normalization/uniqueness and resume hash stability.
- `test_store.py`: empty tracker, job-ID upsert behavior, persisted score/priority, schema migration, and status dating.

Executed command: `python -m pytest tests -q -p no:cacheprovider` using `.venv`. Result: **39 passed, 1 failed in 3.58s**. The failing test is `tests/test_discovery.py::test_senior_role_is_blocked_regardless_of_score` because it calls `score_opportunities()` without an eligibility-results map, and ranking uses its default `0.8, []` fallback rather than deriving a senior blocker.

## Known limitations and potential bugs

Evidence-backed findings, not speculative defects:

1. **Failing test / standalone ranking bypasses eligibility:** the test failure above shows `score_opportunities` does not enforce senior-role blocking unless the caller supplies precomputed `eligibility_results`. Discovery supplies it, but direct callers and the test do not. This contradicts the test name/expectation.
2. **Eligibility details are calculated but not exposed on each `Opportunity`:** `Opportunity` has `eligibility_status`, `eligibility_explanation`, and `eligibility_warnings`; `score_opportunities` does not populate them. The dashboard therefore shows only `blockers`, not the richer `eligible/likely_eligible/uncertain` result.
3. **Two eligibility systems can disagree:** discovery uses `eligibility.analyze_eligibility`, while current-job score/recommendation uses `skills.experience_match`. They inspect different fields/markers and convert uncertainty differently.
4. **Legacy fallback is broken in this checkout:** setting the sidebar to v1 imports `live_pipeline`, but `src/live_pipeline.py` is absent. README also references absent `src/live_pipeline.py` and `src/pipeline.py`. The v2 default path remains available.
5. **README test claim is stale/inaccurate:** it says 40 tests “fully mocked”; 40 tests exist but the observed suite has one failure.
6. **`ALLOW_LEGACY_FALLBACK` is loaded but not consulted by `dashboard.py`;** the sidebar always exposes the legacy choice.
7. **`numpy` is declared but has no direct source import.** It may be transitive support, but direct project use cannot be determined from this code.
8. **Adzuna usage is not represented in `RunMetrics`:** its calls are concurrent HTTP calls but not counted as search calls; only Tavily consumes/records the search budget. Therefore the Performance panel does not report total external retrieval calls and `MAX_SEARCH_CALLS_PER_RUN` cannot cap Adzuna query count.
9. **Retrieved job order is nondeterministic before cap:** `fetch_many` appends futures in completion order, then discovery slices to `DISCOVERY_MAX_JOBS`. When more than the cap is returned, network timing can alter which jobs reach scoring.
10. **Search/LLM company-fact verification is cautious, not ground-truth verification:** it validates claims against Tavily snippets via Gemini and requires URL provenance, but does not fetch/cross-check source pages or validate source authority.
11. **PDF support is text-layer only:** scanned/image-only resumes are rejected; there is no OCR fallback.
12. **Skill coverage is limited to the curated vocabulary:** recognized aliases can be broad (for example `GitHub` maps to Git; `OpenCV` maps to Computer Vision), while unlisted job requirements are invisible to deterministic coverage/roadmap.
13. **Possible cache staleness across behavioral config changes:** profile/gap/company cache keys include model/input hashes but omit temperature and prompt/config revision; PDFs are keyed by path rather than content when using `parse_resume` with a path.
14. **Cover-note validation is intentionally partial:** it cannot deterministically prove all candidate claims are resume-backed or all company claims are sourced when facts do exist; it relies on the prompt and LLM critique for those checks.
15. **Tracker status validation is absent in `store.update_status`:** UI offers `STATUSES`, but the storage function accepts any string and will date any status other than `not_applied` as applied.
16. **The `data/tracker.db` contents were not inspected for personal application data:** its schema/use is audited, but no user data is reproduced.

## Git status and recent changes

Unavailable. `git status --short` and `git log --oneline -8` both return `fatal: not a git repository (or any of the parent directories): .git`. There is no accessible `.git` directory at this project root, so current working-tree changes, commit history, branch, and recent project changes cannot be determined from this checkout.

