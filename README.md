# AI Job Opportunity Intelligence Agent

Understands a candidate's profile from their resume, evaluates fit against a specific job,
researches the employer using **verified, sourced facts only**, generates a grounded cover
note, discovers other live opportunities worth applying to, and prioritises where to spend
limited application time.

It answers four questions:

1. Am I a good fit for this job?
2. What are my actual gaps?
3. Should I apply to this job?
4. What other jobs should I apply to instead or in addition?

---

## Architecture

```
                          Resume (parsed once, cached by SHA-256)
                                        |
                                 Candidate Profile          [1 LLM call, cached]
                                        |
        +-------------------------------+-------------------------------+
        |                               |                               |
   Gap Analysis                 Company Research              Opportunity Discovery
   [1 LLM call]              [1 search + 1 LLM call]     [0 LLM + 1 embedding batch
                                                            + K concurrent HTTP]
        |                               |                               |
        +-------------------------------+-------------------------------+
                                        | join
                                   Cover Note                 [1 LLM call, allow-listed]
                                        |
                            Critique + Revision               [1 LLM call, merged]
                                        |
                           Deterministic Validation           [0 LLM calls]
                                        |
                                 Final Results
```

The three middle branches run in a single LangGraph superstep on separate threads.

### Cost per run

| | Cold | Warm (cached profile + company) |
|---|---|---|
| Gemini calls | 5 | 2 |
| Sequential LLM hops | 3 | 3 |
| Search calls | 1 | 0 |
| Embedding calls | 1 batch | 1 batch |

Opportunity Discovery costs **no additional LLM call** — role families come from the
candidate-profile call, and ranking is fully deterministic.

---

## Opportunity Score (0–100)

```
Score = 45% x rank-normalised resume/JD semantic similarity
      + 25% x required-skill coverage
      + 15% x experience / eligibility match
      + 15% x role relevance
```

Weights are configurable (`WEIGHT_*`). Raw cosine similarity is retained and displayed for
transparency, but rank-normalised within the retrieved pool before weighting: Gemini
resume/JD similarities cluster around 0.65–0.85, so the raw value provides almost no
separation between candidates.

**Application priority** is assigned from evidence, not from the score alone. `Apply Now`
requires ≥70% skill coverage **and** no hard eligibility blocker **and** ≥2 supporting
resume evidence items — a job cannot reach the top tier on similarity alone.

---

## Factuality safeguards

1. **Entity relevance gate.** Search results are discarded unless they can be confirmed to
   concern the target company. A result about a similarly named organisation is worse than
   no result at all, because it is sourced and confident.
2. **Provenance requirement.** A "fact" without a source URL is dropped.
3. **Closed allow-list prompting.** The generator receives verified facts as the only
   permitted company claims. When the list is empty it is explicitly forbidden from making
   any company-specific claim.
4. **Deterministic validation.** After generation, without any LLM: cross-company
   contamination (a note for Company B naming Company A is rejected), unsourced company
   claim markers, banned filler phrases, word count, and target-company/role presence.

---

## Setup

```bash
git clone https://github.com/skapoor3be23/Job-Agent-.git
cd Job-Agent-
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in your keys
streamlit run src/dashboard.py
```

Required credentials: `GOOGLE_API_KEY`, `TAVILY_API_KEY`, and — for Opportunity Discovery —
`ADZUNA_APP_ID` / `ADZUNA_APP_KEY`. The app runs with any subset and reports what was skipped.

### Streamlit Cloud

Point the app at `src/dashboard.py`, then add the same keys under
**App settings → Secrets** in TOML form:

```toml
GOOGLE_API_KEY = "..."
TAVILY_API_KEY = "..."
ADZUNA_APP_ID  = "..."
ADZUNA_APP_KEY = "..."
USE_NEW_PIPELINE = "true"
```

### Switching pipelines

Set `USE_NEW_PIPELINE=false`, or use the radio button in the sidebar. The original
pipeline (`src/live_pipeline.py`, `src/dashboard_legacy.py`) is preserved unchanged.

### Tests

```bash
python -m pytest tests/ -q     # 40 tests, fully mocked, no network or API keys
```

---

## Layout

| Path | Role |
|---|---|
| `src/agent/` | v2 architecture (config, schemas, graph, ranking, discovery, …) |
| `src/dashboard.py` | Streamlit entry point with the pipeline switch |
| `src/dashboard_legacy.py` | Original dashboard, preserved as fallback |
| `src/live_pipeline.py`, `src/pipeline.py` | Original v1 pipeline, unchanged |
| `src/agent/store.py` | Tracker keyed by `job_id` (migrates the old schema in place) |
| `tests/` | Mocked test suite |

## Known limitations

- Skill matching uses a curated vocabulary (`src/agent/skills.py`); skills outside it are
  not counted toward coverage. Extend the vocabulary for other domains.
- Adzuna coverage of Indian internship listings is uneven; discovery quality depends on it.
- The entity gate is conservative by design and will sometimes discard correct results for
  small or newly founded companies, yielding a company-fact-free cover note.
- Match score for the *current* job substitutes an evidence ratio for the semantic
  component, since rank-normalisation needs a comparison pool.
