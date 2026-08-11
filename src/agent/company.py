"""Company research producing VERIFIED FACTS WITH PROVENANCE.

Two failure modes in the original implementation are addressed here:

1. Hallucination. The old drafting prompt said "reference specific company
   facts" unconditionally, so when research returned nothing the model
   invented funding rounds and products to comply.
2. Wrong entity. The query "<company> company funding size recent news"
   returns results about a DIFFERENT organisation with a similar name, and
   nothing checked. That output is sourced, confident, and wrong.

Facts are only released downstream when entity_confirmed is True.
"""
from __future__ import annotations

import logging
from typing import List

from . import cache
from .clients import Clients, LLMUnavailable
from .config import Settings
from .identity import content_hash
from .schemas import CompanyResearch, VerifiedFact, _FactExtraction
from .telemetry import BudgetExceeded, RunMetrics

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are verifying web search results about a specific employer.

TARGET COMPANY: {company}
ROLE BEING APPLIED FOR: {title}

Below are raw web search snippets. Many searches return results about a
DIFFERENT organisation that shares or resembles the name.

Your tasks, in order:

1. ENTITY CHECK. Decide whether these snippets clearly describe the target
   company named above. Set entity_confirmed=true ONLY if you are confident.
   If the snippets are about a similarly named but different organisation,
   are too generic to attribute, or are contradictory, set it to FALSE.
   When in doubt, choose FALSE. A wrong fact is far worse than no fact.

2. FACT EXTRACTION. If and only if entity_confirmed is true, list statements
   that are directly supported by a snippet. Every fact must carry the
   source_url and source_title of the snippet it came from. Do not merge,
   extrapolate, or add background knowledge you have about this company from
   anywhere other than these snippets. Set confidence to "high" only when a
   snippet states the fact plainly.

3. SUMMARY. A neutral two-sentence summary built strictly from those facts.
   Empty string if there are no facts.

SNIPPETS:
{snippets}
"""


def _format_snippets(results: List[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] title: {r.get('title','')}\n"
            f"    url: {r.get('url','')}\n"
            f"    content: {r.get('content','')}\n"
        )
    return "\n".join(lines)


def research_company(
    company: str,
    title: str,
    clients: Clients,
    settings: Settings,
    metrics: RunMetrics,
) -> CompanyResearch:
    """Search + verify. Never raises; degrades to an empty, honest result."""
    company = (company or "").strip()
    if not company:
        return CompanyResearch(
            company="", status="skipped",
            reason="No company name supplied, so no research was attempted.",
        )

    key = cache.make_key("company_research", settings.chat_model, content_hash(company, title))

    def compute() -> CompanyResearch:
        try:
            query = f'"{company}" company overview products what it does'
            results = clients.search.search(query)
        except BudgetExceeded as exc:
            metrics.note(f"Company research: {exc}")
            return CompanyResearch(company=company, status="unavailable", reason=str(exc))

        if not results:
            return CompanyResearch(
                company=company, status="no_results",
                reason="Web search returned no usable results for this company.",
            )

        prompt = EXTRACTION_PROMPT.format(
            company=company, title=title or "(not specified)",
            snippets=_format_snippets(results),
        )
        try:
            extraction = clients.llm.structured(prompt, _FactExtraction)
        except BudgetExceeded as exc:
            metrics.note(f"Company research: {exc}")
            return CompanyResearch(company=company, status="unavailable", reason=str(exc))
        except LLMUnavailable as exc:
            logger.warning("fact extraction failed: %s", exc)
            return CompanyResearch(
                company=company, status="unavailable",
                reason="Could not verify search results, so no company facts were used.",
            )

        if not extraction.entity_confirmed:
            return CompanyResearch(
                company=company, entity_confirmed=False, status="entity_unconfirmed",
                reason=(
                    "Search results could not be confirmed as belonging to this company, "
                    "so they were discarded. "
                    + (extraction.entity_reasoning or "")
                ).strip(),
            )

        # Provenance gate: a "fact" with no source URL is not verified.
        sourced = [f for f in extraction.facts if f.source_url and f.fact.strip()]
        dropped = len(extraction.facts) - len(sourced)
        if dropped:
            logger.info("dropped %d company facts lacking provenance", dropped)

        if not sourced:
            return CompanyResearch(
                company=company, entity_confirmed=True, status="no_results",
                reason="No individually sourced facts could be extracted from the results.",
            )

        return CompanyResearch(
            company=company,
            verified_facts=sourced,
            entity_confirmed=True,
            summary=extraction.summary or "",
            status="ok",
        )

    try:
        return cache.get_or_compute(
            key, compute, ttl_s=settings.company_research_ttl_s,
            metrics=metrics, enabled=settings.cache_enabled,
        )
    except Exception as exc:
        logger.exception("company research failed")
        return CompanyResearch(
            company=company, status="unavailable",
            reason=f"Company research unavailable ({type(exc).__name__}).",
        )
