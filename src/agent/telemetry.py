"""Per-run instrumentation: stage timings, API call counts, cache hits,
and the call budget.

Thread-safe by construction because the pipeline fans out across threads.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised when a run exhausts its allowance for a resource.

    Callers are expected to catch this and degrade gracefully rather than
    propagate it to the user as a crash.
    """

    def __init__(self, resource: str, limit: int):
        self.resource = resource
        self.limit = limit
        super().__init__(
            f"Per-run budget for {resource} exhausted (limit {limit}). "
            f"This stage was skipped to stay within API limits."
        )


@dataclass
class StageTiming:
    name: str
    seconds: float
    ok: bool = True
    detail: str = ""


@dataclass
class RunMetrics:
    """Collects everything the Performance panel displays."""

    max_llm_calls: int = 8
    max_embedding_calls: int = 3
    max_search_calls: int = 4

    llm_calls: int = 0
    embedding_calls: int = 0
    search_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    stages: List[StageTiming] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---------- budget ----------

    def _consume(self, resource: str, count: int, current: int, limit: int) -> None:
        if current + count > limit:
            raise BudgetExceeded(resource, limit)

    def charge_llm(self, count: int = 1) -> None:
        with self._lock:
            self._consume("LLM calls", count, self.llm_calls, self.max_llm_calls)
            self.llm_calls += count

    def charge_embedding(self, count: int = 1) -> None:
        with self._lock:
            self._consume("embedding calls", count, self.embedding_calls, self.max_embedding_calls)
            self.embedding_calls += count

    def charge_search(self, count: int = 1) -> None:
        with self._lock:
            self._consume("search calls", count, self.search_calls, self.max_search_calls)
            self.search_calls += count

    def remaining_llm(self) -> int:
        with self._lock:
            return max(0, self.max_llm_calls - self.llm_calls)

    # ---------- cache ----------

    def record_cache_hit(self, n: int = 1) -> None:
        with self._lock:
            self.cache_hits += n

    def record_cache_miss(self, n: int = 1) -> None:
        with self._lock:
            self.cache_misses += n

    # ---------- timing ----------

    @contextmanager
    def stage(self, name: str) -> Iterator[StageTiming]:
        """Time a pipeline stage. Records the timing even if the body raises,
        so a failed stage still shows up in the performance panel."""
        entry = StageTiming(name=name, seconds=0.0)
        start = time.perf_counter()
        try:
            yield entry
        except Exception as exc:
            entry.ok = False
            entry.detail = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry.seconds = time.perf_counter() - start
            with self._lock:
                self.stages.append(entry)

    def note(self, message: str) -> None:
        """Record a user-facing explanation of something that was skipped."""
        with self._lock:
            if message not in self.notes:
                self.notes.append(message)
        logger.info("run note: %s", message)

    def total_seconds(self) -> float:
        return time.perf_counter() - self._started

    # ---------- reporting ----------

    def as_dict(self) -> Dict[str, object]:
        with self._lock:
            stages = [
                {"stage": s.name, "seconds": round(s.seconds, 3), "ok": s.ok, "detail": s.detail}
                for s in self.stages
            ]
            return {
                "stages": stages,
                "total_seconds": round(self.total_seconds(), 3),
                "llm_calls": self.llm_calls,
                "embedding_calls": self.embedding_calls,
                "search_calls": self.search_calls,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "budget": {
                    "llm": f"{self.llm_calls}/{self.max_llm_calls}",
                    "embedding": f"{self.embedding_calls}/{self.max_embedding_calls}",
                    "search": f"{self.search_calls}/{self.max_search_calls}",
                },
                "notes": list(self.notes),
            }

    def render_text(self) -> str:
        """Plain-text performance table."""
        lines = []
        with self._lock:
            width = max([len(s.name) for s in self.stages] + [12])
            for s in self.stages:
                flag = "" if s.ok else "  (failed)"
                lines.append(f"{s.name.ljust(width)}  {s.seconds:>6.2f}s{flag}")
        lines.append("-" * (width + 12))
        lines.append(f"{'Total'.ljust(width)}  {self.total_seconds():>6.2f}s")
        return "\n".join(lines)


def new_run(settings) -> RunMetrics:
    return RunMetrics(
        max_llm_calls=settings.max_llm_calls_per_run,
        max_embedding_calls=settings.max_embedding_calls_per_run,
        max_search_calls=settings.max_search_calls_per_run,
    )
