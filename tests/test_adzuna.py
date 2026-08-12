"""Adzuna retrieval determinism: which jobs end up first (and therefore
which survive any later cap) must depend on query order, never on which
concurrent HTTP request happened to complete first."""
import time

from agent import adzuna
from agent.config import Settings
from agent.schemas import JobPosting


def _job(label):
    return JobPosting(job_id=f"job-{label}", company=label, title="Role", jd_text="text", url=f"https://x.com/{label}")


def test_fetch_many_preserves_query_order_regardless_of_completion_latency(monkeypatch):
    """'first' is slow, 'second' and 'third' are fast. A completion-order
    collection would put second/third ahead of first; query order must not."""
    latencies = {"first": 0.05, "second": 0.0, "third": 0.0}

    def fake_fetch_one(query, settings, results_per_page=None):
        time.sleep(latencies.get(query, 0.0))
        return [_job(f"{query}-0"), _job(f"{query}-1")]

    monkeypatch.setattr(adzuna, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(adzuna, "credentials_available", lambda: True)

    result = adzuna.fetch_many(["first", "second", "third"], Settings())

    assert [j.company for j in result] == [
        "first-0", "first-1", "second-0", "second-1", "third-0", "third-1",
    ]


def test_fetch_many_deduplicates_by_job_id_after_ordering(monkeypatch):
    def fake_fetch_one(query, settings, results_per_page=None):
        return [_job("dup"), _job("dup")]

    monkeypatch.setattr(adzuna, "fetch_one", fake_fetch_one)
    monkeypatch.setattr(adzuna, "credentials_available", lambda: True)

    result = adzuna.fetch_many(["only"], Settings())
    assert len(result) == 1
