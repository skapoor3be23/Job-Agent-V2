"""LLMClient failure handling: every LLMUnavailable must carry a specific,
actionable reason (never a bare exception class name), and a hard provider
failure (quota/auth/outage) on the native structured-output attempt must
not waste a second real request on a JSON-contract retry that would fail
identically.

Root cause this guards against: the dashboard showed a generic "The cover
note could not be generated." for what was actually a Gemini free-tier
quota exhaustion (429 RESOURCE_EXHAUSTED) -- confirmed by invoking the real
client directly. The generic wrapper message and the hardcoded reason in
graph.py both discarded that detail before it reached the user.
"""
import pytest
from agent.clients import LLMClient, LLMUnavailable
from agent.config import Settings
from agent.schemas import GapAnalysis
from agent.telemetry import new_run


class _FakeBoundStructured:
    def __init__(self, fail_with):
        self.fail_with = fail_with

    def invoke(self, prompt):
        if self.fail_with is not None:
            raise self.fail_with
        return {}


class _FakeRawClient:
    """Minimal stand-in for the real ChatGoogleGenerativeAI client."""

    def __init__(self, structured_fail=None, invoke_fail=None, invoke_text="ok"):
        self.structured_fail = structured_fail
        self.invoke_fail = invoke_fail
        self.invoke_text = invoke_text
        self.invoke_calls = 0

    def with_structured_output(self, schema):
        return _FakeBoundStructured(self.structured_fail)

    def invoke(self, prompt):
        self.invoke_calls += 1
        if self.invoke_fail:
            raise self.invoke_fail
        return type("Response", (), {"content": self.invoke_text})()


def _client(raw):
    settings = Settings()
    return LLMClient(settings, new_run(settings), raw_client=raw)


def test_quota_exhaustion_produces_a_specific_actionable_message():
    quota_error = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded for generativelanguage.googleapis.com")
    client = _client(_FakeRawClient(invoke_fail=quota_error))
    with pytest.raises(LLMUnavailable) as exc_info:
        client.invoke("prompt")
    message = str(exc_info.value).lower()
    assert "quota" in message
    assert message != "llm call failed (runtimeerror)."


def test_permission_denied_produces_a_specific_message():
    auth_error = RuntimeError("403 PERMISSION_DENIED: API key not valid")
    client = _client(_FakeRawClient(invoke_fail=auth_error))
    with pytest.raises(LLMUnavailable) as exc_info:
        client.invoke("prompt")
    assert "api key" in str(exc_info.value).lower()


def test_structured_does_not_waste_a_second_call_on_a_hard_provider_failure():
    """A quota/auth/outage failure on the native structured attempt must
    raise immediately, never fall through to a JSON-contract retry that
    would fail identically and only burn a second real request."""
    quota_error = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
    raw = _FakeRawClient(structured_fail=quota_error, invoke_text="should never be reached")
    client = _client(raw)
    with pytest.raises(LLMUnavailable) as exc_info:
        client.structured("prompt", GapAnalysis)
    assert "quota" in str(exc_info.value).lower()
    assert raw.invoke_calls == 0


def test_structured_still_falls_back_on_an_ordinary_native_output_failure():
    """A plain "structured output not supported" failure is NOT a hard
    provider failure -- it must still fall back to the JSON-contract retry."""
    raw = _FakeRawClient(
        structured_fail=ValueError("with_structured_output not supported for this model"),
        invoke_text='{"matched_requirements": ["Python"]}',
    )
    client = _client(raw)
    result = client.structured("prompt", GapAnalysis)
    assert result.matched_requirements == ["Python"]
    assert raw.invoke_calls == 1
