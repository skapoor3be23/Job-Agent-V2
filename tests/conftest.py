import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from agent import cache, profile_store
from agent.config import Settings


@pytest.fixture(autouse=True)
def clean_cache():
    cache.clear()
    yield
    cache.clear()


@pytest.fixture(autouse=True)
def isolated_profile_store(tmp_path, monkeypatch):
    """The on-disk persisted-profile store defaults to a real repo path
    (data/profile_cache.json). Redirect it to a per-test temp file so tests
    never read a profile persisted by a previous test run or a different
    test in the same session -- see profile_store.DEFAULT_PATH's docstring,
    which is a bare module attribute specifically so this works."""
    monkeypatch.setattr(profile_store, "DEFAULT_PATH", str(tmp_path / "profile_cache.json"))


@pytest.fixture
def settings():
    return Settings(discovery_enabled=True, cache_enabled=True)
