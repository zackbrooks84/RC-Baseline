"""Tests for baseline.keys API key and client helpers."""

import sys
import types

import pytest

from baseline import keys


@pytest.fixture(autouse=True)
def clear_key_env(monkeypatch):
    """Ensure key-related environment variables are reset for each test."""
    for env_var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)


def test_validate_all_partitions_providers(monkeypatch):
    """validate_all should partition all supported providers with no overlap."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")

    result = keys.validate_all()

    all_providers = set(result["available"]) | set(result["missing"])
    overlap = set(result["available"]) & set(result["missing"])

    assert all_providers == {"anthropic", "openai", "google", "groq"}
    assert overlap == set()


def test_get_client_openai_uses_env_key(monkeypatch):
    """get_client should initialize OpenAI client with OPENAI_API_KEY."""
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key):
            captured["api_key"] = api_key

    fake_openai_module = types.SimpleNamespace(OpenAI=FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    client = keys.get_client("openai")

    assert isinstance(client, FakeOpenAI)
    assert captured["api_key"] == "openai-test-key"


def test_get_groq_api_key_raises_clear_error_when_missing():
    """Provider key getter should raise a clear missing key message."""
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        keys.get_groq_api_key()
