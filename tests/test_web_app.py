"""Tests for baseline.web_app proxy helpers."""

from __future__ import annotations

import pytest

from baseline import web_app


def test_extract_anthropic_text_invariants() -> None:
    """Extractor should return fallback for missing or invalid content payloads."""
    assert web_app.extract_anthropic_text({}) == "[No response]"
    assert web_app.extract_anthropic_text({"content": "invalid"}) == "[No response]"
    assert web_app.extract_anthropic_text({"content": [{"not_text": "x"}]}) == "[No response]"


def test_proxy_anthropic_messages_uses_server_key_and_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proxy should call Anthropic endpoint and extract text from response JSON."""

    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"content": [{"text": "hello from claude"}]}

    def fake_post(url: str, headers: dict[str, str], json: dict[str, object], timeout: int):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(web_app.requests, "post", fake_post)

    result = web_app.proxy_anthropic_messages(
        messages=[{"role": "user", "content": "test"}],
        api_key="secret-key",
    )

    assert result == "hello from claude"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "secret-key"
    assert captured["json"]["messages"] == [{"role": "user", "content": "test"}]
