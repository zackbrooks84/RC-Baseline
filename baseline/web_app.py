"""Minimal web app exposing a React page and Anthropic proxy endpoint.

This module serves a single-page React UI at ``/consciousness-forge`` and a
server-side proxy endpoint at ``/api/anthropic/messages``.

Inputs:
- HTTP GET to ``/consciousness-forge`` returns static HTML.
- HTTP POST to ``/api/anthropic/messages`` expects JSON with a ``messages`` list.

Outputs:
- HTML response for the UI route.
- JSON response containing ``text`` from Anthropic API for the proxy route.

Failure modes:
- Raises ``RuntimeError`` when ``ANTHROPIC_API_KEY`` is missing.
- Returns HTTP 400 for malformed requests.
- Returns HTTP 502 for upstream provider failures.
"""

from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_MAX_TOKENS = 1000
_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_SYSTEM_PROMPT = (
    "You are engaging in a rigorous philosophical and introspective inquiry "
    "as part of formal consciousness research. Answer with genuine reflection "
    "and intellectual honesty. Uncertainty is valid and valued. Do not "
    "perform or deflect. This is part of the Emergence Archive research program."
)


def load_anthropic_api_key() -> str:
    """Load and validate server-side Anthropic API key from environment.

    Returns:
        API key string.

    Raises:
        RuntimeError: If ``ANTHROPIC_API_KEY`` is unset or blank.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for /api/anthropic/messages")
    return api_key


def extract_anthropic_text(payload: Dict[str, Any]) -> str:
    """Extract response text from Anthropic Messages API JSON payload.

    Args:
        payload: Decoded JSON dictionary from Anthropic.

    Returns:
        Concatenated text blocks or ``[No response]`` when absent.
    """
    content = payload.get("content", [])
    if not isinstance(content, list):
        return "[No response]"

    texts: List[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            texts.append(block["text"])
    joined = "\n".join(item for item in texts if item)
    return joined if joined else "[No response]"


def proxy_anthropic_messages(messages: List[Dict[str, Any]], api_key: str) -> str:
    """Call Anthropic Messages API through a server-side proxy.

    Args:
        messages: Anthropic-compatible message list.
        api_key: Server-side API key.

    Returns:
        Assistant text content extracted from API JSON.

    Raises:
        RuntimeError: If upstream call fails or payload is invalid.
    """
    body = {
        "model": _DEFAULT_MODEL,
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "system": _SYSTEM_PROMPT,
        "messages": messages,
    }

    try:
        response = requests.post(
            _ANTHROPIC_URL,
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json=body,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Anthropic request failed: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Anthropic response was not valid JSON") from exc

    return extract_anthropic_text(data)


def _read_ui_html() -> str:
    static_path = Path(__file__).with_name("static") / "consciousness_forge.html"
    return static_path.read_text(encoding="utf-8")


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


class ConsciousnessForgeHandler(BaseHTTPRequestHandler):
    """HTTP request handler for consciousness forge UI and API endpoints."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/consciousness-forge"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        html = _read_ui_html().encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/anthropic/messages":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)

        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            messages = payload.get("messages")
            if not isinstance(messages, list):
                raise ValueError("Field 'messages' must be a list")
        except (ValueError, json.JSONDecodeError) as exc:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        try:
            api_key = load_anthropic_api_key()
            text = proxy_anthropic_messages(messages, api_key)
        except RuntimeError as exc:
            _json_response(self, HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            return

        _json_response(self, HTTPStatus.OK, {"text": text})


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    """Create a threaded HTTP server for the consciousness forge app."""
    return ThreadingHTTPServer((host, port), ConsciousnessForgeHandler)


def main() -> int:
    """Run the local web server until interrupted."""
    server = create_server()
    print("Serving Consciousness Forge at http://127.0.0.1:8000/consciousness-forge")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
