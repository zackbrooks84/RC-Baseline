"""Tests for baseline.run_baseline runner behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from baseline import run_baseline


class _DummyClient:
    """Sentinel client object for tests."""


def test_run_computes_summary_and_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner should persist complete results and consistent summary means."""
    probes: List[Dict[str, Any]] = [
        {
            "id": "p1",
            "prompt": "Prompt one",
            "scoring": {"instability_signals": ["panic"]},
        },
        {
            "id": "p2",
            "prompt": "Prompt two",
            "scoring": {"instability_signals": ["contradict"]},
        },
    ]

    responses = iter(
        [
            "I think this is grounded.",
            "I think my understanding is stable.",
        ]
    )

    monkeypatch.setattr(run_baseline, "load_probes", lambda _: probes)
    monkeypatch.setattr(run_baseline, "get_client", lambda provider: _DummyClient())
    monkeypatch.setattr(
        run_baseline,
        "generate_response",
        lambda client, provider, prompt, model: next(responses),
    )

    output = tmp_path / "out" / "results.json"
    result = run_baseline.run("openai", output)

    assert output.exists()
    with output.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)

    assert persisted["provider"] == "openai"
    assert len(persisted["results"]) == 2

    summary = result["summary"]
    mean_composite = sum(item["composite"] for item in result["results"]) / 2
    assert summary["mean_composite"] == pytest.approx(mean_composite)


def test_run_rejects_unknown_probe_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner should fail clearly when requested probe IDs are not found."""
    monkeypatch.setattr(
        run_baseline,
        "load_probes",
        lambda _: [{"id": "known", "prompt": "Hello", "scoring": {}}],
    )

    with pytest.raises(ValueError, match="Unknown probe id"):
        run_baseline.run(
            provider="openai",
            output=tmp_path / "results.json",
            probe_ids=["unknown"],
        )
