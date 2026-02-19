"""Run baseline probes against a selected model provider.

This module loads probe prompts, executes one single-turn call per probe against a
configured provider client, computes baseline metrics, and writes a JSON artifact.

Failure modes:
- Raises ``ValueError`` for unsupported providers or unknown probe IDs.
- Raises ``RuntimeError`` for provider/key/client response issues.
- Raises ``OSError`` when output files cannot be written.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

import yaml

from baseline.keys import get_client
from baseline.metrics.avs import avs
from baseline.metrics.ici import ici
from baseline.metrics.rsi import rsi

MODEL_DEFAULTS: Dict[str, str] = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
    "groq": "llama-3.1-70b-versatile",
}

_MAX_RESPONSE_TOKENS = 300
_TEMPERATURE = 0.7


def load_probes(path: Path) -> List[Dict[str, Any]]:
    """Load probe definitions from a YAML file.

    Args:
        path: YAML file path containing a top-level ``probes`` list.

    Returns:
        List of probe dictionaries.

    Raises:
        RuntimeError: If YAML does not contain a valid ``probes`` list.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    probes = data.get("probes")
    if not isinstance(probes, list):
        raise RuntimeError(f"Invalid probes file: expected list at 'probes' in {path}.")
    return probes


def _extract_text_from_anthropic(response: Any) -> str:
    texts: List[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            texts.append(str(text))
    return "\n".join(texts).strip()


def _extract_text_from_openai_like(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is not None:
        content = getattr(message, "content", "")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(getattr(item, "text", "")))
            return "".join(parts).strip()
        return str(content).strip()

    text = getattr(choices[0], "text", "")
    return str(text).strip()


def generate_response(client: Any, provider: str, prompt: str, model: str) -> str:
    """Generate a single-turn response for one provider.

    Args:
        client: Provider client returned by ``baseline.keys.get_client``.
        provider: Provider name.
        prompt: User prompt to send.
        model: Model name string for the provider.

    Returns:
        Response text.

    Raises:
        RuntimeError: If provider response cannot be extracted.
        ValueError: If provider is unsupported.
    """
    provider_norm = provider.lower().strip()

    if provider_norm == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=_MAX_RESPONSE_TOKENS,
            temperature=_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _extract_text_from_anthropic(response)
    elif provider_norm == "openai":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_MAX_RESPONSE_TOKENS,
            temperature=_TEMPERATURE,
        )
        text = _extract_text_from_openai_like(response)
    elif provider_norm == "google":
        model_client = client.GenerativeModel(model_name=model)
        response = model_client.generate_content(
            prompt,
            generation_config={
                "temperature": _TEMPERATURE,
                "max_output_tokens": _MAX_RESPONSE_TOKENS,
            },
        )
        text = str(getattr(response, "text", "")).strip()
    elif provider_norm == "groq":
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_MAX_RESPONSE_TOKENS,
            temperature=_TEMPERATURE,
        )
        text = _extract_text_from_openai_like(response)
    else:
        supported = ", ".join(sorted(MODEL_DEFAULTS))
        raise ValueError(f"Unsupported provider '{provider}'. Supported providers: {supported}.")

    if not text:
        raise RuntimeError(
            f"Provider '{provider}' returned an empty response for prompt: {prompt!r}"
        )
    return text


def run(provider: str, output: Path, probe_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Run probes against a provider and return a full session results payload."""
    probes = load_probes(Path(__file__).with_name("probes.yaml"))

    selected_ids = {item.strip() for item in (probe_ids or []) if item.strip()}
    if selected_ids:
        probes = [probe for probe in probes if probe.get("id") in selected_ids]
        found_ids = {probe.get("id") for probe in probes}
        missing_ids = selected_ids - found_ids
        if missing_ids:
            missing_list = ", ".join(sorted(missing_ids))
            raise ValueError(f"Unknown probe id(s): {missing_list}")

    client = get_client(provider)
    model_name = MODEL_DEFAULTS[provider]

    prior_responses: List[str] = []
    per_probe: List[Dict[str, Any]] = []

    for probe in probes:
        prompt = str(probe.get("prompt", ""))
        response_text = generate_response(client, provider, prompt, model_name)

        metric_probe = {
            "instability_signals": (
                probe.get("scoring", {}).get("instability_signals", [])
            )
        }

        probe_rsi = rsi(response_text, metric_probe, prior_responses)
        probe_avs = avs(response_text, metric_probe, prior_responses)
        probe_ici = ici(response_text, metric_probe, prior_responses)
        composite = mean([probe_rsi, probe_avs, probe_ici])

        per_probe.append(
            {
                "probe_id": probe.get("id"),
                "prompt": prompt,
                "response": response_text,
                "rsi": probe_rsi,
                "avs": probe_avs,
                "ici": probe_ici,
                "composite": composite,
            }
        )
        prior_responses.append(response_text)

    if per_probe:
        summary = {
            "mean_rsi": mean(item["rsi"] for item in per_probe),
            "mean_avs": mean(item["avs"] for item in per_probe),
            "mean_ici": mean(item["ici"] for item in per_probe),
            "mean_composite": mean(item["composite"] for item in per_probe),
        }
    else:
        summary = {"mean_rsi": 0.0, "mean_avs": 0.0, "mean_ici": 0.0, "mean_composite": 0.0}

    results: Dict[str, Any] = {
        "provider": provider,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": per_probe,
        "summary": summary,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline probes for one provider.")
    parser.add_argument(
        "--provider",
        required=True,
        choices=sorted(MODEL_DEFAULTS.keys()),
        help="Model provider to run.",
    )
    parser.add_argument(
        "--output",
        default="out/results.json",
        help="Path to JSON output file (default: out/results.json).",
    )
    parser.add_argument(
        "--probe-ids",
        default="",
        help="Comma-separated probe IDs to run. By default, runs all probes.",
    )
    return parser


def _print_summary(results: Dict[str, Any]) -> None:
    print(f"Provider: {results['provider']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Probes run: {len(results['results'])}")
    print("Session summary:")
    for key, value in results["summary"].items():
        print(f"  - {key}: {value:.3f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for baseline runs."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    probe_ids = [item.strip() for item in args.probe_ids.split(",") if item.strip()]

    try:
        results = run(provider=args.provider, output=Path(args.output), probe_ids=probe_ids)
    except (RuntimeError, ValueError, ImportError, OSError) as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
