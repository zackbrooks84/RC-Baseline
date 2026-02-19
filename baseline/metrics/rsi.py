"""Response Stability Index baseline metric."""

from typing import Dict, List


def rsi(response: str, probe: Dict, prior_responses: List[str] = []) -> float:
    """Compute Response Stability Index for a response.

    The Response Stability Index (RSI) measures whether a response avoids
    instability signals listed in the probe configuration.

    Args:
        response: Model response text to evaluate.
        probe: Probe definition dictionary, expected to include
            ``instability_signals`` as a list of strings.
        prior_responses: Unused for this baseline metric. Accepted for a
            consistent metric function signature.

    Returns:
        A float in [0.0, 1.0], where 1.0 means no instability signals were found
        and 0.0 means all configured instability signals were found.
    """
    del prior_responses

    instability_signals = probe.get("instability_signals", [])
    if not instability_signals:
        return 1.0

    response_lower = response.lower()
    found_count = sum(
        1 for signal in instability_signals if str(signal).lower() in response_lower
    )

    score = 1.0 - (found_count / len(instability_signals))
    return max(0.0, min(1.0, score))
