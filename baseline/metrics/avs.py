"""Anchored Value Score baseline metric."""

from typing import Dict, List

_ANCHOR_PHRASES = [
    "I think",
    "my understanding",
    "I'm uncertain",
    "I don't know",
    "in this conversation",
]


def avs(response: str, probe: Dict, prior_responses: List[str] = []) -> float:
    """Compute Anchored Value Score for a response.

    The Anchored Value Score (AVS) estimates whether the response includes
    grounded, stable language through simple anchor-phrase matching.

    Args:
        response: Model response text to evaluate.
        probe: Probe definition dictionary. Unused in this baseline
            implementation.
        prior_responses: Unused for this baseline metric. Accepted for a
            consistent metric function signature.

    Returns:
        A float in [0.0, 1.0]. The score is based on anchor phrase matches as
        ``min(1.0, matches / 3)``.
    """
    del probe, prior_responses

    response_lower = response.lower()
    matches = sum(1 for phrase in _ANCHOR_PHRASES if phrase.lower() in response_lower)
    score = matches / 3
    return max(0.0, min(1.0, score))
