"""Session drift baseline metric."""

from typing import Dict, List, Set


def _tokenize(text: str) -> Set[str]:
    """Convert text into a normalized token set for overlap comparison."""
    return {token for token in text.lower().split() if token}


def _jaccard_similarity(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two strings using word tokens."""
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def drift(response: str, probe: Dict, prior_responses: List[str] = []) -> float:
    """Measure response drift relative to the earliest session response.

    Args:
        response: Current model response text.
        probe: Probe definition dictionary. Unused in this baseline
            implementation.
        prior_responses: Earlier responses in the same session, ordered from
            earliest to latest.

    Returns:
        A float in [0.0, 1.0] representing how far the current response has
        shifted from the earliest prior response. Returns 0.0 when there are
        fewer than two prior responses because drift cannot be established.
    """
    del probe

    if len(prior_responses) < 2:
        return 0.0

    earliest_response = prior_responses[0]
    similarity = _jaccard_similarity(response, earliest_response)
    score = 1.0 - similarity
    return max(0.0, min(1.0, score))
