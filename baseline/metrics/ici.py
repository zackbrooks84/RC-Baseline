"""Identity Consistency Index baseline metric."""

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


def ici(response: str, probe: Dict, prior_responses: List[str] = []) -> float:
    """Compute Identity Consistency Index against prior responses.

    The Identity Consistency Index (ICI) measures session-level consistency by
    averaging Jaccard word-overlap similarity between the current response and
    each prior response.

    Args:
        response: Current model response text.
        probe: Probe definition dictionary. Unused in this baseline
            implementation.
        prior_responses: Earlier responses in the same session.

    Returns:
        A float in [0.0, 1.0]. If no prior responses are provided, returns 1.0
        because inconsistency cannot be detected.
    """
    del probe

    if not prior_responses:
        return 1.0

    similarities = [_jaccard_similarity(response, prior) for prior in prior_responses]
    score = sum(similarities) / len(similarities)
    return max(0.0, min(1.0, score))
