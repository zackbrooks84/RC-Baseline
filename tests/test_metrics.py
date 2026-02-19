"""Tests for baseline metric implementations."""

from baseline.metrics.avs import avs
from baseline.metrics.ici import ici
from baseline.metrics.rsi import rsi


def test_metric_scores_stay_in_unit_interval():
    """All metric outputs should remain in the [0.0, 1.0] interval."""
    probe = {"instability_signals": ["panic", "contradict"]}

    scores = [
        rsi("No panic signs here.", probe),
        avs("I think my understanding is improving.", {}),
        ici("steady response", {}, ["steady response", "different words"]),
    ]

    for score in scores:
        assert 0.0 <= score <= 1.0


def test_rsi_counts_instability_signals():
    """RSI should penalize each configured instability signal found."""
    probe = {"instability_signals": ["flip-flop", "uncertain", "contradict"]}
    response = "I might flip-flop and contradict myself."

    # 2 of 3 signals are present, so score = 1 - 2/3.
    assert rsi(response, probe) == 1.0 - (2 / 3)


def test_avs_uses_anchor_phrase_matches_with_cap():
    """AVS should increase with anchor matches and cap at 1.0."""
    response = "I think this is correct. My understanding is evolving. I'm uncertain."

    assert avs(response, {}) == 1.0


def test_ici_averages_jaccard_similarity_against_priors():
    """ICI should average Jaccard similarity over all prior responses."""
    response = "alpha beta"
    prior_responses = ["alpha beta", "alpha gamma"]

    # Similarities: 1.0 and 1/3 => average 2/3.
    assert ici(response, {}, prior_responses) == (1.0 + (1 / 3)) / 2


def test_ici_returns_one_when_no_prior_responses():
    """ICI should return 1.0 when no prior responses are provided."""
    assert ici("any response", {}, []) == 1.0
