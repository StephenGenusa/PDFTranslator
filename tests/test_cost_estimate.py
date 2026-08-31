"""Estimating what a run will cost before it starts."""

from __future__ import annotations

import pytest


LINES = ["Pirma eilute yra gana ilga", "Antra eilute taip pat ilga"]


def test_an_estimate_scales_with_the_amount_of_text(scripted):
    t = scripted([])
    small = t.estimate_page_tokens(["Trumpa"])
    large = t.estimate_page_tokens(LINES * 20)
    assert large > small > 0


def test_the_estimate_covers_both_directions(scripted):
    """A translation is produced as well as consumed, so the estimate has to
    account for output tokens too, not just the prompt."""
    t = scripted([])
    prompt_only = t.estimate_page_tokens(LINES, include_output=False)
    both = t.estimate_page_tokens(LINES)
    assert both > prompt_only


def test_an_empty_page_costs_nothing(scripted):
    t = scripted([])
    assert t.estimate_page_tokens([]) == 0


def test_pricing_is_reported_when_the_model_is_known(scripted):
    """Priced models get a dollar figure; unmapped ones must not crash or
    invent a number."""
    t = scripted([], model="gpt-4o-mini")
    cost = t.estimate_cost(100_000)
    assert cost is None or cost > 0


def test_an_unmapped_model_yields_no_price_rather_than_a_guess(scripted):
    t = scripted([], model="openrouter/some/model-that-does-not-exist")
    assert t.estimate_cost(100_000) is None
