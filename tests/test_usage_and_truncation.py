"""Token usage, spend, and detecting a response cut short.

max_tokens caps the response. When a dense page hits that cap the JSON comes
back truncated and the trailing keys are simply missing -- which is the likely
cause of the 'model drops the last key' behaviour the _SENTINEL padding was
added to work around. finish_reason says so explicitly.
"""

from __future__ import annotations

import pytest


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens, self.completion_tokens = p, c
        self.total_tokens = p + c


def _reply(content, *, finish_reason="stop", usage=(100, 40), cost=None):
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
    r = type("Response", (), {})()
    r.choices = [choice]
    r.usage = _Usage(*usage)
    r._hidden_params = {"response_cost": cost} if cost is not None else {}
    return r


HEALTHY = '{"0": "First line", "1": "Second line", "2": "sentinel"}'
SOURCE = ["Pirma eilute", "Antra eilute"]


# --------------------------------------------------------------------- usage
def test_token_usage_is_accumulated(scripted):
    t = scripted([])
    t.model.replies = [_reply(HEALTHY, usage=(120, 55))]
    t.translate_page(SOURCE)
    assert t.stats["prompt_tokens"] == 120
    assert t.stats["completion_tokens"] == 55


def test_usage_accumulates_across_calls(scripted):
    t = scripted([])
    t.model.replies = [_reply(HEALTHY, usage=(10, 5)),
                       _reply(HEALTHY, usage=(20, 7))]
    t.translate_page(SOURCE)
    t.translate_page(["Kita", "Treti eilute"])
    assert t.stats["prompt_tokens"] == 30
    assert t.stats["completion_tokens"] == 12


def test_reported_spend_is_taken_from_the_response(scripted):
    """The provider returns what was actually charged; use it rather than
    guessing from a price table that may not know the model."""
    t = scripted([])
    t.model.replies = [_reply(HEALTHY, cost=0.000123)]
    t.translate_page(SOURCE)
    assert t.stats["cost_usd"] == pytest.approx(0.000123)


def test_a_response_without_usage_does_not_break_the_run(scripted):
    t = scripted([])
    r = _reply(HEALTHY)
    del r.usage
    t.model.replies = [r]
    assert t.translate_page(SOURCE) == ["First line", "Second line"]


# ---------------------------------------------------------------- truncation
def test_a_truncated_response_is_detected_and_retried(scripted):
    """Breaks if finish_reason is ignored: a page silently loses its tail."""
    truncated = _reply('{"0": "First line", "1": "Second', finish_reason="length")
    t = scripted([], max_attempts=3)
    t.model.replies = [truncated, _reply(HEALTHY)]
    assert t.translate_page(SOURCE) == ["First line", "Second line"]
    assert t.stats["truncated"] == 1


def test_truncation_is_counted_even_when_the_json_parses(scripted):
    """A cut-off response can still parse as valid JSON with keys missing.
    The count is what tells the user to raise --max-tokens."""
    partial = _reply('{"0": "First line"}', finish_reason="length")
    t = scripted([], max_attempts=2)
    t.model.replies = [partial, _reply(HEALTHY)]
    t.translate_page(SOURCE)
    assert t.stats["truncated"] >= 1


def test_a_normal_response_is_not_counted_as_truncated(scripted):
    t = scripted([])
    t.model.replies = [_reply(HEALTHY, finish_reason="stop")]
    t.translate_page(SOURCE)
    assert t.stats["truncated"] == 0
