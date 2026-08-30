"""What the translator is allowed to hand back for rendering.

Anything returned here is stamped onto the page verbatim, so protocol debris --
a JSON wrapper, the internal batch sentinel -- must never survive.
"""

from __future__ import annotations

import pytest

from pdftranslator.translate import _SENTINEL, Translator, _clean_single


class ScriptedTranslator(Translator):
    """Real Translator with a scripted model. Records the messages it sends."""

    def __init__(self, replies, **kwargs):
        kwargs.setdefault("model", "fake/model")
        kwargs.setdefault("target_lang", "English")
        super().__init__(**kwargs)
        self._replies = list(replies)
        self.sent: list[list[dict]] = []

    def _complete(self, messages):
        self.sent.append(messages)
        self.stats["api_calls"] += 1
        return self._replies.pop(0) if self._replies else ""


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "cache")


# ------------------------------------------------------------- _clean_single
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Good morning", "Good morning"),
        ('"Good morning"', "Good morning"),
        ("```\nGood morning\n```", "Good morning"),
        ('["Good morning"]', "Good morning"),
        # The system prompt demands JSON, so a compliant model answers a
        # single-line request with a JSON object. Unwrap it.
        ('{"0": "Good morning"}', "Good morning"),
        ('```json\n{"3": "Good morning"}\n```', "Good morning"),
        # Ambiguous or unusable: the caller must keep the source line.
        ('{"0": "a", "1": "b"}', ""),
        ("{not valid json}", ""),
        ("", ""),
    ],
)
def test_clean_single_never_returns_protocol_debris(raw, expected):
    """Breaks if a JSON wrapper reaches the page as literal text."""
    assert _clean_single(raw) == expected


def test_clean_single_rejects_the_batch_sentinel():
    assert _clean_single(_SENTINEL) == ""
    assert _clean_single(f'{{"7": "{_SENTINEL}"}}') == ""


# ------------------------------------------------------- per-line translation
def test_per_line_reply_wrapped_in_json_is_unwrapped(cache_dir):
    """A JSON-shaped per-line reply must yield the translation, not the wrapper."""
    t = ScriptedTranslator(['{"0": "Good morning everyone"}'], cache_dir=cache_dir)
    try:
        assert t._translate_one("Labas rytas visiems", "context", 0, 1) == (
            "Good morning everyone"
        )
    finally:
        t.close()


def test_per_line_reply_that_is_unusable_keeps_the_source_line(cache_dir):
    t = ScriptedTranslator(["{not valid json}"], cache_dir=cache_dir)
    try:
        assert t._translate_one("Labas rytas visiems", "context", 0, 1) == (
            "Labas rytas visiems"
        )
    finally:
        t.close()


def test_per_line_request_does_not_demand_json(cache_dir):
    """The per-line prompt asks for a bare line; a system prompt ordering
    'output ONLY JSON' contradicts it and is what produced JSON on the page."""
    t = ScriptedTranslator(["Good morning"], cache_dir=cache_dir)
    try:
        t._translate_one("Labas rytas", "context", 0, 1)
        system = t.sent[0][0]["content"]
        assert "JSON" not in system, system
    finally:
        t.close()


# --------------------------------------------------------------- batch output
def test_sentinel_returned_for_a_real_line_is_not_rendered(cache_dir):
    """The model sometimes shifts keys and answers a real line with the
    sentinel. Stamping it produces '__batch_end_a3f2d8e1__' on the page."""
    reply = '{"0": "First line", "1": "%s", "2": "%s"}' % (_SENTINEL, _SENTINEL)
    t = ScriptedTranslator([reply, reply, reply, "Second line"], cache_dir=cache_dir)
    try:
        out = t.translate_page(["Pirma eilute", "Antra eilute"])
    finally:
        t.close()
    assert _SENTINEL not in out[0]
    assert _SENTINEL not in out[1]
    assert out[1] == "Second line"


def test_healthy_batch_is_returned_unchanged(cache_dir):
    reply = '{"0": "First line", "1": "Second line", "2": "ignored"}'
    t = ScriptedTranslator([reply], cache_dir=cache_dir)
    try:
        assert t.translate_page(["Pirma eilute", "Antra eilute"]) == [
            "First line",
            "Second line",
        ]
    finally:
        t.close()


# --------------------------------------------------- repair of poisoned caches
def test_cached_entry_carrying_the_sentinel_is_repaired_on_read(cache_dir):
    """Caches written before the sentinel guard still hold the raw token.

    Returning it would keep stamping '__batch_end_...' onto every re-render, so
    a poisoned entry falls back to the source line rather than the debris.
    Breaks if cache reads stop being sanitised.
    """
    lines = ["Pirma eilute", "Antra eilute"]
    t = ScriptedTranslator([], cache_dir=cache_dir)
    try:
        t._cache.set(t._page_key(lines), ["First line", _SENTINEL])
        out = t.translate_page(lines)
    finally:
        t.close()

    assert out[0] == "First line"      # the good line survives untouched
    assert out[1] == "Antra eilute"    # the debris becomes the source line
    assert t.stats["api_calls"] == 0   # repair costs nothing


def test_clean_cached_entry_is_returned_untouched(cache_dir):
    lines = ["Pirma eilute", "Antra eilute"]
    t = ScriptedTranslator([], cache_dir=cache_dir)
    try:
        t._cache.set(t._page_key(lines), ["First line", "Second line"])
        assert t.translate_page(lines) == ["First line", "Second line"]
        assert t.stats["cache_hits"] == 1
    finally:
        t.close()
