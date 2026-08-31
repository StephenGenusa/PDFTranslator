"""What the translator is allowed to hand back for rendering.

Anything returned here is stamped onto the page verbatim, so protocol debris --
a JSON wrapper, the internal batch sentinel -- must never survive.
"""

from __future__ import annotations

import pytest

from pdftranslator.translate import _SENTINEL, _clean_single


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
def test_per_line_reply_wrapped_in_json_is_unwrapped(scripted):
    """A JSON-shaped per-line reply must yield the translation, not the wrapper."""
    t = scripted(['{"0": "Good morning everyone"}'])
    assert t._translate_one("Labas rytas visiems", "context", 0, 1) == (
        "Good morning everyone"
    )


def test_per_line_reply_that_is_unusable_reports_failure(scripted):
    """None means 'could not translate'; the caller substitutes the source line
    and declines to cache the page (see test_translate_cache.py)."""
    t = scripted(["{not valid json}"])
    assert t._translate_one("Labas rytas visiems", "context", 0, 1) is None


def test_per_line_request_does_not_demand_json(scripted):
    """The per-line prompt asks for a bare line; a system prompt ordering
    'output ONLY JSON' contradicts it and is what produced JSON on the page."""
    t = scripted(["Good morning"])
    t._translate_one("Labas rytas", "context", 0, 1)
    system = t.sent[0][0]["content"]
    assert "JSON" not in system, system


# --------------------------------------------------------------- batch output
def test_sentinel_returned_for_a_real_line_is_not_rendered(scripted):
    """The model sometimes shifts keys and answers a real line with the
    sentinel. Stamping it produces '__batch_end_a3f2d8e1__' on the page."""
    reply = '{"0": "First line", "1": "%s", "2": "%s"}' % (_SENTINEL, _SENTINEL)
    t = scripted([reply, reply, reply, "Second line"])
    out = t.translate_page(["Pirma eilute", "Antra eilute"])
    assert _SENTINEL not in out[0]
    assert _SENTINEL not in out[1]
    assert out[1] == "Second line"


def test_healthy_batch_is_returned_unchanged(scripted):
    reply = '{"0": "First line", "1": "Second line", "2": "ignored"}'
    t = scripted([reply])
    assert t.translate_page(["Pirma eilute", "Antra eilute"]) == [
        "First line",
        "Second line",
    ]


# --------------------------------------------------- repair of poisoned caches
def test_cached_entry_carrying_the_sentinel_is_repaired_on_read(scripted):
    """Caches written before the sentinel guard still hold the raw token.

    Returning it would keep stamping '__batch_end_...' onto every re-render, so
    a poisoned entry falls back to the source line rather than the debris.
    Breaks if cache reads stop being sanitised.
    """
    lines = ["Pirma eilute", "Antra eilute"]
    t = scripted([])
    t._cache.set(t._page_key(lines), ["First line", _SENTINEL])
    out = t.translate_page(lines)

    assert out[0] == "First line"      # the good line survives untouched
    assert out[1] == "Antra eilute"    # the debris becomes the source line
    assert t.stats["api_calls"] == 0   # repair costs nothing


def test_clean_cached_entry_is_returned_untouched(scripted):
    lines = ["Pirma eilute", "Antra eilute"]
    t = scripted([])
    t._cache.set(t._page_key(lines), ["First line", "Second line"])
    assert t.translate_page(lines) == ["First line", "Second line"]
    assert t.stats["cache_hits"] == 1


# ------------------------------------------------------- empty translations
def test_an_empty_batch_value_is_treated_as_missing(scripted):
    """The model sometimes returns "" for a line. Accepting it as a translation
    means the renderer draws nothing -- no mask, no text -- so the untranslated
    original shows through from the base layer, silently.

    Breaks if empty values are accepted as translations again.
    """
    t = scripted(['{"0": "First line", "1": "   ", "2": "sentinel"}', "Second line"],
                 max_attempts=1)
    out = t.translate_page(["Pirma eilute", "Antra eilute"])
    assert out[1] == "Second line", out


def test_a_page_whose_empty_line_cannot_be_recovered_is_not_cached(scripted):
    t = scripted(['{"0": "First line", "1": "", "2": "s"}', "", "", ""],
                 max_attempts=1)
    out = t.translate_page(["Pirma eilute", "Antra eilute"])
    assert out[1] == "Antra eilute", "source text kept"
    assert t._cache.get(t._page_key(["Pirma eilute", "Antra eilute"])) is None


def test_a_cached_empty_translation_is_repaired_on_read(scripted):
    """Caches written before the guard hold empty strings; returning them keeps
    leaving the source showing through on every re-render."""
    lines = ["Pirma eilute", "Antra eilute"]
    t = scripted([])
    t._cache.set(t._page_key(lines), ["First line", ""])
    out = t.translate_page(lines)
    assert out[0] == "First line"
    assert out[1] == "Antra eilute", "empty entry should fall back to the source"
    assert t.stats["api_calls"] == 0


def test_a_blank_source_line_may_stay_blank(scripted):
    """Only a non-empty source with an empty translation is a defect."""
    lines = ["Pirma eilute", "   "]
    t = scripted([])
    t._cache.set(t._page_key(lines), ["First line", ""])
    assert t.translate_page(lines) == ["First line", ""]
