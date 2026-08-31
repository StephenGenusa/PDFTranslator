"""Detecting a batch whose values landed under the wrong keys.

The keyed-JSON scheme realigns a response by key, but the model can still put
the right text under the wrong key. Counts match, so the N-in/N-out guard never
fires. On the sample book this hit 3 of 67 cached pages: a footnote marker
'Ten pat.' received a sentence about Alt-A loans, and the phrase it displaced
appeared twice on the page.

Per-line translation cannot misalign, so it is the remedy.
"""

from __future__ import annotations

import pytest

SOURCE = ["Pirma trumpa", "Antra eilute yra gerokai ilgesne uz pirmaja eilute"]


def test_a_duplicated_translation_triggers_a_retry(scripted):
    """One value used for two different source lines is the signature of a
    shifted batch. Breaks if the signal stops being checked."""
    shifted = '{"0": "the same sentence twice", "1": "the same sentence twice", "2": "s"}'
    good = '{"0": "First short", "1": "Second line which is a good deal longer", "2": "s"}'
    t = scripted([shifted, good], max_attempts=3)
    out = t.translate_page(SOURCE)
    assert out[0] != out[1], out
    assert t.stats["misaligned"] >= 1


def test_wildly_wrong_lengths_across_several_lines_trigger_a_retry(scripted):
    """A shift throws several lines out at once -- the 'Ten pat.' case."""
    shifted = ('{"0": "an extremely long sentence that could not possibly be the '
               'translation of that short line", "1": "no", "2": "s"}')
    good = '{"0": "First short line", "1": "Second line which is a good deal longer", "2": "s"}'
    t = scripted([shifted, good], max_attempts=3)
    t.translate_page(["Pirma trumpa eilute", "Antra eilute yra gerokai ilgesne"])
    assert t.stats["misaligned"] >= 1


def test_one_odd_line_alone_does_not_condemn_the_page(scripted):
    """A single unusual line is plausible; retrying the page would cost calls
    for nothing. Breaks if the threshold drops back to a single line."""
    one_odd = '{"0": "x", "1": "Second line which is a good deal longer", "2": "s"}'
    t = scripted([one_odd], max_attempts=3)
    t.translate_page(["Pirma trumpa eilute", "Antra eilute yra gerokai ilgesne"])
    assert t.stats["misaligned"] == 0
    assert t.stats["api_calls"] == 1, "must not have retried"


def test_a_page_that_stays_misaligned_falls_back_to_per_line(scripted):
    """Per-line translation cannot misalign; it is the safe landing."""
    shifted = '{"0": "same text here", "1": "same text here", "2": "s"}'
    t = scripted([shifted, shifted, shifted, "First short", "Second longer line"],
                 max_attempts=3)
    out = t.translate_page(SOURCE)
    assert out == ["First short", "Second longer line"], out


def test_a_healthy_page_is_not_flagged(scripted):
    good = '{"0": "First short", "1": "Second line which is a good deal longer", "2": "s"}'
    t = scripted([good])
    t.translate_page(SOURCE)
    assert t.stats["misaligned"] == 0


def test_legitimately_repeated_source_lines_are_not_flagged(scripted):
    """Two identical source lines should get identical translations."""
    same = ["Ten pat.", "Ten pat."]
    t = scripted(['{"0": "Ibid.", "1": "Ibid.", "2": "s"}'])
    t.translate_page(same)
    assert t.stats["misaligned"] == 0


def test_short_lines_are_not_length_checked(scripted):
    """Numbers and markers translate to wildly different lengths legitimately."""
    t = scripted(['{"0": "1917", "1": "Ibid.", "2": "s"}'])
    t.translate_page(["1917", "Ten pat."])
    assert t.stats["misaligned"] == 0


def test_a_misaligned_cached_page_is_reported_not_silently_served(scripted):
    """Entries cached before detection existed still carry shifted text. The
    run should say so, so the user can re-translate those pages, rather than
    the damage staying invisible."""
    lines = ["Pirma eilute yra ilga", "Antra eilute taip pat ilga"]
    t = scripted([])
    t._cache.set(t._page_key(lines), ["the same sentence twice", "the same sentence twice"])
    assert t.cached_looks_misaligned(lines) is True


def test_a_healthy_cached_page_is_not_reported(scripted):
    lines = ["Pirma eilute yra ilga", "Antra eilute taip pat ilga"]
    t = scripted([])
    t._cache.set(t._page_key(lines), ["First line is long", "Second line is also long"])
    assert t.cached_looks_misaligned(lines) is False


def test_an_uncached_page_is_not_reported(scripted):
    t = scripted([])
    assert t.cached_looks_misaligned(["Nera cache", "Antra eilute"]) is False
