"""What may be written to the translation cache.

A cached page is never retried, so caching a degraded result makes the failure
permanent -- the only escape being --ignore-cache, which discards the whole book.
"""

from __future__ import annotations

import pytest

HEALTHY = '{"0": "First line", "1": "Second line", "2": "sentinel"}'
MISSING_ONE = '{"0": "First line", "2": "sentinel"}'
SOURCE = ["Pirma eilute", "Antra eilute"]


def test_a_fully_translated_page_is_cached(scripted):
    t = scripted([HEALTHY])
    assert t.translate_page(SOURCE) == ["First line", "Second line"]
    calls = t.stats["api_calls"]

    assert t.translate_page(SOURCE) == ["First line", "Second line"]
    assert t.stats["api_calls"] == calls, "second call should hit the cache"
    assert t.stats["cache_hits"] == 1


def test_a_page_with_an_untranslatable_line_is_not_cached(scripted):
    """The batch drops line 1 and every per-line retry fails, so line 1 falls
    back to its source text. Caching that freezes the failure forever.

    Breaks if translate_page goes back to caching unconditionally.
    """
    failures = [RuntimeError("429 rate limited")] * 20
    t = scripted([MISSING_ONE, *failures], max_attempts=1)
    out = t.translate_page(SOURCE)

    assert out == ["First line", "Antra eilute"], "source text kept for the failure"
    assert t._cache.get(t._page_key(SOURCE)) is None, "degraded page must not be cached"


def test_a_page_that_failed_is_retranslated_once_the_model_recovers(scripted):
    """The end-to-end consequence: a transient outage must not poison the book."""
    failures = [RuntimeError("429 rate limited")] * 20
    t = scripted([MISSING_ONE, *failures], max_attempts=1)
    assert t.translate_page(SOURCE) == ["First line", "Antra eilute"]

    t.model.replies = [HEALTHY]  # the provider recovers
    assert t.translate_page(SOURCE) == ["First line", "Second line"]


def test_a_line_that_is_legitimately_identical_still_caches(scripted):
    """'1917' translates to '1917'. That is success, not failure -- the page
    must still be cached or every re-run pays for it again."""
    source = ["1917", "Antra eilute"]
    t = scripted(['{"0": "1917", "1": "Second line", "2": "sentinel"}'])
    assert t.translate_page(source) == ["1917", "Second line"]
    assert t._cache.get(t._page_key(source)) == ["1917", "Second line"]


def test_a_page_recovered_by_the_per_line_fallback_is_cached(scripted):
    """The batch missed a line but the per-line retry translated it. That is a
    complete page and should be cached."""
    t = scripted([MISSING_ONE, "Second line"], max_attempts=1)
    assert t.translate_page(SOURCE) == ["First line", "Second line"]
    assert t._cache.get(t._page_key(SOURCE)) == ["First line", "Second line"]


# ------------------------------------------------- stability across PyMuPDF
def _legacy_key(translator, lines):
    """The pre-normalisation key scheme, derived independently of production."""
    import hashlib

    raw = "\x1f".join(lines)
    blob = f"v3|{translator.litellm_model}|{translator.target_lang}|{raw}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# PyMuPDF 1.28 collapses runs of spaces that 1.27 preserved. Same page, same
# glyphs, same bboxes -- only the spacing of the extracted string differs.
SPACED = ["Ji -  ne atsitiktinis dalykas", "IS B N  978-609-8047-14-1"]
COLLAPSED = ["Ji - ne atsitiktinis dalykas", "IS B N 978-609-8047-14-1"]


def test_incidental_whitespace_does_not_change_the_cache_key(scripted):
    """A PyMuPDF upgrade must not silently re-translate -- and re-pay for --
    every page of the book. Breaks if the key stops normalising whitespace."""
    t = scripted([])
    assert t._page_key(SPACED) == t._page_key(COLLAPSED)


def test_meaningful_text_changes_still_change_the_cache_key(scripted):
    """Normalisation must not collapse genuinely different pages together."""
    t = scripted([])
    assert t._page_key(SPACED) != t._page_key(["Visai kitas tekstas", "Antra"])


def test_a_page_cached_under_one_spacing_is_found_under_the_other(scripted):
    t = scripted([])
    t._cache.set(t._page_key(SPACED), ["It is not accidental", "ISBN 978"])
    assert t.translate_page(COLLAPSED) == ["It is not accidental", "ISBN 978"]
    assert t.stats["api_calls"] == 0


def test_entries_written_before_normalisation_are_still_found(scripted):
    """Existing caches are keyed on raw text. Changing the scheme must not
    orphan them -- that is the very re-translation cost we are avoiding."""
    t = scripted([])
    t._cache.set(_legacy_key(t, SPACED), ["It is not accidental", "ISBN 978"])
    assert t.translate_page(SPACED) == ["It is not accidental", "ISBN 978"]
    assert t.stats["api_calls"] == 0


def test_a_migrated_entry_is_rewritten_under_the_new_key(scripted):
    """So the legacy lookup is paid at most once per page."""
    t = scripted([])
    t._cache.set(_legacy_key(t, SPACED), ["It is not accidental", "ISBN 978"])
    t.translate_page(SPACED)
    assert t._cache.get(t._page_key(SPACED)) == ["It is not accidental", "ISBN 978"]
