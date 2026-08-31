"""Cache coverage reporting, and migrating entries to the current key scheme.

A cached translation is paid-for work. Migration writes the entry under the new
key, confirms it reads back, and only then removes the old one. Anything that
cannot be re-keyed is left alone -- it may belong to another document.
"""

from __future__ import annotations

import hashlib

import pytest


def _legacy_key(translator, lines):
    """The pre-normalisation scheme, derived independently of production."""
    raw = "\x1f".join(lines)
    blob = f"v3|{translator.litellm_model}|{translator.target_lang}|{raw}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


SPACED = ["Ji -  ne atsitiktinis dalykas", "IS B N  978-609-8047-14-1"]
VALUE = ["It is not accidental", "ISBN 978-609-8047-14-1"]


# ------------------------------------------------------------------ coverage
def test_has_cached_finds_an_entry_under_the_current_key(scripted):
    t = scripted([])
    t._cache.set(t._page_key(SPACED), VALUE)
    assert t.has_cached(SPACED) is True


def test_has_cached_finds_an_entry_under_the_legacy_key(scripted):
    """Breaks if coverage reporting stops consulting the legacy scheme -- the
    mistake that made 311 translated pages look missing."""
    t = scripted([])
    t._cache.set(_legacy_key(t, SPACED), VALUE)
    assert t.has_cached(SPACED) is True


def test_has_cached_is_false_for_an_untranslated_page(scripted):
    t = scripted([])
    assert t.has_cached(["Visai kitas tekstas", "Antra"]) is False


def test_has_cached_does_not_migrate_anything(scripted):
    """Reporting coverage must not mutate the cache."""
    t = scripted([])
    legacy = _legacy_key(t, SPACED)
    t._cache.set(legacy, VALUE)
    t.has_cached(SPACED)
    assert t._cache.get(legacy) == VALUE
    assert t._cache.get(t._page_key(SPACED)) is None


def test_has_cached_rejects_an_entry_of_the_wrong_length(scripted):
    t = scripted([])
    t._cache.set(t._page_key(SPACED), ["only one"])
    assert t.has_cached(SPACED) is False


# ----------------------------------------------------------------- migration
def test_a_legacy_entry_is_rewritten_under_the_current_key(scripted):
    t = scripted([])
    t._cache.set(_legacy_key(t, SPACED), VALUE)
    assert t.translate_page(SPACED) == VALUE
    assert t._cache.get(t._page_key(SPACED)) == VALUE


def test_the_legacy_key_is_removed_once_converted(scripted):
    """Pruning: the duplicate goes only after the new key is readable."""
    t = scripted([])
    legacy = _legacy_key(t, SPACED)
    t._cache.set(legacy, VALUE)
    t.translate_page(SPACED)
    assert t._cache.get(legacy) is None, "legacy duplicate should be pruned"
    assert t._cache.get(t._page_key(SPACED)) == VALUE, "translation must survive"


def test_migration_never_loses_the_translation(scripted):
    """After migration a fresh translator still finds the work, with no API."""
    t = scripted([])
    t._cache.set(_legacy_key(t, SPACED), VALUE)
    t.translate_page(SPACED)
    assert t.translate_page(SPACED) == VALUE
    assert t.stats["api_calls"] == 0


def test_an_entry_for_another_document_is_left_untouched(scripted):
    """Only pages we can re-key are touched; nothing is swept as an orphan."""
    t = scripted(['{"0": "First", "1": "Second", "2": "s"}'], max_attempts=1)
    other = "some other document's cache entry"
    t._cache.set(other, ["not ours"])
    t.translate_page(["Pirma eilute", "Antra eilute"])
    assert t._cache.get(other) == ["not ours"]


def test_a_stale_duplicate_is_pruned_once_the_current_key_holds_it(scripted):
    """An earlier migration wrote the new key without removing the old one, so
    the store carries both. The duplicate goes only because the current key
    already holds the translation -- verified before anything is deleted.
    """
    t = scripted([])
    legacy = _legacy_key(t, SPACED)
    t._cache.set(legacy, VALUE)
    t._cache.set(t._page_key(SPACED), VALUE)

    assert t.translate_page(SPACED) == VALUE
    assert t._cache.get(legacy) is None, "stale duplicate should be pruned"
    assert t._cache.get(t._page_key(SPACED)) == VALUE, "translation must survive"


def test_a_differing_legacy_entry_is_kept_not_silently_dropped(scripted):
    """If the two keys disagree, that is unexplained. Keep both rather than
    discard work we cannot account for."""
    t = scripted([])
    legacy = _legacy_key(t, SPACED)
    t._cache.set(legacy, ["an older, different translation", "second"])
    t._cache.set(t._page_key(SPACED), VALUE)

    t.translate_page(SPACED)
    assert t._cache.get(legacy) == ["an older, different translation", "second"]
