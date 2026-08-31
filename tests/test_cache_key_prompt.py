"""The prompt is part of what a cached translation means.

The key covered the model, the target language and the source text, but not the
instructions. Editing a prompt therefore reused translations produced under the
old one, silently, unless someone remembered to bump _CACHE_VERSION by hand.
"""

from __future__ import annotations

import hashlib

import pytest

from pdftranslator import translate as translate_mod

LINES = ["Pirma eilute yra ilga", "Antra eilute taip pat"]
VALUE = ["First line is long", "Second line as well"]


def test_editing_the_prompt_changes_the_key(scripted, monkeypatch):
    """Breaks if the prompt stops being part of the key."""
    t = scripted([])
    before = t._page_key(LINES)
    monkeypatch.setattr(translate_mod, "_SYSTEM_BATCH",
                        translate_mod._SYSTEM_BATCH + " Also rhyme.")
    assert t._page_key(LINES) != before


def test_editing_the_per_line_prompt_also_changes_the_key(scripted, monkeypatch):
    t = scripted([])
    before = t._page_key(LINES)
    monkeypatch.setattr(translate_mod, "_SYSTEM_LINE",
                        translate_mod._SYSTEM_LINE + " Be terse.")
    assert t._page_key(LINES) != before


def test_the_key_is_stable_when_nothing_changes(scripted):
    a, b = scripted([]), scripted([])
    assert a._page_key(LINES) == b._page_key(LINES)


def test_max_tokens_does_not_change_the_key(scripted):
    """A response limit does not change what a correct translation is, and
    keying on it would throw away good work whenever it is tuned."""
    a = scripted([], max_tokens=4096)
    b = scripted([], max_tokens=16384)
    assert a._page_key(LINES) == b._page_key(LINES)


# ------------------------------------------------------------- not losing work
def _key_without_prompt(t, lines):
    """The scheme in use before the prompt was part of the key."""
    import re
    joined = "\x1f".join(re.sub(r"\s+", " ", ln).strip() for ln in lines)
    blob = f"v{translate_mod._CACHE_VERSION}|{t.litellm_model}|{t.target_lang}|{joined}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_entries_from_before_the_prompt_was_keyed_are_still_found(scripted):
    """Those translations were paid for. They must not be orphaned."""
    t = scripted([])
    t._cache.set(_key_without_prompt(t, LINES), VALUE)
    assert t.translate_page(LINES) == VALUE
    assert t.stats["api_calls"] == 0


def test_such_an_entry_is_migrated_to_the_current_key(scripted):
    t = scripted([])
    old = _key_without_prompt(t, LINES)
    t._cache.set(old, VALUE)
    t.translate_page(LINES)
    assert t._cache.get(t._page_key(LINES)) == VALUE
    assert t._cache.get(old) is None, "the converted duplicate should be pruned"


def test_the_oldest_raw_text_scheme_is_still_found(scripted):
    """Two schemes back: before whitespace normalisation."""
    t = scripted([])
    raw = "\x1f".join(LINES)
    blob = f"v{translate_mod._CACHE_VERSION}|{t.litellm_model}|{t.target_lang}|{raw}"
    t._cache.set(hashlib.sha256(blob.encode("utf-8")).hexdigest(), VALUE)
    assert t.translate_page(LINES) == VALUE
    assert t.stats["api_calls"] == 0
