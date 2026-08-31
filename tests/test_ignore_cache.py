"""--ignore-cache must be scoped to the pages being processed.

It used to call clear_cache(), discarding every entry in the store. Re-doing one
bad page therefore threw away the whole book -- and that is exactly what the
sentinel-repair warning tells users to do.
"""

from __future__ import annotations

import fitz
import pytest
from typer.testing import CliRunner

from pdftranslator import translate as translate_mod
from pdftranslator.cli import app
from pdftranslator.scanner import DocumentScanner
from pdftranslator.translate import Translator

runner = CliRunner()


class _Reply:
    def __init__(self, content: str) -> None:
        message = type("Message", (), {"content": content})()
        self.choices = [type("Choice", (), {"message": message})()]


@pytest.fixture
def book(tmp_path):
    path = tmp_path / "in.pdf"
    with fitz.open() as doc:
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 144), f"Labas rytas puslapis {i + 1}", fontsize=12)
        doc.save(path)
    return path


@pytest.fixture
def provider(monkeypatch):
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)
        return _Reply('{"0": "FRESH TRANSLATION", "1": "sentinel"}')

    monkeypatch.setattr(translate_mod.litellm, "completion", fake)
    monkeypatch.setattr(translate_mod.time, "sleep", lambda *_: None)
    return calls


def _seed(book, cache, page_index, value):
    """Put a known entry in the cache for one page of the book."""
    t = Translator(model="openrouter/google/gemini-2.5-flash-lite",
                   target_lang="English", cache_dir=str(cache))
    with fitz.open(book) as doc:
        content = DocumentScanner().extract(doc[page_index], page_index)
        key = t._page_key([ln.text for ln in content.lines])
    t._cache.set(key, value)
    t.close()
    return key


def _get(cache, key):
    t = Translator(model="x", target_lang="English", cache_dir=str(cache))
    value = t._cache.get(key)
    t.close()
    return value


def _run(book, tmp_path, cache, *extra):
    return runner.invoke(
        app,
        ["-i", str(book), "-o", str(tmp_path / "out.pdf"),
         "--cache-dir", str(cache), *extra],
    )


def test_ignore_cache_leaves_pages_outside_the_range_alone(book, tmp_path, provider):
    """The critical property: fixing page 1 must not discard page 3."""
    cache = tmp_path / "cache"
    key1 = _seed(book, cache, 0, ["STALE ONE"])
    key3 = _seed(book, cache, 2, ["KEEP ME"])

    result = _run(book, tmp_path, cache, "-p", "1", "--ignore-cache")
    assert result.exit_code == 0, result.output

    assert _get(cache, key3) == ["KEEP ME"], "an out-of-range entry was discarded"
    assert _get(cache, key1) == ["FRESH TRANSLATION"], "the target page was not refreshed"


def test_ignore_cache_retranslates_the_pages_in_range(book, tmp_path, provider):
    cache = tmp_path / "cache"
    _seed(book, cache, 0, ["STALE ONE"])
    _run(book, tmp_path, cache, "-p", "1", "--ignore-cache")
    assert len(provider) >= 1, "no API call made despite --ignore-cache"


def test_without_the_flag_a_cached_page_is_not_retranslated(book, tmp_path, provider):
    cache = tmp_path / "cache"
    _seed(book, cache, 0, ["STALE ONE"])
    _run(book, tmp_path, cache, "-p", "1")
    assert provider == [], "cached page should not have hit the API"
