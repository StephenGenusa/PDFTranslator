"""Behaviour when the model provider misbehaves.

A book is hundreds of sequential calls; a transient 429 or timeout is expected,
not exceptional. It must cost a retry, never a page.
"""

from __future__ import annotations

import fitz
import pytest

from pdftranslator.orchestrator import Orchestrator
from pdftranslator.translate import Translator

HEALTHY = '{"0": "First line", "1": "Second line", "2": "sentinel"}'
SOURCE = ["Pirma eilute", "Antra eilute"]


# ------------------------------------------------------------------- retrying
def test_a_transient_api_error_is_retried_and_the_page_survives(scripted):
    """Breaks if _complete stops retrying: the error escapes translate_page and
    the orchestrator emits a blank page for content that was merely delayed."""
    t = scripted([RuntimeError("429 rate limited"), HEALTHY], max_attempts=3)
    assert t.translate_page(SOURCE) == ["First line", "Second line"]


def test_retries_are_bounded_by_max_attempts(scripted):
    """An outage must not spin forever, and must not fan out per line."""
    t = scripted([RuntimeError("503 unavailable")] * 50, max_attempts=2)
    out = t.translate_page(SOURCE)

    assert out == SOURCE, "source text kept when the provider is unreachable"
    assert t.stats["api_calls"] == 2, t.stats
    assert t._cache.get(t._page_key(SOURCE)) is None


def test_a_persistent_outage_does_not_raise_out_of_translate_page(scripted):
    """The exception escaping here is what blanked the page."""
    t = scripted([RuntimeError("503 unavailable")] * 50, max_attempts=1)
    assert t.translate_page(SOURCE) == SOURCE


# ------------------------------------------------- orchestrator failure path
class ExplodingTranslator(Translator):
    def translate_page(self, lines, preceding_context=None):
        raise RuntimeError("provider exploded")

    def _complete(self, messages):
        raise AssertionError("unreachable")


def _make_pdf(path, pages: int = 2) -> None:
    with fitz.open() as doc:
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((72, 144), f"Labas rytas puslapis {i + 1}", fontsize=12)
        doc.save(path)


def test_a_failed_page_keeps_the_original_content_instead_of_going_blank(tmp_path):
    """Losing a page entirely is the worst possible outcome: the reader gets a
    white rectangle where the text was, and only a log line says so.

    Breaks if the page-failure path goes back to emitting an empty page.
    """
    src = tmp_path / "in.pdf"
    _make_pdf(src, pages=2)
    out = tmp_path / "out.pdf"

    t = ExplodingTranslator(model="fake/x", target_lang="English",
                            cache_dir=str(tmp_path / "cache"))
    try:
        Orchestrator(t, mirror=False).run(src, out, "all")
    finally:
        t.close()

    with fitz.open(out) as doc:
        assert doc.page_count == 2
        for i in range(2):
            text = doc[i].get_text().strip()
            assert text, f"page {i + 1} is blank"
            assert f"Labas rytas puslapis {i + 1}" in text
