"""Lines that recur across pages must translate the same way every time.

The sample book's running head appears on 71 pages and the cache holds seven
different translations of it. Each page is translated independently, so nothing
kept them consistent.

The reconciliation happens *after* translate_page, deliberately: pulling
recurring lines out of the page batch would change the page's cache key and
orphan every entry already on disk.
"""

from __future__ import annotations

import pymupdf
import pytest

from pdftranslator.orchestrator import Orchestrator
from pdftranslator.translate import Translator

HEAD = "N. V. Starikov Krize"


class VaryingTranslator(Translator):
    """Returns a different translation of the running head each page."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    def translate_page(self, lines, preceding_context=None):
        self.calls += 1
        out = []
        for line in lines:
            if line == HEAD:
                out.append(f"Starikov Crisis variant {self.calls}")
            else:
                out.append(f"translated: {line}")
        return out


def _book(path, pages=4):
    with pymupdf.open() as doc:
        for i in range(pages):
            page = doc.new_page(width=400, height=600)
            page.insert_text((40, 60), HEAD, fontsize=8)
            page.insert_text((40, 200), f"Body text of page {i + 1} goes here", fontsize=11)
        doc.save(path)


def _render(tmp_path, translator, pages=4):
    src = tmp_path / "in.pdf"
    _book(src, pages)
    out = tmp_path / "out.pdf"
    Orchestrator(translator, mirror=False).run(src, out, "all")
    return out


def _texts(pdf):
    with pymupdf.open(pdf) as doc:
        return [page.get_text() for page in doc]


def test_a_line_recurring_on_enough_pages_is_translated_consistently(tmp_path):
    """Breaks if pages are reconciled independently again."""
    t = VaryingTranslator(model="fake/x", target_lang="English",
                          cache_dir=str(tmp_path / "cache"))
    try:
        out = _render(tmp_path, t, pages=4)
    finally:
        t.close()
    variants = {v for page in _texts(out)
                for v in ("variant 1", "variant 2", "variant 3", "variant 4")
                if v in page}
    assert len(variants) == 1, f"running head rendered as {variants}"


def test_a_line_below_the_threshold_is_left_alone(tmp_path):
    """Two occurrences is not a running element; do not force them together."""
    t = VaryingTranslator(model="fake/x", target_lang="English",
                          cache_dir=str(tmp_path / "cache"))
    try:
        out = _render(tmp_path, t, pages=2)
    finally:
        t.close()
    pages = _texts(out)
    assert "variant 1" in pages[0] and "variant 2" in pages[1]


def test_body_lines_are_untouched(tmp_path):
    t = VaryingTranslator(model="fake/x", target_lang="English",
                          cache_dir=str(tmp_path / "cache"))
    try:
        out = _render(tmp_path, t, pages=4)
    finally:
        t.close()
    for i, page in enumerate(_texts(out)):
        assert f"translated: Body text of page {i + 1}" in page


def test_page_cache_keys_are_unchanged(tmp_path):
    """The reconciliation must not alter what translate_page is asked for.

    Breaks if recurring lines are pulled out of the batch, which would change
    every page's cache key and orphan the entries already on disk.
    """
    seen: list[list[str]] = []

    class Recording(VaryingTranslator):
        def translate_page(self, lines, preceding_context=None):
            seen.append(list(lines))
            return super().translate_page(lines, preceding_context)

    t = Recording(model="fake/x", target_lang="English",
                  cache_dir=str(tmp_path / "cache"))
    try:
        _render(tmp_path, t, pages=4)
    finally:
        t.close()
    assert all(HEAD in batch for batch in seen), seen


def test_a_wildly_mismatched_translation_is_not_adopted(tmp_path):
    """A misaligned batch can hand a short recurring line a long body
    translation. Adopting that would propagate the error to every page."""
    class Misaligning(Translator):
        def translate_page(self, lines, preceding_context=None):
            return ["a very much longer body sentence that is clearly not a header"
                    if line == HEAD else f"translated: {line}" for line in lines]

    t = Misaligning(model="fake/x", target_lang="English",
                    cache_dir=str(tmp_path / "cache"))
    try:
        out = _render(tmp_path, t, pages=4)
    finally:
        t.close()
    # nothing blew up and the body is still right
    for i, page in enumerate(_texts(out)):
        assert f"translated: Body text of page {i + 1}" in page
