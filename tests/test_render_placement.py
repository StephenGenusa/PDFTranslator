"""Where translated text is allowed to land on the output page."""

from __future__ import annotations

import fitz
import pytest

from pdftranslator.fonts import FontBook
from pdftranslator.render import PdfRenderer
from pdftranslator.scanner import DocumentScanner

PAGE_W, PAGE_H = 300.0, 200.0
MARKER = "ZZZQQQ"


@pytest.fixture(scope="module")
def renderer() -> PdfRenderer:
    return PdfRenderer(FontBook())


def _render(renderer, translated: str, *, mirror: bool):
    """Render one synthetic page and return (output_page, keep_alive)."""
    src_doc = fitz.open()
    src = src_doc.new_page(width=PAGE_W, height=PAGE_H)
    src.insert_text((20, 100), "Trumpas", fontsize=12)

    content = DocumentScanner().extract(src, 0)
    assert len(content.lines) == 1, content.lines
    content.lines[0].translated = translated

    writer = fitz.open()
    renderer.render_page(writer, src, content, mirror=mirror)
    return writer[0], (src_doc, writer)


def test_an_overlong_line_never_lands_on_the_original_half(renderer):
    """In side-by-side mode the left half must stay a pristine copy of the
    source. A translation too long to shrink into the column was clamped with
    max(1.0, ...) and drawn from x=1 -- straight across the original.

    Breaks if the horizontal clamp stops being relative to the output half.
    """
    long_line = " ".join([MARKER] * 60)
    page, _keep = _render(renderer, long_line, mirror=True)

    hits = page.search_for(MARKER)
    assert hits, "translated text was not drawn at all"
    leftmost = min(r.x0 for r in hits)
    assert leftmost >= PAGE_W, (
        f"translation starts at x={leftmost:.1f}, inside the original half "
        f"(which ends at x={PAGE_W})"
    )


def test_a_normal_line_keeps_its_original_left_edge(renderer):
    """The clamp must not disturb ordinary lines."""
    page, _keep = _render(renderer, f"{MARKER} trumpas", mirror=True)

    hits = page.search_for(MARKER)
    assert hits
    assert min(r.x0 for r in hits) == pytest.approx(PAGE_W + 20.0, abs=2.0)


def test_substitute_mode_keeps_an_overlong_line_on_the_page(renderer):
    long_line = " ".join([MARKER] * 60)
    page, _keep = _render(renderer, long_line, mirror=False)

    hits = page.search_for(MARKER)
    assert hits
    assert min(r.x0 for r in hits) >= 0.0
