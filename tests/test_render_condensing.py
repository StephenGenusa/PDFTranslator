"""Fitting a translated line to its source line's width.

A translation is usually wider than the line it replaces, and line-by-line
correspondence forbids moving text between lines, so the width must be absorbed
in place. Three levers, cheapest visual cost first: borrow the outer margin,
condense horizontally (width only -- cap height is untouched, so the line still
reads as its neighbours' size), and only then shrink the font.
"""

from __future__ import annotations

import pymupdf
import pytest

from pdftranslator.fonts import FontBook
from pdftranslator.models import Line, PageContent
from pdftranslator.render import PdfRenderer

PAGE_W, PAGE_H = 420.0, 700.0
LEFT, RIGHT = 30.0, 330.0        # text column; 90pt of margin to the right


@pytest.fixture(scope="module")
def fonts() -> FontBook:
    return FontBook()


def _content(specs):
    """specs: (source_size, translated_text); all one left-aligned block."""
    lines, y = [], 60.0
    for size, translated in specs:
        line = Line(
            spans=[], text="source", bbox=(LEFT, y - size, RIGHT, y),
            baseline_y=y, left_x=LEFT, right_x=RIGHT, size=size, serif=False,
            bold=False, color=0, align="left", block_index=0,
        )
        line.translated = translated
        lines.append(line)
        y += size * 1.6
    return PageContent(page_index=0, width=PAGE_W, height=PAGE_H, frozen_spans=[],
                       lines=lines, text_left=LEFT, text_right=RIGHT)


def _drawn(renderer, content):
    """(fontsize, width, x0) of each drawn line, read back from the PDF."""
    src_doc = pymupdf.open()
    src = src_doc.new_page(width=PAGE_W, height=PAGE_H)
    writer = pymupdf.open()
    renderer.render_page(writer, src, content, mirror=False)
    out = []
    for b in writer[0].get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                if s["text"].strip():
                    out.append((s["bbox"][1], s["bbox"][2] - s["bbox"][0], s["bbox"][0]))
    out.sort()
    result = [(w, x0) for _, w, x0 in out]
    src_doc.close(); writer.close()
    return result


SHORT = "A short line."
FITS = "A line of translated text that fits."
LONG = "A translated line that is quite a lot longer than the column allows here"


def test_a_line_that_fits_is_drawn_at_the_original_size(fonts):
    """Breaks if the renderer shrinks text that had no need to shrink."""
    renderer = PdfRenderer(fonts, size_fidelity=0.55)
    content = _content([(11.0, FITS)] * 4)
    natural = fonts.measure(FITS, 11.0, serif=False)
    widths = [w for w, _ in _drawn(renderer, content)]
    assert widths[0] == pytest.approx(natural, abs=1.0), (widths[0], natural)


def test_an_overlong_line_is_condensed_not_shrunk(fonts):
    """Condensing preserves cap height, so the line still reads as full size.

    Breaks if the renderer goes back to absorbing width with font size alone.
    """
    renderer = PdfRenderer(fonts, size_fidelity=0.55)
    content = _content([(11.0, FITS)] * 4 + [(11.0, LONG)])
    drawn = _drawn(renderer, content)
    long_width = drawn[-1][0]
    # it was made to fit
    assert long_width <= (RIGHT - LEFT) + 14.0
    # but by narrowing, not by shrinking: still much wider than a font-shrunk
    # rendering of the same text would be at the same fit factor
    natural = fonts.measure(LONG, 11.0, serif=False)
    assert long_width / natural >= 0.60


def test_condensing_keeps_the_glyph_height(fonts):
    """The point of the lever: a condensed line is the same height as its
    neighbours, so a paragraph still looks uniform."""
    renderer = PdfRenderer(fonts, size_fidelity=0.55)
    content = _content([(11.0, FITS)] * 4 + [(11.0, LONG)])
    src_doc = pymupdf.open()
    src = src_doc.new_page(width=PAGE_W, height=PAGE_H)
    writer = pymupdf.open()
    renderer.render_page(writer, src, content, mirror=False)
    heights = []
    for b in writer[0].get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                if s["text"].strip():
                    heights.append(round(s["bbox"][3] - s["bbox"][1], 1))
    src_doc.close(); writer.close()
    assert len(set(heights)) == 1, heights


def test_an_ordinary_overlong_line_fits_the_column_and_its_margin(fonts):
    renderer = PdfRenderer(fonts, size_fidelity=0.55)
    content = _content([(11.0, FITS)] * 3 + [(11.0, LONG)])
    for width, x0 in _drawn(renderer, content):
        assert x0 + width <= RIGHT + 14.0


def test_a_pathological_line_stays_readable_rather_than_fitting(fonts):
    """A translation twice its line's width can only fit by becoming unreadable.

    The readability floor wins: the line keeps a legible size and runs into the
    blank outer margin instead. Breaks if the floor is removed and such lines
    collapse towards MIN_FONT_SIZE.
    """
    renderer = PdfRenderer(fonts, size_fidelity=0.55)
    content = _content([(11.0, FITS)] * 3 + [(11.0, LONG * 2)])
    drawn = _drawn(renderer, content)
    width, x0 = drawn[-1]
    assert x0 + width <= PAGE_W, "must not run off the page"
    src_doc = pymupdf.open()
    src = src_doc.new_page(width=PAGE_W, height=PAGE_H)
    writer = pymupdf.open()
    renderer.render_page(writer, src, content, mirror=False)
    sizes = sorted(s["size"] for b in writer[0].get_text("dict")["blocks"]
                   for ln in b.get("lines", []) for s in ln["spans"]
                   if s["text"].strip())
    src_doc.close(); writer.close()
    assert sizes[0] >= 11.0 * 0.60 * 0.80 - 0.5, sizes


def test_size_fidelity_raises_the_rendered_size(fonts):
    """The knob must actually do something, in the documented direction."""
    content_a = _content([(11.0, SHORT), (11.0, FITS), (11.0, LONG)])
    content_b = _content([(11.0, SHORT), (11.0, FITS), (11.0, LONG)])
    low = _drawn(PdfRenderer(fonts, size_fidelity=0.10), content_a)
    high = _drawn(PdfRenderer(fonts, size_fidelity=0.55), content_b)
    assert high[0][0] > low[0][0], (high[0][0], low[0][0])


def test_default_size_fidelity_is_the_chosen_value(fonts):
    assert PdfRenderer(fonts).size_fidelity == pytest.approx(0.55)
