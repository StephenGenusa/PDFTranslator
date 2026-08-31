"""Reproducing the source's bold and italic styling.

The scanner has always read the bold flag and the renderer has always ignored
it, so emphasis in the original was flattened to regular weight.
"""

from __future__ import annotations

import pymupdf
import pytest

from pdftranslator.fonts import FontBook
from pdftranslator.models import Line, PageContent
from pdftranslator.render import PdfRenderer
from pdftranslator.scanner import DocumentScanner

PAGE_W, PAGE_H = 420.0, 300.0


@pytest.fixture(scope="module")
def fonts() -> FontBook:
    return FontBook()


# --------------------------------------------------------------- extraction
def test_italic_is_read_from_the_source(tmp_path):
    """PyMuPDF reports italic as bit 1; Line had no field for it at all."""
    scanner = DocumentScanner()
    with pymupdf.open() as doc:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_text((40, 100), "Slanted words here", fontsize=12, fontname="tiit")
        content = scanner.extract(page, 0)
    assert content.lines[0].italic is True


def test_bold_is_read_from_the_source(tmp_path):
    scanner = DocumentScanner()
    with pymupdf.open() as doc:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_text((40, 100), "Heavy words here", fontsize=12, fontname="hebo")
        content = scanner.extract(page, 0)
    assert content.lines[0].bold is True


def test_plain_text_is_neither(tmp_path):
    scanner = DocumentScanner()
    with pymupdf.open() as doc:
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        page.insert_text((40, 100), "Ordinary words here", fontsize=12, fontname="helv")
        content = scanner.extract(page, 0)
    assert content.lines[0].bold is False
    assert content.lines[0].italic is False


# ------------------------------------------------------------------- fonts
@pytest.mark.parametrize(
    ("serif", "bold", "italic", "expected"),
    [
        (False, False, False, "NotoSans-Regular"),
        (False, True, False, "NotoSans-Bold"),
        (False, False, True, "NotoSans-Italic"),
        (True, False, False, "NotoSerif-Regular"),
        (True, True, False, "NotoSerif-Bold"),
        (True, False, True, "NotoSerif-Italic"),
    ],
)
def test_each_style_resolves_to_its_own_face(fonts, serif, bold, italic, expected):
    assert expected in str(fonts.path_for(serif, bold, italic))


def test_bold_measures_wider_than_regular(fonts):
    """A real face is used, not the regular one relabelled."""
    text = "Measured text for comparison"
    assert fonts.measure(text, 12.0, serif=False, bold=True) != pytest.approx(
        fonts.measure(text, 12.0, serif=False), abs=0.01
    )


def test_bold_and_italic_together_prefer_bold(fonts):
    assert "Bold" in str(fonts.path_for(False, True, True))


# ----------------------------------------------------------------- rendering
def _line(text, *, bold=False, italic=False):
    line = Line(spans=[], text="source", bbox=(30, 80, 300, 96), baseline_y=92,
                left_x=30, right_x=300, size=12.0, serif=False, bold=bold,
                color=0, align="left", block_index=0, italic=italic)
    line.translated = text
    return line


def test_a_bold_line_is_drawn_with_the_bold_face(fonts):
    """Breaks if the renderer goes back to ignoring Line.bold."""
    renderer = PdfRenderer(fonts)
    content = PageContent(page_index=0, width=PAGE_W, height=PAGE_H, frozen_spans=[],
                          lines=[_line("Emphasised translation", bold=True)],
                          text_left=30, text_right=300)
    src_doc = pymupdf.open()
    src = src_doc.new_page(width=PAGE_W, height=PAGE_H)
    writer = pymupdf.open()
    renderer.render_page(writer, src, content, mirror=False)
    used = {s["font"] for b in writer[0].get_text("dict")["blocks"]
            for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()}
    src_doc.close(); writer.close()
    assert any("Bold" in f for f in used), used


def test_an_italic_line_is_drawn_with_the_italic_face(fonts):
    renderer = PdfRenderer(fonts)
    content = PageContent(page_index=0, width=PAGE_W, height=PAGE_H, frozen_spans=[],
                          lines=[_line("Emphasised translation", italic=True)],
                          text_left=30, text_right=300)
    src_doc = pymupdf.open()
    src = src_doc.new_page(width=PAGE_W, height=PAGE_H)
    writer = pymupdf.open()
    renderer.render_page(writer, src, content, mirror=False)
    used = {s["font"] for b in writer[0].get_text("dict")["blocks"]
            for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()}
    src_doc.close(); writer.close()
    assert any("Italic" in f for f in used), used
