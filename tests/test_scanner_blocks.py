"""Paragraph grouping and the alignment decided from it.

Alignment used to be decided per line by comparing that line against the page's
text column. An indented, justified block therefore looked centred -- every line
had a left gap, and a full-measure line's midpoint sits near the column midpoint
-- so footnote blocks came out with their lines flip-flopping between left and
centre. Alignment is a property of a block, not of a line.
"""

from __future__ import annotations

import pymupdf
import pytest

from pdftranslator.scanner import DocumentScanner

PAGE_W, PAGE_H = 400.0, 600.0


@pytest.fixture
def scanner() -> DocumentScanner:
    return DocumentScanner()


def _page(doc, lines):
    """lines: (x, baseline_y, size, text)."""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    for x, y, size, text in lines:
        page.insert_text((x, y), text, fontsize=size)
    return page


def _by_text(content, needle):
    for line in content.lines:
        if needle in line.text:
            return line
    raise AssertionError(f"line {needle!r} not extracted")


# --------------------------------------------------------------- grouping
def test_consecutive_lines_of_one_paragraph_share_a_block(scanner):
    with pymupdf.open() as doc:
        page = _page(doc, [
            (60, 200, 11, "First line of a single paragraph of body text here"),
            (60, 216, 11, "second line of that same paragraph continuing along"),
            (60, 232, 11, "and a third line of the very same paragraph again"),
        ])
        content = scanner.extract(page, 0)
    ids = {line.block_index for line in content.lines}
    assert len(ids) == 1, [(l.text[:20], l.block_index) for l in content.lines]


def test_a_distant_paragraph_is_a_separate_block(scanner):
    with pymupdf.open() as doc:
        page = _page(doc, [
            (60, 200, 11, "First paragraph first line of body text goes here"),
            (60, 216, 11, "first paragraph second line of body text goes here"),
            (60, 400, 11, "A much later paragraph after a large vertical gap"),
        ])
        content = scanner.extract(page, 0)
    assert (_by_text(content, "First paragraph").block_index
            != _by_text(content, "much later").block_index)


def test_a_different_font_size_starts_a_new_block(scanner):
    with pymupdf.open() as doc:
        page = _page(doc, [
            (60, 200, 11, "Body text line one of this paragraph here"),
            (60, 216, 11, "body text line two of this paragraph here"),
            (60, 236, 8, "a footnote in smaller type immediately below"),
        ])
        content = scanner.extract(page, 0)
    assert (_by_text(content, "Body text").block_index
            != _by_text(content, "footnote").block_index)


# -------------------------------------------------------------- alignment
def test_an_indented_justified_block_is_left_aligned(scanner):
    """The footnote regression: indented block, lines of differing length."""
    with pymupdf.open() as doc:
        page = _page(doc, [
            (90, 300, 9, "On this occasion I want to remind you that the demand was"),
            (90, 314, 9, "made even of Nicholas Romanov and the demanders achieved"),
            (90, 328, 9, "their goal such a government went down in history as the"),
            (90, 342, 9, "Provisional Government which did not last very long at all"),
            (90, 356, 9, "although they achieved a record in 242 days"),
        ])
        content = scanner.extract(page, 0)
    aligns = [line.align for line in content.lines]
    assert set(aligns) == {"left"}, aligns


def test_a_block_sharing_a_left_edge_is_left_aligned_even_when_indented(scanner):
    with pymupdf.open() as doc:
        page = _page(doc, [
            (40, 100, 11, "Full width body line that reaches across the column"),
            (40, 116, 11, "another full width body line reaching across as well"),
            (40, 132, 11, "and a third full width line of the body text here too"),
            (110, 300, 9, "indented note first line here"),
            (110, 314, 9, "indented note second line here"),
            (110, 328, 9, "indented note third line"),
        ])
        content = scanner.extract(page, 0)
    notes = [l for l in content.lines if "indented note" in l.text]
    assert len(notes) == 3
    assert {l.align for l in notes} == {"left"}, [l.align for l in notes]


def test_a_genuinely_centred_heading_is_still_detected(scanner):
    with pymupdf.open() as doc:
        page = _page(doc, [
            (40, 100, 11, "Full width body line that reaches across the column"),
            (40, 116, 11, "another full width body line reaching across as well"),
            (40, 132, 11, "and a third full width line of the body text here too"),
            # genuinely centred on the 40..289 text column (midpoint 164.7)
            (127, 250, 16, "A Heading"),
        ])
        content = scanner.extract(page, 0)
    assert _by_text(content, "A Heading").align == "center"


def test_a_heading_that_is_not_centred_is_not_called_centred(scanner):
    """Guards the tolerance: an off-centre display line is left aligned."""
    with pymupdf.open() as doc:
        page = _page(doc, [
            (40, 100, 11, "Full width body line that reaches across the column"),
            (40, 116, 11, "another full width body line reaching across as well"),
            (40, 132, 11, "and a third full width line of the body text here too"),
            (150, 250, 16, "A Heading"),
        ])
        content = scanner.extract(page, 0)
    assert _by_text(content, "A Heading").align == "left"
