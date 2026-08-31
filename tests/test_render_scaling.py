"""How translated lines are scaled to fit their column.

A translation is usually longer than its source, so lines must shrink. The
scaling must keep the page a faithful rendering of the original:

  * lines that look alike in the source look alike in the output
  * a heading stays larger than its body text
  * nothing overflows the column
  * and none of it depends on how the PDF parser happens to group text into
    blocks, which is a parser detail that changes between releases

Rendered sizes are read back out of the produced PDF, not computed alongside
the production code.
"""

from __future__ import annotations

import pymupdf
import pytest

from pdftranslator.fonts import FontBook
from pdftranslator.models import Line, PageContent
from pdftranslator.render import PdfRenderer

PAGE_W, PAGE_H = 420.0, 700.0
TEXT_LEFT, TEXT_RIGHT = 20.0, 400.0


@pytest.fixture(scope="module")
def renderer() -> PdfRenderer:
    return PdfRenderer(FontBook())


def _content(specs, *, block_indices=None) -> PageContent:
    """specs: (source_size, translated_text). One line per entry."""
    lines, y = [], 60.0
    for i, (size, translated) in enumerate(specs):
        line = Line(
            spans=[], text="source", bbox=(TEXT_LEFT, y - size, TEXT_LEFT + 10.0, y),
            baseline_y=y, left_x=TEXT_LEFT, right_x=TEXT_RIGHT, size=size,
            serif=False, bold=False, color=0, align="left",
            block_index=i if block_indices is None else block_indices[i],
        )
        line.translated = translated
        lines.append(line)
        y += size * 1.6
    return PageContent(
        page_index=0, width=PAGE_W, height=PAGE_H, frozen_spans=[], lines=lines,
        text_left=TEXT_LEFT, text_right=TEXT_RIGHT,
    )


def _rendered_sizes(renderer, content) -> list[float]:
    """Font sizes actually drawn, in vertical order, read back from the PDF."""
    src_doc = pymupdf.open()
    src = src_doc.new_page(width=PAGE_W, height=PAGE_H)
    writer = pymupdf.open()
    renderer.render_page(writer, src, content, mirror=False)
    spans = [
        (s["bbox"][1], round(s["size"], 3))
        for b in writer[0].get_text("dict")["blocks"]
        for line in b.get("lines", [])
        for s in line["spans"]
        if s["text"].strip()
    ]
    spans.sort()
    out = [size for _, size in spans]
    src_doc.close()
    writer.close()
    return out


BODY = "This is a line of translated body text that is a bit long."
LONG = "This is a very much longer translated line " * 4


# ------------------------------------------------------------------ uniformity
def test_lines_of_one_source_size_render_at_one_size(renderer):
    """The visible defect: neighbouring body lines rendered at different sizes.

    Real body text varies in length, so per-line fitting gives each line its own
    size and the paragraph looks ragged. Breaks if scaling goes back to being
    decided per line or per block.
    """
    content = _content([
        (11.0, "Short line."),
        (11.0, BODY),
        (11.0, "A medium length translated line of text here."),
        (11.0, BODY + " With a little more on the end."),
        (11.0, "Another short one."),
        (11.0, BODY),
    ])
    sizes = _rendered_sizes(renderer, content)
    assert len(set(sizes)) == 1, sizes


def test_block_numbering_does_not_change_the_output(renderer):
    """PyMuPDF 1.28 splits the same text into ~2.2x more blocks than 1.27.

    Breaks if the scale is derived from block grouping again -- which is what
    made rendered output depend on the parser version.
    """
    specs = [(11.0, BODY)] * 4 + [(11.0, LONG)] + [(11.0, BODY)] * 3
    one_block = _rendered_sizes(renderer, _content(specs, block_indices=[0] * 8))
    per_line = _rendered_sizes(renderer, _content(specs, block_indices=list(range(8))))
    paired = _rendered_sizes(renderer, _content(specs, block_indices=[0, 0, 1, 1, 2, 2, 3, 3]))
    assert one_block == per_line == paired


# -------------------------------------------------------------------- hierarchy
def test_a_heading_stays_larger_than_its_body_text(renderer):
    """A heading rendered no larger than body text misrepresents the page."""
    content = _content([(28.0, "A Fairly Long Translated Heading")] + [(10.0, BODY)] * 5)
    sizes = _rendered_sizes(renderer, content)
    assert sizes[0] > max(sizes[1:]), sizes


def test_source_size_order_is_preserved(renderer):
    content = _content(
        [(30.0, "Heading text here"), (18.0, "Subheading text here")]
        + [(10.0, BODY)] * 4
    )
    sizes = _rendered_sizes(renderer, content)
    assert sizes[0] > sizes[1] > max(sizes[2:]), sizes


# -------------------------------------------------------------------------- fit
def test_no_line_is_drawn_wider_than_its_column(renderer):
    content = _content([(11.0, BODY)] * 4 + [(11.0, LONG)])
    src_doc = pymupdf.open()
    src = src_doc.new_page(width=PAGE_W, height=PAGE_H)
    writer = pymupdf.open()
    renderer.render_page(writer, src, content, mirror=False)
    for block in writer[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["text"].strip():
                    assert span["bbox"][2] <= TEXT_RIGHT + 2.0, span["text"][:40]
    src_doc.close()
    writer.close()


def test_one_pathological_line_does_not_shrink_the_whole_page(renderer):
    """A single runaway translation must shrink itself, not the page around it.

    Breaks if the page scale is taken from the single worst line.
    """
    specs = [(11.0, BODY)] * 19 + [(11.0, LONG * 3)]
    sizes = _rendered_sizes(renderer, content := _content(specs))
    normal = _rendered_sizes(renderer, _content([(11.0, BODY)] * 20))
    assert max(sizes) == pytest.approx(max(normal), abs=0.01), (max(sizes), max(normal))
    assert min(sizes) < max(sizes), "the runaway line should shrink on its own"
