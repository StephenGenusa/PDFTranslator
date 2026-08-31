"""Render translated content by overlaying it on the original page.

Composition model (general, layout-preserving):

1. Stamp the original page onto the output (the *base layer*). Everything we do
   not translate — images, rules, backgrounds, page numbers, math, vertical
   labels — is therefore preserved exactly, with no per-element reconstruction.
2. For each translatable line, paint an opaque rectangle over the original text,
   filled with the *sampled local background colour*, so the original text is
   hidden and the box blends into the page (white page -> white box; dark banner
   -> dark box).
3. Draw the translated line at the original baseline and left edge. One shrink
   factor is chosen for the whole page (see ``_page_scale``) so the size
   relationships the source had are preserved; text colour is the original
   colour when it contrasts with the background, otherwise black-on-light /
   white-on-dark.

We deliberately do NOT reflow: each source line maps to one translated line in
its original position, so the page layout is preserved rather than re-typeset.
"""

from __future__ import annotations

import logging

import pymupdf

from .fonts import FontBook
from .models import BBox, Line, PageContent

logger = logging.getLogger(__name__)

MIN_FONT_SIZE = 4.0

# How closely to hold a block to its source font size. It is the quantile of the
# block's per-line width requirements that the chosen size must satisfy using
# gentle condensing alone; the rest are condensed harder or, failing that,
# shrunk. Higher keeps text nearer the original size at the cost of compressing
# the longest lines more. Above roughly 0.70 the size is already pinned at the
# original and the value stops having any effect.
DEFAULT_SIZE_FIDELITY = 0.55

# Condensing scales width only -- cap height is untouched -- so a condensed line
# still reads as the same size as its neighbours. Invisible down to _CONDENSE_SOFT,
# acceptable to _CONDENSE_HARD; past that the font has to give.
_CONDENSE_SOFT = 0.92
_CONDENSE_HARD = 0.80
_MARGIN_BORROW = 12.0  # points of outer margin a line may run into
_SIZE_FLOOR = 0.75     # a block never shrinks below this fraction of source size
_LINE_FLOOR = 0.60     # a single overlong line may go this far
_MASK_PAD_X = 1.5  # points; cover OCR boxes that sit just inside the inked glyphs
_MASK_PAD_Y = 1.5
_CONTRAST_MIN = 0.40  # min luminance gap to keep the original text colour
_SAMPLE_ZOOM = 2.0  # render scale for background sampling


class PdfRenderer:
    def __init__(
        self, fonts: FontBook, size_fidelity: float = DEFAULT_SIZE_FIDELITY
    ) -> None:
        self._fonts = fonts
        self.size_fidelity = size_fidelity

    def render_page(
        self,
        writer: pymupdf.Document,
        source_page: pymupdf.Page,
        content: PageContent,
        *,
        mirror: bool,
    ) -> None:
        w, h = content.width, content.height
        target_w = w * 2 if mirror else w
        offset_x = w if mirror else 0.0

        page = writer.new_page(width=target_w, height=h)
        self._fonts.register(page)

        # Base layer: the original page. In side-by-side mode it appears on both
        # halves (left stays pristine; right gets the overlay).
        parent, pno = source_page.parent, source_page.number
        if mirror:
            page.show_pdf_page(pymupdf.Rect(0, 0, w, h), parent, pno)
        page.show_pdf_page(pymupdf.Rect(offset_x, 0, offset_x + w, h), parent, pno)

        if not content.lines:
            return

        sampler = _BackgroundSampler(source_page)

        # Resolve every line first, then draw all masks, then all text — so a
        # lower line's mask can never clip the line drawn above it.
        ops = [
            op
            for block in self._blocks(content)
            for op in self._plan_block(block, content, sampler, offset_x)
        ]
        for op in ops:
            page.draw_rect(op.mask, color=op.bg, fill=op.bg, width=0)
        for op in ops:
            self._draw(page, op)

    # ----------------------------------------------------------- block sizing
    @staticmethod
    def _blocks(content: PageContent) -> list[list[Line]]:
        grouped: dict[int, list[Line]] = {}
        for line in content.lines:
            grouped.setdefault(line.block_index, []).append(line)
        return [grouped[key] for key in sorted(grouped)]

    def _block_extent(self, block: list[Line], content: PageContent) -> tuple[float, float, float]:
        """(left, right, limit) for a block -- limit includes borrowed margin."""
        left = min(line.left_x for line in block)
        right = max(line.right_x for line in block)
        room = min(_MARGIN_BORROW, max(0.0, content.width - right))
        return left, right, right + room

    def _width_needs(self, block: list[Line], content: PageContent) -> list[float]:
        """Per line: the fraction of its natural width that has to fit."""
        left, _right, limit = self._block_extent(block, content)
        needs = []
        for line in block:
            text = self._text_of(line)
            natural = self._fonts.measure(text, line.size, line.serif)
            available = (limit - line.left_x) if line.align == "left" else (limit - left)
            needs.append(min(1.0, available / natural) if natural > 0 else 1.0)
        return needs

    def _block_scale(self, block: list[Line], content: PageContent) -> float:
        """One font scale for the block, from the size-fidelity quantile."""
        needs = self._width_needs(block, content)
        if not needs:
            return 1.0
        ordered = sorted(needs)
        index = max(0, min(len(ordered) - 1, int(self.size_fidelity * len(ordered))))
        return max(_SIZE_FLOOR, min(1.0, ordered[index] / _CONDENSE_SOFT))

    @staticmethod
    def _text_of(line: Line) -> str:
        return line.translated if line.translated is not None else line.text

    # ------------------------------------------------------------------ lines
    def _plan_block(
        self,
        block: list[Line],
        content: PageContent,
        sampler: "_BackgroundSampler",
        offset_x: float,
    ) -> list["_LineOp"]:
        left, right, _limit = self._block_extent(block, content)
        scale = self._block_scale(block, content)
        needs = self._width_needs(block, content)
        ops: list[_LineOp] = []

        for line, need in zip(block, needs):
            text = self._text_of(line)
            if not text.strip():
                continue

            line_scale = scale
            condense = min(1.0, need / line_scale) if line_scale > 0 else 1.0
            if condense < _CONDENSE_HARD:
                # Condensing alone cannot absorb it; this one line gives up size.
                condense = _CONDENSE_HARD
                line_scale = max(_LINE_FLOOR, need / _CONDENSE_HARD)

            fontsize = max(MIN_FONT_SIZE, line.size * line_scale)
            width = self._fonts.measure(text, fontsize, line.serif) * condense

            bg = sampler.sample(line.bbox)
            mask = pymupdf.Rect(
                line.bbox[0] - _MASK_PAD_X + offset_x,
                line.bbox[1] - _MASK_PAD_Y,
                line.bbox[2] + _MASK_PAD_X + offset_x,
                line.bbox[3] + _MASK_PAD_Y,
            )

            if line.align == "right":
                start_x = right + offset_x - width
            elif line.align == "center":
                start_x = (left + right) / 2.0 + offset_x - width / 2.0
            else:
                start_x = line.left_x + offset_x

            # Never start before this half of the sheet -- that is what used to
            # draw a translation across the pristine original. The start is not
            # pulled leftward to make a line fit: a line keeps its source line's
            # x, so it stays aligned with the rest of its block.
            start_x = max(start_x, offset_x + 1.0)

            # Staying on the page outranks the readability floor: an extreme
            # line shrinks past it rather than running off the sheet.
            page_right = offset_x + content.width
            if start_x + width > page_right - 1.0:
                room = max(1.0, page_right - 1.0 - start_x)
                shrink = room / width
                fontsize = max(MIN_FONT_SIZE, fontsize * shrink)
                width = self._fonts.measure(text, fontsize, line.serif) * condense

            ops.append(_LineOp(
                text=text,
                point=pymupdf.Point(start_x, line.baseline_y),
                fontsize=fontsize,
                serif=line.serif,
                color=self._text_color(bg, line.color),
                mask=mask,
                bg=bg,
                condense=condense,
            ))
        return ops

    def _draw(self, page: pymupdf.Page, op: "_LineOp") -> None:
        try:
            page.insert_text(
                op.point,
                op.text,
                fontname=self._fonts.name_for(op.serif),
                fontsize=op.fontsize,
                color=op.color,
                # Width-only transform: the glyphs narrow, their height does not.
                morph=(op.point, pymupdf.Matrix(op.condense, 1.0))
                if op.condense < 0.999 else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("insert_text failed (%s)", exc)

    # ------------------------------------------------------------------ colour
    @staticmethod
    def _text_color(bg: tuple[float, float, float], color_int: int) -> tuple[float, float, float]:
        orig = _raw_rgb(color_int)
        bg_lum = _luminance(bg)
        if abs(_luminance(orig) - bg_lum) >= _CONTRAST_MIN:
            return orig
        return (0.0, 0.0, 0.0) if bg_lum > 0.5 else (1.0, 1.0, 1.0)


class _LineOp:
    """A resolved draw operation: where to mask and what text to stamp."""

    __slots__ = ("text", "point", "fontsize", "serif", "color", "mask", "bg",
                 "condense")

    def __init__(self, *, text, point, fontsize, serif, color, mask, bg,
                 condense=1.0) -> None:
        self.text = text
        self.point = point
        self.fontsize = fontsize
        self.serif = serif
        self.color = color
        self.mask = mask
        self.bg = bg
        self.condense = condense


class _BackgroundSampler:
    """Samples the original page's local background colour around a text box.

    The page is rasterised once; for a given line we read pixels just above and
    below the box (the inter-line gap, almost always background) and take the
    median, which is robust to stray ink.
    """

    def __init__(self, page: pymupdf.Page, zoom: float = _SAMPLE_ZOOM) -> None:
        self._zoom = zoom
        self._pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)

    def sample(self, bbox: BBox) -> tuple[float, float, float]:
        x0, y0, x1, y1 = bbox
        ys = [y0 - 2.0, y0 - 1.0, y1 + 1.0, y1 + 2.0]
        xs = [x0 + (x1 - x0) * t for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
        samples: list[tuple[int, int, int]] = []
        for py in ys:
            for px in xs:
                rgb = self._read(px, py)
                if rgb is not None:
                    samples.append(rgb)
        if not samples:
            return (1.0, 1.0, 1.0)  # assume white if we couldn't sample
        samples.sort(key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
        r, g, b = samples[len(samples) // 2]
        return (r / 255.0, g / 255.0, b / 255.0)

    def _read(self, x: float, y: float) -> tuple[int, int, int] | None:
        px = int(x * self._zoom)
        py = int(y * self._zoom)
        if px < 0 or py < 0 or px >= self._pix.width or py >= self._pix.height:
            return None
        try:
            return self._pix.pixel(px, py)
        except Exception:  # noqa: BLE001
            return None


def _raw_rgb(color_int: int) -> tuple[float, float, float]:
    """Packed sRGB int -> (r, g, b) floats."""
    b = (color_int & 255) / 255.0
    g = ((color_int >> 8) & 255) / 255.0
    r = ((color_int >> 16) & 255) / 255.0
    return (r, g, b)


def _luminance(rgb: tuple[float, float, float]) -> float:
    return rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
