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
# Quantile of per-line fit factors used as the page-wide shrink factor. 0.0
# would be the worst line on the page, letting one runaway translation shrink
# everything; 0.10 lets the few worst lines shrink on their own instead.
_SCALE_QUANTILE = 0.10
_MASK_PAD_X = 1.5  # points; cover OCR boxes that sit just inside the inked glyphs
_MASK_PAD_Y = 1.5
_CONTRAST_MIN = 0.40  # min luminance gap to keep the original text colour
_SAMPLE_ZOOM = 2.0  # render scale for background sampling


class PdfRenderer:
    def __init__(self, fonts: FontBook) -> None:
        self._fonts = fonts

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
        scale = self._page_scale(content)

        # Resolve every line first, then draw all masks, then all text — so a
        # lower line's mask can never clip the line drawn above it.
        ops = [
            op
            for line in content.lines
            if (op := self._plan_line(line, content, scale, sampler, offset_x)) is not None
        ]
        for op in ops:
            page.draw_rect(op.mask, color=op.bg, fill=op.bg, width=0)
        for op in ops:
            self._draw(page, op)

    # ------------------------------------------------------------- page scale
    def _page_scale(self, content: PageContent) -> float:
        """One shrink factor for the whole page.

        A single factor is what keeps the page faithful: scaling every line by
        the same amount preserves the size *relationships* the source had, so a
        heading stays proportionally larger than its body text and lines that
        looked alike still look alike.

        This used to be one factor per block, averaged over the block's lines,
        with each line then taking ``min(block_factor, its_own_fit)``. Two things
        went wrong. The per-line minimum meant neighbouring body lines still got
        different sizes (measured: 49% of adjacent same-size lines differed).
        And the grouping came from PyMuPDF's block numbering, a parser detail --
        1.28 splits the same text into 2.2x more blocks than 1.27, which moved
        the rendering with it.

        The factor is the ``_SCALE_QUANTILE`` quantile of the per-line fits
        rather than the minimum, so one runaway translation shrinks itself (see
        ``_plan_line``) instead of shrinking the entire page around it.
        """
        fits = sorted(self._fit_scale(line, content) for line in content.lines)
        if not fits:
            return 1.0
        index = max(0, min(len(fits) - 1, int(_SCALE_QUANTILE * len(fits))))
        return min(1.0, fits[index])

    def _fit_scale(self, line: Line, content: PageContent) -> float:
        text = line.translated if line.translated is not None else line.text
        if not text.strip():
            return 1.0
        avail = self._available_width(line, content)
        width = self._fonts.measure(text, line.size, line.serif)
        return min(1.0, avail / width) if width > 0 else 1.0

    @staticmethod
    def _available_width(line: Line, content: PageContent) -> float:
        if line.align == "left":
            avail = content.text_right - line.left_x
        else:
            avail = content.text_right - content.text_left
        return max(avail, 10.0)

    # ------------------------------------------------------------------ lines
    def _plan_line(
        self,
        line: Line,
        content: PageContent,
        page_scale: float,
        sampler: "_BackgroundSampler",
        offset_x: float,
    ) -> "_LineOp | None":
        text = line.translated if line.translated is not None else line.text
        if not text.strip():
            return None

        # The page factor unless this line still would not fit at it, in which
        # case the line shrinks alone rather than dragging the page down.
        scale = min(page_scale, self._fit_scale(line, content))
        fontsize = max(MIN_FONT_SIZE, line.size * scale)
        width = self._fonts.measure(text, fontsize, line.serif)

        # Mask over the original text (source coords, shifted to the output half).
        bg = sampler.sample(line.bbox)
        mask = pymupdf.Rect(
            line.bbox[0] - _MASK_PAD_X + offset_x,
            line.bbox[1] - _MASK_PAD_Y,
            line.bbox[2] + _MASK_PAD_X + offset_x,
            line.bbox[3] + _MASK_PAD_Y,
        )

        # Horizontal placement preserves the original alignment.
        if line.align == "right":
            start_x = line.right_x + offset_x - width
        elif line.align == "center":
            mid = (content.text_left + content.text_right) / 2.0 + offset_x
            start_x = mid - width / 2.0
        else:
            start_x = line.left_x + offset_x
        # Keep the line inside its own half of the output page. Clamping against
        # the whole sheet let a line too long to shrink into the column slide
        # past x=0 and get drawn across the pristine original half.
        half_left, half_right = offset_x, offset_x + content.width
        start_x = min(start_x, half_right - width - 1.0)
        start_x = max(start_x, half_left + 1.0)

        return _LineOp(
            text=text,
            point=pymupdf.Point(start_x, line.baseline_y),
            fontsize=fontsize,
            serif=line.serif,
            color=self._text_color(bg, line.color),
            mask=mask,
            bg=bg,
        )

    def _draw(self, page: pymupdf.Page, op: "_LineOp") -> None:
        try:
            page.insert_text(
                op.point,
                op.text,
                fontname=self._fonts.name_for(op.serif),
                fontsize=op.fontsize,
                color=op.color,
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

    __slots__ = ("text", "point", "fontsize", "serif", "color", "mask", "bg")

    def __init__(self, *, text, point, fontsize, serif, color, mask, bg) -> None:
        self.text = text
        self.point = point
        self.fontsize = fontsize
        self.serif = serif
        self.color = color
        self.mask = mask
        self.bg = bg


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
