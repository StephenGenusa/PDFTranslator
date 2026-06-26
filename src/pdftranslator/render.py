"""Render translated content by overlaying it on the original page.

Composition model (general, layout-preserving):

1. Stamp the original page onto the output (the *base layer*). Everything we do
   not translate — images, rules, backgrounds, page numbers, math, vertical
   labels — is therefore preserved exactly, with no per-element reconstruction.
2. For each translatable line, paint an opaque rectangle over the original text,
   filled with the *sampled local background colour*, so the original text is
   hidden and the box blends into the page (white page -> white box; dark banner
   -> dark box).
3. Draw the translated line at the original baseline and left edge. Font size is
   chosen per block so neighbouring lines share one size (see ``_block_scales``);
   text colour is the original colour when it contrasts with the background,
   otherwise black-on-light / white-on-dark.

We deliberately do NOT reflow: each source line maps to one translated line in
its original position, so the page layout is preserved rather than re-typeset.
"""

from __future__ import annotations

import logging

import fitz

from .fonts import FontBook
from .models import BBox, Line, PageContent

logger = logging.getLogger(__name__)

MIN_FONT_SIZE = 4.0
_MASK_PAD_X = 1.5  # points; cover OCR boxes that sit just inside the inked glyphs
_MASK_PAD_Y = 1.5
_CONTRAST_MIN = 0.40  # min luminance gap to keep the original text colour
_SAMPLE_ZOOM = 2.0  # render scale for background sampling


class PdfRenderer:
    def __init__(self, fonts: FontBook) -> None:
        self._fonts = fonts

    def render_page(
        self,
        writer: fitz.Document,
        source_page: fitz.Page,
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
            page.show_pdf_page(fitz.Rect(0, 0, w, h), parent, pno)
        page.show_pdf_page(fitz.Rect(offset_x, 0, offset_x + w, h), parent, pno)

        if not content.lines:
            return

        sampler = _BackgroundSampler(source_page)
        scales = self._block_scales(content)

        # Resolve every line first, then draw all masks, then all text — so a
        # lower line's mask can never clip the line drawn above it.
        ops = [
            op
            for line in content.lines
            if (op := self._plan_line(line, content, scales, sampler, offset_x)) is not None
        ]
        for op in ops:
            page.draw_rect(op.mask, color=op.bg, fill=op.bg, width=0)
        for op in ops:
            self._draw(page, op)

    # ------------------------------------------------------------ block scale
    def _block_scales(self, content: PageContent) -> dict[int, float]:
        """One shrink factor per block: the average over the lines that shrink.

        Lines that already fit are excluded from the average so they don't pull
        it up; every line is then rendered at ``min(standard, its_own_fit)``.
        """
        by_block: dict[int, list[float]] = {}
        for line in content.lines:
            scale = self._fit_scale(line, content)
            by_block.setdefault(line.block_index, []).append(scale)
        out: dict[int, float] = {}
        for bi, scales in by_block.items():
            compressed = [s for s in scales if s < 0.999]
            out[bi] = sum(compressed) / len(compressed) if compressed else 1.0
        return out

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
        scales: dict[int, float],
        sampler: "_BackgroundSampler",
        offset_x: float,
    ) -> "_LineOp | None":
        text = line.translated if line.translated is not None else line.text
        if not text.strip():
            return None

        standard = scales.get(line.block_index, 1.0)
        scale = min(standard, self._fit_scale(line, content))
        fontsize = max(MIN_FONT_SIZE, line.size * scale)
        width = self._fonts.measure(text, fontsize, line.serif)

        # Mask over the original text (source coords, shifted to the output half).
        bg = sampler.sample(line.bbox)
        mask = fitz.Rect(
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
        start_x = max(1.0, min(start_x, content.width * (2 if offset_x else 1) - width - 1.0))

        return _LineOp(
            text=text,
            point=fitz.Point(start_x, line.baseline_y),
            fontsize=fontsize,
            serif=line.serif,
            color=self._text_color(bg, line.color),
            mask=mask,
            bg=bg,
        )

    def _draw(self, page: fitz.Page, op: "_LineOp") -> None:
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

    def __init__(self, page: fitz.Page, zoom: float = _SAMPLE_ZOOM) -> None:
        self._zoom = zoom
        self._pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)

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
