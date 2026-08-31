"""Internal data models for extracted PDF content.

Plain dataclasses are used (rather than Pydantic) because these objects are
created in tight loops during extraction and never cross a trust boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# A bounding box: (x0, y0, x1, y1) in PDF points, origin top-left.
BBox = tuple[float, float, float, float]


@dataclass
class Span:
    """An atomic run of text with consistent styling, as PyMuPDF reports it."""

    text: str
    bbox: BBox
    origin: tuple[float, float]  # baseline origin (x, y)
    size: float
    color: int  # packed sRGB integer
    # PyMuPDF font flags, a bitmask. Bit index (and value), matching the
    # fitz.TEXT_FONT_* constants:
    #   0 (1) superscript, 1 (2) italic, 2 (4) serif, 3 (8) monospaced, 4 (16) bold
    # scanner.py tests these as `1 << index`, e.g. serif is `1 << 2` == 4.
    flags: int
    font: str


@dataclass
class Line:
    """One visual line of translatable text (a PyMuPDF 'line' within a block).

    The translation unit. We keep the original baseline and left edge so the
    translated line can be drawn in exactly the same place as the source line.
    """

    spans: list[Span]
    text: str
    bbox: BBox
    baseline_y: float
    left_x: float
    right_x: float
    size: float  # dominant font size on the line
    serif: bool  # whether to render with the serif face
    bold: bool
    color: int
    align: str  # 'left' | 'center' | 'right'
    block_index: int = 0  # source block; lines in a block share one shrink factor
    rotation: int = 0
    translated: Optional[str] = None


@dataclass
class PageContent:
    """Everything extracted from a single source page."""

    page_index: int
    width: float
    height: float
    frozen_spans: list[Span]  # not translated; left untouched on the base layer
    lines: list[Line]  # translated, masked and redrawn in place
    text_left: float  # left edge of the page's text column
    text_right: float  # right edge of the page's text column
    is_scanned: bool = False
