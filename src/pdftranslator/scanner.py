"""Extract and structure page content for translation.

Each PyMuPDF "line" (a baseline-grouped run inside a block) becomes one
translation unit. Using PyMuPDF's own line grouping — rather than re-clustering
spans by Y across the whole page — keeps the original line count and prevents
distinct blocks (e.g. a centered title and the first body line) from being
merged, which was a real bug in the earlier code.
"""

from __future__ import annotations

import logging
from collections import Counter

import fitz

from .models import BBox, Line, PageContent, Span

logger = logging.getLogger(__name__)

# Span flag bits (PyMuPDF).
_FLAG_SERIF = 1 << 2
_FLAG_BOLD = 1 << 4

_SCANNED_CHAR_THRESHOLD = 50
_MARGIN = 50.0  # points; page-number detection band at top/bottom
_PAGE_NUM_MAX_LEN = 4
_BASELINE_TOL = 4.0  # points; fragments within this share one visual line


class DocumentScanner:
    """Turns a ``fitz.Page`` into a :class:`PageContent`."""

    def is_scanned(self, page: fitz.Page) -> bool:
        return len(page.get_text().strip()) < _SCANNED_CHAR_THRESHOLD and bool(
            page.get_images()
        )

    def extract(self, page: fitz.Page, page_index: int) -> PageContent:
        width, height = page.rect.width, page.rect.height
        raw = page.get_text(
            "dict",
            flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE,
        )

        frozen: list[Span] = []
        lines: list[Line] = []

        for block in raw.get("blocks", []):
            if block.get("type") != 0:  # not a text block
                continue
            block_index = block.get("number", 0)
            for raw_line in block.get("lines", []):
                line = self._build_line(raw_line, width, height, frozen, block_index)
                if line is not None:
                    lines.append(line)

        # PyMuPDF (and OCR layers especially) often split a single justified line
        # into one "line" per word when inter-word gaps are wide. Reassemble
        # fragments that share a baseline into one visual line before translating.
        lines = self._merge_baseline_fragments(lines)

        text_left, text_right = self._text_column(lines, width)
        for line in lines:
            line.align = self._alignment(line, text_left, text_right)

        return PageContent(
            page_index=page_index,
            width=width,
            height=height,
            frozen_spans=frozen,
            lines=lines,
            text_left=text_left,
            text_right=text_right,
        )

    # ------------------------------------------------------------------ lines
    def _build_line(
        self,
        raw_line: dict,
        page_w: float,
        page_h: float,
        frozen: list[Span],
        block_index: int,
    ) -> Line | None:
        dir_vec = raw_line.get("dir", (1.0, 0.0))
        is_horizontal = 0.9 <= dir_vec[0] <= 1.1 and -0.1 <= dir_vec[1] <= 0.1

        spans: list[Span] = []
        for raw_span in raw_line.get("spans", []):
            text = raw_span["text"]
            if not text.strip():
                continue
            span = Span(
                text=text,
                bbox=tuple(raw_span["bbox"]),
                origin=tuple(raw_span.get("origin", raw_span["bbox"][:2])),
                size=float(raw_span["size"]),
                color=int(raw_span["color"]),
                flags=int(raw_span["flags"]),
                font=raw_span["font"],
            )
            spans.append(span)

        if not spans:
            return None

        full_text = "".join(s.text for s in spans).strip()
        if not full_text:
            return None

        bbox: BBox = (
            min(s.bbox[0] for s in spans),
            min(s.bbox[1] for s in spans),
            max(s.bbox[2] for s in spans),
            max(s.bbox[3] for s in spans),
        )

        # Freeze non-translatable lines (page numbers, lone punctuation, vertical
        # text). They are rendered verbatim at their original position.
        if (not is_horizontal) or self._is_frozen(full_text, bbox, page_h):
            frozen.extend(spans)
            return None

        dominant = self._dominant_span(spans)
        return Line(
            spans=spans,
            text=full_text,
            bbox=bbox,
            baseline_y=float(spans[0].origin[1]),
            left_x=bbox[0],
            right_x=bbox[2],
            size=dominant.size,
            serif=self._is_serif(dominant),
            bold=bool(dominant.flags & _FLAG_BOLD),
            color=dominant.color,
            align="left",  # refined later once the text column is known
            block_index=block_index,
            rotation=0,
        )

    def _merge_baseline_fragments(self, lines: list[Line]) -> list[Line]:
        """Merge line fragments that share a baseline into single visual lines.

        Fragments are grouped when their baselines are within ``_BASELINE_TOL``
        and they are horizontally adjacent (no column-sized gap between them), so
        genuine multi-column layouts are not collapsed into one line.
        """
        if len(lines) <= 1:
            return lines

        lines.sort(key=lambda ln: (ln.baseline_y, ln.left_x))
        groups: list[list[Line]] = [[lines[0]]]
        for ln in lines[1:]:
            row = groups[-1]
            same_baseline = abs(ln.baseline_y - row[0].baseline_y) <= _BASELINE_TOL
            gap = ln.left_x - row[-1].right_x
            max_gap = 3.0 * max(ln.size, row[-1].size)  # column gaps are larger
            if same_baseline and gap <= max_gap:
                row.append(ln)
            else:
                groups.append([ln])

        merged = [self._merge_group(g) for g in groups]
        merged.sort(key=lambda ln: (ln.baseline_y, ln.left_x))
        return merged

    def _merge_group(self, group: list[Line]) -> Line:
        if len(group) == 1:
            return group[0]
        group.sort(key=lambda ln: ln.left_x)
        spans = [s for ln in group for s in ln.spans]
        text = " ".join(ln.text for ln in group).strip()
        bbox: BBox = (
            min(ln.bbox[0] for ln in group),
            min(ln.bbox[1] for ln in group),
            max(ln.bbox[2] for ln in group),
            max(ln.bbox[3] for ln in group),
        )
        dominant = self._dominant_span(spans)
        return Line(
            spans=spans,
            text=text,
            bbox=bbox,
            baseline_y=min(ln.baseline_y for ln in group),
            left_x=bbox[0],
            right_x=bbox[2],
            size=dominant.size,
            serif=self._is_serif(dominant),
            bold=bool(dominant.flags & _FLAG_BOLD),
            color=dominant.color,
            align="left",
            block_index=group[0].block_index,
            rotation=0,
        )

    @staticmethod
    def _dominant_span(spans: list[Span]) -> Span:
        """The span that visually dominates the line (most characters)."""
        return max(spans, key=lambda s: len(s.text.strip()))

    @staticmethod
    def _is_serif(span: Span) -> bool:
        if span.flags & _FLAG_SERIF:
            return True
        low = span.font.lower()
        return "serif" in low or "times" in low or "georgia" in low or "roman" in low

    def _is_frozen(self, text: str, bbox: BBox, page_h: float) -> bool:
        stripped = text.strip()
        if len(stripped) == 1 and not stripped.isalnum():
            return True
        in_top = bbox[1] < _MARGIN
        in_bottom = bbox[3] > (page_h - _MARGIN)
        if (in_top or in_bottom) and stripped.isdigit() and len(stripped) <= _PAGE_NUM_MAX_LEN:
            return True
        return False

    # ------------------------------------------------------------- alignment
    @staticmethod
    def _text_column(lines: list[Line], page_w: float) -> tuple[float, float]:
        """Estimate the page's text column edges from the body lines.

        Use a representative (modal-ish) left edge and the widest right edge so a
        single indented or centered line does not skew the column.
        """
        if not lines:
            return 0.0, page_w
        lefts = sorted(line.left_x for line in lines)
        rights = sorted(line.right_x for line in lines)
        # 10th percentile left, 90th percentile right -> robust body column.
        left = lefts[len(lefts) // 10]
        right = rights[-(max(1, len(rights) // 10))]
        return left, right

    @staticmethod
    def _alignment(line: Line, text_left: float, text_right: float) -> str:
        col_width = max(1.0, text_right - text_left)
        tol = max(4.0, col_width * 0.04)
        left_gap = line.left_x - text_left
        right_gap = text_right - line.right_x

        # Lines that start at (or near) the left margin are left-aligned body
        # text. A small positive left_gap is just a paragraph first-line indent,
        # NOT right-alignment — so require a *large* left gap before calling a
        # line right-aligned.
        center = (text_left + text_right) / 2.0
        line_center = (line.left_x + line.right_x) / 2.0
        if (
            left_gap > tol
            and right_gap > tol
            and abs(line_center - center) <= tol * 2
        ):
            return "center"
        if right_gap <= tol and left_gap > col_width * 0.30:
            return "right"
        return "left"

