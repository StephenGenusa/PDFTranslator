"""Drive extraction -> translation -> rendering across a document."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pymupdf
from tqdm import tqdm

from .fonts import FontBook
from .models import PageContent
from .render import DEFAULT_SIZE_FIDELITY, PdfRenderer
from .scanner import DocumentScanner
from .translate import Translator

logger = logging.getLogger(__name__)


# A line seen on at least this many pages is a running element -- a header,
# footer or repeated caption -- and must read the same on every one of them.
RECURRING_PAGE_THRESHOLD = 3

# A candidate translation is only adopted as the canonical one if its length is
# credible for the source. A misaligned batch can hand a short running head a
# long body translation; adopting that would spread the error over every page.
_CANONICAL_MIN_RATIO = 0.4
_CANONICAL_MAX_RATIO = 2.5


class PageRangeError(ValueError):
    """A --pages value that cannot be interpreted."""


def parse_page_spec(spec: str) -> list[tuple[int, int]] | None:
    """Parse a page spec into 1-based inclusive (start, end) pairs.

    Returns ``None`` for "all". Raises :class:`PageRangeError` with a message
    naming the offending text, so a typo surfaces as an error rather than a
    ValueError traceback out of int().
    """
    text = spec.strip()
    if text.lower() == "all":
        return None
    if not text:
        raise PageRangeError("Empty page range. Use e.g. '1-5,7' or 'all'.")

    pairs: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            start, end = _to_page(a, part), _to_page(b, part)
            if end < start:
                raise PageRangeError(
                    f"Page range {part!r} runs backwards ({start} > {end})."
                )
        else:
            start = end = _to_page(part, part)
        pairs.append((start, end))

    if not pairs:
        raise PageRangeError(f"No pages named in {spec!r}. Use e.g. '1-5,7' or 'all'.")
    return pairs


def _to_page(token: str, part: str) -> int:
    token = token.strip()
    if not token.isdigit():
        raise PageRangeError(
            f"Invalid page number {token!r} in {part!r}. "
            "Expected whole numbers, e.g. '1-5,7' or 'all'."
        )
    number = int(token)
    if number < 1:
        raise PageRangeError(
            f"Invalid page number {number} in {part!r}. Pages are numbered from 1."
        )
    return number


def resolve_page_spec(spec: str, max_pages: int) -> list[int]:
    """Resolve a page spec to sorted 0-based page indices within the document."""
    pairs = parse_page_spec(spec)
    if pairs is None:
        return list(range(max_pages))

    indices = {
        i
        for start, end in pairs
        for i in range(start - 1, end)
        if 0 <= i < max_pages
    }
    if not indices:
        raise PageRangeError(
            f"Page range {spec!r} selects no pages of this {max_pages}-page document."
        )
    return sorted(indices)


class Orchestrator:
    def __init__(
        self,
        translator: Translator,
        *,
        mirror: bool,
        size_fidelity: float = DEFAULT_SIZE_FIDELITY,
    ) -> None:
        self._scanner = DocumentScanner()
        self._renderer = PdfRenderer(FontBook(), size_fidelity=size_fidelity)
        self._translator = translator
        self._mirror = mirror

    def run(self, input_path: Path, output_path: Path, page_range: str = "all") -> bool:
        with pymupdf.open(input_path) as doc, pymupdf.open() as writer:
            return self._run(doc, writer, input_path, output_path, page_range)

    def _run(
        self,
        doc: pymupdf.Document,
        writer: pymupdf.Document,
        input_path: Path,
        output_path: Path,
        page_range: str,
    ) -> bool:
        logger.info("Loaded %s (%d pages)", input_path.name, doc.page_count)

        indices = resolve_page_spec(page_range, doc.page_count)
        logger.info("Processing %d page(s)", len(indices))

        prev_tail: str | None = None
        recurring = self._recurring_lines(doc, indices)
        canonical: dict[str, str] = {}

        for idx in tqdm(indices, desc="Pages"):
            source_page = doc[idx]
            pages_before = writer.page_count
            try:
                content = self._scanner.extract(source_page, idx)
                if content.is_scanned:
                    logger.warning("Page %d looks scanned (no text layer); copying as-is", idx + 1)

                if content.lines:
                    # The whole page is still sent, so the page's cache key is
                    # unchanged; recurring lines are reconciled afterwards.
                    translations = self._translator.translate_page(
                        [ln.text for ln in content.lines], preceding_context=prev_tail
                    )
                    for line, tr in zip(content.lines, translations):
                        line.translated = tr
                    self._apply_canonical(content, recurring, canonical)
                    prev_tail = content.lines[-1].text

                self._renderer.render_page(
                    writer, source_page, content, mirror=self._mirror
                )
            except Exception as exc:  # noqa: BLE001 - never abort the whole run
                logger.error("Page %d failed: %s", idx + 1, exc, exc_info=True)
                # render_page appends its output page before it can fail, so any
                # half-written page must go before the replacement is added --
                # otherwise a failed page shifts the rest of the book by one.
                self._discard_pages_after(writer, pages_before)
                self._untranslated_page(writer, source_page)

        writer.save(output_path, garbage=4, deflate=True)
        logger.info("Saved %s", output_path)
        self._log_stats()
        return True

    def _recurring_lines(self, doc: pymupdf.Document, indices: list[int]) -> set[str]:
        """Source lines that appear on RECURRING_PAGE_THRESHOLD or more pages."""
        counts: Counter[str] = Counter()
        for index in indices:
            content = self._scanner.extract(doc[index], index)
            counts.update({line.text for line in content.lines})
        recurring = {
            text for text, seen in counts.items()
            if seen >= RECURRING_PAGE_THRESHOLD
        }
        if recurring:
            logger.info(
                "%d recurring line(s) will be kept consistent across pages",
                len(recurring),
            )
        return recurring

    @staticmethod
    def _apply_canonical(
        content: PageContent, recurring: set[str], canonical: dict[str, str]
    ) -> None:
        """Give every occurrence of a recurring line the same translation."""
        for line in content.lines:
            if line.text not in recurring or line.translated is None:
                continue
            settled = canonical.get(line.text)
            if settled is not None:
                line.translated = settled
                continue
            source_length = len(line.text.strip())
            ratio = (
                len(line.translated.strip()) / source_length
                if source_length else 1.0
            )
            if _CANONICAL_MIN_RATIO <= ratio <= _CANONICAL_MAX_RATIO:
                canonical[line.text] = line.translated

    @staticmethod
    def _discard_pages_after(writer: pymupdf.Document, count: int) -> None:
        while writer.page_count > count:
            writer.delete_page(writer.page_count - 1)

    def _untranslated_page(self, writer: pymupdf.Document, source_page: pymupdf.Page) -> None:
        """Emit the original page when translation or rendering failed.

        This used to emit an empty page, so one transient provider error
        silently replaced a page of the book with a white rectangle.
        """
        w, h = source_page.rect.width, source_page.rect.height
        page = writer.new_page(width=w * 2 if self._mirror else w, height=h)
        parent, pno = source_page.parent, source_page.number
        if self._mirror:
            page.show_pdf_page(pymupdf.Rect(0, 0, w, h), parent, pno)
            page.show_pdf_page(pymupdf.Rect(w, 0, w * 2, h), parent, pno)
        else:
            page.show_pdf_page(pymupdf.Rect(0, 0, w, h), parent, pno)

    def _log_stats(self) -> None:
        s = self._translator.stats
        logger.info(
            "Translation stats: pages=%d api_calls=%d cache_hits=%d "
            "fallbacks=%d api_errors=%d",
            s["pages"], s["api_calls"], s["cache_hits"], s["fallbacks"],
            s.get("api_errors", 0),
        )
