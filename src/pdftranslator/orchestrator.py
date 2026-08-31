"""Drive extraction -> translation -> rendering across a document."""

from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
        concurrency: int = 1,
    ) -> None:
        self._scanner = DocumentScanner()
        self._renderer = PdfRenderer(FontBook(), size_fidelity=size_fidelity)
        self._translator = translator
        self._mirror = mirror
        self._concurrency = max(1, int(concurrency or 1))
        # One owner for how many requests may be in flight.
        translator.set_max_concurrency(self._concurrency)

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

        recurring = self._survey(doc, indices)
        canonical: dict[str, str] = {}

        # Extract every page first, so each page's preceding context is fixed
        # before translation starts. Threading that context through a running
        # loop would make the result depend on the order threads happened to
        # finish; taken from the source it is the same at any concurrency.
        contents = {index: self._scanner.extract(doc[index], index)
                    for index in indices}
        context = {}
        previous: str | None = None
        for index in indices:
            context[index] = previous
            lines = contents[index].lines
            previous = lines[-1].text if lines else previous

        failed: set[int] = set()

        def translate(index: int) -> None:
            content = contents[index]
            if not content.lines:
                return
            try:
                translations = self._translator.translate_page(
                    [line.text for line in content.lines],
                    preceding_context=context[index],
                )
            except Exception as exc:  # noqa: BLE001 - never lose the whole run
                # A page that cannot be translated is emitted as the original,
                # exactly as before translation moved out of the render loop.
                logger.error("Page %d failed to translate: %s", index + 1, exc,
                             exc_info=True)
                failed.add(index)
                return
            for line, translated in zip(content.lines, translations):
                line.translated = translated

        if self._concurrency > 1:
            logger.info("Translating with up to %d concurrent request(s)",
                        self._concurrency)
            with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
                list(tqdm(pool.map(translate, indices), total=len(indices),
                          desc="Translating"))
        else:
            for index in tqdm(indices, desc="Translating"):
                translate(index)

        # Rendering stays strictly in source order.
        for index in tqdm(indices, desc="Rendering"):
            source_page = doc[index]
            pages_before = writer.page_count
            try:
                if index in failed:
                    self._untranslated_page(writer, source_page)
                    continue
                content = contents[index]
                if content.is_scanned:
                    logger.warning(
                        "Page %d looks scanned (no text layer); copying as-is",
                        index + 1,
                    )
                if content.lines:
                    self._apply_canonical(content, recurring, canonical)
                self._renderer.render_page(
                    writer, source_page, content, mirror=self._mirror
                )
            except Exception as exc:  # noqa: BLE001 - never abort the whole run
                logger.error("Page %d failed: %s", index + 1, exc, exc_info=True)
                self._discard_pages_after(writer, pages_before)
                self._untranslated_page(writer, source_page)

        writer.save(output_path, garbage=4, deflate=True)
        logger.info("Saved %s", output_path)
        self._log_stats()
        return True

    def _survey(self, doc: pymupdf.Document, indices: list[int]) -> set[str]:
        """One pass over the pages: report cache coverage, find running lines.

        Coverage goes through the translator's own lookup so it cannot drift
        from what the run will actually find.
        """
        counts: Counter[str] = Counter()
        already = 0
        pending_tokens = 0
        suspect: list[int] = []
        for index in indices:
            content = self._scanner.extract(doc[index], index)
            counts.update({line.text for line in content.lines})
            if not content.lines:
                continue
            texts = [line.text for line in content.lines]
            if self._translator.has_cached(texts):
                already += 1
                if self._translator.cached_looks_misaligned(texts):
                    suspect.append(index + 1)
            else:
                pending_tokens += self._translator.estimate_page_tokens(texts)
        pending = len(indices) - already
        estimate = self._translator.estimate_cost(pending_tokens) if pending else None
        logger.info(
            "Cache: %d of %d page(s) already translated; %d to translate"
            "%s%s",
            already, len(indices), pending,
            f" (~{pending_tokens:,} tokens" if pending else "",
            (f", ~${estimate:.2f})" if estimate else ")") if pending else "",
        )
        if suspect:
            logger.warning(
                "%d cached page(s) look misaligned (a translation reused, or "
                "badly the wrong length for its line): %s. Re-run those pages "
                "with --ignore-cache to replace them.",
                len(suspect),
                ",".join(str(p) for p in suspect[:20])
                + ("..." if len(suspect) > 20 else ""),
            )
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
            "fallbacks=%d api_errors=%d truncated=%d misaligned=%d cache_migrated=%d "
            "cache_pruned=%d",
            s["pages"], s["api_calls"], s["cache_hits"], s["fallbacks"],
            s.get("api_errors", 0), s.get("truncated", 0),
            s.get("misaligned", 0),
            s.get("cache_migrated", 0), s.get("cache_pruned", 0),
        )
        logger.info(
            "Tokens: prompt=%d completion=%d | reported spend: $%.4f",
            s.get("prompt_tokens", 0), s.get("completion_tokens", 0),
            s.get("cost_usd", 0.0),
        )
