"""Drive extraction -> translation -> rendering across a document."""

from __future__ import annotations

import logging
from pathlib import Path

import fitz
from tqdm import tqdm

from .fonts import FontBook
from .render import PdfRenderer
from .scanner import DocumentScanner
from .translate import Translator

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, translator: Translator, *, mirror: bool) -> None:
        self._scanner = DocumentScanner()
        self._renderer = PdfRenderer(FontBook())
        self._translator = translator
        self._mirror = mirror

    def run(self, input_path: Path, output_path: Path, page_range: str = "all") -> bool:
        doc = fitz.open(input_path)
        logger.info("Loaded %s (%d pages)", input_path.name, doc.page_count)

        indices = self._parse_range(page_range, doc.page_count)
        logger.info("Processing %d page(s)", len(indices))

        writer = fitz.open()
        prev_tail: str | None = None

        for idx in tqdm(indices, desc="Pages"):
            source_page = doc[idx]
            try:
                content = self._scanner.extract(source_page, idx)
                if content.is_scanned:
                    logger.warning("Page %d looks scanned (no text layer); copying as-is", idx + 1)

                if content.lines:
                    translations = self._translator.translate_page(
                        [ln.text for ln in content.lines], preceding_context=prev_tail
                    )
                    for line, tr in zip(content.lines, translations):
                        line.translated = tr
                    prev_tail = content.lines[-1].text

                self._renderer.render_page(
                    writer, source_page, content, mirror=self._mirror
                )
            except Exception as exc:  # noqa: BLE001 - never abort the whole run
                logger.error("Page %d failed: %s", idx + 1, exc, exc_info=True)
                self._blank_page(writer, source_page)

        writer.save(output_path, garbage=4, deflate=True)
        logger.info("Saved %s", output_path)
        self._log_stats()
        return True

    def _blank_page(self, writer: fitz.Document, source_page: fitz.Page) -> None:
        w, h = source_page.rect.width, source_page.rect.height
        writer.new_page(width=w * 2 if self._mirror else w, height=h)

    @staticmethod
    def _parse_range(spec: str, max_pages: int) -> list[int]:
        if spec.strip().lower() == "all":
            return list(range(max_pages))
        out: set[int] = set()
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.update(range(int(a) - 1, int(b)))
            else:
                out.add(int(part) - 1)
        return sorted(p for p in out if 0 <= p < max_pages)

    def _log_stats(self) -> None:
        s = self._translator.stats
        logger.info(
            "Translation stats: pages=%d api_calls=%d cache_hits=%d fallbacks=%d",
            s["pages"], s["api_calls"], s["cache_hits"], s["fallbacks"],
        )
