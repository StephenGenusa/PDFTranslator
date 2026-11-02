"""DocumentScanner extraction behaviour."""

from __future__ import annotations

import base64

import pymupdf
import pytest

from pdftranslator.scanner import DocumentScanner

# A 1x1 PNG, standing in for the scan image on a page with no text layer.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg=="
)


@pytest.fixture
def scanner() -> DocumentScanner:
    return DocumentScanner()


def _text_page(doc: pymupdf.Document, text: str) -> pymupdf.Page:
    page = doc.new_page()
    page.insert_text((72, 144), text, fontsize=12)
    return page


def _image_only_page(doc: pymupdf.Document) -> pymupdf.Page:
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(0, 0, 400, 400), stream=_PNG_1X1)
    return page


def test_image_only_page_is_flagged_as_scanned(scanner):
    """Breaks if extract() stops consulting is_scanned() for the page."""
    with pymupdf.open() as doc:
        content = scanner.extract(_image_only_page(doc), 0)
    assert content.is_scanned is True


def test_page_with_a_text_layer_is_not_scanned(scanner):
    with pymupdf.open() as doc:
        page = _text_page(doc, "Hello world, this is a line of real extractable text.")
        content = scanner.extract(page, 0)
    assert content.is_scanned is False


def test_text_page_still_yields_translatable_lines(scanner):
    """Guards against a scanned-detection change that suppresses extraction."""
    with pymupdf.open() as doc:
        page = _text_page(doc, "Hello world, this is a line of real extractable text.")
        content = scanner.extract(page, 0)
    assert [ln.text for ln in content.lines] == [
        "Hello world, this is a line of real extractable text."
    ]


def _words_with_separate_space_spans(doc: pymupdf.Document, words: list[str]) -> pymupdf.Page:
    """A line whose inter-word spaces are their own spans, as in some OCR layers.

    Each space is set in a different font from the words, so PyMuPDF reports it
    as a separate whitespace-only span between the word spans.
    """
    page = doc.new_page()
    x, y = 72.0, 144.0
    for i, word in enumerate(words):
        if i:
            page.insert_text((x, y), " ", fontname="cour", fontsize=12)
            x += pymupdf.get_text_length(" ", fontname="cour", fontsize=12)
        page.insert_text((x, y), word, fontname="helv", fontsize=12)
        x += pymupdf.get_text_length(word, fontname="helv", fontsize=12)
    return page


def test_spaces_stored_as_separate_spans_are_kept(scanner):
    """Breaks if whitespace-only spans are dropped without leaving a word break.

    The Stasi directive scan (Bundesarchiv, Richtlinie 1/76) stores every space
    this way; dropping them sent "GibtesWidersprüche..." to the model.
    """
    words = ["Gibt", "es", "Widersprüche", "in", "den", "Informationen"]
    with pymupdf.open() as doc:
        page = _words_with_separate_space_spans(doc, words)
        raw_spans = [
            s["text"]
            for b in page.get_text("dict")["blocks"]
            for ln in b.get("lines", [])
            for s in ln["spans"]
        ]
        content = scanner.extract(page, 0)
    assert " " in raw_spans, "fixture must put each space in its own span"
    assert [ln.text for ln in content.lines] == [" ".join(words)]
