"""DocumentScanner extraction behaviour."""

from __future__ import annotations

import base64

import fitz
import pytest

from pdftranslator.scanner import DocumentScanner

# A 1x1 PNG, standing in for the scan image on a page with no text layer.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg=="
)


@pytest.fixture
def scanner() -> DocumentScanner:
    return DocumentScanner()


def _text_page(doc: fitz.Document, text: str) -> fitz.Page:
    page = doc.new_page()
    page.insert_text((72, 144), text, fontsize=12)
    return page


def _image_only_page(doc: fitz.Document) -> fitz.Page:
    page = doc.new_page()
    page.insert_image(fitz.Rect(0, 0, 400, 400), stream=_PNG_1X1)
    return page


def test_image_only_page_is_flagged_as_scanned(scanner):
    """Breaks if extract() stops consulting is_scanned() for the page."""
    with fitz.open() as doc:
        content = scanner.extract(_image_only_page(doc), 0)
    assert content.is_scanned is True


def test_page_with_a_text_layer_is_not_scanned(scanner):
    with fitz.open() as doc:
        page = _text_page(doc, "Hello world, this is a line of real extractable text.")
        content = scanner.extract(page, 0)
    assert content.is_scanned is False


def test_text_page_still_yields_translatable_lines(scanner):
    """Guards against a scanned-detection change that suppresses extraction."""
    with fitz.open() as doc:
        page = _text_page(doc, "Hello world, this is a line of real extractable text.")
        content = scanner.extract(page, 0)
    assert [ln.text for ln in content.lines] == [
        "Hello world, this is a line of real extractable text."
    ]
