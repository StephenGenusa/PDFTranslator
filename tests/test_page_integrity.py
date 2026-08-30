"""The output must have exactly one page per requested source page."""

from __future__ import annotations

import fitz
import pytest

from pdftranslator import render as render_mod
from pdftranslator.orchestrator import Orchestrator
from pdftranslator.translate import Translator


class EchoTranslator(Translator):
    def translate_page(self, lines, preceding_context=None):
        return list(lines)


def _make_pdf(path, pages: int = 3) -> None:
    with fitz.open() as doc:
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((72, 144), f"Labas rytas puslapis {i + 1}", fontsize=12)
        doc.save(path)


def _run(tmp_path, mirror=False, pages=3):
    src = tmp_path / "in.pdf"
    _make_pdf(src, pages)
    out = tmp_path / "out.pdf"
    t = EchoTranslator(model="fake/x", target_lang="English",
                       cache_dir=str(tmp_path / "cache"))
    try:
        Orchestrator(t, mirror=mirror).run(src, out, "all")
    finally:
        t.close()
    return out


def test_a_render_failure_does_not_add_an_extra_page(tmp_path, monkeypatch):
    """render_page appends the output page before it can fail, so the failure
    handler appended a second one -- every failed page shifted the rest of the
    book by one. Breaks if the handler stops discarding the partial page.
    """
    def boom(self, *a, **kw):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(render_mod.PdfRenderer, "_plan_line", boom)

    out = _run(tmp_path, pages=3)
    with fitz.open(out) as doc:
        assert doc.page_count == 3


def test_the_recovered_page_shows_the_original_content(tmp_path, monkeypatch):
    def boom(self, *a, **kw):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(render_mod.PdfRenderer, "_plan_line", boom)

    out = _run(tmp_path, pages=3)
    with fitz.open(out) as doc:
        for i in range(3):
            assert f"Labas rytas puslapis {i + 1}" in doc[i].get_text()


def test_a_render_failure_keeps_page_size_in_side_by_side_mode(tmp_path, monkeypatch):
    def boom(self, *a, **kw):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(render_mod.PdfRenderer, "_plan_line", boom)

    out = _run(tmp_path, mirror=True, pages=2)
    with fitz.open(out) as doc:
        assert doc.page_count == 2
        assert doc[0].rect.width == pytest.approx(595.0 * 2, abs=1.0)


def test_a_healthy_run_has_one_output_page_per_source_page(tmp_path):
    out = _run(tmp_path, pages=3)
    with fitz.open(out) as doc:
        assert doc.page_count == 3
