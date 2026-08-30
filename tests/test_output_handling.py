"""Output-path validation and document lifetime."""

from __future__ import annotations

import os

import fitz
import pytest
from typer.testing import CliRunner

from pdftranslator.cli import app
from pdftranslator.orchestrator import Orchestrator
from pdftranslator.translate import Translator

runner = CliRunner()


def _make_pdf(path, pages: int = 2) -> None:
    with fitz.open() as doc:
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((72, 144), f"Labas rytas puslapis {i + 1}", fontsize=12)
        doc.save(path)


def _open_fds_for(path) -> list[str]:
    real = os.path.realpath(path)
    found = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.path.realpath(f"/proc/self/fd/{fd}") == real:
                found.append(fd)
        except OSError:
            pass
    return found


class EchoTranslator(Translator):
    """Offline translator: returns each line unchanged, never calls out."""

    def translate_page(self, lines, preceding_context=None):
        return list(lines)

    def _complete(self, messages):
        raise AssertionError("no API call expected")


# --------------------------------------------------------------- validation
def test_missing_output_directory_is_rejected_before_any_work(tmp_path):
    """Breaks if the output path is only checked at save() -- after paying for
    every API call. The absent cache directory proves nothing was set up."""
    src = tmp_path / "in.pdf"
    _make_pdf(src)
    cache = tmp_path / "cache"

    result = runner.invoke(
        app,
        [
            "-i", str(src),
            "-o", str(tmp_path / "nope" / "out.pdf"),
            "--cache-dir", str(cache),
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "nope" in result.output
    assert not cache.exists(), "work started before the output path was validated"


def test_output_equal_to_input_is_rejected_and_leaves_the_source_intact(tmp_path):
    src = tmp_path / "book.pdf"
    _make_pdf(src)
    original = src.read_bytes()

    result = runner.invoke(
        app,
        ["-i", str(src), "-o", str(src), "--cache-dir", str(tmp_path / "cache")],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert src.read_bytes() == original


def test_output_pointing_at_a_directory_is_rejected(tmp_path):
    src = tmp_path / "in.pdf"
    _make_pdf(src)
    target = tmp_path / "somedir"
    target.mkdir()

    result = runner.invoke(
        app,
        ["-i", str(src), "-o", str(target), "--cache-dir", str(tmp_path / "cache")],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output


# ----------------------------------------------------------- document lifetime
def test_source_document_is_closed_when_saving_fails(tmp_path):
    """An unclosed fitz.Document holds an OS file descriptor.

    On the happy path CPython's refcounting hides this, but when save() raises,
    the propagating traceback keeps run()'s frame -- and therefore the open
    document -- alive for as long as the caller holds the exception. Breaks if
    run() stops closing the document deterministically.
    """
    src = tmp_path / "in.pdf"
    _make_pdf(src)
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o555)  # exists, but not writable

    translator = EchoTranslator(model="fake/x", target_lang="English",
                                cache_dir=str(tmp_path / "cache"))
    try:
        with pytest.raises(Exception) as excinfo:
            Orchestrator(translator, mirror=False).run(
                src, locked / "out.pdf", "all"
            )
        # excinfo keeps the traceback -- and therefore run()'s frame -- alive,
        # exactly the situation in which a non-deterministic close leaks the fd.
        assert excinfo.value is not None
        assert _open_fds_for(src) == [], "source PDF left open after a failed run"
    finally:
        translator.close()
        locked.chmod(0o755)


def test_run_still_produces_the_expected_output(tmp_path):
    """The lifetime fix must not change what gets written."""
    src = tmp_path / "in.pdf"
    _make_pdf(src, pages=3)
    out = tmp_path / "out.pdf"
    translator = EchoTranslator(model="fake/x", target_lang="English",
                                cache_dir=str(tmp_path / "cache"))
    try:
        Orchestrator(translator, mirror=False).run(src, out, "1-3")
    finally:
        translator.close()

    with fitz.open(out) as doc:
        assert doc.page_count == 3
        assert "Labas rytas puslapis 1" in doc[0].get_text()
