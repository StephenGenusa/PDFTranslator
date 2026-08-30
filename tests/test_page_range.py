"""Page-range parsing and its CLI surface."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from pdftranslator.cli import app
from pdftranslator.orchestrator import PageRangeError, parse_page_spec, resolve_page_spec

runner = CliRunner()


@pytest.mark.parametrize(
    ("spec", "max_pages", "expected"),
    [
        ("all", 3, [0, 1, 2]),
        ("ALL", 3, [0, 1, 2]),
        ("1", 5, [0]),
        ("1-3", 5, [0, 1, 2]),
        ("1-5,7", 10, [0, 1, 2, 3, 4, 6]),
        ("3,1", 5, [0, 2]),          # unordered input, sorted output
        ("2-3,3-4", 10, [1, 2, 3]),  # overlapping ranges de-duplicated
        ("1-99", 3, [0, 1, 2]),      # clamped to the document
        (" 1 - 2 ", 5, [0, 1]),      # surrounding whitespace tolerated
    ],
)
def test_valid_specs_resolve_to_zero_based_indices(spec, max_pages, expected):
    assert resolve_page_spec(spec, max_pages) == expected


@pytest.mark.parametrize(
    "spec",
    [
        "abc",      # not a number
        "3-",       # missing upper bound
        "-3",       # missing lower bound
        "0",        # page numbers are 1-based
        "0-4",      # ditto, as a range start
        "5-1",      # reversed range
        "",         # empty spec
        ",",        # separators only
        "1--3",     # malformed range
        "1.5",      # not an integer
    ],
)
def test_malformed_specs_raise_page_range_error(spec):
    """Breaks if a bad --pages value escapes as a bare ValueError traceback."""
    with pytest.raises(PageRangeError):
        parse_page_spec(spec)


def test_range_outside_the_document_is_rejected():
    """'-p 900' on a 10-page book must fail loudly, not save an empty PDF."""
    with pytest.raises(PageRangeError):
        resolve_page_spec("900-950", 10)


def test_error_names_the_offending_text():
    with pytest.raises(PageRangeError, match="abc"):
        parse_page_spec("1-2,abc")


def test_cli_rejects_a_bad_page_range_without_a_traceback(tmp_path):
    """The user must see a message and a non-zero exit, not a stack trace."""
    src = tmp_path / "in.pdf"
    src.write_bytes(b"%PDF-1.4\n%%EOF\n")
    result = runner.invoke(
        app,
        ["-i", str(src), "-o", str(tmp_path / "out.pdf"), "-p", "abc"],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "abc" in result.output


def test_cli_rejects_an_out_of_document_range_without_a_traceback(tmp_path):
    """'-p 900' on a 3-page document: a message, not a stack trace.

    This range is syntactically valid, so it is only rejected once the document
    is open -- a different code path from the syntax check above.
    """
    import fitz

    src = tmp_path / "in.pdf"
    with fitz.open() as doc:
        for _ in range(3):
            doc.new_page()
        doc.save(src)

    result = runner.invoke(
        app,
        [
            "-i", str(src),
            "-o", str(tmp_path / "out.pdf"),
            "-p", "900-950",
            "--cache-dir", str(tmp_path / "cache"),
        ],
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "900" in result.output
