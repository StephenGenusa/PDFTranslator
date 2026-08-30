"""Font resolution and packaging."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

import pdftranslator
from pdftranslator.fonts import FontBook

REPO = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(pdftranslator.__file__).resolve().parent


def test_font_files_live_inside_the_package():
    """Fonts resolved from outside the package break once pip-installed.

    Breaks if the TTFs are moved back to the repo root, or if _find_font starts
    resolving via the repo layout or the cwd again.
    """
    book = FontBook()
    assert book.sans_path.is_relative_to(PACKAGE_DIR), book.sans_path
    assert book.serif_path.is_relative_to(PACKAGE_DIR), book.serif_path


def test_fontbook_works_with_no_fonts_in_cwd_or_repo_root(tmp_path, monkeypatch):
    """An installed user runs from an arbitrary directory, not the repo."""
    monkeypatch.chdir(tmp_path)
    book = FontBook()
    assert book.measure("Labas rytas", 12.0, serif=False) > 0
    assert book.measure("Labas rytas", 12.0, serif=True) > 0


def test_built_wheel_contains_the_bundled_fonts(tmp_path):
    """The wheel must carry the fonts, or `pdf-translate` crashes when installed."""
    if shutil.which("uv") is None:
        pytest.skip("uv not available")
    result = subprocess.run(
        ["uv", "build", "--wheel", "-o", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-500:]

    wheels = sorted(tmp_path.glob("pdftranslator-*.whl"))
    assert wheels, "no wheel produced"
    names = zipfile.ZipFile(wheels[-1]).namelist()
    assert "pdftranslator/data/NotoSans-Regular.ttf" in names, names
    assert "pdftranslator/data/NotoSerif-Regular.ttf" in names, names
