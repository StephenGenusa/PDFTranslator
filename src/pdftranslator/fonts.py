"""Bundled-font resolution and registration.

Earlier versions shipped NotoSans/NotoSerif TTFs in the repo but never used
them, rendering instead with the PyMuPDF base-14 fonts (Helvetica / Times),
which only cover Latin-1. That is why diacritics and display text turned into
tofu/garbage. We register the Noto faces and render everything with them; Noto
Sans/Serif Regular cover Latin (incl. Lithuanian), Cyrillic and Greek.
"""

from __future__ import annotations

from pathlib import Path

import fitz

# Registered (page-scoped) font names we use when calling insert_text.
SANS = "ptsans"
SERIF = "ptserif"

_SANS_FILE = "NotoSans-Regular.ttf"
_SERIF_FILE = "NotoSerif-Regular.ttf"


# Fonts ship inside the package (see pyproject's wheel config) so they resolve
# identically whether the package is run from a checkout or pip-installed.
_DATA_DIR = Path(__file__).resolve().parent / "data"


def _find_font(filename: str) -> Path:
    """Locate a bundled font file inside the installed package."""
    path = _DATA_DIR / filename
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"Bundled font {filename!r} is missing from the package data directory "
        f"({_DATA_DIR}). The installation is incomplete."
    )


class FontBook:
    """Holds the font files and provides accurate text measurement.

    A single instance is reused across the whole document. ``fitz.Font`` objects
    measure text width precisely for the actual face being drawn, which is what
    the line shrink-to-fit logic relies on.
    """

    def __init__(self) -> None:
        self.sans_path = _find_font(_SANS_FILE)
        self.serif_path = _find_font(_SERIF_FILE)
        self._sans = fitz.Font(fontfile=str(self.sans_path))
        self._serif = fitz.Font(fontfile=str(self.serif_path))

    def register(self, page: fitz.Page) -> None:
        """Register both faces on a freshly created output page."""
        page.insert_font(fontname=SANS, fontfile=str(self.sans_path))
        page.insert_font(fontname=SERIF, fontfile=str(self.serif_path))

    def name_for(self, serif: bool) -> str:
        return SERIF if serif else SANS

    def measure(self, text: str, fontsize: float, serif: bool) -> float:
        """Width of ``text`` in points at the given size, for the chosen face."""
        if not text:
            return 0.0
        font = self._serif if serif else self._sans
        return font.text_length(text, fontsize=fontsize)
