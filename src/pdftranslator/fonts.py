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


def _find_font(filename: str) -> Path:
    """Locate a bundled font file.

    Looks next to the repo root (two levels up from this file under src/), then
    the current working directory. Raises if not found so the failure is loud.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / filename,  # repo root (src/pdftranslator/..)
        Path.cwd() / filename,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"Bundled font {filename!r} not found. Looked in: "
        + ", ".join(str(c) for c in candidates)
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
