"""Bundled-font resolution and registration.

Earlier versions shipped NotoSans/NotoSerif TTFs in the repo but never used
them, rendering instead with the PyMuPDF base-14 fonts (Helvetica / Times),
which only cover Latin-1. That is why diacritics and display text turned into
tofu/garbage. We register the Noto faces and render everything with them; Noto
Sans/Serif Regular cover Latin (incl. Lithuanian), Cyrillic and Greek.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

# Registered (page-scoped) font names we use when calling insert_text.
# Registered (page-scoped) name -> file, one per style we can draw.
_FACES = {
    (False, False, False): ("ptsans", "NotoSans-Regular.ttf"),
    (False, True, False): ("ptsans-b", "NotoSans-Bold.ttf"),
    (False, False, True): ("ptsans-i", "NotoSans-Italic.ttf"),
    (True, False, False): ("ptserif", "NotoSerif-Regular.ttf"),
    (True, True, False): ("ptserif-b", "NotoSerif-Bold.ttf"),
    (True, False, True): ("ptserif-i", "NotoSerif-Italic.ttf"),
}

SANS = _FACES[(False, False, False)][0]
SERIF = _FACES[(True, False, False)][0]


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

    A single instance is reused across the whole document. ``pymupdf.Font``
    objects measure text width precisely for the actual face being drawn, which
    is what the line fitting relies on. Bold wins over italic when a line is
    both, since no bold-italic face is bundled.
    """

    def __init__(self) -> None:
        self._paths = {key: _find_font(name) for key, (_, name) in _FACES.items()}
        self._fonts = {
            key: pymupdf.Font(fontfile=str(path))
            for key, path in self._paths.items()
        }
        # Kept for callers that only care about the regular faces.
        self.sans_path = self._paths[(False, False, False)]
        self.serif_path = self._paths[(True, False, False)]

    @staticmethod
    def _key(serif: bool, bold: bool, italic: bool) -> tuple[bool, bool, bool]:
        return (bool(serif), bool(bold), bool(bold) is False and bool(italic))

    def register(self, page: pymupdf.Page) -> None:
        """Register every face on a freshly created output page."""
        for key, (name, _file) in _FACES.items():
            page.insert_font(fontname=name, fontfile=str(self._paths[key]))

    def path_for(self, serif: bool, bold: bool = False, italic: bool = False):
        return self._paths[self._key(serif, bold, italic)]

    def name_for(self, serif: bool, bold: bool = False, italic: bool = False) -> str:
        return _FACES[self._key(serif, bold, italic)][0]

    def measure(self, text: str, fontsize: float, serif: bool,
                bold: bool = False, italic: bool = False) -> float:
        """Width of ``text`` in points at the given size, for the chosen face."""
        if not text:
            return 0.0
        return self._fonts[self._key(serif, bold, italic)].text_length(
            text, fontsize=fontsize
        )
