"""Layout-preserving PDF translator.

Translates a PDF written in any language into a target language, rendering the
output either side-by-side (original | translation) or as a full substitution.

The core design (see render.py) renders each *visual line* as a single unit at
its original baseline and left edge, shrinking the whole line to fit the text
column only when necessary. This deliberately avoids pinning translated text to
the source document's per-span X positions, which is what produced the
"squeezed fragment" / mid-line-gap glitches in earlier attempts.
"""

__version__ = "0.1.0"
