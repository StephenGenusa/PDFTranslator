"""Objective typography metrics for the rendering pipeline.

Companion to visual_harness.py. Where that answers "did any pixel change?",
this answers "is the page a faithful rendering of the source?" -- useful when a
change (or a PyMuPDF upgrade) moves pixels and you need to judge whether it
moved them in the right direction.

Runs offline against cached translations; any network call raises.

    python tests/typography_metrics.py [--pages P] [--json OUT]

Metrics, all rooted in fidelity to the source document rather than taste:

  over-shrink       Each line has a largest scale at which it still fits its
                    column (its own fit). Because a block shares one scale, a
                    line can be rendered smaller than it needed to be. That
                    discards size information the source had, flattening the
                    document's visual hierarchy. Lower is better.

  overflow          Text rendered wider than the column. The shrink-to-fit is
                    floored at MIN_FONT_SIZE, so a long translation can still
                    spill. Broken layout. Lower is better.

  hierarchy         Pairs of lines whose source sizes differ by >1pt but whose
  inversions        rendered sizes reverse or flatten that order -- a heading
                    rendered no larger than its body text. This is the sharpest
                    accuracy signal: it means the output misrepresents the
                    structure of the original page. Lower is better.

  size jumps        Vertically adjacent lines that share a source size but are
                    rendered at different sizes. _block_scales exists to avoid
                    exactly this. Lower is better.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pymupdf  # noqa: E402

from pdftranslator.fonts import FontBook  # noqa: E402
from pdftranslator.render import MIN_FONT_SIZE, PdfRenderer  # noqa: E402
from pdftranslator.scanner import DocumentScanner  # noqa: E402
from pdftranslator.translate import Translator  # noqa: E402

BOOK = REPO / "Krize, kaip tai daroma (Nikolaj).pdf"
CACHE = REPO / ".translation_cache"
MODEL = "openrouter/google/gemini-2.5-flash-lite"


class OfflineTranslator(Translator):
    def _complete(self, messages):
        raise AssertionError("metrics run attempted a live API call")


def _page_indices(spec: str, page_count: int) -> list[int]:
    from pdftranslator.orchestrator import resolve_page_spec

    return resolve_page_spec(spec, page_count)


def measure(pages: str = "2-25") -> dict:
    translator = OfflineTranslator(model=MODEL, target_lang="English",
                                   cache_dir=str(CACHE))
    scanner, renderer = DocumentScanner(), PdfRenderer(FontBook())
    fonts = renderer._fonts

    over_shrink: list[float] = []
    overflow_pts: list[float] = []
    jumps: list[float] = []
    blocks_per_page: list[int] = []
    inversions = pairs = lines_total = 0

    try:
        with pymupdf.open(BOOK) as doc:
            for idx in _page_indices(pages, doc.page_count):
                content = scanner.extract(doc[idx], idx)
                if not content.lines:
                    continue
                for line, tr in zip(
                    content.lines,
                    translator.translate_page([ln.text for ln in content.lines]),
                ):
                    line.translated = tr

                page_scale = renderer._page_scale(content)
                blocks_per_page.append(
                    len({ln.block_index for ln in content.lines})
                )

                rendered = []
                for line in content.lines:
                    own_fit = renderer._fit_scale(line, content)
                    scale = min(page_scale, own_fit)
                    size = max(MIN_FONT_SIZE, line.size * scale)
                    over_shrink.append(max(0.0, own_fit - scale))
                    lines_total += 1

                    avail = renderer._available_width(line, content)
                    width = fonts.measure(line.translated or line.text, size,
                                          line.serif)
                    if width > avail + 1.0:
                        overflow_pts.append(width - avail)

                    rendered.append((line.baseline_y, line.size, size))

                rendered.sort()
                for (y1, src1, out1), (y2, src2, out2) in zip(rendered, rendered[1:]):
                    if abs(src1 - src2) < 0.1 and abs(y2 - y1) < 40:
                        jumps.append(abs(out1 - out2) / max(src1, 1e-6))

                for i, (_, src_a, out_a) in enumerate(rendered):
                    for _, src_b, out_b in rendered[i + 1:]:
                        if src_a - src_b > 1.0:
                            pairs += 1
                            if out_a <= out_b:
                                inversions += 1
    finally:
        translator.close()

    return {
        "pymupdf": pymupdf.__doc__.splitlines()[0],
        "lines": lines_total,
        "blocks_per_page": round(statistics.mean(blocks_per_page), 3),
        "over_shrink_mean": round(statistics.mean(over_shrink), 5),
        "over_shrink_lines_pct": round(
            100 * sum(1 for v in over_shrink if v > 0.001) / max(lines_total, 1), 2
        ),
        "overflow_lines": len(overflow_pts),
        "overflow_pts_mean": round(statistics.mean(overflow_pts), 2) if overflow_pts else 0.0,
        "hierarchy_pairs": pairs,
        "hierarchy_inversions": inversions,
        "hierarchy_inversion_pct": round(100 * inversions / max(pairs, 1), 3),
        "size_jump_mean": round(statistics.mean(jumps), 5) if jumps else 0.0,
        "size_jump_pct": round(
            100 * sum(1 for j in jumps if j > 0.01) / max(len(jumps), 1), 2
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="2-25")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    result = measure(args.pages)
    width = max(len(k) for k in result)
    for key, value in result.items():
        print(f"{key:<{width}}  {value}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
