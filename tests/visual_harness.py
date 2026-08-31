"""Offline visual-regression harness.

Runs the real pipeline against the sample book using ONLY on-disk cached
translations -- any attempt to reach the network raises -- so a rendered page is
a deterministic function of the rendering code. Renders each output page to PNG
so two revisions can be compared pixel-by-pixel.

Not a pytest module: run it directly.

    python tests/visual_harness.py capture  <outdir> [--mode M] [--pages P]
    python tests/visual_harness.py compare  <dir_a> <dir_b>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pymupdf  # noqa: E402

from pdftranslator.orchestrator import Orchestrator  # noqa: E402
from pdftranslator.translate import Translator  # noqa: E402

BOOK = REPO / "Krize, kaip tai daroma (Nikolaj).pdf"
CACHE = REPO / ".translation_cache"
MODEL = "openrouter/google/gemini-2.5-flash-lite"
RENDER_ZOOM = 1.5


class OfflineTranslator(Translator):
    """Translator that must never touch the network."""

    def _complete(self, messages):  # noqa: D102
        raise AssertionError(
            "visual harness attempted a live API call -- the page is not cached"
        )


def capture(outdir: Path, mode: str, pages: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    translator = OfflineTranslator(
        model=MODEL, target_lang="English", cache_dir=str(CACHE)
    )
    pdf_path = outdir / "output.pdf"
    try:
        Orchestrator(translator, mirror=(mode == "side-by-side")).run(
            BOOK, pdf_path, pages
        )
    finally:
        translator.close()

    doc = pymupdf.open(pdf_path)
    for i in range(doc.page_count):
        pix = doc[i].get_pixmap(matrix=pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM), alpha=False)
        pix.save(outdir / f"page-{i + 1:03d}.png")
    print(f"captured {doc.page_count} page(s) -> {outdir}")
    doc.close()


def compare(a: Path, b: Path) -> int:
    names_a = sorted(p.name for p in a.glob("page-*.png"))
    names_b = sorted(p.name for p in b.glob("page-*.png"))
    if names_a != names_b:
        only_a = set(names_a) - set(names_b)
        only_b = set(names_b) - set(names_a)
        print(f"PAGE SET DIFFERS  only-in-{a.name}={sorted(only_a)} only-in-{b.name}={sorted(only_b)}")
        return 1

    changed = 0
    for name in names_a:
        pa = pymupdf.Pixmap(str(a / name))
        pb = pymupdf.Pixmap(str(b / name))
        if (pa.width, pa.height) != (pb.width, pb.height):
            print(f"{name}: SIZE {pa.width}x{pa.height} -> {pb.width}x{pb.height}")
            changed += 1
            continue
        sa, sb = pa.samples, pb.samples
        if sa == sb:
            continue
        diff = sum(1 for i in range(0, len(sa), 3) if sa[i:i + 3] != sb[i:i + 3])
        pct = 100.0 * diff / (pa.width * pa.height)
        print(f"{name}: {diff} px differ ({pct:.3f}%)")
        changed += 1

    print(f"\n{changed} of {len(names_a)} page(s) changed")
    return 0 if changed == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("outdir", type=Path)
    cap.add_argument("--mode", default="side-by-side")
    cap.add_argument("--pages", default="2-25")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("dir_a", type=Path)
    cmp_.add_argument("dir_b", type=Path)
    args = ap.parse_args()

    if args.cmd == "capture":
        capture(args.outdir, args.mode, args.pages)
        return 0
    return compare(args.dir_a, args.dir_b)


if __name__ == "__main__":
    raise SystemExit(main())
