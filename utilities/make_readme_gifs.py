"""Build the two README demo GIFs from real translator output.

For each showcase page this runs the project's own CLI (``python -m pdftranslator``)
twice -- ``--mode side-by-side`` and ``--mode substitute`` -- on that single page,
renders the result, adds a white caption strip naming the source, and writes two
animated GIFs with identical frame counts and timings so they advance in step:

    docs/images/demo-side-by-side.gif
    docs/images/demo-translated.gif

Source PDFs are expected in corpus/ (see utilities/ia_get.py); intermediate PDFs go
to scratch/readme-gifs/. Translations come from the normal translation cache, so a
rerun costs nothing unless a page or the model changes.

Run (Pillow is only needed here, so it is not a project dependency):

    uv run --with pillow python utilities/make_readme_gifs.py [--model MODEL]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
WORK = ROOT / "scratch" / "readme-gifs"
OUT = ROOT / "docs" / "images"

FRAME_MS = 4000
SBS_SIZE = (1400, 860)      # page area of the side-by-side GIF
SUB_SIZE = (700, 860)       # page area of the translated-only GIF
CAPTION_H = 128
PAGE_BG = (236, 236, 236)
CAPTION_BG = (255, 255, 255)
SUPERSAMPLE = 2

FONT_DIR = Path("/usr/share/fonts/opentype/noto")
BOLD_TTC = FONT_DIR / "NotoSerifCJK-Bold.ttc"
REGULAR_TTC = FONT_DIR / "NotoSerifCJK-Regular.ttc"
# Face index inside the Noto CJK collections; Latin/Cyrillic are in every face.
FACE_JP, FACE_SC = 0, 2


@dataclass(frozen=True)
class Showcase:
    key: str          # corpus/<key>.pdf
    page: int         # 1-based page number
    title: str        # original-language title
    english: str      # English title; shown as "title — english" in bold
    byline: str       # author / publisher, year, language
    face: int = FACE_JP


SHOWCASE = [
    Showcase("onegin", 8, "Евгений Онегин", "Eugene Onegin",
             "Alexander Pushkin  ·  Russian  ·  1874 edition"),
    Showcase("anssi-ia-generative", 12,
             "Recommandations de sécurité pour un système d’IA générative",
             "Security Recommendations for a Generative AI System",
             "ANSSI, French National Cybersecurity Agency  ·  French  ·  2024"),
    Showcase("revolutionibus", 40, "De revolutionibus orbium coelestium",
             "On the Revolutions of the Heavenly Spheres",
             "Nicolaus Copernicus  ·  Latin  ·  1543 first edition"),
    Showcase("meti-ai-guideline-overview", 7, "AI事業者ガイドライン（第1.2版）概要",
             "AI Guidelines for Business (Version 1.2), Overview",
             "Japan MIC & METI  ·  Japanese  ·  2026"),
    Showcase("faust", 23, "Faust. Eine Tragödie", "Faust: A Tragedy",
             "Johann Wolfgang von Goethe  ·  German  ·  1862 edition"),
    Showcase("caict-ai-whitepaper-2018", 9, "人工智能发展白皮书 · 产业应用篇",
             "AI Development White Paper: Industrial Applications",
             "China Academy of Information and Communications Technology  ·  Chinese  ·  2018",
             face=FACE_SC),
    Showcase("sidereus", 17, "Sidereus Nuncius", "The Starry Messenger",
             "Galileo Galilei  ·  Latin  ·  1610 first edition"),
    Showcase("stasi-richtlinie-1-76", 46,
             "Richtlinie Nr. 1/76 zur Entwicklung und Bearbeitung Operativer Vorgänge",
             "Directive No. 1/76 on the Development and Processing of Operational Cases",
             "East German Ministry for State Security (Stasi)  ·  German  ·  1976"),
    Showcase("principe", 131, "Il Principe", "The Prince",
             "Niccolò Machiavelli  ·  Italian  ·  1913 edition"),
]


def translate(item: Showcase, mode: str, model: str | None) -> Path:
    """Run the project's CLI on one page; return the single-page output PDF."""
    out = WORK / f"{item.key}-p{item.page}-{mode}.pdf"
    cmd = [sys.executable, "-m", "pdftranslator",
           "-i", str(CORPUS / f"{item.key}.pdf"), "-o", str(out),
           "-t", "English", "-p", str(item.page), "--mode", mode]
    if model:
        cmd += ["-m", model]
    print("$", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    return out


def render_fit(pdf: Path, box: tuple[int, int]) -> Image.Image:
    """Render page 1 of ``pdf`` as large as fits in ``box``, centred on PAGE_BG."""
    page = pymupdf.open(pdf)[0]
    bw, bh = box[0] - 24, box[1] - 24          # keep a small margin
    zoom = min(bw / page.rect.width, bh / page.rect.height)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom * SUPERSAMPLE, zoom * SUPERSAMPLE), alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img = img.resize((round(pix.width / SUPERSAMPLE), round(pix.height / SUPERSAMPLE)), Image.LANCZOS)
    canvas = Image.new("RGB", box, PAGE_BG)
    x, y = (box[0] - img.width) // 2, (box[1] - img.height) // 2
    canvas.paste(img, (x, y))
    ImageDraw.Draw(canvas).rectangle((x - 1, y - 1, x + img.width, y + img.height), outline=(190, 190, 190))
    return canvas


def fitted_font(path: Path, face: int, text: str, size: int, max_w: int) -> ImageFont.FreeTypeFont:
    """Largest font <= size whose rendering of ``text`` fits in ``max_w`` pixels."""
    while True:
        font = ImageFont.truetype(str(path), size, index=face)
        if font.getlength(text) <= max_w or size <= 10:
            return font
        size -= 1


def title_lines(item: Showcase, max_w: int) -> tuple[list[str], ImageFont.FreeTypeFont]:
    """"Original — English" on one line if it fits at a readable size, else two lines."""
    one = f"{item.title} — {item.english}"
    font = fitted_font(BOLD_TTC, item.face, one, 28, max_w)
    if font.size >= 21:
        return [one], font
    two = [item.title, item.english]
    longest = max(two, key=lambda t: ImageFont.truetype(str(BOLD_TTC), 28, index=item.face).getlength(t))
    return two, fitted_font(BOLD_TTC, item.face, longest, 26, max_w)


def add_caption(page_img: Image.Image, item: Showcase) -> Image.Image:
    w = page_img.width
    frame = Image.new("RGB", (w, page_img.height + CAPTION_H), CAPTION_BG)
    frame.paste(page_img, (0, 0))
    d = ImageDraw.Draw(frame)
    d.line((0, page_img.height, w, page_img.height), fill=(200, 200, 200), width=1)
    lines, title_font = title_lines(item, w - 40)
    byline_font = fitted_font(REGULAR_TTC, item.face, item.byline, 19, w - 40)
    step = title_font.size + 8
    block_h = step * len(lines) + 26                 # title lines + byline
    y = page_img.height + (CAPTION_H - block_h) / 2 + step / 2
    for line in lines:
        d.text((w / 2, y), line, font=title_font, fill=(20, 20, 20), anchor="mm")
        y += step
    d.text((w / 2, y + 4), item.byline, font=byline_font, fill=(90, 90, 90), anchor="mm")
    return frame


def save_gif(frames: list[Image.Image], path: Path) -> None:
    pal = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=FRAME_MS, loop=0, optimize=True)
    print(f"wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB, {len(frames)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="LiteLLM model string (default: the CLI's default)")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    sbs_frames, sub_frames = [], []
    for item in SHOWCASE:
        sbs = translate(item, "side-by-side", args.model)
        sub = translate(item, "substitute", args.model)
        sbs_frames.append(add_caption(render_fit(sbs, SBS_SIZE), item))
        sub_frames.append(add_caption(render_fit(sub, SUB_SIZE), item))

    save_gif(sbs_frames, OUT / "demo-side-by-side.gif")
    save_gif(sub_frames, OUT / "demo-translated.gif")


if __name__ == "__main__":
    main()
