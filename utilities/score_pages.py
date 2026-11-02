"""Score each page's text layer: chars, word-likeness. Prints best candidate pages per PDF."""
import sys, re, pymupdf
for path in sys.argv[1:]:
    doc = pymupdf.open(path)
    rows = []
    for i, pg in enumerate(doc):
        t = pg.get_text()
        words = re.findall(r"\w+", t)
        if not words: continue
        good = sum(1 for w in words if w.isalpha() and 2 <= len(w) <= 14)
        rows.append((good / len(words), len(t), i + 1, t[:70].replace("\n", " ")))
    full = [r for r in rows if r[1] > 900]
    full.sort(reverse=True)
    print(f"== {path}  pages={len(doc)}  size={doc[0].rect.width:.0f}x{doc[0].rect.height:.0f}  with-text={len(rows)}")
    for q, n, p, s in full[:8]:
        print(f"  p{p:<4} q={q:.2f} chars={n:<5} {s}")
