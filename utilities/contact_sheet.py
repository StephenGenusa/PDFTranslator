"""contact_sheet.py OUT.png COLS file.pdf:page ... -> one PNG grid of rendered pages, labelled."""
import sys, pymupdf
out, cols, specs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
W, H = 300, 440
rows = (len(specs) + cols - 1) // cols
sheet = pymupdf.open(); pg = sheet.new_page(width=cols * W, height=rows * (H + 20))
for k, spec in enumerate(specs):
    f, p = spec.rsplit(":", 1)
    src = pymupdf.open(f)
    x, y = (k % cols) * W, (k // cols) * (H + 20)
    pg.show_pdf_page(pymupdf.Rect(x + 5, y + 18, x + W - 5, y + H + 15), src, int(p) - 1)
    pg.insert_text((x + 8, y + 13), f"{f.split('/')[-1]} p{p}", fontsize=11, color=(0.8, 0, 0))
pg.get_pixmap(dpi=110).save(out)
