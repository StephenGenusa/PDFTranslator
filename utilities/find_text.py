"""find_text.py PDF pattern... -> pages whose text layer matches (case-insensitive regex), plus image count."""
import sys, re, pymupdf
doc = pymupdf.open(sys.argv[1])
for pat in sys.argv[2:]:
    hits = [f"p{i+1}(img{len(pg.get_images())},ch{len(pg.get_text())})" for i, pg in enumerate(doc) if re.search(pat, pg.get_text(), re.I)]
    print(f"{sys.argv[1]} /{pat}/: {' '.join(hits[:12])}")
