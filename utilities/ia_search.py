"""ia_search.py OUT.json key="query" ... -- search archive.org (through Playwright's request context) for candidate editions with PDFs."""
import json, sys
from playwright.sync_api import sync_playwright
QUERIES = dict(a.split("=", 1) for a in sys.argv[2:])  # key="lucene query"
out = {}
with sync_playwright() as p:
    ctx = p.request.new_context()
    for key, q in QUERIES.items():
        r = ctx.get("https://archive.org/advancedsearch.php", params={
            "q": q, "fl[]": "identifier,title,year,language,downloads", "rows": "12",
            "sort[]": "downloads desc", "output": "json"})
        docs = r.json()["response"]["docs"]
        rows = []
        for d in docs[:8]:
            m = ctx.get(f"https://archive.org/metadata/{d['identifier']}/files").json()
            pdfs = [(f["name"], int(f.get("size", 0)) // 1024) for f in m.get("result", []) if f["name"].lower().endswith(".pdf")]
            rows.append({k: d.get(k) for k in ("identifier", "title", "year", "language", "downloads")} | {"pdfs": pdfs[:3]})
        out[key] = rows
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
for k, rows in out.items():
    print("==", k)
    for r in rows:
        t = r["title"] if isinstance(r["title"], str) else r["title"][0]
        print(f"  {r['identifier'][:45]:45} {str(r['year'])[:6]:6} {str(r['language'])[:12]:12} dl={r['downloads']} | {t[:60]} | pdfs={[(n[:40], s) for n, s in r['pdfs']]}")
