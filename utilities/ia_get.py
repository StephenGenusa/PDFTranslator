"""Download PDFs via Playwright's request context.

ia_get.py OUTDIR key=identifier/filename ...   (archive.org item file)
ia_get.py OUTDIR key=https://host/path.pdf ...  (any direct URL)
"""
import sys, pathlib, urllib.parse
from playwright.sync_api import sync_playwright
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    ctx = p.request.new_context(timeout=600_000)
    for spec in sys.argv[2:]:
        key, path = spec.split("=", 1)
        dest = out / f"{key}.pdf"
        if dest.exists(): print("have", dest.name); continue
        url = path if path.startswith("http") else "https://archive.org/download/" + urllib.parse.quote(path)
        r = ctx.get(url)
        if not r.ok or not r.body().startswith(b"%PDF"): print("FAIL", key, r.status, r.headers.get("content-type")); continue
        dest.write_bytes(r.body()); print("got", dest.name, len(r.body()) // 1024, "KB")
