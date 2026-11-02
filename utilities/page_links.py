"""page_links.py URL [regex] -> list link text + href on a JS-rendered page (Playwright Chromium)."""
import re, sys
from playwright.sync_api import sync_playwright
url, pat = sys.argv[1], re.compile(sys.argv[2] if len(sys.argv) > 2 else ".", re.I)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto(url, wait_until="networkidle", timeout=60_000)
    for a in pg.eval_on_selector_all("a", "els => els.map(e => [e.innerText.trim(), e.href])"):
        if pat.search(a[0] + " " + a[1]): print(f"{a[0][:70]:70} {a[1]}")
    b.close()
