#!/usr/bin/env python3
"""Screenshot KPViz with Playwright (verification, and the README images).

    python tools/screenshot.py [BASE_URL] [OUT_DIR]             # every page
    python tools/screenshot.py --readme [BASE_URL] [docs/img]   # README images

Run it against an app serving the sample data (tools/make_sample_data.py),
so the published images show no one's own documents. Needs
`pip install playwright` and a Chromium (`playwright install chromium`).
"""
import asyncio
import pathlib
import sys

from playwright.async_api import async_playwright

args = [a for a in sys.argv[1:] if not a.startswith("--")]
README = "--readme" in sys.argv
BASE = args[0] if args else "http://127.0.0.1:8050"
OUT = pathlib.Path(args[1] if len(args) > 1 else ("docs/img" if README else "shots"))

PAGES = [
    ("home", "/", 3.0),
    ("datasets", "/datasets", 4.0),
    ("models", "/models", 3.5),
    ("architectures", "/architectures", 3.0),
    ("insights-rq1", "/insights#rq1", 4.0),
    ("insights-rq2", "/insights#rq2", 6.0),
    ("insights-rq3", "/insights#rq3", 8.0),
    ("insights-rq4", "/insights#rq4", 6.0),
    ("insights-rq5", "/insights#rq5", 4.0),
]


def _chromium():
    for p in ("/opt/pw-browsers/chromium",
              "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"):
        if pathlib.Path(p).exists():
            return p
    return None


async def _go(page, path, wait):
    """Client-side navigation (the app keeps every page mounted)."""
    await page.evaluate(
        "p => { window.history.pushState({}, '', p);"
        "       window.dispatchEvent(new PopStateEvent('popstate')); }", path)
    await asyncio.sleep(wait)


async def every_page(page):
    for name, path, wait in PAGES:
        await _go(page, path, wait)
        await page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        print("shot", name)


async def readme(page):
    """Three images: the cost–quality workbench, a document with its gold
    marked and a run's score explained, the Overview."""
    async def clip(first, last, name):
        """From the top of `first` to the bottom of `last`, in page
        coordinates (bounding boxes are relative to the viewport)."""
        y0 = await page.evaluate("window.scrollY")
        a = await page.locator(first).first.bounding_box()
        b = await page.locator(last).first.bounding_box()
        await page.screenshot(path=str(OUT / f"{name}.png"), full_page=True,
                              clip={"x": a["x"] - 18, "y": a["y"] + y0 - 10,
                                    "width": a["width"] + 36,
                                    "height": b["y"] + b["height"] - a["y"] + 20})
        print("shot", name)

    await _go(page, "/insights#rq4", 7.0)
    await page.evaluate("window.scrollTo(0, 0)")
    await clip("#panel-rq4 .rq-question", "#panel-rq4 .rq-table", "insights")

    await _go(page, "/datasets", 4.0)
    await page.click("#ds-pick")
    await asyncio.sleep(0.5)
    await page.get_by_role("option", name="talnarchives").click()
    await asyncio.sleep(4.0)
    await page.locator("button.row-open").first.click()
    await asyncio.sleep(3.0)
    await clip("#ds-doc-view", "#ds-explain-card", "document")

    await _go(page, "/", 4.0)
    await page.evaluate("window.scrollTo(0, 0)")
    await page.mouse.move(0, 0)
    await asyncio.sleep(0.5)
    await page.screenshot(path=str(OUT / "overview.png"))
    print("shot overview")
    _shrink([OUT / f"{n}.png" for n in ("insights", "document", "overview")])


def _shrink(paths):
    """256-colour PNGs: an interface screenshot loses nothing visible and
    the repository stays light (about 150 KB an image at 2x)."""
    from PIL import Image
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.NONE).save(p, optimize=True)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    errors = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=_chromium())
        page = await browser.new_page(viewport={"width": 1440, "height": 900},
                                      device_scale_factor=1 if not README else 2)
        page.on("console", lambda m: errors.append(f"[{m.type}] {m.text}")
                if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
        await page.goto(BASE + "/")
        await asyncio.sleep(2.0)
        await (readme(page) if README else every_page(page))
        await browser.close()
    # Navigating away cancels the callback request in flight, which
    # dash-renderer reports as "the server did not respond": that is this
    # tool moving on, not the app failing.
    real = [e for e in errors if "did not respond" not in e]
    if real:
        print("\n--- console errors ---")
        for e in real[:30]:
            print(e[:300])
        return 1
    print("no console errors")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
