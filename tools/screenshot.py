#!/usr/bin/env python3
"""Screenshot every KPViz page with Playwright (dev/verification helper)."""
import asyncio
import sys

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8085"
OUT = "/tmp/shots"

PAGES = [
    ("home", "/", 3.0, None),
    ("datasets", "/datasets", 4.0, None),
    ("models", "/models", 3.5, None),
    ("architectures", "/architectures", 3.0, None),
    ("insights-rq4", "/insights", 6.0, None),
    ("insights-rq1", "/insights", 4.0, "tab-rq1"),
    ("insights-rq2", "/insights", 6.0, "tab-rq2"),
    ("insights-rq3", "/insights", 8.0, "tab-rq3"),
    ("insights-rq5", "/insights", 4.0, "tab-rq5"),
]


async def main():
    import pathlib
    pathlib.Path(OUT).mkdir(exist_ok=True)
    errors = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium"
                                          if pathlib.Path("/opt/pw-browsers/chromium").exists()
                                          else None)
        page = await browser.new_page(viewport={"width": 1620, "height": 1150})
        page.on("console", lambda m: errors.append(f"[{m.type}] {m.text}")
                if m.type in ("error",) else None)
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
        for name, path, wait, click in PAGES:
            if page.url != BASE + path:
                await page.goto(BASE + path)
                await asyncio.sleep(1.5)
            if click:
                await page.click(f"#{click}")
            await asyncio.sleep(wait)
            await page.screenshot(path=f"{OUT}/{name}.png", full_page=True)
            print("shot", name)
        await browser.close()
    # Navigating away cancels the callback POST that was in flight, and
    # dash-renderer reports the cancelled XHR as "the server did not respond".
    # That is this tool moving on, not the app failing — the server log shows
    # no 5xx for these. Count them separately so a real error stands out.
    aborted = [e for e in errors if "did not respond" in e]
    real = [e for e in errors if "did not respond" not in e]
    if aborted:
        print(f"\n{len(aborted)} callback(s) aborted by navigation (expected)")
    if real:
        print("\n--- console errors ---")
        for e in real[:30]:
            print(e[:300])
        return 1
    print("no console errors")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
