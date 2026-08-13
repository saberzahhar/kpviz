#!/usr/bin/env python3
"""Playwright interaction test: scan button, doc drill-down, PDF download,
LaTeX clipboard content presence."""
import asyncio
import pathlib

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8085"
OUT = "/tmp/shots"


async def main():
    pathlib.Path(OUT).mkdir(exist_ok=True)
    async with async_playwright() as p:
        exe = "/opt/pw-browsers/chromium"
        browser = await p.chromium.launch(
            executable_path=exe if pathlib.Path(exe).exists() else None)
        page = await browser.new_page(viewport={"width": 1620, "height": 1100})

        # 1) Home: click "Scan for changes", watch it complete
        await page.goto(BASE + "/")
        await asyncio.sleep(2)
        await page.click("#btn-scan")
        await asyncio.sleep(2.5)
        await page.screenshot(path=f"{OUT}/scan-running.png")
        for _ in range(40):
            body = await page.inner_text("#scan-progress")
            if "scan complete" in body:
                break
            await asyncio.sleep(1)
        print("scan status:", "complete" if "scan complete" in body else body[:80])
        await page.screenshot(path=f"{OUT}/scan-done.png")

        # 2) Datasets: click a document row -> drill-down opens
        await page.goto(BASE + "/datasets")
        await asyncio.sleep(9)
        rows = page.locator("table.kp-table tr.row-click")
        n = await rows.count()
        print("clickable doc rows:", n)
        if n:
            await rows.first.click()
            await asyncio.sleep(2)
            txt = await page.inner_text("#ds-doc-view")
            print("doc view shows:", txt[:80].replace("\n", " "))
            await page.screenshot(path=f"{OUT}/doc-view.png", full_page=True)

        # 3) Insights RQ4: check clipboard content + download PDF
        await page.goto(BASE + "/insights")
        await asyncio.sleep(6)
        clip = page.locator("div.export-bar").first
        print("export bar present:", await clip.count() > 0)
        pdf_btn = page.get_by_role("button", name="PDF").first
        async with page.expect_download(timeout=90000) as dl_info:
            await pdf_btn.click()
        dl = await dl_info.value
        path = f"{OUT}/dl-{dl.suggested_filename}"
        await dl.save_as(path)
        size = pathlib.Path(path).stat().st_size
        print("downloaded:", dl.suggested_filename, size, "bytes")

        # bundle zip too
        zip_btn = page.get_by_role("button", name="Bundle").first
        async with page.expect_download(timeout=120000) as dl_info:
            await zip_btn.click()
        dl = await dl_info.value
        path = f"{OUT}/dl-{dl.suggested_filename}"
        await dl.save_as(path)
        print("downloaded:", dl.suggested_filename,
              pathlib.Path(path).stat().st_size, "bytes")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
