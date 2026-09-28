#!/usr/bin/env python3
"""Drive a running KPViz server in headless Chromium and measure what a user
feels: callbacks per action, time until the page settles, idle traffic,
payload bytes, console errors, and the server's RSS.

    python app.py --data sample_data --no-browser --port 8050 &
    python tools/bench/probe_ui.py http://127.0.0.1:8050 [--chromium PATH]
                                   [--json out.json] [--budget budget.json]

Setup: `pip install playwright psutil` and either `playwright install
chromium` or pass --chromium to an existing Chromium binary.

--budget takes {"first_load_s": 1.5, "interaction_s": 0.6, "idle_requests": 0}
(any subset) and exits 1 when a measurement exceeds 120 % of its budget.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter

MB = 2 ** 20


def _server_proc(url: str):
    try:
        import psutil
        port = int(url.rsplit(":", 1)[1].split("/")[0])
        for c in psutil.net_connections("tcp"):
            if c.laddr and c.laddr.port == port and c.status == "LISTEN" and c.pid:
                return psutil.Process(c.pid)
    except Exception:
        return None
    return None


def _kind(name: str) -> str:
    if "scan-poll" in name or "scan-progress" in name or name.startswith("sidebar-scan"):
        return "poll"
    if "exp-clip" in name:
        return "export clipboards"
    if "rq-graph" in name:
        return "workbench compute"
    if '"rq' in name or "rq" in name.split(".")[0]:
        return "workbench options"
    if "page-" in name or "ins-active" in name:
        return "routing"
    return "pages"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--chromium", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--budget", default=None)
    ap.add_argument("--screenshots", default=None, help="directory")
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright

    base = a.url.rstrip("/")
    srv = _server_proc(base)
    calls, pending, errors = [], {}, []
    results: dict = {"actions": {}}

    def rss():
        return round(srv.memory_info().rss / MB) if srv else None

    def name_of(req):
        try:
            return json.loads(req.post_data or "{}").get("output") or "?"
        except Exception:
            return "?"

    with sync_playwright() as p:
        kw = {"executable_path": a.chromium} if a.chromium else {}
        browser = p.chromium.launch(**kw)
        page = browser.new_page(viewport={"width": 1440, "height": 1000},
                                accept_downloads=True)

        def on_req(r):
            if "_dash-update-component" in r.url:
                pending[r] = (time.time(), name_of(r))

        def on_done(r):
            hit = pending.pop(r, None)
            if hit:
                try:
                    size = len(r.response().body()) if r.response() else 0
                    status = r.response().status if r.response() else 0
                except Exception:
                    size, status = 0, 0
                # the browser's own timing of the round trip: Playwright
                # delivers events late while the page is busy rendering, so
                # time.time() at delivery overstated server latency by the
                # render time of whatever the response drew
                try:
                    end = r.timing.get("responseEnd", -1)
                    dur = end / 1000.0 if end and end > 0 else time.time() - hit[0]
                except Exception:
                    dur = time.time() - hit[0]
                calls.append((hit[0], hit[0] + dur, hit[1], size, status))

        page.on("request", on_req)
        page.on("requestfinished", on_done)
        page.on("requestfailed", on_done)
        page.on("console", lambda m: errors.append(m.text)
                if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        def settle(quiet=1.0, timeout=120):
            t0, last = time.time(), time.time()
            while time.time() - t0 < timeout:
                if pending:
                    last = time.time()
                elif time.time() - last > quiet:
                    return max(0.0, time.time() - t0 - quiet)
                page.wait_for_timeout(25)
            return time.time() - t0

        def act(label, fn, quiet=1.0):
            t = time.time()
            fn()
            took = settle(quiet)
            sel = [c for c in calls if c[0] >= t]
            kinds = Counter(_kind(c[2]) for c in sel)
            slow = max(sel, key=lambda c: c[1] - c[0]) if sel else None
            bad = [c for c in sel if c[4] and c[4] != 200]
            results["actions"][label] = {
                "callbacks": len(sel), "settle_s": round(took, 3),
                "kb": round(sum(c[3] for c in sel) / 1024),
                "slowest_s": round(slow[1] - slow[0], 3) if slow else 0,
                "slowest": slow[2][:70] if slow else "",
                "non_200": len(bad), "kinds": dict(kinds), "rss_mb": rss()}
            r = results["actions"][label]
            print(f"{label:<34} {r['callbacks']:4d} cb  settle {r['settle_s']:6.2f}s  "
                  f"slowest {r['slowest_s']:5.2f}s  {r['kb']:5d} KB  "
                  f"RSS {r['rss_mb']} MB  {dict(kinds)}")
            if a.screenshots:
                import re
                fname = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
                page.screenshot(path=f"{a.screenshots}/{fname}.png",
                                full_page=True)

        print(f"server RSS before load: {rss()} MB")
        results["rss_before_mb"] = rss()
        act("first load /", lambda: page.goto(base + "/"), quiet=1.5)

        t = time.time()
        page.wait_for_timeout(10_000)
        idle = len([c for c in calls if c[0] >= t])
        results["idle_requests_10s"] = idle
        print(f"{'idle 10 s on Overview':<34} {idle:4d} requests")

        act("open Insights", lambda: page.click("#nav-insights"))
        for tab in ("rq1", "rq2", "rq3", "rq5", "rq4"):
            act(f"tab {tab}", lambda tab=tab: page.click(f"#tab-{tab}"))
        for rq in ("rq1", "rq2", "rq3", "rq4", "rq5"):
            page.click(f"#tab-{rq}")
            settle(0.6)

            def change_k(rq=rq):
                page.click(f"#{rq}-metric")
                opt = page.get_by_role("option", name="F1@M", exact=True)
                if opt.count():
                    opt.first.click()
                else:                       # dropdowns without ARIA options
                    page.keyboard.type("F1@M")
                    page.keyboard.press("Enter")
                txt = page.inner_text(f"#{rq}-metric")
                if "@M" not in txt:
                    print(f"  ! {rq}: the metric did not change (shows {txt.strip()!r})")
            act(f"{rq}: metric -> F1@M", change_k)
            act(f"{rq}: tab away and back", lambda rq=rq: (
                page.click("#tab-rq1" if rq != "rq1" else "#tab-rq2"),
                page.click(f"#tab-{rq}")))

        # exports on the visible workbench (rq5 is last shown)
        page.click("#tab-rq4")
        settle(0.6)
        for what in ("png", "pdf"):
            t = time.time()
            try:
                with page.expect_download(timeout=60_000) as dl:
                    # Dash renders dict ids as sorted-key JSON strings
                    page.click(f'[id=\'{{"rq":"rq4","type":"exp-btn",'
                               f'"what":"{what}"}}\']')
                size = len(open(dl.value.path(), "rb").read())
                took = time.time() - t
                results["actions"][f"export {what}"] = {"s": round(took, 2),
                                                         "bytes": size}
                print(f"{'export ' + what:<34} {took:6.2f}s  {size} bytes")
            except Exception as e:
                results["actions"][f"export {what}"] = {"error": str(e)[:120]}
                print(f"export {what}: FAILED {str(e)[:120]}")

        act("open Datasets", lambda: page.click("#nav-datasets"))
        act("open Models", lambda: page.click("#nav-models"))
        act("open Architectures", lambda: page.click("#nav-architectures"))
        act("back to Overview", lambda: page.click("#nav-home"))
        t = time.time()
        page.wait_for_timeout(5_000)
        results["idle_requests_5s_end"] = len([c for c in calls if c[0] >= t])
        results["rss_end_mb"] = rss()
        results["console_errors"] = errors[:20]
        print(f"server RSS at end: {rss()} MB · console errors: {len(errors)}")
        for e in errors[:10]:
            print("   !", e[:160])
        browser.close()

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
    rc = 1 if errors else 0
    if a.budget:
        b = json.load(open(a.budget))
        acts = results["actions"]
        checks = []
        if "first_load_s" in b:
            checks.append(("first load", acts["first load /"]["settle_s"],
                           b["first_load_s"]))
        if "interaction_s" in b:
            for k, v in acts.items():
                if "@k" in k:
                    checks.append((k, v["settle_s"], b["interaction_s"]))
        if "idle_requests" in b:
            checks.append(("idle requests", results["idle_requests_10s"],
                           b["idle_requests"]))
        for label, got, want in checks:
            ok = got <= want * 1.2 + 1e-9
            print(f"budget {label:<30} {got} vs {want}  {'ok' if ok else 'OVER'}")
            rc |= 0 if ok else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
