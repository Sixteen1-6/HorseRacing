"""
Fetch confirmed race dates per track from Equibase's calendar endpoint.

Produces track_race_dates.json used by config.is_race_day() to skip
date/track combos where no card was run — saving hundreds of wasted
requests during the main scrape.

Usage:
  python3 -m scraper.fetch_track_dates                     # current year
  python3 -m scraper.fetch_track_dates --years 2022,2023,2024,2025,2026
  python3 -m scraper.fetch_track_dates --years 2022-2026   # range shorthand
"""

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
import time
import platform
from datetime import datetime
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    sys.exit("Run: pip install playwright && playwright install chromium")

from .config import TRACKS
from .page_utils import setup_page_blocking

CALENDAR_URL = "https://www.equibase.com/premium/eqbRaceChartCalendar.cfm"

# Match dates in href like dt=4/3/2025 or dt=04/03/2025 (with or without leading zeros)
DATE_IN_HREF = re.compile(r"(?:dt|raceDate)=(\d{1,2})/(\d{1,2})/(\d{4})")


async def _extract_dates_from_page(page, year_set: set[int]) -> set[str]:
    """Pull all race dates from the currently loaded calendar page."""
    cal_html = await page.evaluate("""
        () => {
            const cal = document.getElementById('racechart-calendar');
            return cal ? cal.innerHTML : document.body.innerHTML;
        }
    """)
    dates: set[str] = set()
    for m in DATE_IN_HREF.finditer(cal_html):
        mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
        if int(yyyy) in year_set:
            dates.add(f"{yyyy}-{int(mm):02d}-{int(dd):02d}")
    return dates


async def fetch_dates_for_track(page, track: str, track_name: str,
                                years: list[int], debug: bool = False) -> dict[str, list[str]]:
    """Select a track and year on the calendar form, submit it, and extract
    all race-day links. One form submit per year gives all 12 months.
    Returns {year_str: [date, ...]}."""
    year_set = set(years)
    all_dates: set[str] = set()

    for year in years:
        for attempt in range(3):
            try:
                await page.goto(CALENDAR_URL, wait_until="domcontentloaded", timeout=15000)
                await asyncio.sleep(0.5)

                # Find the track option value (format: "KEE;USA;KEENELAND")
                # Note: Equibase pads some codes with spaces (e.g. "CD ;USA;...")
                track_value = await page.evaluate("""
                    (track) => {
                        const sel = document.querySelector('select[name="trackid"]');
                        if (!sel) return null;
                        for (const opt of sel.options) {
                            const code = opt.value.split(';')[0].trim();
                            if (code === track) return opt.value;
                        }
                        return null;
                    }
                """, track)
                if not track_value:
                    return {}

                # Select track and year from the actual form dropdowns
                await page.select_option('select[name="trackid"]', track_value)
                await page.select_option('select[name="YEAR"]', str(year))
                await asyncio.sleep(0.3)

                # Submit the form
                await page.evaluate('document.myForm.submit()')
                await page.wait_for_load_state("domcontentloaded", timeout=15000)
                await asyncio.sleep(0.8)

                # Check we didn't land on bot block
                body = await page.inner_text("body")
                if "pardon our interruption" in body.lower():
                    await asyncio.sleep(3)
                    continue

                if debug and year == years[0] and attempt == 0:
                    html = await page.content()
                    Path("debug_calendar_page.html").write_text(html)
                    print(f"  [debug] Saved {track} {year} -> debug_calendar_page.html")

                found = await _extract_dates_from_page(page, year_set)
                all_dates.update(found)
                break  # success

            except Exception as e:
                if attempt == 2 and debug:
                    print(f"  [warn] {track} {year}: {e}")
                await asyncio.sleep(1)

    # Group by year
    result: dict[str, list[str]] = {}
    for d in sorted(all_dates):
        yr = d[:4]
        result.setdefault(yr, []).append(d)

    return result


def parse_years(raw: str) -> list[int]:
    """Parse '2022,2023' or '2022-2026' into a list of years."""
    if "-" in raw and "," not in raw:
        parts = raw.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(y.strip()) for y in raw.split(",")]


def _find_chrome() -> str | None:
    """Find Chrome/Chromium executable (handles WSL)."""
    if Path("/proc/version").exists() and "microsoft" in Path("/proc/version").read_text().lower():
        for prefix in ["/mnt/c/Program Files/Google/Chrome/Application",
                       "/mnt/c/Program Files (x86)/Google/Chrome/Application"]:
            p = Path(prefix) / "chrome.exe"
            if p.exists():
                return str(p)
    for name in ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"]:
        found = shutil.which(name)
        if found:
            return found
    return None


def _launch_chrome(port: int) -> subprocess.Popen | None:
    chrome = _find_chrome()
    if not chrome:
        return None
    print(f"Launching Chrome with debug port {port}: {chrome}")
    try:
        proc = subprocess.Popen(
            [chrome, f"--remote-debugging-port={port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
        if proc.poll() is not None:
            print(f"Chrome exited immediately (code {proc.returncode})")
            return None
        return proc
    except Exception as e:
        print(f"Failed to launch Chrome: {e}")
        return None


async def main_async(years: list[int], output: str, concurrency: int,
                     cdp_port: int = 9222, debug=False, track_filter=None):
    results: dict = {}
    async with async_playwright() as pw:
        browser = None
        is_cdp = False
        for url in [f"http://localhost:{cdp_port}", f"http://127.0.0.1:{cdp_port}"]:
            try:
                browser = await pw.chromium.connect_over_cdp(url)
                is_cdp = True
                print(f"Connected to Chrome via CDP on {url}")
                break
            except Exception:
                pass

        if not browser:
            chrome_proc = _launch_chrome(cdp_port)
            if chrome_proc:
                for url in [f"http://localhost:{cdp_port}", f"http://127.0.0.1:{cdp_port}"]:
                    try:
                        browser = await pw.chromium.connect_over_cdp(url)
                        is_cdp = True
                        print(f"Connected to auto-launched Chrome on {url}")
                        break
                    except Exception:
                        pass

        if not browser:
            print("CDP not available — launching fresh browser (may get bot-blocked)")
            browser = await pw.chromium.launch(headless=True, args=[
                "--disable-blink-features=AutomationControlled", "--no-sandbox",
            ])

        # Warmup: pass bot check
        if is_cdp:
            warmup_ctx = browser.contexts[0]
        else:
            warmup_ctx = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            )
        warmup_page = await warmup_ctx.new_page()
        print("Warming up: visiting equibase.com to pass bot check...")
        try:
            await warmup_page.goto("https://www.equibase.com/",
                                   wait_until="domcontentloaded", timeout=20000)
            await asyncio.sleep(5)
            body = await warmup_page.inner_text("body")
            if "pardon our interruption" in body.lower():
                print("WARNING: Bot detection active. Waiting 15s for challenge...")
                await asyncio.sleep(15)
                body = await warmup_page.inner_text("body")
                if "pardon our interruption" in body.lower():
                    print("ERROR: Still blocked. Try with Chrome CDP.")
                    return
            print("Bot check passed!")
        except Exception as e:
            print(f"Warmup failed: {e}")
        finally:
            await warmup_page.close()

        tracks_to_fetch = sorted(track_filter) if track_filter else sorted(TRACKS.keys())

        # Process tracks sequentially (each one does a form submit on the same page type)
        # but use concurrency for parallelism across tracks
        sem = asyncio.Semaphore(concurrency)
        total_dates = 0

        async def process_track(track):
            nonlocal total_dates
            async with sem:
                page = await warmup_ctx.new_page()
                await setup_page_blocking(page)
                track_name = TRACKS.get(track, track)
                do_debug = debug and (debug == "FIRST" or debug.upper() == track.upper())
                try:
                    yr_map = await fetch_dates_for_track(page, track, track_name, years, debug=do_debug)
                    for yr, dates in yr_map.items():
                        total_dates += len(dates)
                        print(f"  {track} {yr}: {len(dates)} race days")
                    if not yr_map:
                        print(f"  {track}: --")
                    results[track] = yr_map
                except Exception as e:
                    print(f"  {track}: ERROR {e}")
                    results[track] = {}
                finally:
                    await page.close()

        await asyncio.gather(*[process_track(t) for t in tracks_to_fetch])

        if not is_cdp:
            await browser.close()

    # Merge with existing data
    out_path = Path(output)
    existing: dict = {}
    if out_path.exists():
        try:
            raw = json.loads(out_path.read_text())
            existing = raw.get("per_track_per_year") or raw.get("tracks") or {}
        except Exception:
            pass

    for track, yr_map in results.items():
        if track not in existing:
            existing[track] = {}
        existing[track].update(yr_map)

    out_path.write_text(json.dumps({"per_track_per_year": existing}, indent=2))
    print(f"\nSaved {total_dates} new race dates -> {out_path}")
    if existing:
        all_years = set()
        for ym in existing.values():
            all_years.update(ym.keys())
        print(f"File now contains years: {sorted(all_years)}")


def main():
    p = argparse.ArgumentParser(description="Fetch race-day calendar from Equibase")
    p.add_argument("--years", default=str(datetime.now().year),
                   help="Years to fetch, e.g. '2024' or '2022-2026' or '2023,2024'")
    p.add_argument("--output", default="track_race_dates.json",
                   help="Output JSON path (default: track_race_dates.json)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Number of concurrent page workers (default: 4)")
    p.add_argument("--cdp-port", type=int, default=9222,
                   help="Chrome DevTools port to connect to (default: 9222)")
    p.add_argument("--debug", nargs="?", const="FIRST", default=None,
                   help="Dump calendar page HTML. Optionally specify track code (e.g. --debug KEE)")
    p.add_argument("--tracks", help="Comma-separated track codes to fetch (default: all)")
    args = p.parse_args()

    years = parse_years(args.years)
    track_list = args.tracks.split(",") if args.tracks else None
    debug = args.debug if args.debug else False
    print(f"Fetching race dates for {len(track_list or TRACKS)} tracks, years: {years}")
    asyncio.run(main_async(years, args.output, args.concurrency,
                           cdp_port=args.cdp_port, debug=debug,
                           track_filter=track_list))


if __name__ == "__main__":
    main()
