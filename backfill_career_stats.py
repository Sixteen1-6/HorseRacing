"""Backfill num_past_{starts,wins,seconds,thirds} on a races CSV by looking
up each unique horse name on Equibase.

Design:
- Extract unique horse names across the whole CSV (batch-by-meet implicit).
- Disk cache at horse_career_cache.json — resume / re-run safe, survives
  across the 4 distributed PCs if you copy the file.
- Ad/tracker blocklist via Playwright route interception (~3x speedup vs.
  letting ads load).
- N concurrent Playwright pages (default 8).
- Attaches to the CDP Chrome on localhost:9222 so you use your signed-in
  session.

Usage:
    # Run backfill against the merged 2024-2026 CSV
    python backfill_career_stats.py \
        --input races_2024_2026_merged.csv \
        --output races_2024_2026_enriched.csv \
        --parallel 8

    # Test run on a small file
    python backfill_career_stats.py --input career_week_test.csv \
        --output career_week_test_enriched.csv --parallel 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from playwright.async_api import async_playwright, Page, BrowserContext

CACHE_FILE = Path("horse_career_cache.json")
CAREER_COLS = ["num_past_starts", "num_past_wins", "num_past_seconds", "num_past_thirds"]

# Domain/path fragments to abort during rendering. These are ads, trackers,
# consent frameworks, real-time bidders, and the Fuse ad platform Equibase
# uses. Blocking them cuts profile page load from ~3s to ~0.7-1s.
BLOCKED_FRAGMENTS = (
    "doubleclick", "googletagmanager", "google-analytics", "googlesyndication",
    "googleadservices", "adservice.google", "criteo", "pubmatic", "rubicon",
    "adnxs", "adsrvr", "fuseplatform", "bloodhorse", "uniconsent", "amazon-adsystem",
    "gumgum", "casalemedia", "sodar", "safeframe", "ingage.tech", "media.net",
    "richaudience", "kueezrtb", "cootlogix", "lijit", "33across", "openrtb",
    "pbxai", "unrulymedia", "optable.co", "dns-finder", "adtrafficquality",
    "servenobid", "smartadserver", "hbopenbid", "pagead", "ad-delivery",
    "cmp.uniconsent", "scorecardresearch", "taboola", "outbrain", "yieldmo",
    "analytics.google", "recaptcha", "gstatic.com/recaptcha", "html-load.cc",
    "/pagead/", "/cm.g.doubleclick", "fonts.googleapis", "fonts.gstatic",
)

# Resource types to block regardless of domain.
BLOCKED_RESOURCE_TYPES = {"image", "font", "media", "stylesheet"}


def should_block_url(url: str) -> bool:
    return any(frag in url for frag in BLOCKED_FRAGMENTS)


def load_cache() -> Dict[str, Dict]:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"  WARN: {CACHE_FILE} corrupt; starting fresh")
    return {}


def save_cache(cache: Dict[str, Dict]):
    tmp = CACHE_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    tmp.replace(CACHE_FILE)


def parse_career_from_html(html: str) -> Optional[Dict]:
    """Extract career totals from a horse profile page's HTML.

    The profile page renders the block as:
        CAREER STATISTICS*
        Starts  Firsts  Seconds  Thirds  Earnings
        14      7       2       2       $1,234,567
    Strip tags first so the regex works across both table and flex layouts.
    """
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    m = re.search(
        r"CAREER\s+STATISTICS\*?\s*"
        r"Starts\s+Firsts\s+Seconds\s+Thirds\s+Earnings\s+"
        r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
        text, re.I,
    )
    if m:
        return {
            "num_past_starts":  int(m.group(1)),
            "num_past_wins":    int(m.group(2)),
            "num_past_seconds": int(m.group(3)),
            "num_past_thirds":  int(m.group(4)),
        }
    return None


async def setup_page_blocking(page: Page):
    """Intercept and abort ads/trackers on this page."""
    async def handler(route):
        req = route.request
        try:
            if req.resource_type in BLOCKED_RESOURCE_TYPES:
                await route.abort()
                return
            if should_block_url(req.url):
                await route.abort()
                return
            await route.continue_()
        except Exception:
            pass
    await page.route("**/*", handler)


async def fetch_one(page: Page, horse_name: str) -> Dict:
    """Fetch one horse's career totals via the homepage form-click flow.
    Returns {} on any failure.

    Handles two landing outcomes:
    - Direct profile hit: URL contains `refno=` — parse inline.
    - Disambiguation list: URL has no `refno` — pick the first candidate
      `refno=` link, navigate to it, and parse. This is a heuristic for
      horses that share names; we accept the most-listed candidate.
    """
    try:
        await page.goto("https://www.equibase.com/", wait_until="domcontentloaded", timeout=15000)
        await page.fill("input[name='searchInput']", horse_name, timeout=5000)
        try:
            await page.click("form button[type='submit'], form input[type='submit']", timeout=3000)
        except Exception:
            await page.press("input[name='searchInput']", "Enter")
        await page.wait_for_load_state("domcontentloaded", timeout=10000)
        # Small settle for JS-rendered career block.
        await asyncio.sleep(0.6)

        # Disambiguation fallback.
        if "refno=" not in page.url:
            html = await page.content()
            m = re.search(r'href="(/profiles/Results\.cfm\?type=Horse[^"]*refno=\d+[^"]*)"', html)
            if not m:
                return {}
            href = m.group(1).replace("&amp;", "&")
            await page.goto("https://www.equibase.com" + href,
                            wait_until="domcontentloaded", timeout=15000)
            await asyncio.sleep(0.6)

        html = await page.content()
    except Exception:
        return {}
    return parse_career_from_html(html) or {}


async def worker(wid: int, ctx: BrowserContext, queue: asyncio.Queue,
                 cache: Dict[str, Dict], stats: Dict, save_lock: asyncio.Lock):
    page = await ctx.new_page()
    await setup_page_blocking(page)
    try:
        while True:
            try:
                name = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            t0 = time.time()
            result = await fetch_one(page, name)
            dt = time.time() - t0

            cache[name.strip().lower()] = result
            stats["fetched"] += 1
            if result:
                stats["hits"] += 1
            else:
                stats["misses"] += 1

            total_done = stats["fetched"] + stats["cached"]
            elapsed = time.time() - stats["start"]
            rate = stats["fetched"] / max(0.1, elapsed)
            eta = (stats["total"] - total_done) / max(0.01, rate)
            print(f"  [w{wid}] {name[:28]:28s} "
                  f"{'OK' if result else '—':3s} "
                  f"{dt:.2f}s  "
                  f"[{total_done}/{stats['total']}] "
                  f"rate={rate:.1f}/s eta={eta:.0f}s",
                  flush=True)

            # Checkpoint the cache every 50 fetches (any worker).
            if stats["fetched"] % 50 == 0:
                async with save_lock:
                    save_cache(cache)

            queue.task_done()
    finally:
        try:
            await page.close()
        except Exception:
            pass


async def run_backfill(input_path: Path, output_path: Path, parallel: int,
                       cdp_port: int):
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} rows from {input_path}")
    if "horse_name" not in df.columns:
        raise SystemExit("Input CSV has no 'horse_name' column.")

    names = df["horse_name"].dropna().astype(str).str.strip()
    unique = sorted({n for n in names if n}, key=str.lower)
    print(f"Unique horses: {len(unique):,}")

    cache = load_cache()
    print(f"Disk cache: {len(cache):,} entries")

    # Only count a cached horse as "done" if it has a successful parse OR
    # we want to respect past failures. Here: always skip anything cached,
    # so we don't re-hit horses we couldn't resolve previously.
    to_fetch = [n for n in unique if n.lower() not in cache]
    print(f"To fetch:   {len(to_fetch):,}")
    already_cached = len(unique) - len(to_fetch)

    if to_fetch:
        queue: asyncio.Queue = asyncio.Queue()
        for n in to_fetch:
            queue.put_nowait(n)

        stats = {
            "total": len(to_fetch),
            "fetched": 0, "cached": already_cached,
            "hits": 0, "misses": 0,
            "start": time.time(),
        }
        save_lock = asyncio.Lock()

        async with async_playwright() as pw:
            print(f"Connecting to CDP Chrome on localhost:{cdp_port}...")
            browser = await pw.chromium.connect_over_cdp(f"http://localhost:{cdp_port}")
            ctx = browser.contexts[0]

            print(f"Launching {parallel} parallel workers with ad blocking...")
            workers = [asyncio.create_task(worker(i, ctx, queue, cache, stats, save_lock))
                       for i in range(parallel)]
            await asyncio.gather(*workers, return_exceptions=True)

        save_cache(cache)
        elapsed = time.time() - stats["start"]
        print(f"\nDone fetching in {elapsed:.1f}s "
              f"(hits={stats['hits']}, misses={stats['misses']})")
    else:
        print("All horses already cached. Applying cache to CSV...")

    # Apply cache back onto the dataframe.
    def lookup(name):
        if not isinstance(name, str) or not name.strip():
            return (None, None, None, None)
        r = cache.get(name.strip().lower()) or {}
        return (r.get("num_past_starts"), r.get("num_past_wins"),
                r.get("num_past_seconds"), r.get("num_past_thirds"))

    results = df["horse_name"].apply(lookup)
    df[CAREER_COLS] = pd.DataFrame(results.tolist(), index=df.index)
    df.to_csv(output_path, index=False)

    filled = df[CAREER_COLS[0]].notna().sum()
    print(f"Wrote {output_path} — {len(df):,} rows, {filled:,} "
          f"({filled/max(1,len(df))*100:.1f}%) with career stats")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="Input CSV (must have horse_name column)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output CSV with career columns populated")
    ap.add_argument("--parallel", type=int, default=8,
                    help="Concurrent Playwright pages (default 8)")
    ap.add_argument("--cdp-port", type=int, default=9222,
                    help="CDP port for your signed-in Chrome (default 9222)")
    args = ap.parse_args()

    asyncio.run(run_backfill(args.input, args.output, args.parallel, args.cdp_port))


if __name__ == "__main__":
    main()
