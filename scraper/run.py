"""
Horse Racing Data Scraper v3 — Google-Entry, Human-Like Browsing
=================================================================

Browses exactly like you would:
  1. Google search → "Keeneland April 3 2026 race results"
  2. Click the Equibase result (organic traffic, not direct URL)
  3. Land on race day chart index page
  4. Click each race link → opens full chart (PDF or HTML)
  5. Parse chart for all data columns
  6. Move to next date, repeat

Anti-detection:
  - Random delays with jitter (2-8s between actions)
  - Random mouse movements before clicks
  - Random scrolling behavior
  - Viewport size variation
  - User-agent rotation
  - Occasional "distraction" browsing (visit homepage, etc.)
  - Typing speed variation for search queries
  - Referrer chain looks natural (Google → site)

Setup:
  pip install playwright pdfplumber
  playwright install chromium

Usage:
  # Test on one day (visible browser to debug)
  python3 -m scraper --tracks KEE --start 2026-04-03 --end 2026-04-03 --visible

  # Keeneland spring 2023
  python3 -m scraper --tracks KEE --start 2023-04-07 --end 2023-04-28

  # Resume interrupted scrape
  python3 -m scraper --resume

  # Full 10-year scrape
  python3 -m scraper --start 2016-01-01 --end 2026-04-04

  # Concurrent scraping (4 browser contexts) with verbose output
  python3 -m scraper --start 2024-01-01 --end 2024-12-31 --concurrency 4 -v
"""

import asyncio
import contextvars
import csv
import json
import os
import re
import random
import subprocess
import sys
import time
import logging
import argparse
import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple

try:
    from playwright.async_api import async_playwright, Page
except ImportError:
    sys.exit("Run: pip install playwright && playwright install chromium")

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    print("[WARN] pdfplumber not installed — PDF parsing disabled. pip install pdfplumber")

from .config import OUTPUT_COLUMNS, TRACKS, USER_AGENTS, VIEWPORTS, is_race_day
from .human_behavior import HumanBehavior
from .pdf_parser import ChartPDFParser
from .checkpoint import Checkpoint

# ── Per-worker context variable for log prefixing ────────────
# Each async worker sets this; the log filter reads it automatically.
_worker_id: contextvars.ContextVar[str] = contextvars.ContextVar('worker_id', default='')


class _WorkerFilter(logging.Filter):
    """Injects the current async worker ID into every log record."""
    def filter(self, record):
        wid = _worker_id.get('')
        record.worker = f" [{wid}]" if wid else ""
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]%(worker)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("scraper_v3.log")],
)
log = logging.getLogger("scraper")
log.addFilter(_WorkerFilter())


# ═══════════════════════════════════════════════════════════════
# MAIN SCRAPER
# ═══════════════════════════════════════════════════════════════

class RaceScraper:
    """
    The main scraper. Flow for each (track, date):

    1. Google search: "{track_name} {date} race results equibase"
    2. Click the Equibase result link
    3. On the race day index page, find links to each race chart
    4. For each race:
       a. Click the race link
       b. If PDF → download bytes → parse with pdfplumber
       c. If HTML → parse tables from DOM
       d. Extract all columns
    5. Save entries to CSV
    6. Checkpoint progress
    """

    # Equibase race day index URL pattern (used as fallback if Google fails)
    INDEX_URL = "https://www.equibase.com/premium/eqbPDFChartPlusIndex.cfm?tid={track}&dt={date}&ctry=USA"
    # Summary results — all races on one page (free, HTML)
    # Format: {TRACK}{MMDDYY}USA-EQB.html  e.g. KEE040326USA-EQB.html
    SUMMARY_URL = "https://www.equibase.com/static/chart/summary/{track}{date}USA-EQB.html"
    # Race card index (shows race list with links to individual races)
    RACECARD_INDEX_URL = "https://www.equibase.com/static/chart/summary/RaceCardIndex{track}{date}USA-EQB.html"
    # Direct PDF chart URL — RECENT races (free, public)
    # Format: {TRACK}{MMDDYY}USA{RACE_NUM}.pdf  e.g. KEE040326USA1.pdf
    PDF_CHART_URL = "https://www.equibase.com/static/chart/pdf/{track}{date}USA{race_num}.pdf"
    # Direct PDF chart URL — HISTORICAL races (free, public)
    # Format: /static/chart/{YEAR}/usa/{track_lower}/{YYYYMMDD}-usa-{track_lower}-{race_num}-d.standard.pdf
    PDF_CHART_HISTORICAL_URL = "https://www.equibase.com/static/chart/{year}/usa/{track_lower}/{date_ymd}-usa-{track_lower}-{race_num}-d.standard.pdf"

    def __init__(self, output_file="historical_races.csv", headless=True, use_google=True,
                 cdp_port=9222, checkpoint_path="scraper_v3_checkpoint.json",
                 career_stats=False, concurrency=1):
        self.output_file = output_file
        self.headless = headless
        self._use_google = use_google
        self._cdp_port = cdp_port
        self._concurrency = max(1, concurrency)
        self.checkpoint = Checkpoint(checkpoint_path)
        self.pdf_parser = ChartPDFParser()
        self.human = HumanBehavior()
        self.pw = None
        self.browser = None
        self._csv_file = None
        self._csv_writer = None
        # Career stats enrichment
        self._career_stats_enabled = career_stats
        # Cache: horse_name_lower -> {num_past_starts, num_past_wins, num_past_seconds, num_past_thirds}
        # Keeps scrape fast when the same horse races repeatedly.
        self._career_cache: Dict[str, Dict] = {}
        self._career_locks: Dict[str, asyncio.Lock] = {}
        self._career_lock_creation = asyncio.Lock()
        self._career_miss = 0
        self._career_hit = 0
        # Atomic progress counter for concurrent workers
        self._completed_count = 0
        self._run_start_time = 0.0

    # ── Browser lifecycle ────────────────────────────────────

    @staticmethod
    def _find_chrome_executable() -> Optional[str]:
        """Find the Chrome/Chromium executable on this system."""
        system = platform.system()
        candidates = []
        if system == "Windows":
            candidates = [
                Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "Google" / "Chrome" / "Application" / "chrome.exe",
                Path(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)")) / "Google" / "Chrome" / "Application" / "chrome.exe",
                Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
            ]
        elif system == "Darwin":
            candidates = [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            ]
        else:  # Linux / WSL
            # Check if we're in WSL and can launch Windows Chrome
            wsl = Path("/proc/version").exists() and "microsoft" in Path("/proc/version").read_text().lower()
            if wsl:
                for prefix in ["/mnt/c/Program Files/Google/Chrome/Application",
                               "/mnt/c/Program Files (x86)/Google/Chrome/Application"]:
                    p = Path(prefix) / "chrome.exe"
                    if p.exists():
                        candidates.append(p)
            # Native Linux Chrome
            for name in ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"]:
                import shutil
                found = shutil.which(name)
                if found:
                    candidates.append(Path(found))

        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def _launch_chrome_debug(self) -> Optional[subprocess.Popen]:
        """Auto-launch Chrome with --remote-debugging-port if not already running."""
        chrome = self._find_chrome_executable()
        if not chrome:
            log.debug("Chrome executable not found — skipping auto-launch")
            return None

        log.info(f"Launching Chrome with debug port {self._cdp_port}: {chrome}")
        try:
            proc = subprocess.Popen(
                [chrome, f"--remote-debugging-port={self._cdp_port}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Give Chrome a moment to start and open the debug port
            time.sleep(2)
            if proc.poll() is not None:
                log.warning(f"Chrome exited immediately (code {proc.returncode})")
                return None
            log.info(f"Chrome launched (pid={proc.pid})")
            return proc
        except Exception as e:
            log.warning(f"Failed to launch Chrome: {e}")
            return None

    async def _start_browser(self):
        """Launch the browser only. Context/page creation is handled by _create_context()."""
        self.pw = await async_playwright().start()
        self._is_cdp = False
        self._chrome_proc = None

        # Try connecting to running Chrome via CDP first
        if self._cdp_port:
            for url in [f"http://localhost:{self._cdp_port}", f"http://127.0.0.1:{self._cdp_port}"]:
                try:
                    self.browser = await self.pw.chromium.connect_over_cdp(url)
                    self._is_cdp = True
                    log.info(f"Connected to your running Chrome on {url}! "
                             "(assuming you're already signed in)")
                    return
                except Exception as e:
                    log.debug(f"CDP connect to {url} failed: {e}")

            # Chrome isn't running — try to auto-launch it
            self._chrome_proc = self._launch_chrome_debug()
            if self._chrome_proc:
                for url in [f"http://localhost:{self._cdp_port}", f"http://127.0.0.1:{self._cdp_port}"]:
                    try:
                        self.browser = await self.pw.chromium.connect_over_cdp(url)
                        self._is_cdp = True
                        log.info(f"Connected to auto-launched Chrome on {url}!")
                        return
                    except Exception as e:
                        log.debug(f"CDP connect after auto-launch failed: {e}")

            log.warning(f"Could not connect to Chrome on port {self._cdp_port}. Falling back to fresh browser.")

        # Fallback: launch fresh Playwright browser
        self.browser = await self.pw.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        log.info(f"Fresh browser launched (concurrency={self._concurrency})")

    async def _create_context(self):
        """Create a new browser context + page with randomized fingerprint."""
        ua = random.choice(USER_AGENTS)
        vp = random.choice(VIEWPORTS)

        if self._is_cdp:
            # CDP: reuse the existing context, just open a new page
            context = self.browser.contexts[0]
            page = await context.new_page()
            log.debug(f"New CDP page opened")
            return context, page

        context = await self.browser.new_context(
            user_agent=ua,
            viewport=vp,
            locale="en-US",
            timezone_id="America/New_York",
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
        """)
        page = await context.new_page()
        log.debug(f"New context | UA: {ua[ua.rfind(')')-15:ua.rfind(')')]}... | "
                  f"Viewport: {vp['width']}x{vp['height']}")
        return context, page

    async def _setup_session(self):
        """Run one-time login and bot-detection using a temporary page.
        Called once before workers start."""
        context, page = await self._create_context()
        try:
            await self._pass_bot_detection(page)
            # Only wait for login when there's a visible browser to sign into
            if not self.headless or self._is_cdp:
                await self._wait_for_equibase_login(page)
            else:
                log.debug("Headless mode — skipping Equibase login wait")
        finally:
            await page.close()
            if not self._is_cdp:
                await context.close()

    async def _wait_for_equibase_login(self, page: Page):
        """Open equibase.com in the attached Chrome and wait for the user to
        sign in manually. Polls until a signed-in indicator appears, or user
        presses Enter in the terminal."""
        try:
            await page.goto("https://www.equibase.com/", wait_until="domcontentloaded", timeout=20000)
            await asyncio.sleep(1.5)
        except Exception as e:
            log.warning(f"Could not open equibase.com: {e}")

        async def _is_signed_in() -> bool:
            try:
                text = await page.inner_text("body")
                tl = text.lower()
                if "sign out" in tl or "my account" in tl or "welcome," in tl:
                    return True
            except Exception:
                pass
            return False

        if await _is_signed_in():
            log.info("Already signed into Equibase. Continuing...")
            return

        log.info("=" * 60)
        log.info("Please sign in to Equibase in the Chrome window that opened.")
        log.info("Polling every 3s for sign-in (5 min timeout)...")
        log.info("=" * 60)

        # Poll the page every 3 seconds for up to 5 minutes.
        for i in range(100):
            await asyncio.sleep(3)
            if await _is_signed_in():
                log.info("Sign-in detected. Continuing with scrape...")
                return
            if i % 5 == 4:
                log.info(f"  still waiting... ({(100 - i - 1) * 3}s remaining)")
        log.warning("Sign-in wait timed out — continuing anyway (career stats may fail).")

    async def _prompt_login(self, page: Page):
        """Prompt user to sign in via the URL bar, then wait."""
        # Check if already signed in
        await page.goto("https://accounts.google.com", wait_until="domcontentloaded", timeout=15000)
        await asyncio.sleep(2)
        page_url = page.url
        if "myaccount" in page_url or "signin" not in page_url.lower():
            text = await page.inner_text("body")
            if "sign in" not in text.lower()[:300] and len(text) > 100:
                log.info("Already signed into Google. Continuing...")
                return

        # Use the URL bar + page content as a message to the user
        await page.goto("data:text/html,<html><head><title>SIGN IN TO CHROME - Scraper waiting...</title></head>"
                             "<body style='display:flex;align-items:center;justify-content:center;height:100vh;"
                             "font-family:Arial;background:%23222;color:white'>"
                             "<div style='text-align:center'>"
                             "<h1>Please sign in to your Chrome profile</h1>"
                             "<p style='font-size:24px;color:%23aaa'>Click your profile icon (top right) and sign in to Google.</p>"
                             "<p style='font-size:20px;color:%23888'>The scraper will continue automatically once you're signed in.</p>"
                             "<p style='font-size:18px;color:%23666'>You have 2 minutes.</p>"
                             "</div></body></html>")

        log.info("=" * 50)
        log.info("WAITING FOR SIGN-IN -- Check the Chrome window")
        log.info("=" * 50)

        # Poll: check if user signed in every 5 seconds for 2 minutes
        context = page.context
        for i in range(24):
            await asyncio.sleep(5)
            try:
                # Open a new tab to check sign-in status
                check_page = await context.new_page()
                await check_page.goto("https://accounts.google.com", wait_until="domcontentloaded", timeout=10000)
                await asyncio.sleep(2)
                check_url = check_page.url
                check_text = await check_page.inner_text("body")
                await check_page.close()

                if "myaccount" in check_url or ("sign in" not in check_text.lower()[:300] and len(check_text) > 100):
                    log.info("Sign-in detected! Continuing scraper...")
                    # Show success message briefly
                    await page.goto("data:text/html,<html><body style='display:flex;align-items:center;"
                                         "justify-content:center;height:100vh;font-family:Arial;background:%23222;"
                                         "color:%2300ff00'><h1>Signed in! Starting scraper...</h1></body></html>")
                    await asyncio.sleep(2)
                    return
            except Exception:
                pass
            remaining = 120 - (i + 1) * 5
            if i % 4 == 3:
                log.info(f"Still waiting for sign-in... ({remaining}s remaining)")

        log.warning("Timed out waiting for sign-in (2 min). Continuing anyway...")

    async def _pass_bot_detection(self, page: Page):
        """Navigate to Equibase and let the user solve any captcha."""
        log.info("Opening equibase.com to check for bot detection...")
        await page.goto("https://www.equibase.com", wait_until="domcontentloaded", timeout=20000)
        await asyncio.sleep(3)

        text = await page.inner_text("body")
        bot_phrases = ["security check", "captcha", "i am human", "pardon our interruption"]
        if any(p in text.lower() for p in bot_phrases):
            log.info("=" * 50)
            log.info("CAPTCHA DETECTED in the browser window!")
            log.info("Please solve it manually. Scraper will wait...")
            log.info("=" * 50)
            print("\n>>> SOLVE THE CAPTCHA in the browser window, then wait... <<<\n")

            # Poll every 3 seconds for up to 2 minutes
            for i in range(40):
                await asyncio.sleep(3)
                try:
                    text = await page.inner_text("body")
                    if not any(p in text.lower() for p in bot_phrases):
                        log.info("Captcha solved! Continuing scraper...")
                        await asyncio.sleep(2)
                        return
                except Exception:
                    pass
                if i % 10 == 9:
                    log.info("Still waiting for captcha...")

            log.warning("Timed out waiting for captcha (2 min). Continuing anyway...")
        else:
            log.info("No bot detection -- equibase.com loaded fine!")

    @staticmethod
    def _find_chrome_profile() -> Optional[str]:
        """Find the real Chrome user data directory on this system."""
        import platform
        system = platform.system()
        home = Path.home()

        candidates = []
        if system == "Windows":
            candidates = [
                home / "AppData" / "Local" / "Google" / "Chrome" / "User Data",
            ]
        elif system == "Darwin":
            candidates = [
                home / "Library" / "Application Support" / "Google" / "Chrome",
            ]
        else:  # Linux
            candidates = [
                home / ".config" / "google-chrome",
                home / ".config" / "chromium",
            ]

        for path in candidates:
            if path.exists():
                return str(path)
        return None

    async def _stop_browser(self):
        if self.browser:
            await self.browser.close()
        if self.pw:
            await self.pw.stop()

    # ── Career stats enrichment (horse profile scrape) ───────
    #
    # The chart PDFs contain only the result of a single race — they don't
    # report a horse's career totals. To fill num_past_{starts,wins,seconds,
    # thirds}, we hit the Equibase horse profile page per unique horse and
    # read the career stats table. This mirrors scraper_entries.py's
    # _fetch_horse_pp, but we need a URL (not a name) since Equibase has no
    # public name-search endpoint. We harvest those URLs from the chart
    # index HTML page (eqbPDFChartPlusIndex.cfm) which embeds
    # /profiles/Results.cfm?type=Horse&refno=... links for each runner.

    async def _fetch_horse_career_by_name(self, horse_name: str, page: Page) -> Dict:
        """Search Equibase by horse name via the homepage form and parse the
        career stats block from whatever page lands. Returns {} on failure."""
        result: Dict = {}
        try:
            await page.goto("https://www.equibase.com/", wait_until="domcontentloaded", timeout=20000)
            await asyncio.sleep(0.5)
            await page.fill("input[name='searchInput']", horse_name)
            # Click the submit button (JS-driven form → direct profile page for unique matches)
            try:
                await page.click("form button[type='submit'], form input[type='submit']", timeout=3000)
            except Exception:
                await page.press("input[name='searchInput']", "Enter")
            await page.wait_for_load_state("domcontentloaded", timeout=15000)
            await asyncio.sleep(0.8)

            landed = page.url
            # Case A: direct hit — we're on a horse profile page already.
            if "type=Horse" in landed and "refno=" in landed:
                result = await self._parse_career_from_profile(page)
                return result

            # Case B: multi-match search result — click the first TB match.
            profile_href = await page.evaluate(
                r"""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href*="type=Horse"]'))
                        .filter(a => /refno=\d+/i.test(a.href));
                    if (!links.length) return null;
                    const tb = links.find(a => /registry=T(?:&|$)/i.test(a.href)) || links[0];
                    return tb.href;
                }
                """
            )
            if profile_href:
                await page.goto(profile_href, wait_until="domcontentloaded", timeout=20000)
                await asyncio.sleep(0.8)
                result = await self._parse_career_from_profile(page)
        except Exception as e:
            log.debug(f"career search failed for {horse_name!r}: {e}")
        return result

    async def _parse_career_from_profile(self, page: Page) -> Dict:
        """Extract career totals from the currently-loaded horse profile page.
        Tries DOM tables first, then falls back to body-text regex because
        Equibase sometimes renders the block as a flex layout without a
        standards-compliant table header row."""
        # Attempt 1: table with canonical header
        try:
            data = await page.evaluate(
                r"""
                () => {
                    const tables = Array.from(document.querySelectorAll('table'))
                        .filter(t => {
                            const h = t.rows[0];
                            if (!h) return false;
                            const txt = Array.from(h.cells).map(c => c.innerText.trim()).join('|');
                            return txt === 'Starts|Firsts|Seconds|Thirds|Earnings';
                        });
                    return tables.map(t =>
                        Array.from(t.rows).map(r =>
                            Array.from(r.cells).map(c => c.innerText.trim())
                        )
                    );
                }
                """
            )
            career_row = None
            if data and len(data) >= 2 and len(data[1]) > 1:
                career_row = data[1][1]
            elif data and len(data) == 1 and len(data[0]) > 1:
                career_row = data[0][1]
            if career_row and len(career_row) >= 4:
                try:
                    return {
                        "num_past_starts":  int(re.sub(r"[^\d]", "", career_row[0] or "0") or "0"),
                        "num_past_wins":    int(re.sub(r"[^\d]", "", career_row[1] or "0") or "0"),
                        "num_past_seconds": int(re.sub(r"[^\d]", "", career_row[2] or "0") or "0"),
                        "num_past_thirds":  int(re.sub(r"[^\d]", "", career_row[3] or "0") or "0"),
                    }
                except ValueError:
                    pass
        except Exception:
            pass

        # Attempt 2: body-text regex. On Fierceness's profile the text had the form
        #   "CAREER STATISTICS*\nStarts\tFirsts\tSeconds\tThirds\tEarnings\n14\t7\t2\t2"
        try:
            body = await page.evaluate("() => document.body.innerText || ''")
            m = re.search(
                r"CAREER\s+STATISTICS\*?[\s\S]{0,200}?"
                r"Starts\s+Firsts\s+Seconds\s+Thirds\s+Earnings\s+"
                r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
                body, re.I,
            )
            if m:
                return {
                    "num_past_starts":  int(m.group(1)),
                    "num_past_wins":    int(m.group(2)),
                    "num_past_seconds": int(m.group(3)),
                    "num_past_thirds":  int(m.group(4)),
                }
        except Exception:
            pass
        return {}

    async def _get_career_stats(self, name: str, page: Page) -> Dict:
        """Fetch career stats with per-horse locking to avoid redundant fetches
        when multiple concurrent workers encounter the same horse."""
        key = name.lower()

        # Fast path: already cached
        if key in self._career_cache:
            self._career_hit += 1
            stats = self._career_cache[key]
            if stats:
                log.debug(f"  Career CACHE HIT: {name} "
                          f"({stats.get('num_past_starts',0)} starts, "
                          f"{stats.get('num_past_wins',0)} wins)")
            else:
                log.debug(f"  Career CACHE HIT: {name} (no data)")
            return stats

        # Get or create a lock for this horse
        async with self._career_lock_creation:
            if key not in self._career_locks:
                self._career_locks[key] = asyncio.Lock()
            lock = self._career_locks[key]

        # Acquire the per-horse lock; double-check cache after acquiring
        async with lock:
            if key in self._career_cache:
                self._career_hit += 1
                return self._career_cache[key]
            self._career_miss += 1
            log.debug(f"  Career FETCH: {name} (searching Equibase...)")
            stats = await self._fetch_horse_career_by_name(name, page)
            self._career_cache[key] = stats
            if stats:
                log.debug(f"  Career FOUND: {name} -> "
                          f"{stats.get('num_past_starts',0)} starts, "
                          f"{stats.get('num_past_wins',0)} wins, "
                          f"{stats.get('num_past_seconds',0)} 2nds, "
                          f"{stats.get('num_past_thirds',0)} 3rds")
            else:
                log.debug(f"  Career NOT FOUND: {name} (no profile on Equibase)")
            await self.human.random_delay(0.4, 1.0)
            return stats

    async def _enrich_with_career_stats(self, entries: List[Dict], page: Page):
        """Populate career fields on each entry by searching each unique horse
        by name and parsing the profile. Cached across the run with per-horse locking."""
        if not self._career_stats_enabled or not entries:
            return

        unique_names = []
        seen = set()
        for e in entries:
            name = (e.get("horse_name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_names.append(name)

        log.debug(f"  Enriching career stats for {len(unique_names)} unique horses "
                  f"(cache: {len(self._career_cache)} horses, "
                  f"{self._career_hit}/{self._career_hit + self._career_miss} hit rate)")

        for idx, name in enumerate(unique_names):
            key = name.lower()
            stats = await self._get_career_stats(name, page)

            if stats:
                for e in entries:
                    if (e.get("horse_name") or "").strip().lower() == key:
                        for k, v in stats.items():
                            e[k] = v

    # ── CSV output ───────────────────────────────────────────

    def _open_csv(self):
        exists = os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0
        self._csv_file = open(self.output_file, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore"
        )
        if not exists:
            self._csv_writer.writeheader()

    def _write_entries(self, entries: List[Dict]):
        if not self._csv_writer:
            self._open_csv()
        for e in entries:
            self._csv_writer.writerow(e)
        self._csv_file.flush()

    def _close_csv(self):
        if self._csv_file:
            self._csv_file.close()

    # ── Google search entry ──────────────────────────────────

    async def _google_search(self, query: str, page: Page) -> Optional[str]:
        """
        Search Google and return the first Equibase result URL.
        Returns None if no result found.
        """
        try:
            await page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=15000)
            await self.human.random_delay(1.5, 3.0)

            # Accept cookies if prompted
            try:
                accept_btn = page.locator("button:has-text('Accept'), button:has-text('I agree')")
                if await accept_btn.count() > 0:
                    await accept_btn.first.click()
                    await self.human.random_delay(1.0, 2.0)
            except Exception:
                pass

            # Type the search query like a human
            search_box = page.locator('textarea[name="q"], input[name="q"]').first
            await search_box.click()
            await asyncio.sleep(random.uniform(0.3, 0.8))

            for char in query:
                await page.keyboard.type(char, delay=random.randint(40, 180))
                if random.random() < 0.03:
                    await asyncio.sleep(random.uniform(0.2, 0.6))

            await asyncio.sleep(random.uniform(0.5, 1.5))
            await page.keyboard.press("Enter")
            await page.wait_for_load_state("domcontentloaded")
            await self.human.random_delay(1.2, 2.5)

            # Find Equibase links in results
            links = await page.evaluate("""
                () => {
                    const results = [];
                    document.querySelectorAll('a[href]').forEach(a => {
                        const href = a.href || '';
                        if (href.includes('equibase.com') && !href.includes('google')) {
                            results.push(href);
                        }
                    });
                    return results;
                }
            """)

            if links:
                # Prefer chart/summary links
                for link in links:
                    if "chart" in link or "summary" in link or "PDFChart" in link:
                        return link
                return links[0]

            return None

        except Exception as e:
            log.warning(f"Google search failed: {e}")
            return None

    # ── Navigate to race day ─────────────────────────────────

    async def _navigate_to_race_day(self, track_code: str, date: datetime, page: Page) -> bool:
        """
        Navigate to race day results.
        Order: Direct summary URL first → Chart index → Google (last resort).
        Google is avoided by default because Playwright gets CAPTCHA'd.
        Returns True if we landed on a page with data.
        """
        track_name = TRACKS.get(track_code, track_code)
        date_str = date.strftime("%B %d, %Y").replace(" 0", " ")

        # Helper to check if a page loaded successfully
        async def _check_page(label: str) -> bool:
            landed_url = page.url
            log.info(f"{label} -> landed on: {landed_url}")
            text = await page.inner_text("body")
            text_preview = text[:200].replace('\n', ' ').strip()

            # Handle bot detection — wait and retry once
            if "Pardon Our Interruption" in text:
                log.info(f"{label}: Bot detection triggered, waiting 10s and reloading...")
                await asyncio.sleep(10)
                await page.reload(wait_until="domcontentloaded", timeout=15000)
                await self.human.random_delay(3.0, 5.0)
                text = await page.inner_text("body")
                text_preview = text[:200].replace('\n', ' ').strip()
                if "Pardon Our Interruption" in text:
                    log.warning(f"{label}: Bot detection persists after retry")
                    return False

            if self._page_has_no_data(text):
                log.info(f"{label}: No data detected. Page text: {text_preview}")
                return False

            log.info(f"{label}: Page has data! ({len(text)} chars)")
            return True

        # Tier-1 optimization: skip the SUMMARY_URL and RACECARD_INDEX_URL paths.
        # Both return 404 for every date in the 2024-2026 target window — they
        # wasted ~8s per race day with zero data yield. Jump straight to the
        # chart embed which is the first URL that actually works.

        # ── Attempt 1 (primary): Premium chart embed ──
        chart_embed_url = f"https://www.equibase.com/premium/chartEmb.cfm?track={track_code}&raceDate={date.strftime('%m/%d/%Y')}&cy=USA"
        log.info(f"Trying premium chart embed: {chart_embed_url}")
        try:
            await page.goto(chart_embed_url, wait_until="domcontentloaded", timeout=15000)
            await self.human.random_delay(1.2, 2.5)
            if await _check_page("Premium chart embed"):
                log.debug(f"  Nav strategy: chart embed succeeded for {track_code} {date.strftime('%Y-%m-%d')}")
                return True
        except Exception as e:
            log.info(f"Premium chart embed URL failed: {e}")

        # ── Attempt 3b: Premium chart index URL (older format) ──
        index_url = self.INDEX_URL.format(
            track=track_code, date=date.strftime("%m/%d/%Y")
        )
        log.info(f"Trying premium chart index: {index_url}")
        try:
            await page.goto(index_url, wait_until="domcontentloaded", timeout=15000)
            await self.human.random_delay(1.2, 2.5)
            if await _check_page("Premium chart index"):
                log.debug(f"  Nav strategy: chart index succeeded for {track_code} {date.strftime('%Y-%m-%d')}")
                return True
        except Exception as e:
            log.info(f"Premium chart index URL failed: {e}")

        # ── Attempt 4: Google search (last resort — may trigger CAPTCHA) ──
        if self._use_google:
            query = f"{track_name} {date_str} race results equibase"
            log.info(f"Falling back to Google: {query}")
            url = await self._google_search(query, page)

            if url:
                log.info(f"Google found: {url}")
                try:
                    link_clicked = await page.evaluate("""
                        () => {
                            const links = document.querySelectorAll('a[href]');
                            for (const a of links) {
                                if (a.href && a.href.includes('equibase.com')) {
                                    a.click();
                                    return true;
                                }
                            }
                            return false;
                        }
                    """)

                    if link_clicked:
                        await page.wait_for_load_state("domcontentloaded", timeout=15000)
                        await self.human.random_delay(1.2, 2.5)
                        landed_url = page.url
                        log.info(f"Google -> landed on: {landed_url}")
                        text = await page.inner_text("body")
                        if not self._page_has_no_data(text):
                            log.debug(f"  Nav strategy: Google search succeeded for {track_code} {date.strftime('%Y-%m-%d')}")
                            return True
                except Exception as e:
                    log.debug(f"Google click-through failed: {e}")

                    # If Google CAPTCHA'd us, disable Google for the rest of the run
                    page_text = await page.inner_text("body")
                    if "unusual traffic" in page_text.lower() or "captcha" in page_text.lower():
                        log.warning("Google CAPTCHA detected — disabling Google search for this run")
                        self._use_google = False

        log.warning(f"All navigation attempts failed for {track_code} {date.strftime('%Y-%m-%d')}")
        return False

    def _page_has_no_data(self, text: str) -> bool:
        text_lower = text.lower().strip()
        # Empty page or trivially small
        if len(text_lower) < 50:
            return True
        # Check for explicit "no data" messages, but be careful with "404"
        # which could appear in other contexts (e.g. purse amounts)
        no_data_phrases = [
            "no charts available", "no results found", "page not found",
            "no data available", "no races scheduled", "error 404",
            "we can't find the page", "this page doesn't exist",
        ]
        return any(p in text_lower for p in no_data_phrases)

    # ── Extract race data from current page ──────────────────

    async def _extract_from_page(self, track_code: str, date: datetime, page: Page) -> List[Dict]:
        """
        Extract race data from the current page.
        Tries multiple strategies:
          1. Find PDF chart links → download and parse PDFs
          2. Parse Equibase summary HTML tables directly
          3. Click into individual race pages
        """
        entries = []
        race_date_str = date.strftime("%m/%d/%Y")
        current_url = page.url
        log.info(f"Extracting from: {current_url}")

        # ── Strategy 1: Look for PDF chart links ──
        pdf_links = await page.evaluate("""
            () => {
                const links = [];
                document.querySelectorAll('a[href]').forEach(a => {
                    const href = a.href || '';
                    const text = a.innerText || '';
                    if (href.includes('.pdf') || href.includes('PDF') ||
                        href.includes('chart') || text.toLowerCase().includes('chart')) {
                        links.push({href, text: text.trim()});
                    }
                });
                return links;
            }
        """)

        if pdf_links and pdfplumber:
            # Filter to only real PDF URLs (the /static/chart/pdf/ ones are free)
            real_pdf_links = [l for l in pdf_links if "/static/chart/" in l["href"] and l["href"].endswith(".pdf")]
            if real_pdf_links:
                log.info(f"Found {len(real_pdf_links)} direct PDF chart links")
                for link_info in real_pdf_links:
                    href = link_info["href"]
                    pdf_entries = await self._download_and_parse_pdf(href, track_code, race_date_str, page)
                    if pdf_entries:
                        entries.extend(pdf_entries)
                        self.checkpoint.stats["pdfs"] += 1
                        await self.human.random_delay(2.0, 5.0)
            else:
                log.debug(f"Found {len(pdf_links)} links but none are direct PDF URLs")

        # ── Strategy 1b: Navigate to chart embed pages which redirect to PDFs ──
        # The chartEmb.cfm?...&rn=N URLs redirect directly to the PDF file.
        # The page URL after navigation IS the PDF URL.
        if not entries and pdfplumber:
            # eqbPDFChartPlus.cfm redirects directly to the PDF file
            chart_base = f"https://www.equibase.com/premium/eqbPDFChartPlus.cfm?BorP=P&TID={track_code}&CTRY=USA&DT={date.strftime('%m/%d/%Y')}&DAY=D&STYLE=EQB"
            saved_url = page.url
            consecutive_failures = 0

            log.info(f"Trying chart PDF download strategy with base: {chart_base}")
            for race_num in range(1, 16):
                chart_url = f"{chart_base}&RACE={race_num}"
                log.info(f"Race {race_num}: downloading from {chart_url}")
                try:
                    # Use page.request API to download the PDF bytes directly
                    # This avoids the "Download is starting" navigation error
                    response = await page.request.get(chart_url, timeout=30000)
                    status = response.status
                    content_type = response.headers.get("content-type", "")
                    pdf_bytes = await response.body()

                    log.info(f"Race {race_num}: HTTP {status}, type={content_type}, size={len(pdf_bytes)}")

                    if pdf_bytes[:5].startswith(b"%PDF") and len(pdf_bytes) > 1000:
                        parsed = self.pdf_parser.parse_pdf_bytes(pdf_bytes, track_code, race_date_str)
                        if parsed:
                            entries.extend(parsed)
                            self.checkpoint.stats["pdfs"] += 1
                            consecutive_failures = 0
                            log.info(f"Race {race_num}: Parsed {len(parsed)} entries from PDF")
                            # Verbose: show horses parsed from this race
                            if log.isEnabledFor(logging.DEBUG):
                                horses = [e.get("horse_name", "?") for e in parsed]
                                log.debug(f"  Race {race_num} horses: {', '.join(horses)}")
                        else:
                            log.info(f"Race {race_num}: PDF valid but parsing returned 0 entries")
                            consecutive_failures += 1
                    elif status == 404 or len(pdf_bytes) < 500:
                        log.info(f"Race {race_num}: No chart available (404 or tiny response)")
                        consecutive_failures += 1
                    else:
                        log.info(f"Race {race_num}: Response not a PDF (starts with {pdf_bytes[:30]!r})")
                        consecutive_failures += 1

                    await self.human.random_delay(0.8, 1.5)

                    if consecutive_failures >= 2:
                        log.info(f"2 consecutive failures after race {race_num}, stopping")
                        break
                except Exception as e:
                    log.info(f"Race {race_num} chart download failed: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= 2:
                        break

            if entries:
                log.info(f"Chart embed PDFs found {len(entries)} entries from {self.checkpoint.stats['pdfs']} PDFs")
            else:
                # Navigate back for further strategies
                try:
                    await page.goto(saved_url, wait_until="domcontentloaded", timeout=15000)
                except Exception:
                    pass

        # ── Strategy 1c: Try direct PDF URLs by race number ──
        if not entries and pdfplumber:
            date_mmddyy = date.strftime("%m%d%y")
            date_ymd = date.strftime("%Y%m%d")
            track_lower = track_code.lower()
            year = date.strftime("%Y")

            test_urls = [
                self.PDF_CHART_URL.format(track=track_code, date=date_mmddyy, race_num=1),
                self.PDF_CHART_HISTORICAL_URL.format(
                    year=year, track_lower=track_lower, date_ymd=date_ymd, race_num=1
                ),
            ]

            working_pattern = None
            for test_url in test_urls:
                log.info(f"Testing PDF URL: {test_url}")
                test_entries = await self._download_and_parse_pdf(test_url, track_code, race_date_str, page)
                if test_entries:
                    entries.extend(test_entries)
                    self.checkpoint.stats["pdfs"] += 1
                    working_pattern = test_url.replace("1.pdf", "{race_num}.pdf").replace("-1-d.", "-{race_num}-d.")
                    log.info(f"PDF pattern works! Using: {working_pattern}")
                    break

            if working_pattern:
                for race_num in range(2, 16):
                    pdf_url = working_pattern.format(race_num=race_num)
                    pdf_entries = await self._download_and_parse_pdf(pdf_url, track_code, race_date_str, page)
                    if pdf_entries:
                        entries.extend(pdf_entries)
                        self.checkpoint.stats["pdfs"] += 1
                        await self.human.random_delay(1.0, 3.0)
                    else:
                        break
                log.info(f"Direct PDF parsing found {len(entries)} entries from {self.checkpoint.stats['pdfs']} PDFs")
            else:
                log.debug("No direct PDF URLs accessible, falling back to HTML")

        # ── Strategy 2: Parse HTML tables on current page ──
        if not entries:
            entries = await self._parse_html_tables(track_code, race_date_str, page)
            if entries:
                log.info(f"HTML table parsing found {len(entries)} entries")
            else:
                log.debug("HTML table parsing found no entries on current page")

        # ── Strategy 3: Navigate to all-races summary page if not already on it ──
        if not entries:
            date_mmddyy = date.strftime("%m%d%y")

            # If we're on a RaceCardIndex page, click "View All Races" or navigate to summary
            if "RaceCardIndex" in current_url:
                # Try clicking "View All Races" link
                try:
                    view_all_url = await page.evaluate(r"""
                        () => {
                            const link = document.querySelector('a[href*="EQB.html"]:not([href*="RaceCardIndex"])');
                            return link ? link.href : null;
                        }
                    """)
                    if view_all_url:
                        log.info(f"Clicking 'View All Races': {view_all_url}")
                        await page.goto(view_all_url, wait_until="domcontentloaded", timeout=15000)
                        await self.human.random_delay(1.2, 2.5)
                        entries = await self._parse_html_tables(track_code, race_date_str, page)
                except Exception as e:
                    log.debug(f"View All Races click failed: {e}")

            # If still no entries, go directly to summary URL
            if not entries:
                summary_url = self.SUMMARY_URL.format(track=track_code, date=date_mmddyy)
                if summary_url not in current_url:
                    log.info(f"Trying Equibase summary page: {summary_url}")
                    try:
                        await page.goto(summary_url, wait_until="domcontentloaded", timeout=15000)
                        await self.human.random_delay(1.2, 2.5)
                        log.info(f"Summary page landed on: {page.url}")
                        entries = await self._parse_html_tables(track_code, race_date_str, page)
                        if entries:
                            log.info(f"Summary page parsing found {len(entries)} entries")
                    except Exception as e:
                        log.debug(f"Summary page navigation failed: {e}")

        # ── Strategy 4: Click into individual race links ──
        if not entries:
            race_links = await page.evaluate(r"""
                () => {
                    const links = [];
                    document.querySelectorAll('a[href]').forEach(a => {
                        const text = (a.innerText || '').trim();
                        const href = a.href || '';
                        if (/race\s*\d+/i.test(text) || /R\d+/i.test(text) ||
                            /race\s*#?\s*\d+/i.test(text)) {
                            links.push({href, text});
                        }
                    });
                    return links;
                }
            """)

            if race_links:
                log.info(f"Found {len(race_links)} race links to click into")
            for link_info in race_links[:15]:  # max 15 races per card
                try:
                    await page.goto(link_info["href"],
                                         wait_until="domcontentloaded", timeout=15000)
                    await self.human.random_delay(1.2, 2.5)
                    await self.human.random_scroll(page)

                    race_entries = await self._parse_html_tables(track_code, race_date_str, page)
                    entries.extend(race_entries)

                    await page.go_back(wait_until="domcontentloaded", timeout=10000)
                    await self.human.random_delay(1.5, 3.0)
                except Exception as e:
                    log.debug(f"Race link click failed: {e}")

        if not entries:
            # Dump page diagnostics for debugging
            await self._dump_page_diagnostics(page)

        # Verbose: summarize all entries found
        if entries and log.isEnabledFor(logging.DEBUG):
            races = {}
            for e in entries:
                rn = e.get("race_number", "?")
                races.setdefault(rn, []).append(e.get("horse_name", "?"))
            for rn in sorted(races.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
                horses = races[rn]
                log.debug(f"  Race {rn}: {len(horses)} entries -> "
                          f"{', '.join(horses[:5])}{'...' if len(horses) > 5 else ''}")

        return entries

    async def _dump_page_diagnostics(self, page: Page):
        """Log diagnostic info about current page to help debug parsing failures."""
        try:
            url = page.url
            title = await page.title()
            table_info = await page.evaluate("""
                () => {
                    const tables = document.querySelectorAll('table');
                    return Array.from(tables).map((t, i) => ({
                        index: i,
                        rows: t.rows.length,
                        firstRowCells: t.rows[0] ? Array.from(t.rows[0].cells).map(c => c.innerText.trim().substring(0, 30)) : [],
                        className: t.className,
                        id: t.id,
                    }));
                }
            """)
            log.warning(f"Page diagnostics — URL: {url}")
            log.warning(f"  Title: {title}")
            log.warning(f"  Tables found: {len(table_info)}")
            for t in table_info[:5]:
                log.warning(f"  Table #{t['index']} (class='{t['className']}', id='{t['id']}'): "
                           f"{t['rows']} rows, headers: {t['firstRowCells']}")
        except Exception as e:
            log.debug(f"Page diagnostics failed: {e}")

    async def _download_and_parse_pdf(self, url: str, track_code: str, race_date: str, page: Page) -> List[Dict]:
        """Download a PDF and parse it. Validates response is actually a PDF."""
        try:
            # Use page context to download (maintains cookies/session)
            response = await page.request.get(url, timeout=30000)
            if response.status != 200:
                log.debug(f"PDF download HTTP {response.status}: {url}")
                return []

            # Check Content-Type header
            content_type = response.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                log.debug(f"PDF link returned non-PDF content-type '{content_type}': {url}")
                return []

            pdf_bytes = await response.body()
            if len(pdf_bytes) < 1000:  # too small to be a real PDF
                log.debug(f"PDF too small ({len(pdf_bytes)} bytes): {url}")
                return []

            # Validate PDF magic bytes (%PDF)
            if not pdf_bytes[:5].startswith(b"%PDF"):
                log.debug(f"PDF link returned non-PDF content (no %%PDF header, got {pdf_bytes[:20]!r}): {url}")
                return []

            return self.pdf_parser.parse_pdf_bytes(pdf_bytes, track_code, race_date)
        except Exception as e:
            log.debug(f"PDF download failed: {e}")
            return []

    async def _parse_html_tables(self, track_code: str, race_date: str, page: Page) -> List[Dict]:
        """Parse race results from HTML tables on the current page."""
        # Grab tables WITH their preceding context (race headers above tables)
        raw_tables = await page.evaluate("""
            () => {
                const results = [];
                document.querySelectorAll('table').forEach(table => {
                    const rows = [];
                    table.querySelectorAll('tr').forEach(tr => {
                        const cells = [];
                        tr.querySelectorAll('td, th').forEach(cell => {
                            cells.push(cell.innerText.trim());
                        });
                        if (cells.length > 0) rows.push(cells);
                    });
                    if (rows.length > 1) results.push(rows);
                });
                return results;
            }
        """)

        # Also grab surrounding text for race metadata
        page_text = await page.inner_text("body")

        entries = []
        for table in (raw_tables or []):
            parsed = self._parse_generic_table(table, track_code, race_date, page_text)
            entries.extend(parsed)

        # If generic parsing found nothing, try Equibase-specific summary parsing
        if not entries:
            entries = await self._parse_equibase_summary(track_code, race_date, page_text, page)

        return entries

    async def _parse_equibase_summary(self, track_code: str, race_date: str, page_text: str, page: Page) -> List[Dict]:
        """
        Parse Equibase summary results pages.

        Actual Equibase page structure (confirmed by live inspection):
        +------------------------------------------------------------+
        | RACE 1                                                      |
        | Off at: 1:05  Race type: Maiden Special Weight              |
        | Age Restriction: Two Year Old                               |
        | Purse: $90,000                                              |
        | Distance: Four And One Half Furlongs On The Dirt            |
        | Track Condition: Sloppy                                     |
        | Winning Time: 52.74                                         |
        +------+--------------+--------------+------+-------+--------+
        | Pgm  | Horse        | Jockey       | Win  | Place | Show   |
        +------+--------------+--------------+------+-------+--------+
        | 8    | Suspicions   | Pietro Moran | 4.60 | 2.82  | 2.54   |
        | 5    | Bourbon Town | Luis Saez    |      | 2.76  | 2.26   |
        | 7    | Tigrado      | Evin Roman   |      |       | 4.92   |
        +------+--------------+--------------+------+-------+--------+
        Also ran: 3 - Super Saiyajin , 2 - Joe Joe Dude , 9 - Cross Power
        Winning Breeder: ... Winning Owner: ... Winning Trainer: ...

        Key facts:
        - Tables have headers: Pgm | Horse | Jockey | Win | Place | Show
        - Only top 3 finishers are in the table
        - Remaining finishers are in "Also ran:" text
        - Race metadata is in text blocks between tables
        """
        entries = []

        # Use JavaScript to extract structured race data matching actual Equibase layout
        race_data = await page.evaluate(r"""
            () => {
                const races = [];

                // Split page text into race blocks using "RACE \d+" pattern
                const bodyText = document.body.innerText;
                const racePattern = /RACE\s+(\d+)/g;
                const racePositions = [];
                let match;
                while ((match = racePattern.exec(bodyText)) !== null) {
                    racePositions.push({num: parseInt(match[1]), pos: match.index});
                }

                // For each race block, extract the text content
                for (let i = 0; i < racePositions.length; i++) {
                    const start = racePositions[i].pos;
                    const end = i + 1 < racePositions.length ? racePositions[i + 1].pos : bodyText.length;
                    const blockText = bodyText.substring(start, Math.min(end, start + 2000));
                    races.push({num: racePositions[i].num, text: blockText});
                }

                // Also get all results tables (Pgm | Horse | Jockey | Win | Place | Show)
                const resultTables = [];
                document.querySelectorAll('table').forEach(table => {
                    const headerRow = table.querySelector('tr');
                    if (!headerRow) return;
                    const headerText = headerRow.innerText;
                    if (headerText.includes('Pgm') && headerText.includes('Horse') && headerText.includes('Jockey')) {
                        const rows = [];
                        table.querySelectorAll('tr').forEach(tr => {
                            const cells = Array.from(tr.querySelectorAll('td,th')).map(c => c.innerText.trim());
                            rows.push(cells);
                        });
                        resultTables.push(rows);
                    }
                });

                return {races, resultTables};
            }
        """)

        if not race_data:
            return []

        race_blocks = race_data.get("races", [])
        result_tables = race_data.get("resultTables", [])

        log.info(f"Found {len(race_blocks)} race blocks and {len(result_tables)} result tables")

        # Match each race block with its corresponding result table
        for idx, race_block in enumerate(race_blocks):
            race_num = race_block["num"]
            block_text = race_block["text"]

            # Extract race metadata from the text block
            race_meta = {
                "race_number": race_num,
                "track_code": track_code,
                "track_name": TRACKS.get(track_code, track_code),
                "race_date": race_date,
            }

            # Parse metadata from text
            rt_m = re.search(r"Race type:\s*(.+?)(?:\n|$)", block_text)
            if rt_m:
                race_meta["race_type"] = rt_m.group(1).strip()

            purse_m = re.search(r"Purse:\s*\$?([\d,]+)", block_text)
            if purse_m:
                race_meta["purse"] = purse_m.group(1).replace(",", "")

            dist_m = re.search(r"Distance:\s*(.+?)(?:\n|$)", block_text)
            if dist_m:
                race_meta["distance"] = dist_m.group(1).strip()

            cond_m = re.search(r"Track Condition:\s*(\w+)", block_text)
            if cond_m:
                race_meta["track_condition"] = cond_m.group(1).strip()

            surf_m = re.search(r"On The\s+(Dirt|Turf|Synthetic|All Weather)", block_text, re.I)
            if surf_m:
                race_meta["surface"] = surf_m.group(1)[0].upper()
            elif "Turf" in block_text:
                race_meta["surface"] = "T"
            else:
                race_meta["surface"] = "D"

            time_m = re.search(r"Winning Time:\s*([\d:.]+)", block_text)
            if time_m:
                race_meta["final_time_secs"] = self.pdf_parser._parse_time(time_m.group(1))

            # Verbose: log race metadata
            log.debug(f"  Race {race_num}: type={race_meta.get('race_type','?')}, "
                      f"purse=${race_meta.get('purse','?')}, "
                      f"surface={race_meta.get('surface','?')}, "
                      f"condition={race_meta.get('track_condition','?')}")

            # Parse the "Also ran:" section for non-placed horses
            also_ran = []
            also_m = re.search(r"Also ran:\s*(.+?)(?:\n|Scratched|Wager|$)", block_text, re.S)
            if also_m:
                # Format: "3 - Super Saiyajin , 2 - Joe Joe Dude , 9 - Cross Power"
                for horse_m in re.finditer(r"(\d+)\s*-\s*([^,]+)", also_m.group(1)):
                    also_ran.append({
                        "program_num": horse_m.group(1).strip(),
                        "horse_name": horse_m.group(2).strip(),
                    })

            # Parse winning connections
            owner_m = re.search(r"Winning Owner:\s*(.+?)(?:\n|$)", block_text)
            trainer_m = re.search(r"Winning Trainer:\s*(.+?)(?:\n|$)", block_text)

            # Get the corresponding result table (Pgm | Horse | Jockey | Win | Place | Show)
            if idx < len(result_tables):
                table = result_tables[idx]
                finish_pos = 1
                for row in table[1:]:  # skip header
                    if len(row) < 3:
                        continue
                    pgm = row[0].strip()
                    if not pgm or not re.match(r'\d+', pgm):
                        continue

                    horse = row[1].strip() if len(row) > 1 else ""
                    jockey = row[2].strip() if len(row) > 2 else ""
                    win = row[3].strip() if len(row) > 3 else ""
                    place = row[4].strip() if len(row) > 4 else ""
                    show = row[5].strip() if len(row) > 5 else ""

                    if not horse or horse.lower() == "horse":
                        continue

                    entry = {**race_meta}
                    entry["program_num"] = pgm
                    entry["horse_name"] = horse
                    entry["jockey"] = jockey
                    entry["finish"] = str(finish_pos)
                    # Use Win payoff as a proxy for odds (Win / 2 ~ odds)
                    if win:
                        try:
                            entry["dollar_odds"] = str(round(float(win) / 2, 2))
                        except ValueError:
                            entry["dollar_odds"] = ""
                    if owner_m and finish_pos == 1:
                        entry["owner"] = owner_m.group(1).strip()
                    if trainer_m and finish_pos == 1:
                        entry["trainer"] = trainer_m.group(1).strip()

                    for c in OUTPUT_COLUMNS:
                        if c not in entry:
                            entry[c] = ""
                    entries.append(entry)
                    finish_pos += 1

                # Add "Also ran" horses
                for also_horse in also_ran:
                    entry = {**race_meta}
                    entry["program_num"] = also_horse["program_num"]
                    entry["horse_name"] = also_horse["horse_name"]
                    entry["finish"] = str(finish_pos)
                    for c in OUTPUT_COLUMNS:
                        if c not in entry:
                            entry[c] = ""
                    entries.append(entry)
                    finish_pos += 1

        if entries:
            log.info(f"Equibase summary parsing found {len(entries)} entries across {len(race_blocks)} races")

        return entries

    def _parse_generic_table(self, table, track_code, race_date, page_text) -> List[Dict]:
        """Auto-detect columns and parse a results table."""
        if not table or len(table) < 2:
            return []

        headers = [str(h).lower().strip() for h in table[0]]
        col = {}
        for i, h in enumerate(headers):
            if any(k in h for k in ["horse", "name", "starter"]) and "horse" not in col:
                col["horse"] = i
            elif any(k in h for k in ["fin", "pos"]) and "finish" not in col:
                col["finish"] = i
            elif any(k in h for k in ["pgm", "no.", "#"]) and "pgm" not in col:
                col["pgm"] = i
            elif "pp" == h or "post" in h:
                col["pp"] = i
            elif "jock" in h:
                col["jockey"] = i
            elif "train" in h:
                col["trainer"] = i
            elif any(k in h for k in ["wgt", "wt", "weight"]):
                col["weight"] = i
            elif "odds" in h or "ml" in h:
                col["odds"] = i
            elif "owner" in h:
                col["owner"] = i
            elif any(k in h for k in ["comment", "remark"]):
                col["comment"] = i
            elif any(k in h for k in ["margin", "btn", "behind", "lengths"]):
                col["margin"] = i

        if "horse" not in col:
            return []

        # Extract race-level metadata from page text
        race_meta = self._extract_race_metadata(page_text)

        entries = []
        for row in table[1:]:
            if len(row) <= col.get("horse", 0):
                continue
            def get(k):
                idx = col.get(k)
                return str(row[idx]).strip() if idx is not None and idx < len(row) and row[idx] else ""

            horse = get("horse")
            if not horse or len(horse) < 2 or horse.lower() in ("horse", "name"):
                continue

            entry = {
                "horse_name": horse,
                "finish": get("finish"),
                "program_num": get("pgm"),
                "post_position": get("pp"),
                "jockey": get("jockey"),
                "trainer": get("trainer"),
                "owner": get("owner"),
                "weight": get("weight"),
                "dollar_odds": get("odds").replace("$", ""),
                "comment": get("comment"),
                "margin_finish": get("margin"),
                "track_code": track_code,
                "track_name": TRACKS.get(track_code, track_code),
                "race_date": race_date,
                **race_meta,
            }
            for c in OUTPUT_COLUMNS:
                if c not in entry:
                    entry[c] = ""
            entries.append(entry)

        return entries

    def _extract_race_metadata(self, text: str) -> Dict:
        """Pull race-level info (distance, surface, purse, etc.) from page text."""
        meta = {}
        dist_m = re.search(r"(\d+\.?\d*)\s*(furlong|mile|yard)", text, re.I)
        if dist_m:
            meta["distance"] = dist_m.group(0)
        surf_m = re.search(r"\b(Dirt|Turf|Synthetic)\b", text, re.I)
        if surf_m:
            meta["surface"] = surf_m.group(1)[0].upper()
        purse_m = re.search(r"\$[\d,]+", text)
        if purse_m:
            meta["purse"] = purse_m.group(0).replace("$", "").replace(",", "")
        cond_m = re.search(r"\b(Fast|Good|Muddy|Sloppy|Firm|Yielding|Soft)\b", text, re.I)
        if cond_m:
            meta["track_condition"] = cond_m.group(1)
        type_m = re.search(
            r"\b(Maiden Special Weight|Maiden Claiming|Claiming|Allowance|Stakes|Handicap)\b",
            text, re.I)
        if type_m:
            meta["race_type"] = type_m.group(1)
        return meta

    # ── Progress reporter ─────────────────────────────────────

    async def _progress_reporter(self, remaining_count: int, total: int,
                                 skipped: int, interval: float = 60.0):
        """Background coroutine that logs progress and ETA periodically."""
        while True:
            await asyncio.sleep(interval)
            completed = self._completed_count
            if completed == 0:
                continue
            if completed >= remaining_count:
                break

            elapsed = time.perf_counter() - self._run_start_time
            rate = completed / elapsed  # targets per second
            remaining = remaining_count - completed
            eta_seconds = remaining / rate if rate > 0 else 0
            pct = (skipped + completed) / total * 100 if total > 0 else 0

            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds / 60:.0f}m"
            else:
                eta_str = f"{eta_seconds / 3600:.1f}h"

            s = self.checkpoint.stats
            log.info(
                f"[PROGRESS] {pct:.1f}% ({completed}/{remaining_count}) | "
                f"{s['entries']:,} entries | "
                f"{elapsed / 60:.1f}m elapsed | "
                f"ETA: {eta_str} | "
                f"{rate * 3600:.0f} pages/h"
            )

    # ── Per-target worker ────────────────────────────────────

    async def _process_race_day(self, page: Page, track: str, date: datetime,
                                total: int, skipped: int):
        """Process a single (track, date) target using the given page."""
        date_str = date.strftime("%Y-%m-%d")
        t0 = time.perf_counter()

        log.debug(f"[START] {track} {date_str}")

        # Occasional distraction browsing (~5% of the time)
        if random.random() < 0.05:
            log.debug(f"  Distraction browse before {track} {date_str}")
            await self.human.browse_distraction(page)

        # Navigate to race day
        has_data = await self._navigate_to_race_day(track, date, page)

        if has_data:
            # Human-like: scroll around, look at the page
            await self.human.random_scroll(page)
            await self.human.random_mouse_move(page)

            # Extract data
            entries = await self._extract_from_page(track, date, page)

            # Optionally enrich with career stats from horse profile pages
            if entries and self._career_stats_enabled:
                await self._enrich_with_career_stats(entries, page)

            elapsed = time.perf_counter() - t0

            if entries:
                self._write_entries(entries)
                self.checkpoint.mark(track, date, len(entries))
                self._completed_count += 1
                pct = (skipped + self._completed_count) / total * 100

                # Count unique races
                race_nums = {e.get("race_number") for e in entries}
                log.info(
                    f"[{pct:.1f}%] {track} {date_str}: {len(entries)} entries, "
                    f"{len(race_nums)} races ({elapsed:.1f}s) "
                    f"(total: {self.checkpoint.stats['entries']:,})"
                )
            else:
                self.checkpoint.mark(track, date, 0)
                self._completed_count += 1
                log.debug(f"[DONE] {track} {date_str}: page found but no parseable data ({elapsed:.1f}s)")
        else:
            elapsed = time.perf_counter() - t0
            self.checkpoint.mark(track, date, 0)
            self._completed_count += 1
            log.debug(f"[SKIP] {track} {date_str}: no racing ({elapsed:.1f}s)")

        # Variable delay between pages
        await self.human.random_delay(1.5, 3.5)

    # ── Main run loop ────────────────────────────────────────

    async def run(self, targets: List[Tuple[str, datetime]]):
        """Main scraping loop. Runs up to self._concurrency workers concurrently."""
        remaining = [(t, d) for t, d in targets if not self.checkpoint.is_done(t, d)]
        total = len(targets)
        skipped = total - len(remaining)

        log.info(f"Targets: {total:,} total, {skipped:,} done, {len(remaining):,} remaining")
        log.info(f"Concurrency: {self._concurrency} browser context(s)")

        self._run_start_time = time.perf_counter()
        self._open_csv()
        try:
            await self._start_browser()
            await self._setup_session()

            semaphore = asyncio.Semaphore(self._concurrency)

            # Pool of reusable worker IDs (W1..WN) so logs show which slot is active
            _id_pool = asyncio.Queue()
            for _i in range(1, self._concurrency + 1):
                _id_pool.put_nowait(_i)

            async def worker(track, date):
                async with semaphore:
                    wid = await _id_pool.get()
                    if self._concurrency > 1:
                        _worker_id.set(f"W{wid}")
                    context, page = await self._create_context()
                    try:
                        await self._process_race_day(page, track, date, total, skipped)
                    except Exception as e:
                        log.error(f"Worker error for {track} {date.strftime('%Y-%m-%d')}: {e}",
                                  exc_info=True)
                        self.checkpoint.stats["errors"] += 1
                    finally:
                        await page.close()
                        if not self._is_cdp:
                            await context.close()
                        _id_pool.put_nowait(wid)
                        _worker_id.set('')

            # Start background progress reporter (every 60s)
            progress_task = asyncio.create_task(
                self._progress_reporter(len(remaining), total, skipped, interval=60.0)
            )

            # Launch all workers; semaphore limits actual concurrency
            results = await asyncio.gather(
                *[worker(t, d) for t, d in remaining],
                return_exceptions=True,
            )

            # Stop progress reporter
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

            # Log any unhandled exceptions from gather
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    track, date = remaining[i]
                    log.error(f"Unhandled exception for {track} {date.strftime('%Y-%m-%d')}: {result}")

        except KeyboardInterrupt:
            log.info("Interrupted — saving checkpoint")
        except Exception as e:
            log.error(f"Fatal: {e}", exc_info=True)
        finally:
            self.checkpoint.save()
            self._close_csv()
            await self._stop_browser()

            elapsed = time.perf_counter() - self._run_start_time
            s = self.checkpoint.stats
            throughput = s['pages'] / elapsed * 3600 if elapsed > 0 else 0
            log.info(f"\n{'='*50}")
            log.info(f"DONE | Pages: {s['pages']:,} | Entries: {s['entries']:,} | "
                     f"PDFs: {s['pdfs']} | Errors: {s['errors']}")
            log.info(f"Time: {elapsed:.0f}s ({elapsed/3600:.1f}h) | "
                     f"Throughput: {throughput:.0f} pages/h")
            if self._career_stats_enabled:
                total_lookups = self._career_hit + self._career_miss
                hit_rate = (self._career_hit / total_lookups * 100) if total_lookups > 0 else 0
                log.info(f"Career stats | {len(self._career_cache)} unique horses | "
                         f"cache hit rate: {hit_rate:.0f}% ({self._career_hit}/{total_lookups})")
            log.info(f"Output: {self.output_file}")
            log.info(f"{'='*50}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def _generate_targets(start_date, end_date, track_codes):
    """Generate (track, date) targets filtered by race-day calendar."""
    targets = []
    d = start_date
    while d <= end_date:
        for t in track_codes:
            if is_race_day(t, d):
                targets.append((t, d))
        d += timedelta(days=1)
    return targets


def main():
    p = argparse.ArgumentParser(
        description="Scrape 10 years of horse racing data via Google -> Equibase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Debug one day (visible browser, verbose)
  python3 -m scraper --tracks KEE --start 2026-04-03 --end 2026-04-03 --visible -v

  # One month of Keeneland
  python3 -m scraper --tracks KEE --start 2023-04-01 --end 2023-04-30

  # Concurrent scraping with 4 browser contexts
  python3 -m scraper --tracks KEE --start 2023-04-01 --end 2023-04-30 --concurrency 4

  # Top 5 tracks, full 10 years
  python3 -m scraper --tracks KEE,CD,GP,SA,SAR --start 2016-01-01 --end 2026-04-04

  # All tracks, full decade (run in background)
  nohup python3 -m scraper --start 2016-01-01 --end 2026-04-04 &

  # Resume after interruption
  python3 -m scraper --resume
        """,
    )
    p.add_argument("--start", help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD")
    p.add_argument("--tracks", help="Comma-separated track codes (default: all)")
    p.add_argument("--output", default="historical_races.csv")
    p.add_argument("--visible", action="store_true", help="Show browser (only for fallback mode)")
    p.add_argument("--no-google", action="store_true",
                   help="Skip Google search, go direct to Equibase URLs (avoids CAPTCHA)")
    p.add_argument("--cdp-port", type=int, default=9222,
                   help="Chrome DevTools port (default: 9222)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--list-tracks", action="store_true")
    p.add_argument("--checkpoint", default="scraper_v3_checkpoint.json",
                   help="Checkpoint file path (use a separate file for isolated runs)")
    p.add_argument("--career-stats", action="store_true",
                   help="After parsing each race day, enrich entries with career stats "
                        "(num_past_starts/wins/seconds/thirds) from horse profile pages. "
                        "Slower — adds ~1s per unique horse — but populates those fields.")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Number of concurrent browser contexts (default: 1). "
                        "Use 4 for ~3-4x speedup. Beyond 8 risks rate limiting.")
    p.add_argument("--targets-file",
                   help="JSON file with [[track, date], ...] targets (used by launcher)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show detailed output: per-horse career lookups, race metadata, "
                        "entry rosters, timing per race day, cache hit rates, "
                        "navigation strategy used, and worker lifecycle events.")

    args = p.parse_args()

    # Set log level based on --verbose
    if args.verbose:
        log.setLevel(logging.DEBUG)
        # Also set the root scraper logger so all modules pick it up
        logging.getLogger("scraper").setLevel(logging.DEBUG)
        log.debug("Verbose mode enabled")
    else:
        log.setLevel(logging.INFO)

    if args.list_tracks:
        for code, name in sorted(TRACKS.items()):
            print(f"  {code:<6} {name}")
        return

    if args.targets_file:
        # Load targets from JSON file (used by multi-process launcher)
        with open(args.targets_file) as f:
            raw = json.load(f)
        targets = [(t, datetime.strptime(d, "%Y-%m-%d")) for t, d in raw]
    elif args.resume and not args.start:
        cp = Checkpoint(args.checkpoint)
        if not cp.done:
            p.error("No checkpoint found. Use --start and --end.")
        dates = [datetime.strptime(k.split("_")[1], "%Y%m%d") for k in cp.done]
        tracks_seen = list({k.split("_")[0] for k in cp.done})
        start_date, end_date = min(dates), max(dates) + timedelta(days=30)
        targets = _generate_targets(start_date, end_date, tracks_seen)
    else:
        if not args.start or not args.end:
            p.error("Use --start and --end (or --resume)")
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        track_codes = args.tracks.split(",") if args.tracks else list(TRACKS.keys())
        targets = _generate_targets(start_date, end_date, track_codes)

    # Shuffle to avoid hammering one track consecutively
    random.shuffle(targets)

    est_hours = len(targets) * 4.0 / 3600 / max(1, args.concurrency)
    log.info(
        f"Targets: {len(targets):,} | Concurrency: {args.concurrency} "
        f"| Est: {est_hours:.1f}h"
    )

    scraper = RaceScraper(
        output_file=args.output,
        headless=not args.visible,
        use_google=not args.no_google,
        cdp_port=args.cdp_port,
        checkpoint_path=args.checkpoint,
        career_stats=args.career_stats,
        concurrency=args.concurrency,
    )
    asyncio.run(scraper.run(targets))


if __name__ == "__main__":
    main()
