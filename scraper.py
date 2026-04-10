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
  python3 scraper_v3.py --tracks KEE --start 2026-04-03 --end 2026-04-03 --visible

  # Keeneland spring 2023
  python3 scraper_v3.py --tracks KEE --start 2023-04-07 --end 2023-04-28

  # Resume interrupted scrape
  python3 scraper_v3.py --resume

  # Full 10-year scrape
  python3 scraper_v3.py --start 2016-01-01 --end 2026-04-04
"""

import asyncio
import csv
import json
import os
import re
import random
import sys
import math
import time
import logging
import argparse
import tempfile
import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Set

try:
    from playwright.async_api import async_playwright, Page
except ImportError:
    sys.exit("Run: pip install playwright && playwright install chromium")

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    print("[WARN] pdfplumber not installed — PDF parsing disabled. pip install pdfplumber")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("scraper_v3.log")],
)
log = logging.getLogger("scraper")


# ═══════════════════════════════════════════════════════════════
# OUTPUT SCHEMA
# ═══════════════════════════════════════════════════════════════

OUTPUT_COLUMNS = [
    "race_number", "race_type", "purse", "distance", "distance_unit",
    "course", "surface", "track_condition", "weather", "post_time", "win_time",
    "horse_name", "breed", "weight", "age", "sex", "medication",
    "program_num", "post_position", "finish", "comment",
    "jockey", "trainer", "owner",
    "last_race_track", "last_race_date", "last_race_number", "last_race_finish",
    "track_code", "track_name", "race_date", "dollar_odds",
    "num_past_starts", "num_past_wins", "num_past_seconds", "num_past_thirds",
]

TRACKS = {
    "KEE": "Keeneland", "CD": "Churchill Downs", "GP": "Gulfstream Park",
    "SA": "Santa Anita Park", "AQU": "Aqueduct", "BEL": "Belmont Park",
    "SAR": "Saratoga", "DMR": "Del Mar", "OP": "Oaklawn Park",
    "FG": "Fair Grounds", "TAM": "Tampa Bay Downs", "LRL": "Laurel Park",
    "PRX": "Parx Racing", "PEN": "Penn National", "CT": "Charles Town",
    "IND": "Horseshoe Indianapolis", "WO": "Woodbine", "TP": "Turfway Park",
    "RP": "Remington Park", "LS": "Lone Star Park", "EVD": "Evangeline Downs",
    "DED": "Delta Downs", "LAD": "Louisiana Downs", "GG": "Golden Gate Fields",
    "MVR": "Mahoning Valley", "MNR": "Mountaineer", "PRM": "Prairie Meadows",
    "FL": "Finger Lakes", "TDN": "Thistledown", "HOU": "Sam Houston",
    "TUP": "Turf Paradise", "BTP": "Belterra Park", "SUN": "Sunland Park",
    "PIM": "Pimlico", "DEL": "Delaware Park", "MTH": "Monmouth Park",
    "CBY": "Canterbury Park", "KD": "Kentucky Downs",
}


# ═══════════════════════════════════════════════════════════════
# TRACK RACE-DAY CALENDAR — loaded from track_race_dates.json
# ═══════════════════════════════════════════════════════════════
# Populated at module load from a cache produced by fetch_all_track_dates.py,
# which pulls confirmed race dates directly from Equibase's
# eqbRaceChartCalendar.cfm endpoint. Each entry is a set of "YYYY-MM-DD"
# strings. Tracks missing from the cache fall through to "always try".

TRACK_RACE_DATES: dict[str, set[str]] = {}


def _load_track_race_dates():
    path = Path(__file__).with_name("track_race_dates.json")
    if not path.exists():
        return
    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        print(f"[warn] could not read {path.name}: {e}", file=sys.stderr)
        return
    ppp = raw.get("per_track_per_year") or raw.get("tracks") or {}
    for code, yr_map in ppp.items():
        if not isinstance(yr_map, dict):
            continue
        dates: set[str] = set()
        for v in yr_map.values():
            if isinstance(v, list):
                dates.update(v)
        if dates:
            TRACK_RACE_DATES[code] = dates


_load_track_race_dates()


def is_race_day(track_code: str, date) -> bool:
    """Return True if the track actually ran a card on `date`.

    Backed by confirmed Equibase calendar data. If a track has no entry in
    the cache (cache missing or older than the target year), we fall back
    to True so the scraper still attempts the date rather than silently
    skipping real races.
    """
    dates = TRACK_RACE_DATES.get(track_code)
    if not dates:
        return True
    return date.strftime("%Y-%m-%d") in dates

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

VIEWPORTS = [
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1920, "height": 1080},
    {"width": 1280, "height": 800},
]


# ═══════════════════════════════════════════════════════════════
# HUMAN-LIKE BEHAVIOR ENGINE
# ═══════════════════════════════════════════════════════════════

class HumanBehavior:
    """Simulates realistic human browsing patterns."""

    @staticmethod
    async def random_delay(min_s=1.5, max_s=3.0):
        """Wait a random amount with occasional longer pauses."""
        # 10% chance of a longer "reading" pause
        if random.random() < 0.10:
            delay = random.uniform(8.0, 15.0)
        else:
            delay = random.uniform(min_s, max_s)
        await asyncio.sleep(delay)

    @staticmethod
    async def type_like_human(page: Page, selector: str, text: str):
        """Type text with variable speed like a real person."""
        await page.click(selector)
        for char in text:
            await page.keyboard.type(char, delay=random.randint(50, 200))
            # Occasional brief pause mid-word
            if random.random() < 0.05:
                await asyncio.sleep(random.uniform(0.3, 0.8))

    @staticmethod
    async def random_scroll(page: Page):
        """Scroll the page randomly like a person reading."""
        scroll_amount = random.randint(200, 600)
        direction = random.choice([1, 1, 1, -1])  # mostly scroll down
        await page.evaluate(f"window.scrollBy(0, {scroll_amount * direction})")
        await asyncio.sleep(random.uniform(0.5, 1.5))

    @staticmethod
    async def random_mouse_move(page: Page):
        """Move mouse to a random position on the page."""
        x = random.randint(100, 1200)
        y = random.randint(100, 700)
        await page.mouse.move(x, y)
        await asyncio.sleep(random.uniform(0.1, 0.3))

    @staticmethod
    async def hover_before_click(page: Page, selector: str):
        """Hover over element briefly before clicking (human behavior)."""
        try:
            elem = page.locator(selector).first
            await elem.hover()
            await asyncio.sleep(random.uniform(0.3, 0.8))
            await elem.click()
        except Exception:
            await page.click(selector)

    @staticmethod
    async def browse_distraction(page: Page):
        """
        Occasionally visit an unrelated page (mimics tabbed browsing).
        Called randomly ~5% of the time between scrape targets.
        """
        distractions = [
            "https://www.google.com",
            "https://www.weather.com",
            "https://news.ycombinator.com",
        ]
        url = random.choice(distractions)
        log.debug(f"Distraction browse: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        await asyncio.sleep(random.uniform(2, 5))


# ═══════════════════════════════════════════════════════════════
# PDF CHART PARSER
# ═══════════════════════════════════════════════════════════════

class ChartPDFParser:
    """
    Parses Equibase full chart PDFs into structured data.
    
    Equibase chart PDF layout:
    ┌──────────────────────────────────────────────────┐
    │ TRACK NAME - DATE - RACE N                       │
    │ Distance, Surface, Condition, Race Type, Purse   │
    │ Fractional Times: :22.40  :45.60  1:10.20        │
    │ Final Time: 1:36.80                              │
    ├──┬───┬────────────┬──────┬────┬────┬────┬────┬───┤
    │PP│Pgm│Horse       │Jockey│Wgt │Start│1C  │2C  │Str│Fin│Odds│Comment│
    │  │   │            │      │    │Pos/M│P/M │P/M │P/M│P/M│    │       │
    ├──┼───┼────────────┼──────┼────┼────┼────┼────┼───┤
    │ 2│ 2 │Winner      │Smith │122 │1 - │1 Hd│1 1 │1 2│1  │2.50│led...  │
    │ 5│ 5 │Runner Up   │Jones │120 │3 2½│2 1 │2 Nk│2 1│2 3│5.00│chased..│
    └──┴───┴────────────┴──────┴────┴────┴────┴────┴───┘
    """

    # Regex patterns for Equibase chart PDFs
    RACE_HEADER = re.compile(
        r"(?:RACE|Race)\s*(\d+)", re.IGNORECASE
    )
    DISTANCE_PATTERN = re.compile(
        r"((?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|"
        r"About|And|A|One\s+And)[\w\s-]*(?:Furlong|Mile|Yard)s?)",
        re.IGNORECASE,
    )
    # Also match no-space PDF format: "FourAndOneHalfFurlongsOnTheDirt"
    DISTANCE_PATTERN_NOSPACE = re.compile(
        r"Distance:([\w]+(?:Furlong|Mile|Yard)s?)",
        re.IGNORECASE,
    )
    SURFACE_PATTERN = re.compile(r"\b(Dirt|Turf|Synthetic|All Weather)\b", re.IGNORECASE)
    # Also match no-space: "OnTheDirt", "OnTheTurf"
    SURFACE_PATTERN_NOSPACE = re.compile(r"OnThe(Dirt|Turf)", re.IGNORECASE)
    CONDITION_PATTERN = re.compile(
        r"(?:Track:|Track\s*Condition:)\s*(Fast|Good|Muddy|Sloppy|Firm|Yielding|Soft|Heavy|Wet\s*Fast|Slow)",
        re.IGNORECASE,
    )
    PURSE_PATTERN = re.compile(r"Purse:\$?([\d,]+)")
    FRACTIONAL_PATTERN = re.compile(r"(?:FractionalTimes:|Fractional\s*Times:)\s*([\d\.\s:]+)")
    FINAL_TIME_PATTERN = re.compile(r"(?:FinalTime:|Final\s*Time:)\s*([\d\.:]+)")
    RACE_TYPE_PATTERN = re.compile(
        r"\b(Maiden Special Weight|Maiden Claiming|Claiming|Allowance Optional Claiming|"
        r"Allowance|Stakes|Starter Allowance|Starter Optional Claiming|Optional Claiming|"
        r"Handicap|Starter|MAIDEN\s*SPECIAL\s*WEIGHT|MAIDEN\s*CLAIMING|CLAIMING|"
        r"ALLOWANCE|STAKES|HANDICAP|MAIDENSPECIALWEIGHT|MAIDENCLAIMING|ALLOWANCEOPTIONALCLAIMING)\b",
        re.IGNORECASE,
    )

    def parse_pdf_bytes(self, pdf_bytes: bytes, track_code: str, race_date: str) -> List[Dict]:
        """Parse PDF bytes into list of entry dicts."""
        if not pdfplumber:
            log.warning("pdfplumber not available — skipping PDF parse")
            return []

        entries = []
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                full_text = ""
                all_tables = []
                for page_obj in pdf.pages:
                    text = page_obj.extract_text() or ""
                    full_text += text + "\n"
                    tables = page_obj.extract_tables()
                    if tables:
                        all_tables.extend(tables)

                entries = self._parse_chart_content(full_text, all_tables, track_code, race_date)
        except Exception as e:
            log.error(f"PDF parse error: {e}")

        return entries

    # ─── Training-schema lookups ───
    CONDITION_MAP = {
        "fast": "FT", "good": "GD", "muddy": "MY", "sloppy": "SY",
        "firm": "FM", "yielding": "YL", "soft": "SF", "heavy": "HY",
        "wetfast": "WF", "wet fast": "WF", "slow": "SL", "frozen": "FZ",
    }
    WORD_NUM = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12,
    }
    WORD_FRAC = {
        "half": 0.5, "quarter": 0.25, "threequarters": 0.75,
        "sixteenth": 1/16, "eighth": 1/8, "threeeighths": 3/8,
        "onesixteenth": 1/16, "oneeighth": 1/8,
    }

    def _split_camel(self, text: str) -> str:
        """Insert spaces in CamelCase: 'BourbonTown' -> 'Bourbon Town'.
        Preserves common name prefixes (Mc, Mac, O', De, La, Van, Le)."""
        if not text:
            return ""
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
        # Unsplit name-prefix artifacts like "Mc Peek" -> "McPeek"
        s = re.sub(r"\bMc\s+([A-Z])", r"Mc\1", s)
        s = re.sub(r"\bMac\s+([A-Z])", r"Mac\1", s)
        s = re.sub(r"\bDe\s+([A-Z])", r"De\1", s)
        s = re.sub(r"\bDi\s+([A-Z])", r"Di\1", s)
        s = re.sub(r"\bLa\s+([A-Z])", r"La\1", s)
        s = re.sub(r"\bO'\s*([A-Z])", r"O'\1", s)
        return s

    def _parse_distance_integer(self, raw: str) -> Tuple[int, str]:
        """
        Convert Equibase distance text to (integer_distance, unit).
        F unit = furlongs*100 (plus any extra yards).
        Y unit = raw yards.

        Examples:
          'FourAndOneHalfFurlongs' -> (450, 'F')
          'SevenFurlongs'          -> (700, 'F')
          'OneMile'                -> (800, 'F')
          'OneAndOneSixteenthMiles'-> (850, 'F')
          'OneAndOneEighthMiles'   -> (900, 'F')
          'TwoHundredAndTwentyYards' -> (220, 'Y')
        """
        if not raw:
            return (0, "F")
        t = raw.lower().replace(" ", "").replace("-", "")
        # Strip surface tail
        t = re.sub(r"onthe(dirt|turf|allweather|synthetic).*$", "", t)
        t = re.sub(r"(innerturf|outerturf|turfcourse|dirtcourse).*$", "", t)

        is_mile = "mile" in t
        is_yard = "yard" in t and "furlong" not in t and "mile" not in t
        t_clean = re.sub(r"(furlongs?|miles?|yards?)", "", t)

        # About prefix
        t_clean = t_clean.replace("about", "")

        # Parse "ABC and DEF" into whole + fraction
        whole = 0
        frac = 0.0
        # Split on 'and'
        parts = t_clean.split("and")
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            # First part could be integer word or fraction word
            if i == 0:
                # Whole number
                if part in self.WORD_NUM:
                    whole = self.WORD_NUM[part]
                else:
                    # Try to parse composite like 'twohundredandtwenty' collapsed
                    m = re.match(r"(" + "|".join(self.WORD_NUM.keys()) + r")hundred", part)
                    if m:
                        whole = self.WORD_NUM[m.group(1)] * 100
                        rest = part[len(m.group(0)):]
                        if rest in self.WORD_NUM:
                            whole += self.WORD_NUM[rest]
            else:
                # Subsequent parts: "one half" etc
                # "onehalf" or "onesixteenth"
                if part in self.WORD_FRAC:
                    frac += self.WORD_FRAC[part]
                    continue
                # Try numerator+denominator
                for key, val in self.WORD_FRAC.items():
                    if part.endswith(key):
                        num_word = part[:-len(key)]
                        n = self.WORD_NUM.get(num_word, 1)
                        frac += n * val
                        break

        if is_yard:
            # Distance is yards
            return (int(round(whole + frac * 1)), "Y")
        if is_mile:
            # Miles → furlongs*100
            furlongs = (whole + frac) * 8
            return (int(round(furlongs * 100)), "F")
        # Default furlongs
        furlongs = whole + frac
        return (int(round(furlongs * 100)), "F")

    def _parse_course(self, raw_block: str) -> str:
        """Determine course from block text. Returns Dirt/Turf/Inner turf/Outer turf/Hurdle/Timber."""
        t = raw_block.lower().replace(" ", "")
        if "innerturf" in t:
            return "Inner turf"
        if "outerturf" in t:
            return "Outer turf"
        if "hurdle" in t:
            return "Hurdle"
        if "timber" in t:
            return "Timber"
        if "onthedirt" in t or "dirtcourse" in t:
            return "Dirt"
        if "ontheturf" in t or "turfcourse" in t:
            return "Turf"
        return ""

    def _parse_surface(self, course: str) -> str:
        """Derive single-letter surface from course."""
        if not course:
            return ""
        c = course.lower()
        if "turf" in c or "hurdle" in c or "timber" in c:
            return "T"
        return "D"

    def _parse_condition_code(self, raw: str) -> str:
        """Convert 'Sloppy(Sealed)' or 'Fast' to 2-letter training code."""
        if not raw:
            return ""
        cleaned = re.sub(r"\(.*\)", "", raw).strip().lower()
        cleaned = cleaned.replace(" ", "")
        return self.CONDITION_MAP.get(cleaned, "")

    def _parse_race_type_clean(self, raw: str) -> str:
        """Split 'MAIDENSPECIALWEIGHT' into 'Maiden Special Weight' (title case)."""
        if not raw:
            return ""
        # Known training-data race types
        known = [
            "Maiden Special Weight", "Maiden Claiming", "Maiden",
            "Allowance Optional Claiming", "Allowance", "Starter Allowance",
            "Starter Optional Claiming", "Optional Claiming", "Claiming",
            "Handicap", "Invitational Stakes", "Invitational",
            "Claiming Stakes", "Stakes", "Derby Trial", "Derby",
            "Futurity Trial", "Futurity", "Championship", "Final",
            "Match Race", "Speed Index Final", "Speed Index Race",
        ]
        # Normalize by stripping whitespace
        collapsed = re.sub(r"\s+", "", raw).upper()
        for kt in known:
            if re.sub(r"\s+", "", kt).upper() == collapsed:
                return kt
        # Fallback: camel split + title case
        return " ".join(self._split_camel(raw).title().split())

    def _parse_sex_from_winner(self, winner_text: str) -> str:
        """From 'Suspicions,BayColt,byCorniche...' extract 'Colt'."""
        if not winner_text:
            return ""
        t = winner_text.lower()
        for sex in ["Ridgling", "Gelding", "Filly", "Colt", "Mare", "Horse"]:
            if sex.lower() in t:
                return sex
        return ""

    def _parse_sex_from_conditions(self, block: str) -> str:
        """From 'FORFILLIESTHREEYEARSOLD' or 'FORFILLIESANDMARES' etc."""
        t = block.upper().replace(" ", "")
        # Find the "FOR ..." eligibility line
        m = re.search(r"FOR([A-Z,]+?)(?:WHICH|\.|WEIGHT|NON-WINNER|CLAIMING|\d)", t)
        scope = m.group(1) if m else t[:500]
        if "FILLIESANDMARES" in scope:
            return "Filly"  # default; mares would need per-horse age
        if "FILLIES" in scope and "COLTS" not in scope:
            return "Filly"
        if "MARES" in scope and "HORSES" not in scope:
            return "Mare"
        if "COLTSANDGELDINGS" in scope:
            return "Colt"
        return ""

    def _parse_age_from_conditions(self, block: str) -> Optional[int]:
        """From race conditions 'FORMAIDENS,TWOYEARSOLD' or 'FORFILLIESTHREEYEARSOLD'."""
        t = block.upper().replace(" ", "").replace(",", "")
        ages = [
            ("TWOYEARSOLD", 2), ("THREEYEARSOLD", 3), ("FOURYEARSOLD", 4),
            ("FIVEYEARSOLD", 5), ("SIXYEARSOLD", 6),
            ("TWOYEAROLDS", 2), ("THREEYEAROLDS", 3), ("FOURYEAROLDS", 4),
        ]
        for k, v in ages:
            if k in t:
                return v
        if "THREEYEARSOLDANDUPWARD" in t or "THREEYEARSOLDSANDUPWARD" in t:
            return 3
        if "FOURYEARSOLDANDUPWARD" in t or "FOURYEARSOLDSANDUPWARD" in t:
            return 4
        return None

    def _format_jockey(self, raw: str) -> str:
        """'Moran,Pietro' or 'Ortiz,Jr.,Irad' -> 'Pietro Moran' / 'Irad Ortiz Jr.'"""
        if not raw:
            return ""
        raw = raw.strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) == 1:
            return self._split_camel(parts[0])
        if len(parts) == 2:
            last, first = parts
            return f"{self._split_camel(first)} {self._split_camel(last)}".strip()
        # 3 parts like "Ortiz,Jr.,Irad"
        last, suffix, first = parts[0], parts[1], parts[-1]
        return f"{self._split_camel(first)} {self._split_camel(last)} {suffix}".strip()

    def _format_track_name_upper(self, code: str) -> str:
        name = TRACKS.get(code, code)
        return name.upper()

    def _format_date_slashy(self, iso_date: str) -> str:
        """'2026-04-03' -> '4/3/2026'."""
        try:
            d = datetime.strptime(iso_date, "%Y-%m-%d")
            return f"{d.month}/{d.day}/{d.year}"
        except Exception:
            return iso_date

    def _parse_last_raced_token(self, tok: str) -> Dict[str, str]:
        """
        Parse Equibase 'last raced' token like '20Dec2510FG6' or '---'.
        Format: DDMMMYY + race_number + track_code + finish_position
        e.g. '20Dec2510FG6' = Dec 20 2025, race 10, FG track, finish 6.
        Note: race_number can be 1-2 digits, track_code 2-5 letters, finish 1-2 digits.
        """
        out = {"last_race_date": "", "last_race_number": "",
               "last_race_track": "", "last_race_finish": ""}
        if not tok or tok.strip() in ("---", "--", "-"):
            return out
        m = re.match(
            r"^(\d{1,2})([A-Z][a-z]{2})(\d{2})(\d{1,2})([A-Z]{2,5})(\d{1,2})$",
            tok.strip(),
        )
        if not m:
            return out
        day, mon, yy, race_num, track, finish = m.groups()
        month_names = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                       "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        mn = month_names.get(mon, 0)
        year = 2000 + int(yy)
        out["last_race_date"] = f"{mn}/{int(day)}/{year}"
        out["last_race_number"] = race_num
        out["last_race_track"] = self._format_track_name_upper(track)
        out["last_race_finish"] = finish
        return out

    def _parse_trainer_owner_list(self, block: str, key: str) -> Dict[str, str]:
        """
        Parse 'Trainers: 8-Ward,Wesley;5-Hernandez,Rey;...' or 'Owners: ...'
        Returns dict keyed by program number -> formatted name.
        """
        out = {}
        # Section stops at the next labelled section header
        if key.lower() == "trainers":
            stop = r"(?:Owners:|Footnotes|Copyright|ViewGlossary|Breeder:|Scratched|\Z)"
        else:  # owners
            stop = r"(?:Trainers:|Footnotes|Copyright|ViewGlossary|Breeder:|Scratched|\Z)"
        m = re.search(rf"{key}:\s*(.+?){stop}", block, re.DOTALL | re.IGNORECASE)
        if not m:
            return out
        raw = m.group(1).replace("\n", "")
        for piece in raw.split(";"):
            piece = piece.strip()
            if not piece or "-" not in piece:
                continue
            pgm, _, name = piece.partition("-")
            pgm = pgm.strip()
            name = name.strip().rstrip(",;")
            if not pgm.isdigit() or not name:
                continue
            if key.lower() == "trainers":
                out[pgm] = self._format_jockey(name)  # Last,First,(Suffix)
            else:
                out[pgm] = self._split_camel(name)
        return out

    def _parse_chart_content(self, text: str, tables: list,
                             track_code: str, race_date: str) -> List[Dict]:
        """Parse extracted text and tables from a chart PDF."""
        entries = []

        # Split text into race blocks by the 'TRACK-Date-Race N' header
        race_blocks = re.split(r"(?=[A-Z]+-[A-Z][a-z]+\d+,\d{4}-Race\s*\d+)", text)
        if len(race_blocks) <= 1:
            race_blocks = re.split(r"(?=(?:RACE|Race)\s*\d+)", text)

        for block in race_blocks:
            if not block.strip():
                continue

            race_num_m = self.RACE_HEADER.search(block)
            race_num = int(race_num_m.group(1)) if race_num_m else 0
            if race_num == 0:
                continue

            # ─── Extract race-level fields ───
            # Distance: 'Distance:FourAndOneHalfFurlongsOnTheDirtCurrentTrackRecord:...'
            dist_m = re.search(r"Distance:([A-Za-z]+?)(?:CurrentTrackRecord|$)", block)
            dist_raw = dist_m.group(1) if dist_m else ""
            dist_int, dist_unit = self._parse_distance_integer(dist_raw)

            course = self._parse_course(dist_raw + " " + block[:2000])
            surface = self._parse_surface(course)

            # Track condition: 'Track:Sloppy(Sealed)' or 'Track:Fast'
            cond_m = re.search(r"Track:([A-Za-z]+(?:\([^)]*\))?)", block)
            track_condition = self._parse_condition_code(cond_m.group(1) if cond_m else "")

            # Weather: 'Weather:Showery,67°'
            weather_m = re.search(r"Weather:([A-Za-z]+)", block)
            weather = weather_m.group(1).strip() if weather_m else ""

            # Post time: 'Offat:1:05'
            off_m = re.search(r"Offat:([\d:]+)", block)
            post_time = off_m.group(1) if off_m else ""

            # Purse
            purse_m = re.search(r"Purse:\$?([\d,]+)", block)
            purse = purse_m.group(1).replace(",", "") if purse_m else ""

            # Race type: first line after the race header, e.g. 'MAIDENSPECIALWEIGHT-Thoroughbred' or 'STAKESCentralBankAshlandS.Grade1-Thoroughbred'
            type_m = re.search(r"Race\s*\d+\s*\n?(?:\*|\[)?\s*\n?([A-Z][A-Z0-9&\- ]+?)(?:CentralBank|-Thoroughbred|-QuarterHorse|-Arabian|\n|FOR)", block)
            race_type_raw = ""
            if type_m:
                race_type_raw = type_m.group(1).strip()
            # Fall back to known-types regex
            if not race_type_raw:
                rt_m = re.search(
                    r"(MAIDENSPECIALWEIGHT|MAIDENCLAIMING|ALLOWANCEOPTIONALCLAIMING|"
                    r"STARTERALLOWANCE|STARTEROPTIONALCLAIMING|OPTIONALCLAIMING|"
                    r"ALLOWANCE|CLAIMING|STAKES|HANDICAP|MAIDEN)", block)
                if rt_m:
                    race_type_raw = rt_m.group(1)
            race_type = self._parse_race_type_clean(race_type_raw)

            # Breed: 'MAIDENSPECIALWEIGHT-Thoroughbred'
            breed_raw_m = re.search(r"-(Thoroughbred|QuarterHorse|Arabian|Paint|AppaloosaAPHA|Appaloosa)", block)
            breed_map = {"Thoroughbred": "TB", "QuarterHorse": "QH",
                         "Arabian": "AR", "Paint": "PT", "Appaloosa": "AP", "AppaloosaAPHA": "AP"}
            breed = breed_map.get(breed_raw_m.group(1), "TB") if breed_raw_m else "TB"

            # Age and sex from conditions
            age_val = self._parse_age_from_conditions(block)
            sex_from_conditions = self._parse_sex_from_conditions(block)

            # Fractional times
            frac_m = re.search(r"FractionalTimes:([\d\.\s:]+?)FinalTime:", block)
            frac_times_raw = frac_m.group(1).strip().split() if frac_m else []
            final_m = re.search(r"FinalTime:([\d:.]+)", block)
            win_time = self._parse_time(final_m.group(1)) if final_m else ""

            # Winner details block: 'Winner: Suspicions,BayColt,byCornicheoutofManaoag...'
            winner_m = re.search(r"Winner:\s*([^\n]+?)(?=Breeder:|Owner:|\n)", block)
            winner_text = winner_m.group(1) if winner_m else ""
            winner_sex = self._parse_sex_from_winner(winner_text)

            # Trainers and owners by program number
            trainer_map = self._parse_trainer_owner_list(block, "Trainers")
            owner_map = self._parse_trainer_owner_list(block, "Owners")

            # Claiming price: 'ClaimingPrice:$50,000-$0'
            claim_m = re.search(r"ClaimingPrice:\$?([\d,]+)", block)
            claimed_price = claim_m.group(1).replace(",", "") if claim_m else ""

            race_info = {
                "race_number": race_num,
                "track_code": track_code,
                "track_name": self._format_track_name_upper(track_code),
                "race_date": self._format_date_slashy(race_date),
                "distance": dist_int,
                "distance_unit": dist_unit,
                "course": course,
                "surface": surface,
                "track_condition": track_condition,
                "weather": weather,
                "post_time": post_time,
                "win_time": win_time,
                "purse": purse,
                "race_type": race_type,
                "breed": breed,
                "_age_default": age_val,
                "_winner_sex": winner_sex,
                "_sex_from_conditions": sex_from_conditions,
                "_trainer_map": trainer_map,
                "_owner_map": owner_map,
                "claimed_price": claimed_price,
                "frac_1": self._parse_time(frac_times_raw[0]) if len(frac_times_raw) > 0 else "",
                "frac_2": self._parse_time(frac_times_raw[1]) if len(frac_times_raw) > 1 else "",
                "frac_3": self._parse_time(frac_times_raw[2]) if len(frac_times_raw) > 2 else "",
                "frac_4": self._parse_time(frac_times_raw[3]) if len(frac_times_raw) > 3 else "",
                "final_time_secs": win_time,
            }

            race_entries = self._parse_results_from_text(block, race_info)
            if not race_entries:
                race_entries = self._parse_results_from_tables(tables, race_info)

            # Strip internal-only keys before emitting
            for e in race_entries:
                for k in ("_age_default", "_winner_sex", "_sex_from_conditions",
                          "_trainer_map", "_owner_map"):
                    e.pop(k, None)
            entries.extend(race_entries)

        return entries

    def _parse_results_from_tables(self, tables: list, race_info: dict) -> List[Dict]:
        """Try to find and parse the results table for this race."""
        entries = []
        for table in tables:
            if not table or len(table) < 3:
                continue

            # Check if this table looks like a race results table
            header = [str(c).lower().strip() if c else "" for c in table[0]]
            has_horse = any("horse" in h or "name" in h for h in header)
            has_pgm = any("pgm" in h or "#" in h or "no" in h for h in header)

            if not (has_horse or has_pgm):
                continue

            # Map columns
            col = {}
            for i, h in enumerate(header):
                if "horse" in h or "name" in h:
                    col["horse"] = i
                elif "pgm" in h or "no." in h:
                    col["pgm"] = i
                elif "pp" == h or "post" in h:
                    col["pp"] = i
                elif "jock" in h:
                    col["jockey"] = i
                elif "train" in h:
                    col["trainer"] = i
                elif "wgt" in h or "wt" in h or "weight" in h:
                    col["weight"] = i
                elif "odds" in h or "ml" in h:
                    col["odds"] = i
                elif "fin" in h:
                    col["finish"] = i
                elif "comment" in h or "remark" in h:
                    col["comment"] = i
                elif "owner" in h:
                    col["owner"] = i
                elif "margin" in h or "btn" in h or "behind" in h:
                    col["margin"] = i
                elif "1st" in h or "first" in h:
                    col["pos_1st"] = i
                elif "2nd" in h or "second" in h:
                    col["pos_2nd"] = i
                elif "str" in h and "start" not in h:
                    col["pos_str"] = i
                elif "start" in h:
                    col["pos_start"] = i
                elif "med" in h:
                    col["med"] = i
                elif "age" in h:
                    col["age"] = i
                elif "sex" in h:
                    col["sex"] = i

            if "horse" not in col:
                continue

            for row_idx, row in enumerate(table[1:], 1):
                if not row or len(row) <= col.get("horse", 0):
                    continue

                def get(key):
                    idx = col.get(key)
                    if idx is not None and idx < len(row) and row[idx]:
                        return str(row[idx]).strip()
                    return ""

                horse = get("horse")
                if not horse or len(horse) < 2 or horse.lower() == "horse":
                    continue

                entry = {**race_info}
                entry["horse_name"] = horse
                entry["program_num"] = get("pgm")
                entry["post_position"] = get("pp")
                entry["finish"] = get("finish") or str(row_idx)
                entry["jockey"] = get("jockey")
                entry["trainer"] = get("trainer")
                entry["owner"] = get("owner")
                entry["weight"] = get("weight")
                entry["dollar_odds"] = self._clean_odds(get("odds"))
                entry["comment"] = get("comment")
                entry["medication"] = get("med")
                entry["age"] = get("age")
                entry["sex"] = get("sex")
                entry["margin_finish"] = self._parse_margin(get("margin"))
                entry["pos_1st_call"] = self._extract_pos(get("pos_1st"))
                entry["pos_2nd_call"] = self._extract_pos(get("pos_2nd"))
                entry["pos_stretch"] = self._extract_pos(get("pos_str"))
                entry["margin_1st_call"] = self._extract_margin(get("pos_1st"))
                entry["margin_2nd_call"] = self._extract_margin(get("pos_2nd"))
                entry["margin_stretch"] = self._extract_margin(get("pos_str"))

                # Fill missing columns
                for c in OUTPUT_COLUMNS:
                    if c not in entry:
                        entry[c] = ""

                entries.append(entry)

            if entries:
                break  # found the right table

        return entries

    def _parse_results_from_text(self, block: str, race_info: dict) -> List[Dict]:
        """Parse horse entries from Equibase chart PDF text.

        Equibase PDF line format (no-space variant):
          --- 8 Suspicions(Moran,Pietro) 119 b 5 2 22 12 12 1.30* bmpst,2w,clrd,hldswy
          20Dec2510FG6 5 MissCall(Hernandez,Jr.,Brian) 120 L 4 6 31/2 41 11/2 163/4 2.90 2p,bid...

        Fields: LastRaced Pgm HorseName(Jockey) Wgt M/E PP Start 1/4 [1/2] [3/4] Str Fin Odds Comment
        """
        entries = []
        lines = block.split("\n")
        finish_pos = 0

        # LastRaced token is either '---' or a compact date/track/race/finish code.
        # HorseName is CamelCase (no spaces) followed by (Jockey,Name).
        horse_line_re = re.compile(
            r"^(---|[0-9A-Za-z]+?)\s+"          # Last raced
            r"(\d{1,2})\s+"                      # Program number
            r"([A-Z][A-Za-z'\.0-9\- ]*?)\(([^)]+)\)\s*"  # HorseName(Jockey)
            r"(\d{2,3})\s*"                      # Weight
            r"([A-Za-z\-]{1,5}|--)\s+"           # Medication/equip
            r"(\d{1,2})\s+"                      # Post position
            r"(\d{1,2})\s+"                      # Start position
            r"(.+)$"                             # Rest: running positions + odds + comment
        )

        trainer_map = race_info.get("_trainer_map", {})
        owner_map = race_info.get("_owner_map", {})
        age_default = race_info.get("_age_default")
        winner_sex = race_info.get("_winner_sex", "")
        sex_from_conditions = race_info.get("_sex_from_conditions", "")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip header/section lines
            if line.startswith("LastRaced") or line.startswith("FractionalTimes") or line.startswith("Pgm "):
                continue
            m = horse_line_re.match(line)
            if not m:
                continue

            finish_pos += 1
            last_raced, pgm, horse, jockey, weight, med_equip, pp, _start, rest = m.groups()

            # Running positions + odds + comment
            rest_parts = rest.split()
            odds = ""
            comment_parts = []
            positions = []
            for i, part in enumerate(rest_parts):
                # Odds: 0.78*, *1.40, 26.07
                if re.match(r"^\*?\d+\.\d+\*?$", part) and "/" not in part:
                    odds = part.replace("*", "")
                    comment_parts = rest_parts[i + 1:]
                    break
                positions.append(part)

            # Odds as integer dollar form (training schema uses int)
            try:
                dollar_odds_int = int(round(float(odds))) if odds else ""
            except ValueError:
                dollar_odds_int = ""

            # Last raced parse
            lr = self._parse_last_raced_token(last_raced)

            # Medication: 'b', 'L', 'Lb', 'Lbf', '--'
            med_clean = med_equip if med_equip and med_equip != "--" else ""
            # Training schema uses L, BL, B, LA - map by presence
            med_upper = med_clean.upper()
            if "L" in med_upper and "B" in med_upper:
                medication = "BL"
            elif "L" in med_upper and "A" in med_upper:
                medication = "LA"
            elif "L" in med_upper:
                medication = "L"
            elif "B" in med_upper:
                medication = "B"
            else:
                medication = ""

            entry = {**race_info}
            entry["horse_name"] = self._split_camel(horse.strip())
            entry["program_num"] = int(pgm) if pgm.isdigit() else pgm
            entry["post_position"] = int(pp) if pp.isdigit() else pp
            entry["finish"] = finish_pos
            entry["jockey"] = self._format_jockey(jockey)
            entry["trainer"] = trainer_map.get(pgm, "")
            entry["owner"] = owner_map.get(pgm, "")
            entry["weight"] = int(weight) if weight.isdigit() else weight
            entry["dollar_odds"] = dollar_odds_int
            entry["medication"] = medication
            entry["comment"] = " ".join(comment_parts).replace(",", " ").strip()
            entry["age"] = age_default if age_default is not None else ""
            # Winner gets exact sex from Winner line; others use condition-derived default.
            if finish_pos == 1 and winner_sex:
                entry["sex"] = winner_sex
            else:
                entry["sex"] = sex_from_conditions
            entry.update(lr)

            for c in OUTPUT_COLUMNS:
                if c not in entry:
                    entry[c] = ""
            entries.append(entry)

        return entries

    def _extract_margin_from_pos(self, text):
        """Extract margin from position text like '111/2' or '3Head' or '21/4'."""
        if not text:
            return ""
        # Remove leading position number to get margin
        # "111/2" = 1 and 1/2 lengths, "3Head" = head margin at 3rd, etc
        text = str(text).strip()
        specials = {"head": 0.1, "hd": 0.1, "nose": 0.05, "neck": 0.25, "nk": 0.25}
        text_lower = text.lower()
        for key, val in specials.items():
            if key in text_lower:
                return val
        # Try to extract fraction: "11/2" = 0.5, "31/4" = 0.25
        m = re.search(r"(\d+)/(\d+)", text)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            # Could be "11/2" meaning "1 and 1/2" or just "1/2"
            prefix = text[:m.start()]
            whole = int(prefix) if prefix and prefix.isdigit() else 0
            return whole + num / den
        return ""

    def _parse_time(self, text):
        if not text:
            return ""
        text = str(text).strip().lstrip(":")
        try:
            if ":" in text:
                parts = text.split(":")
                return round(int(parts[0]) * 60 + float(parts[1]), 2)
            return round(float(text), 2)
        except ValueError:
            return ""

    def _parse_margin(self, text):
        if not text:
            return ""
        text = str(text).strip()
        if text in ("---", "—", "-"):
            return 0.0
        specials = {"head": 0.1, "hd": 0.1, "nose": 0.05, "neck": 0.25, "nk": 0.25}
        if text.lower() in specials:
            return specials[text.lower()]
        fracs = {"¼": 0.25, "½": 0.5, "¾": 0.75}
        for ch, val in fracs.items():
            if ch in text:
                w = text.replace(ch, "").strip()
                return (int(w) if w else 0) + val
        try:
            return float(text)
        except ValueError:
            return ""

    def _extract_pos(self, text):
        """Extract running position from '3 2½' → 3"""
        if not text:
            return ""
        m = re.match(r"(\d+)", str(text).strip())
        return int(m.group(1)) if m else ""

    def _extract_margin(self, text):
        """Extract margin from '3 2½' → 2.5"""
        if not text:
            return ""
        parts = str(text).strip().split()
        if len(parts) >= 2:
            return self._parse_margin(parts[-1])
        return ""

    def _clean_odds(self, text):
        text = str(text).strip().replace("$", "")
        m = re.match(r"(\d+)-(\d+)", text)
        if m:
            return str(round(int(m.group(1)) / int(m.group(2)), 2))
        m = re.search(r"[\d.]+", text)
        return m.group(0) if m else ""


# ═══════════════════════════════════════════════════════════════
# CHECKPOINT
# ═══════════════════════════════════════════════════════════════

class Checkpoint:
    def __init__(self, path="scraper_v3_checkpoint.json"):
        self.path = path
        self.done: Set[str] = set()
        self.stats = {"pages": 0, "entries": 0, "errors": 0, "pdfs": 0}
        self._load()

    def _key(self, track, date):
        return f"{track}_{date.strftime('%Y%m%d')}"

    def is_done(self, track, date):
        return self._key(track, date) in self.done

    def mark(self, track, date, n_entries=0):
        self.done.add(self._key(track, date))
        self.stats["pages"] += 1
        self.stats["entries"] += n_entries
        if self.stats["pages"] % 25 == 0:
            self.save()

    def save(self):
        with open(self.path, "w") as f:
            json.dump({"done": list(self.done), "stats": self.stats}, f)

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                d = json.load(f)
            self.done = set(d.get("done", []))
            self.stats = d.get("stats", self.stats)
            log.info(f"Checkpoint: {len(self.done)} done, {self.stats['entries']:,} entries")


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

    def __init__(self, output_file="historical_races.csv", headless=True, use_google=True, cdp_port=9222, checkpoint_path="scraper_v3_checkpoint.json", career_stats=False):
        self.output_file = output_file
        self.headless = headless
        self._use_google = use_google
        self._cdp_port = cdp_port
        self.checkpoint = Checkpoint(checkpoint_path)
        self.pdf_parser = ChartPDFParser()
        self.human = HumanBehavior()
        self.pw = None
        self.browser = None
        self.page = None
        self._csv_file = None
        self._csv_writer = None
        # Career stats enrichment
        self._career_stats_enabled = career_stats
        # Cache: horse_name_lower -> {num_past_starts, num_past_wins, num_past_seconds, num_past_thirds}
        # Keeps scrape fast when the same horse races repeatedly.
        self._career_cache: Dict[str, Dict] = {}
        self._career_miss = 0
        self._career_hit = 0

    # ── Browser lifecycle ────────────────────────────────────

    async def _start_browser(self):
        self.pw = await async_playwright().start()
        vp = random.choice(VIEWPORTS)

        # Try connecting to running Chrome via CDP first
        if self._cdp_port:
            for url in [f"http://localhost:{self._cdp_port}", f"http://127.0.0.1:{self._cdp_port}"]:
                try:
                    self.browser = await self.pw.chromium.connect_over_cdp(url)
                    self.context = self.browser.contexts[0]
                    self.page = await self.context.new_page()
                    log.info(f"Connected to your running Chrome on {url}! "
                             "(assuming you're already signed in)")
                    return
                except Exception as e:
                    log.debug(f"CDP connect to {url} failed: {e}")
            log.warning(f"Could not connect to Chrome on port {self._cdp_port}. Falling back to fresh browser.")

        # Fallback: launch fresh Playwright browser
        ua = random.choice(USER_AGENTS)
        self.browser = await self.pw.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )
        self.context = await self.browser.new_context(
            user_agent=ua,
            viewport=vp,
            locale="en-US",
            timezone_id="America/New_York",
        )
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = { runtime: {} };
        """)
        self.page = await self.context.new_page()
        log.info(f"Fresh browser launched | Viewport: {vp['width']}x{vp['height']}")

        # Navigate to Equibase and handle bot detection upfront
        await self._pass_bot_detection()
        # Wait for the user to sign in to Equibase in the fresh browser window
        await self._wait_for_equibase_login()

    async def _wait_for_equibase_login(self):
        """Open equibase.com in the attached Chrome and wait for the user to
        sign in manually. Polls until a signed-in indicator appears, or user
        presses Enter in the terminal."""
        try:
            await self.page.goto("https://www.equibase.com/", wait_until="domcontentloaded", timeout=20000)
            await asyncio.sleep(1.5)
        except Exception as e:
            log.warning(f"Could not open equibase.com: {e}")

        async def _is_signed_in() -> bool:
            try:
                text = await self.page.inner_text("body")
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

    async def _prompt_login(self):
        """Prompt user to sign in via the URL bar, then wait."""
        # Check if already signed in
        await self.page.goto("https://accounts.google.com", wait_until="domcontentloaded", timeout=15000)
        await asyncio.sleep(2)
        page_url = self.page.url
        if "myaccount" in page_url or "signin" not in page_url.lower():
            text = await self.page.inner_text("body")
            if "sign in" not in text.lower()[:300] and len(text) > 100:
                log.info("Already signed into Google. Continuing...")
                return

        # Use the URL bar + page content as a message to the user
        await self.page.goto("data:text/html,<html><head><title>SIGN IN TO CHROME - Scraper waiting...</title></head>"
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
        for i in range(24):
            await asyncio.sleep(5)
            try:
                # Open a new tab to check sign-in status
                check_page = await self.context.new_page()
                await check_page.goto("https://accounts.google.com", wait_until="domcontentloaded", timeout=10000)
                await asyncio.sleep(2)
                check_url = check_page.url
                check_text = await check_page.inner_text("body")
                await check_page.close()

                if "myaccount" in check_url or ("sign in" not in check_text.lower()[:300] and len(check_text) > 100):
                    log.info("Sign-in detected! Continuing scraper...")
                    # Show success message briefly
                    await self.page.goto("data:text/html,<html><body style='display:flex;align-items:center;"
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

    async def _pass_bot_detection(self):
        """Navigate to Equibase and let the user solve any captcha."""
        log.info("Opening equibase.com to check for bot detection...")
        await self.page.goto("https://www.equibase.com", wait_until="domcontentloaded", timeout=20000)
        await asyncio.sleep(3)

        text = await self.page.inner_text("body")
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
                    text = await self.page.inner_text("body")
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
        elif self.context:
            await self.context.close()
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

    async def _fetch_horse_career_by_name(self, horse_name: str) -> Dict:
        """Search Equibase by horse name via the homepage form and parse the
        career stats block from whatever page lands. Returns {} on failure."""
        result: Dict = {}
        try:
            await self.page.goto("https://www.equibase.com/", wait_until="domcontentloaded", timeout=20000)
            await asyncio.sleep(0.5)
            await self.page.fill("input[name='searchInput']", horse_name)
            # Click the submit button (JS-driven form → direct profile page for unique matches)
            try:
                await self.page.click("form button[type='submit'], form input[type='submit']", timeout=3000)
            except Exception:
                await self.page.press("input[name='searchInput']", "Enter")
            await self.page.wait_for_load_state("domcontentloaded", timeout=15000)
            await asyncio.sleep(0.8)

            landed = self.page.url
            # Case A: direct hit — we're on a horse profile page already.
            if "type=Horse" in landed and "refno=" in landed:
                result = await self._parse_career_from_profile()
                return result

            # Case B: multi-match search result — click the first TB match.
            profile_href = await self.page.evaluate(
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
                await self.page.goto(profile_href, wait_until="domcontentloaded", timeout=20000)
                await asyncio.sleep(0.8)
                result = await self._parse_career_from_profile()
        except Exception as e:
            log.debug(f"career search failed for {horse_name!r}: {e}")
        return result

    async def _parse_career_from_profile(self) -> Dict:
        """Extract career totals from the currently-loaded horse profile page.
        Tries DOM tables first, then falls back to body-text regex because
        Equibase sometimes renders the block as a flex layout without a
        standards-compliant table header row."""
        # Attempt 1: table with canonical header
        try:
            data = await self.page.evaluate(
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
            body = await self.page.evaluate("() => document.body.innerText || ''")
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

    async def _enrich_with_career_stats(self, entries: List[Dict]):
        """Populate career fields on each entry by searching each unique horse
        by name and parsing the profile. Cached across the run."""
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

        for name in unique_names:
            key = name.lower()
            if key in self._career_cache:
                self._career_hit += 1
                stats = self._career_cache[key]
            else:
                self._career_miss += 1
                stats = await self._fetch_horse_career_by_name(name)
                self._career_cache[key] = stats
                await self.human.random_delay(0.4, 1.0)

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

    async def _google_search(self, query: str) -> Optional[str]:
        """
        Search Google and return the first Equibase result URL.
        Returns None if no result found.
        """
        try:
            await self.page.goto("https://www.google.com", wait_until="domcontentloaded", timeout=15000)
            await self.human.random_delay(1.5, 3.0)

            # Accept cookies if prompted
            try:
                accept_btn = self.page.locator("button:has-text('Accept'), button:has-text('I agree')")
                if await accept_btn.count() > 0:
                    await accept_btn.first.click()
                    await self.human.random_delay(1.0, 2.0)
            except Exception:
                pass

            # Type the search query like a human
            search_box = self.page.locator('textarea[name="q"], input[name="q"]').first
            await search_box.click()
            await asyncio.sleep(random.uniform(0.3, 0.8))

            for char in query:
                await self.page.keyboard.type(char, delay=random.randint(40, 180))
                if random.random() < 0.03:
                    await asyncio.sleep(random.uniform(0.2, 0.6))

            await asyncio.sleep(random.uniform(0.5, 1.5))
            await self.page.keyboard.press("Enter")
            await self.page.wait_for_load_state("domcontentloaded")
            await self.human.random_delay(1.2, 2.5)

            # Find Equibase links in results
            links = await self.page.evaluate("""
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

    async def _navigate_to_race_day(self, track_code: str, date: datetime) -> bool:
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
            landed_url = self.page.url
            log.info(f"{label} -> landed on: {landed_url}")
            text = await self.page.inner_text("body")
            text_preview = text[:200].replace('\n', ' ').strip()

            # Handle bot detection — wait and retry once
            if "Pardon Our Interruption" in text:
                log.info(f"{label}: Bot detection triggered, waiting 10s and reloading...")
                await asyncio.sleep(10)
                await self.page.reload(wait_until="domcontentloaded", timeout=15000)
                await self.human.random_delay(3.0, 5.0)
                text = await self.page.inner_text("body")
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
            await self.page.goto(chart_embed_url, wait_until="domcontentloaded", timeout=15000)
            await self.human.random_delay(1.2, 2.5)
            if await _check_page("Premium chart embed"):
                return True
        except Exception as e:
            log.info(f"Premium chart embed URL failed: {e}")

        # ── Attempt 3b: Premium chart index URL (older format) ──
        index_url = self.INDEX_URL.format(
            track=track_code, date=date.strftime("%m/%d/%Y")
        )
        log.info(f"Trying premium chart index: {index_url}")
        try:
            await self.page.goto(index_url, wait_until="domcontentloaded", timeout=15000)
            await self.human.random_delay(1.2, 2.5)
            if await _check_page("Premium chart index"):
                return True
        except Exception as e:
            log.info(f"Premium chart index URL failed: {e}")

        # ── Attempt 4: Google search (last resort — may trigger CAPTCHA) ──
        if self._use_google:
            query = f"{track_name} {date_str} race results equibase"
            log.info(f"Falling back to Google: {query}")
            url = await self._google_search(query)

            if url:
                log.info(f"Google found: {url}")
                try:
                    link_clicked = await self.page.evaluate("""
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
                        await self.page.wait_for_load_state("domcontentloaded", timeout=15000)
                        await self.human.random_delay(1.2, 2.5)
                        landed_url = self.page.url
                        log.info(f"Google -> landed on: {landed_url}")
                        text = await self.page.inner_text("body")
                        if not self._page_has_no_data(text):
                            return True
                except Exception as e:
                    log.debug(f"Google click-through failed: {e}")

                    # If Google CAPTCHA'd us, disable Google for the rest of the run
                    page_text = await self.page.inner_text("body")
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

    async def _extract_from_page(self, track_code: str, date: datetime) -> List[Dict]:
        """
        Extract race data from the current page.
        Tries multiple strategies:
          1. Find PDF chart links → download and parse PDFs
          2. Parse Equibase summary HTML tables directly
          3. Click into individual race pages
        """
        entries = []
        race_date_str = date.strftime("%m/%d/%Y")
        current_url = self.page.url
        log.info(f"Extracting from: {current_url}")

        # ── Strategy 1: Look for PDF chart links ──
        pdf_links = await self.page.evaluate("""
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
                    pdf_entries = await self._download_and_parse_pdf(href, track_code, race_date_str)
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
            saved_url = self.page.url
            consecutive_failures = 0

            log.info(f"Trying chart PDF download strategy with base: {chart_base}")
            for race_num in range(1, 16):
                chart_url = f"{chart_base}&RACE={race_num}"
                log.info(f"Race {race_num}: downloading from {chart_url}")
                try:
                    # Use page.request API to download the PDF bytes directly
                    # This avoids the "Download is starting" navigation error
                    response = await self.page.request.get(chart_url, timeout=30000)
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
                    await self.page.goto(saved_url, wait_until="domcontentloaded", timeout=15000)
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
                test_entries = await self._download_and_parse_pdf(test_url, track_code, race_date_str)
                if test_entries:
                    entries.extend(test_entries)
                    self.checkpoint.stats["pdfs"] += 1
                    working_pattern = test_url.replace("1.pdf", "{race_num}.pdf").replace("-1-d.", "-{race_num}-d.")
                    log.info(f"PDF pattern works! Using: {working_pattern}")
                    break

            if working_pattern:
                for race_num in range(2, 16):
                    pdf_url = working_pattern.format(race_num=race_num)
                    pdf_entries = await self._download_and_parse_pdf(pdf_url, track_code, race_date_str)
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
            entries = await self._parse_html_tables(track_code, race_date_str)
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
                    view_all_url = await self.page.evaluate(r"""
                        () => {
                            const link = document.querySelector('a[href*="EQB.html"]:not([href*="RaceCardIndex"])');
                            return link ? link.href : null;
                        }
                    """)
                    if view_all_url:
                        log.info(f"Clicking 'View All Races': {view_all_url}")
                        await self.page.goto(view_all_url, wait_until="domcontentloaded", timeout=15000)
                        await self.human.random_delay(1.2, 2.5)
                        entries = await self._parse_html_tables(track_code, race_date_str)
                except Exception as e:
                    log.debug(f"View All Races click failed: {e}")

            # If still no entries, go directly to summary URL
            if not entries:
                summary_url = self.SUMMARY_URL.format(track=track_code, date=date_mmddyy)
                if summary_url not in current_url:
                    log.info(f"Trying Equibase summary page: {summary_url}")
                    try:
                        await self.page.goto(summary_url, wait_until="domcontentloaded", timeout=15000)
                        await self.human.random_delay(1.2, 2.5)
                        log.info(f"Summary page landed on: {self.page.url}")
                        entries = await self._parse_html_tables(track_code, race_date_str)
                        if entries:
                            log.info(f"Summary page parsing found {len(entries)} entries")
                    except Exception as e:
                        log.debug(f"Summary page navigation failed: {e}")

        # ── Strategy 4: Click into individual race links ──
        if not entries:
            race_links = await self.page.evaluate(r"""
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
                    await self.page.goto(link_info["href"],
                                         wait_until="domcontentloaded", timeout=15000)
                    await self.human.random_delay(1.2, 2.5)
                    await self.human.random_scroll(self.page)

                    race_entries = await self._parse_html_tables(track_code, race_date_str)
                    entries.extend(race_entries)

                    await self.page.go_back(wait_until="domcontentloaded", timeout=10000)
                    await self.human.random_delay(1.5, 3.0)
                except Exception as e:
                    log.debug(f"Race link click failed: {e}")

        if not entries:
            # Dump page diagnostics for debugging
            await self._dump_page_diagnostics()

        return entries

    async def _dump_page_diagnostics(self):
        """Log diagnostic info about current page to help debug parsing failures."""
        try:
            url = self.page.url
            title = await self.page.title()
            table_info = await self.page.evaluate("""
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

    async def _download_and_parse_pdf(self, url: str, track_code: str, race_date: str) -> List[Dict]:
        """Download a PDF and parse it. Validates response is actually a PDF."""
        try:
            # Use page context to download (maintains cookies/session)
            response = await self.page.request.get(url, timeout=30000)
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

    async def _parse_html_tables(self, track_code: str, race_date: str) -> List[Dict]:
        """Parse race results from HTML tables on the current page."""
        # Grab tables WITH their preceding context (race headers above tables)
        raw_tables = await self.page.evaluate("""
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
        page_text = await self.page.inner_text("body")

        entries = []
        for table in (raw_tables or []):
            parsed = self._parse_generic_table(table, track_code, race_date, page_text)
            entries.extend(parsed)

        # If generic parsing found nothing, try Equibase-specific summary parsing
        if not entries:
            entries = await self._parse_equibase_summary(track_code, race_date, page_text)

        return entries

    async def _parse_equibase_summary(self, track_code: str, race_date: str, page_text: str) -> List[Dict]:
        """
        Parse Equibase summary results pages.

        Actual Equibase page structure (confirmed by live inspection):
        ┌────────────────────────────────────────────────────────┐
        │ RACE 1                                                  │
        │ Off at: 1:05  Race type: Maiden Special Weight          │
        │ Age Restriction: Two Year Old                           │
        │ Purse: $90,000                                          │
        │ Distance: Four And One Half Furlongs On The Dirt        │
        │ Track Condition: Sloppy                                 │
        │ Winning Time: 52.74                                     │
        ├──────┬──────────────┬──────────────┬──────┬───────┬─────┤
        │ Pgm  │ Horse        │ Jockey       │ Win  │ Place │ Show│
        ├──────┼──────────────┼──────────────┼──────┼───────┼─────┤
        │ 8    │ Suspicions   │ Pietro Moran │ 4.60 │ 2.82  │2.54 │
        │ 5    │ Bourbon Town │ Luis Saez    │      │ 2.76  │2.26 │
        │ 7    │ Tigrado      │ Evin Roman   │      │       │4.92 │
        └──────┴──────────────┴──────────────┴──────┴───────┴─────┘
        Also ran: 3 - Super Saiyajin , 2 - Joe Joe Dude , 9 - Cross Power
        Winning Breeder: ... Winning Owner: ... Winning Trainer: ...
        └────────────────────────────────────────────────────────┘

        Key facts:
        - Tables have headers: Pgm | Horse | Jockey | Win | Place | Show
        - Only top 3 finishers are in the table
        - Remaining finishers are in "Also ran:" text
        - Race metadata is in text blocks between tables
        """
        entries = []

        # Use JavaScript to extract structured race data matching actual Equibase layout
        race_data = await self.page.evaluate(r"""
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
                    # Use Win payoff as a proxy for odds (Win / 2 ≈ odds)
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

    # ── Main run loop ────────────────────────────────────────

    async def run(self, targets: List[Tuple[str, datetime]]):
        """Main scraping loop."""
        remaining = [(t, d) for t, d in targets if not self.checkpoint.is_done(t, d)]
        total = len(targets)
        skipped = total - len(remaining)

        log.info(f"Targets: {total:,} total, {skipped:,} done, {len(remaining):,} remaining")

        self._open_csv()
        try:
            await self._start_browser()

            for i, (track, date) in enumerate(remaining):
                pct = (skipped + i) / total * 100
                date_str = date.strftime("%Y-%m-%d")

                # Occasional distraction browsing (~5% of the time)
                if random.random() < 0.05:
                    await self.human.browse_distraction(self.page)

                # Navigate to race day
                has_data = await self._navigate_to_race_day(track, date)

                if has_data:
                    # Human-like: scroll around, look at the page
                    await self.human.random_scroll(self.page)
                    await self.human.random_mouse_move(self.page)

                    # Extract data
                    entries = await self._extract_from_page(track, date)

                    # Optionally enrich with career stats from horse profile pages
                    if entries and self._career_stats_enabled:
                        await self._enrich_with_career_stats(entries)

                    if entries:
                        self._write_entries(entries)
                        self.checkpoint.mark(track, date, len(entries))
                        log.info(
                            f"[{pct:.1f}%] {track} {date_str}: {len(entries)} entries "
                            f"(total: {self.checkpoint.stats['entries']:,})"
                        )
                    else:
                        self.checkpoint.mark(track, date, 0)
                        log.debug(f"[{pct:.1f}%] {track} {date_str}: page found but no parseable data")
                else:
                    self.checkpoint.mark(track, date, 0)
                    log.debug(f"[{pct:.1f}%] {track} {date_str}: no racing")

                # Variable delay between pages
                await self.human.random_delay(1.5, 3.5)

        except KeyboardInterrupt:
            log.info("Interrupted — saving checkpoint")
        except Exception as e:
            log.error(f"Fatal: {e}", exc_info=True)
        finally:
            self.checkpoint.save()
            self._close_csv()
            await self._stop_browser()

            s = self.checkpoint.stats
            log.info(f"\n{'='*50}")
            log.info(f"DONE | Pages: {s['pages']:,} | Entries: {s['entries']:,} | "
                     f"PDFs: {s['pdfs']} | Errors: {s['errors']}")
            if self._career_stats_enabled:
                log.info(f"Career stats | cache hits: {self._career_hit} | "
                         f"misses: {self._career_miss} | unique horses: {len(self._career_cache)}")
            log.info(f"Output: {self.output_file}")
            log.info(f"{'='*50}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Scrape 10 years of horse racing data via Google → Equibase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Debug one day (visible browser)
  python3 scraper_v3.py --tracks KEE --start 2026-04-03 --end 2026-04-03 --visible

  # One month of Keeneland
  python3 scraper_v3.py --tracks KEE --start 2023-04-01 --end 2023-04-30

  # Top 5 tracks, full 10 years
  python3 scraper_v3.py --tracks KEE,CD,GP,SA,SAR --start 2016-01-01 --end 2026-04-04

  # All tracks, full decade (run in background)
  nohup python3 scraper_v3.py --start 2016-01-01 --end 2026-04-04 &

  # Resume after interruption
  python3 scraper_v3.py --resume
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

    args = p.parse_args()

    if args.list_tracks:
        for code, name in sorted(TRACKS.items()):
            print(f"  {code:<6} {name}")
        return

    if args.resume and not args.start:
        cp = Checkpoint(args.checkpoint)
        if not cp.done:
            p.error("No checkpoint found. Use --start and --end.")
        dates = [datetime.strptime(k.split("_")[1], "%Y%m%d") for k in cp.done]
        tracks_seen = list({k.split("_")[0] for k in cp.done})
        start_date, end_date = min(dates), max(dates) + timedelta(days=30)
        track_codes = tracks_seen
    else:
        if not args.start or not args.end:
            p.error("Use --start and --end (or --resume)")
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
        track_codes = args.tracks.split(",") if args.tracks else list(TRACKS.keys())

    # Generate targets — pre-filter by track meet calendar to skip non-race days
    targets = []
    skipped = 0
    d = start_date
    while d <= end_date:
        for t in track_codes:
            if is_race_day(t, d):
                targets.append((t, d))
            else:
                skipped += 1
        d += timedelta(days=1)

    # Shuffle to avoid hammering one track consecutively
    random.shuffle(targets)

    est_hours = len(targets) * 4.0 / 3600
    log.info(
        f"Targets: {len(targets):,} (filtered out {skipped:,} non-race days) "
        f"| Est: {est_hours:.1f}h | Tracks: {len(track_codes)}"
    )

    scraper = RaceScraper(
        output_file=args.output,
        headless=not args.visible,
        use_google=not args.no_google,
        cdp_port=args.cdp_port,
        checkpoint_path=args.checkpoint,
        career_stats=args.career_stats,
    )
    asyncio.run(scraper.run(targets))


if __name__ == "__main__":
    main()