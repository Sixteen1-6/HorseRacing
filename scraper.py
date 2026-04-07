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
    # Extended columns
    "margin_finish", "pos_1st_call", "pos_2nd_call", "pos_stretch",
    "margin_1st_call", "margin_2nd_call", "margin_stretch",
    "frac_1", "frac_2", "frac_3", "frac_4", "final_time_secs",
    "speed_figure_equibase", "claimed_price",
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
    async def random_delay(min_s=2.0, max_s=6.0):
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

    def _parse_chart_content(self, text: str, tables: list,
                             track_code: str, race_date: str) -> List[Dict]:
        """Parse extracted text and tables from a chart PDF."""
        entries = []

        # Split text into race blocks
        race_blocks = re.split(r"(?=(?:RACE|Race)\s*\d+)", text)

        for block in race_blocks:
            if not block.strip():
                continue

            # Extract race-level info
            race_num_m = self.RACE_HEADER.search(block)
            race_num = int(race_num_m.group(1)) if race_num_m else 0
            if race_num == 0:
                continue

            distance_m = self.DISTANCE_PATTERN.search(block) or self.DISTANCE_PATTERN_NOSPACE.search(block)
            surface_m = self.SURFACE_PATTERN.search(block) or self.SURFACE_PATTERN_NOSPACE.search(block)
            condition_m = self.CONDITION_PATTERN.search(block)
            purse_m = self.PURSE_PATTERN.search(block)
            race_type_m = self.RACE_TYPE_PATTERN.search(block)

            # Fractional and final times
            frac_m = self.FRACTIONAL_PATTERN.search(block)
            frac_times_raw = frac_m.group(1).strip().split() if frac_m else []
            final_m = self.FINAL_TIME_PATTERN.search(block)

            # Insert CamelCase spaces for distance: "FourAndOneHalf" -> "Four And One Half"
            distance_raw = distance_m.group(1).strip() if distance_m else ""
            if distance_raw and " " not in distance_raw:
                distance_raw = re.sub(r"([a-z])([A-Z])", r"\1 \2", distance_raw)

            race_info = {
                "race_number": race_num,
                "track_code": track_code,
                "track_name": TRACKS.get(track_code, track_code),
                "race_date": race_date,
                "distance": distance_raw,
                "surface": (surface_m.group(1)[0].upper() if surface_m else ""),
                "track_condition": (condition_m.group(1) if condition_m else ""),
                "purse": (purse_m.group(1).replace(",", "") if purse_m else ""),
                "race_type": re.sub(r"([a-z])([A-Z])", r"\1 \2", race_type_m.group(1)) if race_type_m else "",
                "frac_1": self._parse_time(frac_times_raw[0]) if len(frac_times_raw) > 0 else "",
                "frac_2": self._parse_time(frac_times_raw[1]) if len(frac_times_raw) > 1 else "",
                "frac_3": self._parse_time(frac_times_raw[2]) if len(frac_times_raw) > 2 else "",
                "frac_4": self._parse_time(frac_times_raw[3]) if len(frac_times_raw) > 3 else "",
                "final_time_secs": self._parse_time(final_m.group(1)) if final_m else "",
            }

            # Parse the results table for this race from the tables list
            race_entries = self._parse_results_from_tables(tables, race_info)
            if race_entries:
                entries.extend(race_entries)
            else:
                # Fallback: try to parse from text lines
                race_entries = self._parse_results_from_text(block, race_info)
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

        Equibase PDF lines look like:
          --- 3 Bledsoe(Rosario,Joel) 119 b 2 6 3Head 1Head 111/2 0.78* ins,split
          03/25/23KEE 5 WestSaratoga(Calleja,Andres) 112 b 3 5 51/2 5Head 41 42.86 midpack
        Format: LastRaced Pgm HorseName(Jockey) Wgt M/E PP Start 1/4 Str Fin Odds Comments
        """
        entries = []
        lines = block.split("\n")
        finish_pos = 0

        # Pattern: last-raced (--- or 18Feb232FG1 etc), pgm#, Name(Jockey), weight
        # Last raced can be: ---, or date+track like "18Feb232FG1", "4Mar2312GP6"
        horse_line_re = re.compile(
            r"(\S+)\s+"                             # Last raced (any non-space token)
            r"(\d{1,2})\s+"                         # Program number
            r"(\w[^(]+)\(([^)]+)\)\s+"              # HorseName(Jockey)
            r"(\d{2,3})\s+"                         # Weight
            r"(\S+)\s+"                             # M/E (medication/equipment: b, Lb, Lbf, --)
            r"(\d{1,2})\s+"                         # Post Position
            r"(\d{1,2})\s+"                         # Start position
            r"(.+)"                                 # Rest: running positions, odds, comments
        )

        for line in lines:
            m = horse_line_re.match(line.strip())
            if not m:
                continue

            finish_pos += 1
            last_raced, pgm, horse, jockey, weight, med_equip, pp, _start, rest = m.groups()

            # Parse the rest: running positions, finish, odds, comments
            # Rest looks like: "3Head 1Head 111/2 0.78* ins,split3/16,cleared"
            rest_parts = rest.split()
            odds = ""
            comment_parts = []
            positions = []

            for part in rest_parts:
                # Odds look like: 0.78* or 26.07 or *1.40
                if re.match(r"^\*?\d+\.\d+\*?$", part):
                    odds = part.replace("*", "")
                    # Everything after odds is comments
                    idx = rest_parts.index(part)
                    comment_parts = rest_parts[idx + 1:]
                    break
                else:
                    positions.append(part)

            entry = {**race_info}
            entry["horse_name"] = horse.strip()
            entry["program_num"] = pgm
            entry["post_position"] = pp
            entry["finish"] = str(finish_pos)
            entry["jockey"] = jockey.replace(",", ", ") if jockey else ""
            entry["weight"] = weight
            entry["dollar_odds"] = odds
            entry["medication"] = med_equip if med_equip != "--" else ""
            entry["comment"] = " ".join(comment_parts)
            # Running positions: typically 1/4, stretch, finish margin
            if len(positions) >= 1:
                entry["pos_1st_call"] = self._extract_pos(positions[0])
                entry["margin_1st_call"] = self._extract_margin_from_pos(positions[0])
            if len(positions) >= 2:
                entry["pos_stretch"] = self._extract_pos(positions[1])
                entry["margin_stretch"] = self._extract_margin_from_pos(positions[1])
            if len(positions) >= 3:
                entry["margin_finish"] = self._extract_margin_from_pos(positions[2])

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

    def __init__(self, output_file="historical_races.csv", headless=True, use_google=True, cdp_port=9222):
        self.output_file = output_file
        self.headless = headless
        self._use_google = use_google
        self._cdp_port = cdp_port
        self.checkpoint = Checkpoint()
        self.pdf_parser = ChartPDFParser()
        self.human = HumanBehavior()
        self.pw = None
        self.browser = None
        self.page = None
        self._csv_file = None
        self._csv_writer = None

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
                    log.info(f"Connected to your running Chrome on {url}!")
                    await self._prompt_login()
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
            await self.human.random_delay(2.0, 4.0)

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
        date_mmddyy = date.strftime("%m%d%y")

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

        # ── Attempt 1: All-races summary URL (free, best for scraping) ──
        summary_url = self.SUMMARY_URL.format(track=track_code, date=date_mmddyy)
        log.info(f"[{track_code} {date.strftime('%Y-%m-%d')}] Trying summary URL: {summary_url}")
        try:
            await self.page.goto(summary_url, wait_until="domcontentloaded", timeout=15000)
            await self.human.random_delay(2.0, 4.0)
            if await _check_page("Summary"):
                return True
        except Exception as e:
            log.info(f"Summary URL failed: {e}")

        # ── Attempt 2: Race card index URL (free, has links per race) ──
        racecard_url = self.RACECARD_INDEX_URL.format(track=track_code, date=date_mmddyy)
        log.info(f"Trying race card index: {racecard_url}")
        try:
            await self.page.goto(racecard_url, wait_until="domcontentloaded", timeout=15000)
            await self.human.random_delay(2.0, 4.0)
            if await _check_page("Race card index"):
                return True
        except Exception as e:
            log.info(f"Race card index URL failed: {e}")

        # ── Attempt 3: Premium chart embed (works for historical data) ──
        chart_embed_url = f"https://www.equibase.com/premium/chartEmb.cfm?track={track_code}&raceDate={date.strftime('%m/%d/%Y')}&cy=USA"
        log.info(f"Trying premium chart embed: {chart_embed_url}")
        try:
            await self.page.goto(chart_embed_url, wait_until="domcontentloaded", timeout=15000)
            await self.human.random_delay(2.0, 4.0)
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
            await self.human.random_delay(2.0, 4.0)
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
                        await self.human.random_delay(2.0, 4.0)
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

                    await self.human.random_delay(1.0, 2.0)

                    if consecutive_failures >= 3:
                        log.info(f"3 consecutive failures after race {race_num}, stopping")
                        break
                except Exception as e:
                    log.info(f"Race {race_num} chart download failed: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
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
                        await self.human.random_delay(2.0, 4.0)
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
                        await self.human.random_delay(2.0, 4.0)
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
                    await self.human.random_delay(2.0, 4.0)
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
                await self.human.random_delay(2.5, 6.0)

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

    args = p.parse_args()

    if args.list_tracks:
        for code, name in sorted(TRACKS.items()):
            print(f"  {code:<6} {name}")
        return

    if args.resume and not args.start:
        cp = Checkpoint()
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

    # Generate targets
    targets = []
    d = start_date
    while d <= end_date:
        for t in track_codes:
            targets.append((t, d))
        d += timedelta(days=1)

    # Shuffle to avoid hammering one track consecutively
    random.shuffle(targets)

    est_hours = len(targets) * 4.0 / 3600
    log.info(f"Targets: {len(targets):,} | Est: {est_hours:.1f}h | Tracks: {len(track_codes)}")

    scraper = RaceScraper(
        output_file=args.output,
        headless=not args.visible,
        use_google=not args.no_google,
        cdp_port=args.cdp_port,
    )
    asyncio.run(scraper.run(targets))


if __name__ == "__main__":
    main()