"""
Shared constants and configuration for the horse racing scraper.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Set


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

CAREER_STAT_COLUMNS = ["num_past_starts", "num_past_wins", "num_past_seconds", "num_past_thirds"]
GENERAL_COLUMNS = [c for c in OUTPUT_COLUMNS if c not in CAREER_STAT_COLUMNS]
CAREER_CSV_COLUMNS = ["horse_name"] + CAREER_STAT_COLUMNS

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
# TRACK RACE-DAY CALENDAR — loaded from track_race_dates.json
# ═══════════════════════════════════════════════════════════════
# Populated at module load from a cache produced by fetch_all_track_dates.py,
# which pulls confirmed race dates directly from Equibase's
# eqbRaceChartCalendar.cfm endpoint. Each entry is a set of "YYYY-MM-DD"
# strings. Tracks missing from the cache fall through to "always try".

TRACK_RACE_DATES: Dict[str, Set[str]] = {}


def _load_track_race_dates():
    path = Path(__file__).resolve().parent.parent / "track_race_dates.json"
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
