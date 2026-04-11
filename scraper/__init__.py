"""
Horse racing data scraper package.

Modules:
  run         — Main scraper (RaceScraper + CLI)
  entries     — Pre-race entries scraper
  backfill    — Career stats backfill
  config      — Shared constants (tracks, output schema, timeouts)
  parsing     — Shared parsing utilities (distance, odds, time, table columns)
  page_utils  — Shared browser utilities (ad blocking, bot detection, career stats)
"""

from .config import TRACKS, OUTPUT_COLUMNS
