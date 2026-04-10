"""
Horse racing data scraper package.

Modules:
  run      — Main scraper (RaceScraper + CLI)
  entries  — Pre-race entries scraper
  backfill — Career stats backfill
  config   — Shared constants (tracks, output schema)
"""

from .config import TRACKS, OUTPUT_COLUMNS
