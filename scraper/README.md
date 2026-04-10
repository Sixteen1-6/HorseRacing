# Scraper Package

Web scraper for collecting historical and pre-race horse racing data from Equibase. Supports concurrent scraping with multiple browser contexts and distributed multi-process execution.

## Setup

```bash
# Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install browser binaries
playwright install chromium
sudo playwright install-deps chromium
```

## Quick Start

```bash
# Scrape one week of data (all tracks, 4 concurrent contexts, verbose)
python -m scraper --start 2026-04-03 --end 2026-04-09 --concurrency 4 --no-google -v

# Scrape specific tracks
python -m scraper --tracks KEE,CD,GP --start 2024-01-01 --end 2024-03-31 --concurrency 4

# Include career stats (starts, wins, 2nds, 3rds per horse)
python -m scraper --start 2026-04-03 --end 2026-04-09 --concurrency 4 --career-stats

# Resume after interruption
python -m scraper --resume

# Save output to log file while watching live
python -m scraper --start 2026-04-03 --end 2026-04-09 --concurrency 4 -v 2>&1 | tee scrape.log
```

## Modules

| File | Purpose |
|---|---|
| `run.py` | Main scraper — `RaceScraper` class + CLI entry point |
| `launcher.py` | Multi-process launcher for distributed scraping |
| `entries.py` | Pre-race entries scraper (upcoming races) |
| `pdf_parser.py` | Equibase chart PDF parser (pdfplumber) |
| `human_behavior.py` | Anti-detection delays, scrolling, mouse movement |
| `config.py` | Shared constants — tracks, output schema, race calendar |
| `checkpoint.py` | Checkpoint/resume support |
| `backfill.py` | Career stats backfill for existing datasets |
| `merge_runs.py` | Merge CSV outputs from distributed workers |
| `debug_equibase.py` | Diagnostic tool for Equibase page structure |

## CLI Reference

```
python -m scraper [options]

Required (unless --resume):
  --start DATE          Start date YYYY-MM-DD
  --end DATE            End date YYYY-MM-DD

Options:
  --tracks CODES        Comma-separated track codes (default: all 36 tracks)
  --output FILE         Output CSV path (default: historical_races.csv)
  --concurrency N       Browser contexts to run in parallel (default: 1, recommended: 4)
  --career-stats        Fetch career stats per horse from Equibase profiles
  --no-google           Skip Google search, go direct to Equibase URLs
  --cdp-port PORT       Chrome DevTools port (default: 9222, 0 to disable)
  --checkpoint FILE     Checkpoint file path for resume support
  --targets-file FILE   JSON target list (used by launcher)
  --resume              Resume from last checkpoint
  --visible             Show browser window (for debugging)
  -v, --verbose         Detailed output: horse rosters, career cache hits,
                        race metadata, navigation strategy, worker IDs, timing
  --list-tracks         Print all supported track codes
```

## Concurrency

The scraper supports two layers of parallelism:

**Layer 1 — Async contexts (within one process):**
Multiple browser contexts in a single event loop. While one context waits on network I/O or anti-detection delays, others do useful work.

```bash
# 4 concurrent contexts (~3-4x speedup)
python -m scraper --start 2024-01-01 --end 2024-12-31 --concurrency 4
```

**Layer 2 — Multi-process (across cores/machines):**
The launcher partitions targets across N worker processes, each running its own concurrent scraper.

```bash
# 2 processes x 4 contexts = 8 effective workers
python -m scraper.launcher --workers 2 --concurrency 4 \
    --start 2024-01-01 --end 2024-12-31
```

| Config | Est. speed |
|---|---|
| `--concurrency 1` (default) | 1x baseline |
| `--concurrency 4` | ~3-4x |
| `--workers 2 --concurrency 4` | ~6-8x |
| `--workers 2 --concurrency 4` on 4 PCs | ~24-32x |

## Chrome Debug Port

For premium Equibase content, connect the scraper to your signed-in Chrome:

```powershell
# Windows PowerShell — start Chrome with debug port
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
```

Then run the scraper normally — it auto-connects to port 9222. If Chrome isn't running, the scraper will attempt to auto-launch it, then fall back to headless Playwright.

## Verbose Output (`-v`)

With `--verbose`, the scraper logs:

- **Worker IDs** `[W1]`-`[W4]` showing which concurrent context produced each line
- **Per race day** `[START]`/`[DONE]`/`[SKIP]` with timing
- **Navigation strategy** used (chart embed, chart index, Google)
- **Race metadata** (type, purse, surface, track condition)
- **Horse rosters** per race
- **Career cache** hits/misses per horse with stats
- **Progress reports** every 60s with ETA and throughput

## Output Schema

The scraper outputs CSV with these columns:

```
race_number, race_type, purse, distance, distance_unit, course, surface,
track_condition, weather, post_time, win_time, horse_name, breed, weight,
age, sex, medication, program_num, post_position, finish, comment,
jockey, trainer, owner, last_race_track, last_race_date, last_race_number,
last_race_finish, track_code, track_name, race_date, dollar_odds,
num_past_starts, num_past_wins, num_past_seconds, num_past_thirds
```

## Supported Tracks

36 North American tracks including KEE (Keeneland), CD (Churchill Downs), GP (Gulfstream Park), SA (Santa Anita), SAR (Saratoga), BEL (Belmont), DMR (Del Mar), and more. Run `python -m scraper --list-tracks` for the full list.
