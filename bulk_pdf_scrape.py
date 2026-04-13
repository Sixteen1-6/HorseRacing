"""
Bulk PDF downloader + parser for Equibase chart PDFs.

Downloads full-card PDFs directly via HTTP (no browser needed) and parses
them with ChartPDFParser. Much faster than the Playwright-based scraper.

Features:
  - Checkpoint/resume: interrupted runs pick up where they left off
  - Per-worker logging with worker IDs
  - Live progress with ETA, rates, per-track stats
  - Retry logic for transient failures
  - Validation: flags tracks with suspiciously low entry counts

Usage:
  # Download 2014-2021 + 2023, 12 threads, save to Z:
  python bulk_pdf_scrape.py --years 2014-2021,2023 --threads 12 --output Z:/bulk_scrape_2014_2023.csv

  # Resume interrupted run
  python bulk_pdf_scrape.py --years 2014-2021,2023 --threads 12 --output Z:/bulk_scrape_2014_2023.csv --resume

  # Specific tracks only
  python bulk_pdf_scrape.py --years 2016-2020 --tracks KEE,CD,GP --threads 4
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock, local

from scraper.config import OUTPUT_COLUMNS, TRACKS
from scraper.pdf_parser import ChartPDFParser

# ── Logging ────────────────────────────────────────────────────
log = logging.getLogger("bulk_scrape")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bulk_scrape.log", encoding="utf-8"),
    ],
)

# ── Constants ──────────────────────────────────────────────────
PDF_URL = "https://www.equibase.com/static/chart/{year}/usa/{track}/{date}-usa-{track}-a-d.standard.pdf"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds between retries
REQUEST_INTERVAL = 0.3  # minimum seconds between requests (global throttle)


# ── Checkpoint ─────────────────────────────────────────────────
class BulkCheckpoint:
    """Track completed targets for resume support."""

    def __init__(self, path):
        self.path = path
        self.done = set()
        self.failed = set()
        self.stats = {"downloaded": 0, "parsed": 0, "entries": 0,
                      "errors": 0, "skipped": 0}
        self._lock = Lock()
        self._load()

    def key(self, track, date_str):
        return f"{track}_{date_str}"

    def is_done(self, track, date_str):
        return self.key(track, date_str) in self.done

    def mark_done(self, track, date_str, n_entries=0):
        with self._lock:
            k = self.key(track, date_str)
            self.done.add(k)
            self.failed.discard(k)
            self.stats["parsed"] += 1
            self.stats["entries"] += n_entries
            if len(self.done) % 50 == 0:
                self._save()

    def mark_downloaded(self):
        with self._lock:
            self.stats["downloaded"] += 1

    def mark_skipped(self):
        with self._lock:
            self.stats["skipped"] += 1

    def mark_failed(self, track, date_str):
        with self._lock:
            k = self.key(track, date_str)
            if k not in self.done:
                self.failed.add(k)
            self.stats["errors"] += 1

    def save(self):
        with self._lock:
            self._save()

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "done": list(self.done),
                "failed": list(self.failed),
                "stats": self.stats,
            }, f)
        os.replace(tmp, self.path)

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                d = json.load(f)
            self.done = set(d.get("done", []))
            self.failed = set(d.get("failed", []))
            self.stats = d.get("stats", self.stats)
            log.info(f"Checkpoint loaded: {len(self.done):,} done, "
                     f"{len(self.failed):,} failed, "
                     f"{self.stats['entries']:,} entries")


# ── Per-thread parser (thread-local to avoid contention) ──────
_thread_local = local()

# ── Global rate limiter ───────────────────────────────────────
_throttle_lock = Lock()
_last_request_time = 0.0


def _throttle():
    """Ensure minimum interval between HTTP requests across all threads."""
    global _last_request_time
    with _throttle_lock:
        now = time.time()
        wait = REQUEST_INTERVAL - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.time()


def _get_parser():
    if not hasattr(_thread_local, "parser"):
        _thread_local.parser = ChartPDFParser()
    return _thread_local.parser


# ── Core download + parse ─────────────────────────────────────
def download_and_parse(track_code, date_str, writer, write_lock, checkpoint, worker_id):
    """Download one full-card PDF, parse it, write entries to CSV."""
    year = date_str[:4]
    track_lower = track_code.lower()
    date_compact = date_str.replace("-", "")
    url = PDF_URL.format(year=year, track=track_lower, date=date_compact)
    tag = f"[W{worker_id:02d}]"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _throttle()
            req = urllib.request.Request(url, headers=HEADERS)
            resp = urllib.request.urlopen(req, timeout=20)
            pdf_bytes = resp.read()

            # Validate it's actually a PDF
            if len(pdf_bytes) < 1000 or not pdf_bytes[:5].startswith(b"%PDF"):
                log.debug(f"{tag} {track_code} {date_str}: not a PDF ({len(pdf_bytes)} bytes)")
                checkpoint.mark_skipped()
                checkpoint.mark_done(track_code, date_str, 0)
                return 0

            checkpoint.mark_downloaded()

            # Parse
            parser = _get_parser()
            entries = parser.parse_pdf_bytes(pdf_bytes, track_code, date_str)

            if not entries:
                log.debug(f"{tag} {track_code} {date_str}: PDF parsed but 0 entries")
                checkpoint.mark_done(track_code, date_str, 0)
                return 0

            # Write to CSV
            with write_lock:
                for entry in entries:
                    row = [str(entry.get(col, "")) for col in OUTPUT_COLUMNS]
                    writer.writerow(row)

            n = len(entries)
            races = len(set(e.get("race_number") for e in entries))
            log.debug(f"{tag} {track_code} {date_str}: {races} races, {n} entries")
            checkpoint.mark_done(track_code, date_str, n)
            return n

        except urllib.error.HTTPError as e:
            if e.code == 404:
                log.debug(f"{tag} {track_code} {date_str}: 404")
                checkpoint.mark_skipped()
                checkpoint.mark_done(track_code, date_str, 0)
                return 0
            if attempt < MAX_RETRIES:
                log.warning(f"{tag} {track_code} {date_str}: HTTP {e.code}, "
                            f"retry {attempt}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY * attempt)
            else:
                log.error(f"{tag} {track_code} {date_str}: HTTP {e.code} after "
                          f"{MAX_RETRIES} retries")
                checkpoint.mark_failed(track_code, date_str)
                return 0

        except Exception as e:
            if attempt < MAX_RETRIES:
                log.warning(f"{tag} {track_code} {date_str}: {e}, "
                            f"retry {attempt}/{MAX_RETRIES}")
                time.sleep(RETRY_DELAY * attempt)
            else:
                log.error(f"{tag} {track_code} {date_str}: FAILED after "
                          f"{MAX_RETRIES} retries: {e}")
                checkpoint.mark_failed(track_code, date_str)
                return 0

    return 0


# ── Year parser ───────────────────────────────────────────────
def parse_years(raw):
    years = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            years.extend(range(int(a), int(b) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


# ── Main ──────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Bulk download & parse Equibase chart PDFs")
    p.add_argument("--years", required=True,
                   help="Years to download, e.g. '2014-2021,2023'")
    p.add_argument("--tracks", default=None,
                   help="Comma-separated track codes (default: all)")
    p.add_argument("--threads", type=int, default=8,
                   help="Download threads (default: 8)")
    p.add_argument("--output", default=None,
                   help="Output CSV path (default: bulk_scrape_YEARS.csv)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint file (default: <output>.checkpoint.json)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Show per-PDF debug output")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger("bulk_scrape").setLevel(logging.DEBUG)

    years = parse_years(args.years)
    track_list = args.tracks.split(",") if args.tracks else sorted(TRACKS.keys())
    output_path = args.output or f"bulk_scrape_{years[0]}_{years[-1]}.csv"
    ckpt_path = args.checkpoint or f"{output_path}.checkpoint.json"

    # ── Load calendar ──
    cal_path = Path("track_race_dates.json")
    if not cal_path.exists():
        log.error("track_race_dates.json not found. Run fetch_track_dates first.")
        sys.exit(1)

    with open(cal_path) as f:
        cal = json.load(f)["per_track_per_year"]

    # ── Build work queue ──
    all_tasks = []
    for track in track_list:
        if track not in cal:
            continue
        for yr in years:
            yr_str = str(yr)
            if yr_str not in cal[track]:
                continue
            for date_str in cal[track][yr_str]:
                all_tasks.append((track, date_str))

    # ── Checkpoint ──
    checkpoint = BulkCheckpoint(ckpt_path)

    if args.resume:
        remaining = [(t, d) for t, d in all_tasks if not checkpoint.is_done(t, d)]
        skipped = len(all_tasks) - len(remaining)
        log.info(f"Resuming: {skipped:,} already done, {len(remaining):,} remaining")
    else:
        remaining = all_tasks

    if not remaining:
        log.info("Nothing to do — all targets already completed!")
        return

    # ── Summary ──
    log.info("=" * 60)
    log.info("BULK PDF SCRAPER")
    log.info("=" * 60)
    log.info(f"  Tracks:      {len(track_list)}")
    log.info(f"  Years:       {years[0]}-{years[-1]}")
    log.info(f"  Targets:     {len(remaining):,} track-dates")
    log.info(f"  Threads:     {args.threads}")
    log.info(f"  Output:      {output_path}")
    log.info(f"  Checkpoint:  {ckpt_path}")
    est_minutes = len(remaining) / args.threads / 60
    log.info(f"  Est. time:   {est_minutes:.0f} min ({est_minutes/60:.1f}h)")
    log.info("=" * 60)

    # ── Open CSV (append if resuming, write header if new) ──
    file_exists = os.path.exists(output_path) and args.resume
    csv_file = open(output_path, "a" if file_exists else "w",
                    newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(OUTPUT_COLUMNS)
        csv_file.flush()

    write_lock = Lock()
    start_time = time.time()
    completed = 0

    # ── Per-track stats for validation ──
    track_entries = {}

    # ── Worker pool ──
    try:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures = {}
            for i, (track, date_str) in enumerate(remaining):
                worker_id = i % args.threads
                fut = pool.submit(download_and_parse, track, date_str,
                                  writer, write_lock, checkpoint, worker_id)
                futures[fut] = (track, date_str)

            for fut in as_completed(futures):
                completed += 1
                track, date_str = futures[fut]

                try:
                    n = fut.result()
                    track_entries[track] = track_entries.get(track, 0) + n
                except Exception as e:
                    log.error(f"Unexpected error for {track} {date_str}: {e}")

                # ── Progress reporting ──
                if completed % 100 == 0 or completed == len(remaining):
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta_sec = (len(remaining) - completed) / rate if rate > 0 else 0
                    s = checkpoint.stats

                    if eta_sec < 60:
                        eta = f"{eta_sec:.0f}s"
                    elif eta_sec < 3600:
                        eta = f"{eta_sec / 60:.0f}m"
                    else:
                        eta = f"{eta_sec / 3600:.1f}h"

                    log.info(
                        f"[PROGRESS] {100*completed/len(remaining):.1f}% "
                        f"({completed:,}/{len(remaining):,}) | "
                        f"{s['entries']:,} entries | "
                        f"{s['downloaded']:,} PDFs | "
                        f"{s['errors']} errors | "
                        f"{s['skipped']} skipped | "
                        f"{rate:.1f}/sec | "
                        f"ETA: {eta}"
                    )

                # ── Periodic CSV flush ──
                if completed % 50 == 0:
                    csv_file.flush()

    except KeyboardInterrupt:
        log.warning("Interrupted! Saving checkpoint...")
    finally:
        csv_file.flush()
        csv_file.close()
        checkpoint.save()

    # ── Final summary ──
    elapsed = time.time() - start_time
    s = checkpoint.stats

    log.info("")
    log.info("=" * 60)
    log.info("SCRAPE COMPLETE")
    log.info("=" * 60)
    log.info(f"  Time:        {elapsed/60:.1f} min ({elapsed/3600:.1f}h)")
    log.info(f"  Downloaded:  {s['downloaded']:,} PDFs")
    log.info(f"  Parsed:      {s['parsed']:,} race cards")
    log.info(f"  Entries:     {s['entries']:,} rows")
    log.info(f"  Skipped:     {s['skipped']:,} (404 / not PDF)")
    log.info(f"  Errors:      {s['errors']:,}")
    log.info(f"  Output:      {output_path}")
    log.info(f"  Checkpoint:  {ckpt_path}")

    # ── Validation: flag tracks with suspiciously low counts ──
    if track_entries:
        log.info("")
        log.info("Per-track entry counts:")
        for track in sorted(track_entries):
            count = track_entries[track]
            name = TRACKS.get(track, track)
            # Estimate expected: ~9 horses * ~10 races * years of dates
            track_dates = sum(
                len(cal.get(track, {}).get(str(y), []))
                for y in years
            )
            expected = track_dates * 9 * 8  # rough: 9 horses, 8 races avg
            ratio = count / expected if expected > 0 else 0
            flag = " !! LOW" if ratio < 0.3 and count > 0 else ""
            flag = flag or (" XX EMPTY" if count == 0 and track_dates > 10 else "")
            log.info(f"    {track:5s} {name:30s} {count:>8,} entries "
                     f"({track_dates} dates){flag}")

    # ── Report failed targets ──
    if checkpoint.failed:
        log.warning(f"\n{len(checkpoint.failed)} failed targets. "
                    f"Re-run with --resume to retry.")


if __name__ == "__main__":
    main()
