"""
Multi-process launcher for distributed scraping (Approach 2A).

Partitions targets across N worker processes, each running the async-concurrent
scraper internally. Replaces the manual distributed_run.md approach.

Usage:
  # 2 processes, each with 4 concurrent browser contexts = 8 effective workers
  python -m scraper.launcher --workers 2 --concurrency 4 \
      --start 2024-01-01 --end 2024-12-31

  # 4 processes across all tracks
  python -m scraper.launcher --workers 4 --concurrency 4 \
      --start 2016-01-01 --end 2026-04-04 --no-google
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

from .config import TRACKS, is_race_day, OUTPUT_COLUMNS

log = logging.getLogger("launcher")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def generate_targets(start_date, end_date, track_codes) -> List[Tuple[str, str]]:
    """Generate (track_code, date_str) targets filtered by race-day calendar."""
    targets = []
    d = start_date
    while d <= end_date:
        for t in track_codes:
            if is_race_day(t, d):
                targets.append((t, d.strftime("%Y-%m-%d")))
        d += timedelta(days=1)
    return targets


def partition_targets(targets: List, n_workers: int) -> List[List]:
    """Round-robin partition targets into n roughly equal groups."""
    random.shuffle(targets)
    partitions = [[] for _ in range(n_workers)]
    for i, target in enumerate(targets):
        partitions[i % n_workers].append(target)
    return [p for p in partitions if p]  # drop empty


def merge_csv_files(input_files: List[str], output_file: str):
    """Merge multiple CSV files into one, deduplicating by header."""
    import csv
    seen_header = False
    with open(output_file, "w", newline="", encoding="utf-8") as out:
        writer = None
        for fpath in input_files:
            if not os.path.exists(fpath):
                log.warning(f"Worker output not found: {fpath}")
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if writer is None:
                    writer = csv.DictWriter(out, fieldnames=reader.fieldnames or OUTPUT_COLUMNS,
                                            extrasaction="ignore")
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
    log.info(f"Merged {len(input_files)} files -> {output_file}")


def main():
    p = argparse.ArgumentParser(
        description="Launch multiple scraper processes in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2 processes x 4 contexts each on one machine
  python -m scraper.launcher --workers 2 --concurrency 4 \\
      --start 2024-01-01 --end 2024-12-31

  # 4 processes, specific tracks
  python -m scraper.launcher --workers 4 --concurrency 4 \\
      --tracks KEE,CD,GP,SA --start 2016-01-01 --end 2026-04-04
        """,
    )
    p.add_argument("--workers", type=int, default=2,
                   help="Number of worker processes (default: 2)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Browser contexts per worker (default: 4)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--tracks", help="Comma-separated track codes (default: all)")
    p.add_argument("--output", default="historical_races.csv",
                   help="Final merged output file")
    p.add_argument("--no-google", action="store_true")
    p.add_argument("--career-stats", action="store_true")
    p.add_argument("--visible", action="store_true")
    p.add_argument("--cdp-port", type=int, default=0,
                   help="Chrome DevTools port (0 = don't use CDP, default: 0)")

    args = p.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    track_codes = args.tracks.split(",") if args.tracks else list(TRACKS.keys())

    # Generate all targets
    all_targets = generate_targets(start_date, end_date, track_codes)
    log.info(f"Total targets: {len(all_targets):,} across {len(track_codes)} tracks")

    # Partition across workers
    partitions = partition_targets(all_targets, args.workers)
    actual_workers = len(partitions)
    log.info(f"Launching {actual_workers} worker processes, {args.concurrency} contexts each")
    log.info(f"Effective parallelism: {actual_workers * args.concurrency} concurrent contexts")

    # Create work directory for temp files
    work_dir = Path(".launcher_work")
    work_dir.mkdir(exist_ok=True)

    # Write target files and launch workers
    processes = []
    worker_outputs = []
    for i, partition in enumerate(partitions):
        targets_file = work_dir / f"targets_worker_{i}.json"
        output_file = work_dir / f"races_worker_{i}.csv"
        checkpoint_file = work_dir / f"checkpoint_worker_{i}.json"
        worker_outputs.append(str(output_file))

        # Write targets as JSON
        with open(targets_file, "w") as f:
            json.dump(partition, f)

        log.info(f"Worker {i}: {len(partition):,} targets -> {output_file}")

        cmd = [
            sys.executable, "-m", "scraper",
            "--targets-file", str(targets_file),
            "--output", str(output_file),
            "--checkpoint", str(checkpoint_file),
            "--concurrency", str(args.concurrency),
        ]
        if args.no_google:
            cmd.append("--no-google")
        if args.career_stats:
            cmd.append("--career-stats")
        if args.visible:
            cmd.append("--visible")
        if args.cdp_port:
            cmd.extend(["--cdp-port", str(args.cdp_port)])

        proc = subprocess.Popen(cmd)
        processes.append((i, proc))

    # Wait for all workers
    log.info(f"All {actual_workers} workers launched. Waiting for completion...")
    start_time = time.time()

    failed = []
    for i, proc in processes:
        rc = proc.wait()
        elapsed = time.time() - start_time
        if rc == 0:
            log.info(f"Worker {i} completed successfully ({elapsed:.0f}s elapsed)")
        else:
            log.error(f"Worker {i} exited with code {rc} ({elapsed:.0f}s elapsed)")
            failed.append(i)

    total_elapsed = time.time() - start_time

    # Merge outputs
    log.info(f"All workers done in {total_elapsed:.0f}s. Merging outputs...")
    merge_csv_files(worker_outputs, args.output)

    # Summary
    log.info(f"\n{'='*50}")
    log.info(f"LAUNCHER DONE")
    log.info(f"  Workers: {actual_workers} ({len(failed)} failed)")
    log.info(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    log.info(f"  Output: {args.output}")
    if failed:
        log.warning(f"  Failed workers: {failed}")
        log.warning(f"  Re-run with --resume to retry failed targets")
    log.info(f"{'='*50}")


if __name__ == "__main__":
    main()
