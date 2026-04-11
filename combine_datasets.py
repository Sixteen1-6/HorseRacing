"""
Combine multiple scraped CSV files into one historical_races_general.csv,
then run the date-aware career stats merge.

Usage:
  python3 combine_datasets.py
  python3 combine_datasets.py --run-merge
"""

import argparse
import csv
import os
import sys

# The 32 raw race data columns (no career stats)
RAW_COLUMNS = [
    "race_number", "race_type", "purse", "distance", "distance_unit",
    "course", "surface", "track_condition", "weather", "post_time",
    "win_time", "horse_name", "breed", "weight", "age", "sex",
    "medication", "program_num", "post_position", "finish", "comment",
    "jockey", "trainer", "owner", "last_race_track", "last_race_date",
    "last_race_number", "last_race_finish", "track_code", "track_name",
    "race_date", "dollar_odds",
]

INPUT_FILES = [
    "all_tracks_hackathon.csv",        # 2023 data (has career stats — will be dropped)
    "historical_races_general.csv",     # 2022 - mid 2024 (this machine)
    "races_2024_2026_general.csv",      # mid 2024 - 2026 (other machine)
]

OUTPUT_FILE = "historical_races_general_combined.csv"


def main():
    p = argparse.ArgumentParser(description="Combine scraped CSV files and deduplicate")
    p.add_argument("--run-merge", action="store_true",
                   help="After combining, run career stats merge via scraper")
    p.add_argument("--output", default=OUTPUT_FILE)
    args = p.parse_args()

    seen = set()
    total = 0
    dupes = 0

    with open(args.output, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for path in INPUT_FILES:
            if not os.path.exists(path):
                print(f"  SKIP: {path} not found")
                continue

            count = 0
            file_dupes = 0
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Dedup key: same horse in same race
                    key = (
                        (row.get("horse_name") or "").strip().lower(),
                        (row.get("track_code") or "").strip(),
                        (row.get("race_date") or "").strip(),
                        (row.get("race_number") or "").strip(),
                    )
                    if key in seen:
                        file_dupes += 1
                        continue
                    seen.add(key)

                    # Write only raw columns (strips career stats if present)
                    writer.writerow(row)
                    count += 1

            total += count
            dupes += file_dupes
            print(f"  {path}: {count:,} rows added, {file_dupes:,} duplicates skipped")

    print(f"\nCombined: {total:,} rows -> {args.output}")
    if dupes:
        print(f"Duplicates removed: {dupes:,}")

    if args.run_merge:
        print("\nRunning career stats merge...")
        # Copy combined file to the expected location
        if args.output != "historical_races_general.csv":
            import shutil
            shutil.copy(args.output, "historical_races_general.csv")
            print(f"  Copied {args.output} -> historical_races_general.csv")

        # Update meta file so scraper skips Phase 1
        import json
        from datetime import datetime
        meta = {
            "start_date": "2022-01-01",
            "end_date": "2026-04-10",
            "scraped_at": datetime.now().isoformat(),
            "entry_count": total,
        }
        with open("historical_races_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        os.system("python3 -m scraper --start 2022-01-01 --end 2026-04-10 --concurrency 1")


if __name__ == "__main__":
    main()
