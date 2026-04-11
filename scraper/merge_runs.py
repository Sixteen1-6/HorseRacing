"""Merge the 8 distributed scrape CSVs into one file, then (separately)
combine with the 2023 hackathon dataset for training.

Usage:
    python merge_distributed_runs.py

Inputs expected in the current directory:
    races_pc1a.csv ... races_pc4b.csv   (the 8 scrape outputs)
    hackdata_tmp/hackathon_data/all_tracks_hackathon.csv   (2023 training data)

Outputs:
    races_2024_2026_merged.csv           -- all 8 scrape files combined & deduped
    races_2023_2026_combined.csv         -- scraped + 2023 hackathon data combined
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


# Defaults — used when no CLI args are given
DEFAULT_SCRAPE_PATTERN = "races_pc*.csv"
DEFAULT_HACKATHON_PATH = Path("hackdata_tmp/hackathon_data/all_tracks_hackathon.csv")
DEFAULT_MERGED_PATH = Path("races_2024_2026_merged.csv")
DEFAULT_COMBINED_PATH = Path("races_2023_2026_combined.csv")


def load_scrape_files(file_list) -> pd.DataFrame:
    frames = []
    for name in file_list:
        p = Path(name)
        if not p.exists():
            print(f"  [skip] {name} not found")
            continue
        df = pd.read_csv(p, low_memory=False)
        print(f"  [{name}] {len(df):,} rows")
        df["__source"] = name
        frames.append(df)
    if not frames:
        sys.exit("No scrape files found. Pass --input with a glob pattern or file list.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\n  merged raw: {len(combined):,} rows")
    return combined


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    # Dedupe key: same horse running the same race at the same track on the same day
    # is a duplicate. The boundary dates of adjacent chunks should never overlap
    # because we split inclusively, but be defensive.
    key_cols = [c for c in ("track_code", "race_date", "race_number", "horse_name")
                if c in df.columns]
    if not key_cols:
        print("  WARNING: cannot dedupe — missing key columns")
        return df
    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="first")
    after = len(df)
    if before != after:
        print(f"  deduped: {before:,} -> {after:,}  ({before-after:,} removed)")
    return df


def merge_scrape(file_list, merged_path) -> pd.DataFrame:
    print(f"=== Merging {len(file_list)} scrape files ===")
    df = load_scrape_files(file_list)
    df = dedupe(df)
    df = df.drop(columns=["__source"], errors="ignore")
    df.to_csv(merged_path, index=False)
    print(f"\n  wrote {merged_path} ({len(df):,} rows)")
    return df


def combine_with_hackathon(scraped: pd.DataFrame, hackathon_path, combined_path) -> None:
    print("\n=== Combining with 2023 hackathon data ===")
    if not hackathon_path.exists():
        print(f"  [skip] {hackathon_path} not found — final combine skipped")
        return
    hk = pd.read_csv(hackathon_path, low_memory=False)
    print(f"  hackathon rows: {len(hk):,}")
    print(f"  scraped  rows: {len(scraped):,}")

    # Column alignment — take the intersection so each row has a consistent
    # schema; the rest are dropped. Anything in one but not the other becomes
    # a per-row NaN if kept, which breaks LightGBM categorical handling.
    scraped_cols = set(scraped.columns)
    hk_cols = set(hk.columns)
    shared = sorted(scraped_cols & hk_cols)
    only_scrape = sorted(scraped_cols - hk_cols)
    only_hk = sorted(hk_cols - scraped_cols)

    print(f"\n  shared columns: {len(shared)}")
    print(f"  only in scrape: {len(only_scrape)} -> {only_scrape[:10]}{'...' if len(only_scrape)>10 else ''}")
    print(f"  only in hack:   {len(only_hk)} -> {only_hk[:10]}{'...' if len(only_hk)>10 else ''}")

    combined = pd.concat([hk[shared], scraped[shared]], ignore_index=True, sort=False)
    combined = dedupe(combined)
    combined.to_csv(combined_path, index=False)
    print(f"\n  wrote {combined_path} ({len(combined):,} rows)")


def main():
    p = argparse.ArgumentParser(description="Merge distributed scrape CSVs")
    p.add_argument("--input", default=DEFAULT_SCRAPE_PATTERN,
                   help="Glob pattern or comma-separated file list (default: races_pc*.csv)")
    p.add_argument("--output", type=Path, default=DEFAULT_MERGED_PATH,
                   help="Merged output file")
    p.add_argument("--hackathon", type=Path, default=DEFAULT_HACKATHON_PATH,
                   help="Hackathon CSV to combine with")
    p.add_argument("--combined", type=Path, default=DEFAULT_COMBINED_PATH,
                   help="Combined output file")
    args = p.parse_args()

    # Resolve input files from glob or comma-separated list
    if "," in args.input:
        file_list = [f.strip() for f in args.input.split(",")]
    else:
        file_list = sorted(glob.glob(args.input))
        if not file_list:
            sys.exit(f"No files matched pattern: {args.input}")

    scraped = merge_scrape(file_list, args.output)
    combine_with_hackathon(scraped, args.hackathon, args.combined)
    print("\ndone.")


if __name__ == "__main__":
    main()
