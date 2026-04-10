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

import sys
from pathlib import Path

import pandas as pd


SCRAPE_FILES = [
    "races_pc1a.csv", "races_pc1b.csv",
    "races_pc2a.csv", "races_pc2b.csv",
    "races_pc3a.csv", "races_pc3b.csv",
    "races_pc4a.csv", "races_pc4b.csv",
]

HACKATHON_PATH = Path("hackdata_tmp/hackathon_data/all_tracks_hackathon.csv")
MERGED_PATH = Path("races_2024_2026_merged.csv")
COMBINED_PATH = Path("races_2023_2026_combined.csv")


def load_scrape_files() -> pd.DataFrame:
    frames = []
    for name in SCRAPE_FILES:
        p = Path(name)
        if not p.exists():
            print(f"  [skip] {name} not found")
            continue
        df = pd.read_csv(p, low_memory=False)
        print(f"  [{name}] {len(df):,} rows")
        df["__source"] = name
        frames.append(df)
    if not frames:
        sys.exit("No scrape files found. Copy races_pc*.csv files here first.")
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


def merge_scrape() -> pd.DataFrame:
    print("=== Merging 8 scrape files ===")
    df = load_scrape_files()
    df = dedupe(df)
    df = df.drop(columns=["__source"], errors="ignore")
    df.to_csv(MERGED_PATH, index=False)
    print(f"\n  wrote {MERGED_PATH} ({len(df):,} rows)")
    return df


def combine_with_hackathon(scraped: pd.DataFrame) -> None:
    print("\n=== Combining with 2023 hackathon data ===")
    if not HACKATHON_PATH.exists():
        print(f"  [skip] {HACKATHON_PATH} not found — final combine skipped")
        return
    hk = pd.read_csv(HACKATHON_PATH, low_memory=False)
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
    combined.to_csv(COMBINED_PATH, index=False)
    print(f"\n  wrote {COMBINED_PATH} ({len(combined):,} rows)")


def main():
    scraped = merge_scrape()
    combine_with_hackathon(scraped)
    print("\ndone.")


if __name__ == "__main__":
    main()
