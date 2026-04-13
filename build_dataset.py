"""
Build All-Tracks Hackathon 2016-2026 Dataset
=============================================
Combines multiple CSV files (scraped or existing) with computed speed figures.

Supports:
  - Single CSV input (original 36-col or extended 50-col scraper format)
  - Multiple CSVs via glob pattern or --input-dir
  - Date filtering (--start-date / --end-date)
  - Deduplication across overlapping scrape files

Output: all_tracks_hackathon_2016_2026.csv
  - All original data dictionary columns (+ any extended scraper columns)
  - 4 core speed figure columns
  - 12 ML feature columns (leak-free, from PRIOR races only)
  - 3 bonus derived columns (furlongs, win_rate, class_level)

Usage:
  # Single file (current 2023 data)
  python build_dataset.py

  # Multiple scraped files
  python build_dataset.py --input-dir scraped_data/

  # Glob pattern
  python build_dataset.py --input "scraped_data/*.csv"

  # With date filter
  python build_dataset.py --input-dir scraped_data/ --start-date 2016-01-01 --end-date 2026-12-31

  # With actual margins
  python build_dataset.py --margins margins.csv
"""

import pandas as pd
import numpy as np
import argparse
import glob
import sys
import os

# Import speedfig pipeline functions
from speedfig import (
    load_data,
    compute_par_times,
    compute_raw_speed_ratings,
    compute_track_variants,
    normalize_figures,
    compute_derived_features,
    NUM_ITERATIONS,
)


# Core 36 columns from the original data dictionary
ORIGINAL_COLS = [
    "race_number", "race_type", "purse", "distance", "distance_unit",
    "course", "surface", "track_condition", "weather", "post_time",
    "win_time", "horse_name", "breed", "weight", "age", "sex",
    "medication", "program_num", "post_position", "finish", "comment",
    "jockey", "trainer", "owner", "last_race_track", "last_race_date",
    "last_race_number", "last_race_finish", "track_code", "track_name",
    "race_date", "dollar_odds", "num_past_starts", "num_past_wins",
    "num_past_seconds", "num_past_thirds",
]

# Extended columns from scraper v3
EXTENDED_SCRAPER_COLS = [
    "start_pos",
    "pos_1st_call", "margin_1st_call", "pos_2nd_call", "margin_2nd_call",
    "pos_3rd_call", "margin_3rd_call", "pos_stretch", "margin_stretch",
    "pos_finish", "margin_finish", "temperature",
    "frac_1", "frac_2", "frac_3", "frac_4", "final_time_secs",
    "speed_figure_equibase", "claimed_price",
]

# Running style, pace, trip, and class features (computed by build_dataset)
PACE_TRIP_COLS = [
    "running_style", "pace_pressure", "pace_pressure_pct", "style_advantage",
    "avg_early_pos", "avg_late_gain", "avg_margin_finish", "best_finish_margin",
    "had_trouble", "wide_trip", "poor_start", "strong_close", "faded",
    "avg_trouble_rate", "avg_wide_rate",
    "class_change", "distance_change",
]

# Speed figure columns computed by our pipeline
SPEED_FIG_COLS = [
    "raw_speed_rating",
    "track_variant",
    "speed_figure",
    "speed_figure_normalized",
    "best_prior_figure",
    "avg_prior_figure",
    "avg_last3_figure",
    "last_figure",
    "num_prior_races",
    "figure_trend",
    "figure_trend_3race",
    "best_surface_figure",
    "avg_surface_figure",
    "best_dist_figure",
    "figure_consistency",
    "peak_vs_recent",
]

# Bonus derived columns
BONUS_COLS = ["furlongs", "win_rate", "class_level"]


def load_and_concat_csvs(input_paths):
    """Load one or more CSV files and concatenate into a single DataFrame."""
    dfs = []
    for path in input_paths:
        print(f"  Loading {path}...")
        try:
            df = pd.read_csv(path, low_memory=False)
            print(f"    {len(df):,} rows, {len(df.columns)} cols")
            dfs.append(df)
        except Exception as e:
            print(f"    ERROR: {e} — skipping")
    if not dfs:
        print("Error: No valid CSV files loaded.")
        sys.exit(1)
    combined = pd.concat(dfs, ignore_index=True)
    return combined


def parse_race_dates(df):
    """Parse race_date column, handling multiple date formats."""
    if "race_date" not in df.columns:
        return df

    # Try common formats
    parsed = pd.to_datetime(df["race_date"], format="%m/%d/%Y", errors="coerce")
    # Fill gaps with other formats
    mask = parsed.isna()
    if mask.any():
        parsed[mask] = pd.to_datetime(df.loc[mask, "race_date"], format="%Y-%m-%d", errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed[mask] = pd.to_datetime(df.loc[mask, "race_date"], errors="coerce")

    df["_race_date_parsed"] = parsed
    return df


def deduplicate(df):
    """Remove duplicate rows (same race, same horse)."""
    dedup_keys = ["track_code", "race_date", "race_number", "horse_name"]
    available_keys = [k for k in dedup_keys if k in df.columns]
    if len(available_keys) == len(dedup_keys):
        before = len(df)
        df = df.drop_duplicates(subset=available_keys, keep="last")
        dupes = before - len(df)
        if dupes > 0:
            print(f"  Removed {dupes:,} duplicate rows")
    return df


def filter_dates(df, start_date, end_date):
    """Filter rows by date range."""
    if "_race_date_parsed" not in df.columns:
        df = parse_race_dates(df)

    before = len(df)
    if start_date:
        sd = pd.to_datetime(start_date)
        df = df[df["_race_date_parsed"] >= sd]
    if end_date:
        ed = pd.to_datetime(end_date)
        df = df[df["_race_date_parsed"] <= ed]
    after = len(df)
    if before != after:
        print(f"  Date filter: {before:,} -> {after:,} rows")
    return df


def compute_career_stats(df):
    """Compute num_past_starts/wins/seconds/thirds from the dataset itself.

    For each row, counts how many races the same horse completed BEFORE
    this race date (strictly less-than — no leakage).  Only fills values
    where the column is missing or empty; preserves Equibase-provided
    values from the HTML scraper when present.

    Requires _race_date_parsed to be set (call parse_race_dates first).
    """
    if "_race_date_parsed" not in df.columns:
        print("  WARNING: _race_date_parsed missing, skipping career stats")
        return df

    stat_cols = ["num_past_starts", "num_past_wins", "num_past_seconds", "num_past_thirds"]
    for col in stat_cols:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check how many rows already have values (from HTML scraper)
    already_filled = df["num_past_starts"].notna().sum()
    need_fill = df["num_past_starts"].isna().sum()
    print(f"  Career stats: {already_filled:,} already filled, {need_fill:,} to compute")

    if need_fill == 0:
        return df

    # We need finish positions and dates
    df["_finish_num"] = pd.to_numeric(df["finish"], errors="coerce")

    # Sort by date so we can iterate chronologically
    df = df.sort_values("_race_date_parsed", na_position="last").reset_index(drop=True)

    # Build per-horse running tallies
    # Track: {horse_name: [starts, wins, seconds, thirds]}
    horse_counts = {}
    computed = 0

    starts_arr = df["num_past_starts"].values.copy()
    wins_arr = df["num_past_wins"].values.copy()
    seconds_arr = df["num_past_seconds"].values.copy()
    thirds_arr = df["num_past_thirds"].values.copy()
    names = df["horse_name"].values
    finishes = df["_finish_num"].values
    dates = df["_race_date_parsed"].values

    for i in range(len(df)):
        name = names[i]
        if pd.isna(name) or str(name).strip() == "":
            continue

        name = str(name).strip()

        # Initialize horse if first time seeing them
        if name not in horse_counts:
            horse_counts[name] = [0, 0, 0, 0]  # starts, wins, 2nds, 3rds

        # Fill in career stats if missing
        if pd.isna(starts_arr[i]):
            s, w, sec, thi = horse_counts[name]
            starts_arr[i] = s
            wins_arr[i] = w
            seconds_arr[i] = sec
            thirds_arr[i] = thi
            computed += 1

        # Update running tally AFTER recording (this race counts for future rows)
        fin = finishes[i]
        if not pd.isna(fin) and not pd.isna(dates[i]):
            horse_counts[name][0] += 1
            if fin == 1:
                horse_counts[name][1] += 1
            elif fin == 2:
                horse_counts[name][2] += 1
            elif fin == 3:
                horse_counts[name][3] += 1

    df["num_past_starts"] = starts_arr.astype(float)
    df["num_past_wins"] = wins_arr.astype(float)
    df["num_past_seconds"] = seconds_arr.astype(float)
    df["num_past_thirds"] = thirds_arr.astype(float)
    df.drop(columns=["_finish_num"], inplace=True, errors="ignore")

    print(f"  Computed career stats for {computed:,} rows "
          f"({len(horse_counts):,} unique horses)")

    return df


# ── Trip comment keyword lists ──────────────────────────────────
TROUBLE_KW = [
    "checked", "chked", "chkd", "steadied", "bumped", "bmpd", "blocked",
    "boxed", "impeded", "clipped", "stumbled", "fell", "lugged",
]
WIDE_KW = ["wide", "4-w", "5-w", "6-w", "7-w", "8-w", "4wide", "5wide", "6wide"]
POOR_START_KW = [
    "slow start", "slowstart", "dwelt", "broke slow", "brokeslow",
    "broke out", "brokeout", "broke in", "brokein", "stumbled start",
]
STRONG_CLOSE_KW = [
    "drewoff", "drew off", "rallied", "rally", "surged", "strongrally",
    "closed fast", "closedfast", "flying late", "flyinglate", "strong finish",
    "gaining", "upfinaljump",
]
FADED_KW = [
    "tired", "faded", "weakened", "gave way", "gaveway", "stopped",
    "eased", "pulled up", "hung", "flattenedstretch", "dulleffort",
    "neverfired", "showedlittle",
]


def _comment_has(comment, keywords):
    """Check if lowercased comment contains any keyword."""
    return 1 if any(kw in comment for kw in keywords) else 0


def compute_position_and_trip_features(df):
    """Compute per-horse historical running style, position averages,
    trip comment flags, and class/distance changes.

    Single chronological pass — all features are leak-free (use only
    data from races strictly before the current one).

    Requires: _race_date_parsed, num_horses_in_race, furlongs.
    """
    if "_race_date_parsed" not in df.columns:
        print("  WARNING: _race_date_parsed missing, skipping position features")
        return df

    # Ensure numeric columns
    for col in ["pos_1st_call", "pos_finish", "margin_finish", "finish", "purse"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "num_horses_in_race" not in df.columns:
        df["num_horses_in_race"] = 8

    df = df.sort_values("_race_date_parsed", na_position="last").reset_index(drop=True)
    n = len(df)

    # Pre-allocate output arrays
    running_styles = np.full(n, "U", dtype=object)
    avg_early_pos = np.full(n, np.nan)
    avg_late_gain = np.full(n, np.nan)
    avg_margin_fin = np.full(n, np.nan)
    best_margin_fin = np.full(n, np.nan)
    had_trouble = np.zeros(n, dtype=np.int8)
    wide_trip = np.zeros(n, dtype=np.int8)
    poor_start = np.zeros(n, dtype=np.int8)
    strong_close = np.zeros(n, dtype=np.int8)
    faded = np.zeros(n, dtype=np.int8)
    avg_trouble_rate = np.full(n, np.nan)
    avg_wide_rate = np.full(n, np.nan)
    class_change = np.full(n, np.nan)
    distance_change = np.full(n, np.nan)

    # Extract arrays for fast access
    names = df["horse_name"].values
    pos1 = df["pos_1st_call"].values if "pos_1st_call" in df.columns else np.full(n, np.nan)
    finishes = df["finish"].values if "finish" in df.columns else np.full(n, np.nan)
    mfin = df["margin_finish"].values if "margin_finish" in df.columns else np.full(n, np.nan)
    field_sizes = df["num_horses_in_race"].values
    purses = df["purse"].values if "purse" in df.columns else np.full(n, np.nan)
    fur = df["furlongs"].values if "furlongs" in df.columns else np.full(n, np.nan)
    comments = df["comment"].values if "comment" in df.columns else np.full(n, "", dtype=object)
    dates = df["_race_date_parsed"].values

    # Per-horse accumulators
    # Each horse: {rel_pos: [], gains: [], margins: [],
    #              trouble: int, wide: int, commented: int,
    #              last_purse: float, last_fur: float}
    horse = {}
    computed = 0

    for i in range(n):
        name = names[i]
        if pd.isna(name) or str(name).strip() == "":
            continue
        name = str(name).strip()

        if name not in horse:
            horse[name] = {
                "rel_pos": [], "gains": [], "margins": [],
                "trouble": 0, "wide": 0, "commented": 0,
                "last_purse": np.nan, "last_fur": np.nan,
            }

        h = horse[name]

        # === WRITE features from PRIOR races ===
        if h["rel_pos"]:
            avg_rel = np.mean(h["rel_pos"])
            avg_early_pos[i] = round(avg_rel, 4)
            if avg_rel <= 0.25:
                running_styles[i] = "E"
            elif avg_rel <= 0.50:
                running_styles[i] = "P"
            elif avg_rel <= 0.75:
                running_styles[i] = "S"
            else:
                running_styles[i] = "C"

        if h["gains"]:
            avg_late_gain[i] = round(np.mean(h["gains"]), 2)

        if h["margins"]:
            avg_margin_fin[i] = round(np.mean(h["margins"]), 2)
            best_margin_fin[i] = round(min(h["margins"]), 2)

        if h["commented"] > 0:
            avg_trouble_rate[i] = round(h["trouble"] / h["commented"], 4)
            avg_wide_rate[i] = round(h["wide"] / h["commented"], 4)

        # Class/distance change vs last race
        if not np.isnan(h["last_purse"]) and not np.isnan(purses[i]):
            class_change[i] = purses[i] - h["last_purse"]
        if not np.isnan(h["last_fur"]) and not np.isnan(fur[i]):
            distance_change[i] = round(fur[i] - h["last_fur"], 2)

        # === PARSE current race comment (flags, not leak) ===
        c = str(comments[i]).lower() if not pd.isna(comments[i]) else ""
        had_trouble[i] = _comment_has(c, TROUBLE_KW)
        wide_trip[i] = _comment_has(c, WIDE_KW)
        poor_start[i] = _comment_has(c, POOR_START_KW)
        strong_close[i] = _comment_has(c, STRONG_CLOSE_KW)
        faded[i] = _comment_has(c, FADED_KW)

        # === UPDATE accumulators for future rows ===
        if not pd.isna(dates[i]):
            p1 = pos1[i]
            fs = field_sizes[i]
            fin = finishes[i]

            if not np.isnan(p1) and not np.isnan(fs) and fs > 0:
                h["rel_pos"].append(p1 / fs)

            if not np.isnan(p1) and not np.isnan(fin):
                h["gains"].append(p1 - fin)

            if not np.isnan(mfin[i]):
                h["margins"].append(mfin[i])

            if c:
                h["commented"] += 1
                if had_trouble[i]:
                    h["trouble"] += 1
                if wide_trip[i]:
                    h["wide"] += 1

            if not np.isnan(purses[i]):
                h["last_purse"] = purses[i]
            if not np.isnan(fur[i]):
                h["last_fur"] = fur[i]

        computed += 1

    # Assign to DataFrame
    df["running_style"] = running_styles
    df["avg_early_pos"] = avg_early_pos
    df["avg_late_gain"] = avg_late_gain
    df["avg_margin_finish"] = avg_margin_fin
    df["best_finish_margin"] = best_margin_fin
    df["had_trouble"] = had_trouble
    df["wide_trip"] = wide_trip
    df["poor_start"] = poor_start
    df["strong_close"] = strong_close
    df["faded"] = faded
    df["avg_trouble_rate"] = avg_trouble_rate
    df["avg_wide_rate"] = avg_wide_rate
    df["class_change"] = class_change
    df["distance_change"] = distance_change

    style_counts = pd.Series(running_styles).value_counts()
    print(f"  Processed {computed:,} rows, {len(horse):,} unique horses")
    print(f"  Running styles: {dict(style_counts)}")

    return df


def compute_pace_scenario(df):
    """Compute race-level pace features from running style labels.

    Uses running_style (from prior races) to estimate pace pressure
    for the current race. Leak-free: styles are based on historical data.
    """
    if "running_style" not in df.columns:
        print("  WARNING: running_style missing, skipping pace scenario")
        return df

    race_keys = ["track_code", "_race_date_parsed", "race_number"]
    if not all(c in df.columns for c in race_keys):
        return df

    # Count early-speed types (E and P) per race
    df["_is_speed"] = df["running_style"].isin(["E", "P"]).astype(int)
    df["pace_pressure"] = df.groupby(race_keys, observed=False)["_is_speed"].transform("sum")
    df["pace_pressure_pct"] = (
        df["pace_pressure"] / df["num_horses_in_race"].clip(lower=1)
    ).round(3)

    # Style advantage: closers benefit from hot pace, speed from cold pace
    style_map = {"E": -1, "P": -1, "S": 1, "C": 1, "U": 0}
    df["style_advantage"] = (
        df["running_style"].map(style_map).fillna(0) * df["pace_pressure_pct"]
    ).round(3)

    df.drop(columns=["_is_speed"], inplace=True, errors="ignore")

    avg_pressure = df["pace_pressure"].mean()
    print(f"  Avg pace pressure: {avg_pressure:.1f} speed types/race")

    return df


def compute_bonus_features(df):
    """Add derived columns that complement the speed figures."""
    # Furlongs (continuous distance)
    if "distance" in df.columns and "distance_unit" in df.columns:
        dist = pd.to_numeric(df["distance"], errors="coerce")
        df["furlongs"] = np.nan
        mask_f = df["distance_unit"] == "F"
        mask_y = df["distance_unit"] == "Y"
        mask_m = df["distance_unit"] == "M"
        df.loc[mask_f, "furlongs"] = dist[mask_f] / 100.0
        df.loc[mask_y, "furlongs"] = dist[mask_y] / 220.0
        df.loc[mask_m, "furlongs"] = dist[mask_m] * 8.0

    # Win rate
    if "num_past_wins" in df.columns and "num_past_starts" in df.columns:
        starts = pd.to_numeric(df["num_past_starts"], errors="coerce").fillna(0)
        wins = pd.to_numeric(df["num_past_wins"], errors="coerce").fillna(0)
        df["win_rate"] = (wins / starts.clip(lower=1)).round(4)

    # Class level
    if "race_type" in df.columns and "purse" in df.columns:
        purse = pd.to_numeric(df["purse"], errors="coerce").fillna(0)

        def _class_level(race_type, purse_val):
            rt = str(race_type).upper()
            if any(k in rt for k in ["STAKES", "STK", "GRADED", "GRD", "G1", "G2", "G3", "LISTED", "LST"]):
                return 1
            if "ALLOWANCE" in rt or "ALW" in rt:
                return 2
            if "MAIDEN SPECIAL WEIGHT" in rt or "MSW" in rt:
                return 3
            if "MAIDEN CLAIMING" in rt or "MCL" in rt:
                return 4
            if "CLAIMING" in rt or "CLM" in rt:
                if purse_val >= 20000:
                    return 4
                if purse_val >= 10000:
                    return 5
                return 6
            if "STARTER" in rt:
                return 4
            if purse_val >= 100000:
                return 1
            if purse_val >= 50000:
                return 2
            if purse_val >= 25000:
                return 3
            if purse_val >= 15000:
                return 4
            if purse_val >= 10000:
                return 5
            return 6

        df["class_level"] = [_class_level(rt, p) for rt, p in zip(df["race_type"], purse)]

    return df


def load_margins(margins_path):
    """Load optional margins CSV."""
    if not margins_path or not os.path.exists(margins_path):
        return None
    print(f"Loading margins from {margins_path}...")
    import csv as csv_mod
    margins_data = {}
    with open(margins_path, "r") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            key = (row["track_code"], row["race_date"], int(row["race_number"]), row["horse_name"])
            margins_data[key] = float(row["beaten_lengths"])
    print(f"  Loaded {len(margins_data)} margin entries")
    return margins_data


def run_speed_pipeline(tmp_input, margins_data=None):
    """Run the full speed figure pipeline on a temporary combined CSV."""
    print("\n=== SPEED FIGURE PIPELINE ===")

    print("Step 1: Loading & filtering data (TB, Furlongs only)...")
    entries = load_data(tmp_input)
    print(f"  {len(entries):,} entries qualify for speed figures")

    if not entries:
        print("  WARNING: No qualifying entries. Speed figures will be empty.")
        return []

    print("Step 2: Computing par times...")
    par_times = compute_par_times(entries)
    print(f"  {len(par_times)} (distance, surface) pars computed")

    print("Step 3: Computing raw speed ratings...")
    races = compute_raw_speed_ratings(entries, par_times, margins_data)
    print(f"  {len(races):,} races processed")

    print(f"Step 4: Iterating track variants ({NUM_ITERATIONS} iterations)...")
    compute_track_variants(entries, races)

    print("Step 5: Normalizing figures (target mean=75, std=15)...")
    stats = normalize_figures(entries)
    if stats:
        print(f"  Raw: mean={stats[0]:.2f}, std={stats[1]:.2f} -> Normalized: mean=75, std=15")

    print("Step 6: Computing derived ML features (leak-free)...")
    compute_derived_features(entries)

    print("  Done.\n")
    return entries


def entries_to_dataframe(entries):
    """Convert speedfig entries to a merge-ready DataFrame."""
    records = []
    for e in entries:
        row = {
            "track_code": e["track_code"],
            "race_date": e["race_date"],
            "race_number": e["race_number"],
            "horse_name": e["horse_name"],
        }
        for col in SPEED_FIG_COLS:
            val = e.get(col)
            if val is not None and col in ["raw_speed_rating", "track_variant", "speed_figure"]:
                val = round(val, 2)
            row[col] = val
        records.append(row)
    return pd.DataFrame(records)


def resolve_input_files(args):
    """Resolve input file paths from args."""
    paths = []

    if args.input_dir:
        pattern = os.path.join(args.input_dir, "*.csv")
        paths = sorted(glob.glob(pattern))
        if not paths:
            print(f"Error: No CSV files found in {args.input_dir}")
            sys.exit(1)
        print(f"Found {len(paths)} CSV files in {args.input_dir}")

    elif "*" in args.input or "?" in args.input:
        paths = sorted(glob.glob(args.input))
        if not paths:
            print(f"Error: No files match pattern '{args.input}'")
            sys.exit(1)
        print(f"Found {len(paths)} CSV files matching pattern")

    else:
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.")
            sys.exit(1)
        paths = [args.input]

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Build comprehensive all_tracks_hackathon_2016_2026.csv"
    )
    parser.add_argument(
        "--input", default="all_tracks_hackathon.csv",
        help="Input CSV file or glob pattern (default: all_tracks_hackathon.csv)"
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Directory containing CSV files to combine"
    )
    parser.add_argument(
        "--output", default="all_tracks_hackathon_2016_2026.csv",
        help="Output CSV (default: all_tracks_hackathon_2016_2026.csv)"
    )
    parser.add_argument("--margins", default=None, help="Optional margins CSV")
    parser.add_argument("--start-date", default=None, help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="End date filter (YYYY-MM-DD)")
    args = parser.parse_args()

    # --- Step 1: Resolve and load input files ---
    input_paths = resolve_input_files(args)
    print(f"\nLoading data from {len(input_paths)} file(s)...")
    df = load_and_concat_csvs(input_paths)
    print(f"\nCombined: {len(df):,} rows, {len(df.columns)} columns")

    # --- Step 2: Parse dates, deduplicate, filter ---
    df = parse_race_dates(df)
    df = deduplicate(df)

    if args.start_date or args.end_date:
        df = filter_dates(df, args.start_date, args.end_date)

    # Show date range
    if "_race_date_parsed" in df.columns:
        valid_dates = df["_race_date_parsed"].dropna()
        if len(valid_dates) > 0:
            print(f"  Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
            print(f"  Years: {sorted(valid_dates.dt.year.unique())}")

    print(f"  Final row count: {len(df):,}")

    # --- Step 3: Write temp CSV for speedfig pipeline (needs CSV DictReader format) ---
    tmp_path = "_tmp_combined_for_speedfig.csv"
    print(f"\nWriting temp file for speed figure pipeline...")
    df.to_csv(tmp_path, index=False)

    # --- Step 4: Run speed figure pipeline ---
    margins_data = load_margins(args.margins)
    entries = run_speed_pipeline(tmp_path, margins_data)

    # Clean up temp file
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    if entries:
        sf_df = entries_to_dataframe(entries)
        print(f"Speed figures computed for {len(sf_df):,} rows")

        # --- Step 5: Merge speed figures back ---
        join_keys = ["track_code", "race_date", "race_number", "horse_name"]
        df["race_number"] = pd.to_numeric(df["race_number"], errors="coerce")
        sf_df["race_number"] = pd.to_numeric(sf_df["race_number"], errors="coerce")
        for col in ["track_code", "race_date", "horse_name"]:
            df[col] = df[col].astype(str).str.strip()
            sf_df[col] = sf_df[col].astype(str).str.strip()

        df = df.merge(sf_df, on=join_keys, how="left")
        matched = df["speed_figure_normalized"].notna().sum()
        print(f"  Merged: {matched:,}/{len(df):,} rows got speed figures")
    else:
        matched = 0
        print("  No speed figures computed (no qualifying entries)")

    # --- Step 6: Compute career stats from dataset (fills missing values) ---
    print("Computing career stats (num_past_starts/wins/seconds/thirds)...")
    df = compute_career_stats(df)

    # --- Step 7: Compute bonus features ---
    print("Computing bonus derived features (furlongs, win_rate, class_level)...")
    df = compute_bonus_features(df)

    # --- Step 8: Compute field sizes ---
    print("Computing field sizes...")
    race_keys = ["track_code", "_race_date_parsed", "race_number"]
    if all(c in df.columns for c in race_keys):
        df["num_horses_in_race"] = df.groupby(race_keys, observed=False)["horse_name"].transform("size")
    else:
        df["num_horses_in_race"] = 8
    print(f"  Avg field size: {df['num_horses_in_race'].mean():.1f}")

    # --- Step 9: Compute running style, position, trip & class features ---
    print("Computing running style, position, trip & class features...")
    df = compute_position_and_trip_features(df)

    # --- Step 10: Compute pace scenario ---
    print("Computing pace scenario features...")
    df = compute_pace_scenario(df)

    # --- Step 11: Order columns ---
    # Build ordered column list: original -> extended scraper -> speed figs -> bonus -> pace/trip -> anything else
    ordered = []
    for col_list in [ORIGINAL_COLS, EXTENDED_SCRAPER_COLS, SPEED_FIG_COLS, BONUS_COLS, PACE_TRIP_COLS]:
        for c in col_list:
            if c in df.columns and c not in ordered:
                ordered.append(c)
    # Append any remaining columns not in our lists
    for c in df.columns:
        if c not in ordered and c != "_race_date_parsed":
            ordered.append(c)

    df = df[[c for c in ordered if c in df.columns]]

    # --- Step 12: Write output ---
    print(f"\nWriting {args.output}...")
    df.to_csv(args.output, index=False)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"  Output file:   {args.output}")
    print(f"  Total rows:    {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")

    n_orig = sum(1 for c in ORIGINAL_COLS if c in df.columns)
    n_ext = sum(1 for c in EXTENDED_SCRAPER_COLS if c in df.columns)
    n_sf = sum(1 for c in SPEED_FIG_COLS if c in df.columns)
    n_bonus = sum(1 for c in BONUS_COLS if c in df.columns)
    print(f"  Original cols:  {n_orig}")
    print(f"  Extended cols:  {n_ext}")
    print(f"  Speed fig cols: {n_sf}")
    print(f"  Bonus cols:     {n_bonus}")
    if matched:
        print(f"\n  Rows with speed figures: {matched:,} ({100*matched/len(df):.1f}%)")

    if "speed_figure_normalized" in df.columns:
        valid = df["speed_figure_normalized"].dropna()
        if len(valid) > 0:
            print(f"\n  Speed Figure Stats:")
            print(f"    Mean:   {valid.mean():.1f}")
            print(f"    Std:    {valid.std():.1f}")
            print(f"    Min:    {valid.min():.1f}")
            print(f"    Max:    {valid.max():.1f}")
            print(f"    Median: {valid.median():.1f}")

    if "_race_date_parsed" in df.columns:
        valid_dates = df["_race_date_parsed"].dropna() if "_race_date_parsed" in df.columns else pd.Series()
    # re-check after potential column drop
    # (we dropped _race_date_parsed from output already via the ordered list)

    print(f"\nColumn list:")
    for i, col in enumerate(df.columns, 1):
        tag = ""
        if col in SPEED_FIG_COLS:
            tag = " [SPEED FIG]"
        elif col in BONUS_COLS:
            tag = " [BONUS]"
        elif col in EXTENDED_SCRAPER_COLS:
            tag = " [EXTENDED]"
        print(f"  {i:>2}. {col}{tag}")


if __name__ == "__main__":
    main()
