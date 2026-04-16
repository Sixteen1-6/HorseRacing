"""
Enrich today's entries with historical features from the full dataset.

For each horse in today's entries, looks up their most recent race in
all_tracks_2014_2026.csv and carries forward all computed features
(speed figures, running style, pace, trip rates, etc.).

Then computes race-level features (pace_pressure, style_advantage,
num_horses_in_race) based on today's assembled field.

Usage:
  python enrich_entries.py --entries test_data.csv --history all_tracks_2014_2026.csv
  python enrich_entries.py --entries test_data.csv  # uses default history path
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# Features to carry forward from the horse's most recent race
HORSE_HISTORY_FEATURES = [
    # Speed figures (these are already "prior" features — leak-free)
    "best_prior_figure", "avg_prior_figure", "avg_last3_figure", "last_figure",
    "num_prior_races", "figure_trend", "figure_trend_3race",
    "best_surface_figure", "avg_surface_figure", "best_dist_figure",
    "figure_consistency", "peak_vs_recent",
    # Running style / position
    "running_style", "avg_early_pos", "avg_late_gain",
    "avg_margin_finish", "best_finish_margin",
    # Trip history
    "avg_trouble_rate", "avg_wide_rate",
]

# Features that need the CURRENT race's data for computation
# (we recompute these based on today's field, not carry forward)
RECOMPUTE_RACE_LEVEL = [
    "pace_pressure", "pace_pressure_pct", "style_advantage", "num_horses_in_race",
]


def load_history_index(history_path):
    """Load the historical dataset and build a per-horse index of their latest race."""
    print(f"Loading history from {history_path}...")

    cols_needed = ["horse_name", "race_date", "track_code", "race_number",
                   "finish", "purse", "distance", "furlongs", "surface",
                   "num_past_starts", "num_past_wins", "num_past_seconds", "num_past_thirds",
                   "win_rate", "class_level"] + HORSE_HISTORY_FEATURES

    # Only load columns we actually need
    df = pd.read_csv(history_path, low_memory=False,
                     usecols=lambda c: c in cols_needed)
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    print(f"  {len(df):,} historical rows, {df['horse_name'].nunique():,} unique horses")

    # Sort by date descending — first occurrence per horse is their latest race
    df.sort_values("race_date", ascending=False, inplace=True)
    df.drop_duplicates(subset=["horse_name"], keep="first", inplace=True)
    df.set_index("horse_name", inplace=True)

    print(f"  Index built: {len(df):,} horses with history")
    return df


def enrich(entries_df, history_idx):
    """Enrich entries with historical features."""
    n_total = len(entries_df)
    n_matched = 0

    # Carry forward per-horse features
    for feat in HORSE_HISTORY_FEATURES:
        if feat in history_idx.columns:
            entries_df[feat] = entries_df["horse_name"].map(history_idx[feat])

    # Carry forward career stats if missing from entries
    for col in ["num_past_starts", "num_past_wins", "num_past_seconds", "num_past_thirds", "win_rate"]:
        if col in history_idx.columns:
            existing = pd.to_numeric(entries_df.get(col), errors="coerce")
            missing = existing.isna()
            if missing.any():
                entries_df.loc[missing, col] = entries_df.loc[missing, "horse_name"].map(
                    history_idx[col]
                )

    # Class change: current purse vs last race purse
    if "purse" in history_idx.columns and "purse" in entries_df.columns:
        current_purse = pd.to_numeric(entries_df["purse"], errors="coerce")
        last_purse = entries_df["horse_name"].map(history_idx["purse"])
        entries_df["class_change"] = current_purse - last_purse

    # Distance change: current furlongs vs last race furlongs
    if "furlongs" in history_idx.columns:
        if "furlongs" not in entries_df.columns and "distance" in entries_df.columns:
            dist = pd.to_numeric(entries_df["distance"], errors="coerce")
            unit = entries_df.get("distance_unit", "F").astype(str).str.upper()
            entries_df["furlongs"] = np.where(unit == "F", dist / 100, np.nan)
            entries_df.loc[unit == "Y", "furlongs"] = dist[unit == "Y"] / 220
            entries_df.loc[unit == "M", "furlongs"] = dist[unit == "M"] * 8

        if "furlongs" in entries_df.columns:
            current_fur = pd.to_numeric(entries_df["furlongs"], errors="coerce")
            last_fur = entries_df["horse_name"].map(history_idx["furlongs"])
            entries_df["distance_change"] = (current_fur - last_fur).round(2)

    n_matched = entries_df["horse_name"].isin(history_idx.index).sum()

    # Compute race-level features from today's field
    race_keys = ["race_number"]
    if "track_code" in entries_df.columns:
        race_keys = ["track_code"] + race_keys

    entries_df["num_horses_in_race"] = entries_df.groupby(race_keys)["horse_name"].transform("size")

    # Pace scenario from running styles
    if "running_style" in entries_df.columns:
        is_speed = entries_df["running_style"].isin(["E", "P"]).astype(int)
        entries_df["pace_pressure"] = entries_df.groupby(race_keys)["horse_name"].transform("size")
        # Recompute properly
        entries_df["pace_pressure"] = is_speed.groupby(
            [entries_df[k] for k in race_keys]
        ).transform("sum")
        entries_df["pace_pressure_pct"] = (
            entries_df["pace_pressure"] / entries_df["num_horses_in_race"].clip(lower=1)
        ).round(3)

        style_map = {"E": -1, "P": -1, "S": 1, "C": 1, "U": 0}
        entries_df["style_advantage"] = (
            entries_df["running_style"].map(style_map).fillna(0) * entries_df["pace_pressure_pct"]
        ).round(3)

    print(f"  Enriched: {n_matched}/{n_total} horses found in history "
          f"({100*n_matched/n_total:.0f}%)")

    return entries_df


def main():
    p = argparse.ArgumentParser(description="Enrich entries with historical features")
    p.add_argument("--entries", default="test_data.csv", help="Today's entries CSV")
    p.add_argument("--history", default="all_tracks_2014_2026.csv",
                   help="Historical dataset with computed features")
    p.add_argument("--output", default=None,
                   help="Output path (default: overwrites entries file)")
    args = p.parse_args()

    if not os.path.exists(args.entries):
        print(f"Error: {args.entries} not found")
        sys.exit(1)
    if not os.path.exists(args.history):
        print(f"Error: {args.history} not found")
        sys.exit(1)

    # Load
    entries_df = pd.read_csv(args.entries, low_memory=False)
    print(f"Entries: {len(entries_df)} horses across "
          f"{entries_df['race_number'].nunique()} races")

    history_idx = load_history_index(args.history)

    # Enrich
    entries_df = enrich(entries_df, history_idx)

    # Save
    output_path = args.output or args.entries
    entries_df.to_csv(output_path, index=False)
    print(f"Saved enriched entries to {output_path}")

    # Summary
    feat_filled = {}
    for feat in HORSE_HISTORY_FEATURES + RECOMPUTE_RACE_LEVEL + ["class_change", "distance_change"]:
        if feat in entries_df.columns:
            filled = entries_df[feat].notna().sum()
            feat_filled[feat] = f"{filled}/{len(entries_df)}"

    print(f"\nFeature fill rates:")
    for feat, rate in feat_filled.items():
        print(f"  {feat:25s}: {rate}")


if __name__ == "__main__":
    main()
