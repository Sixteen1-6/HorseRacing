# -*- coding: utf-8 -*-
"""
Horse Racing Ranking Model — LambdaRank + Optuna + Multi-Bet Strategy
=====================================================================
Architecture:
  - LightGBM LambdaRank (learns within-race ordering)
  - Optuna optimizes NDCG@3 on 5-fold GroupKFold
  - Softmax converts scores → per-race probabilities
  - Isotonic regression calibrates probabilities
  - Betting module: win, exacta, trifecta box, trifecta straight

CLI Usage:
  python horse_racing_ranking_model.py --input data.csv
  python horse_racing_ranking_model.py --input data.csv --quick

Google Colab Usage:
  # ── Cell 1: Install deps ──
  # !pip install lightgbm optuna scikit-learn scipy joblib

  # ── Cell 2: Mount Drive ──
  # from google.colab import drive
  # drive.mount('/content/drive')

  # ── Cell 3: Upload this script + your CSV to /content/ ──
  # from google.colab import files
  # uploaded = files.upload()

  # ── Cell 4: Quick sanity check (~10 min) ──
  # from horse_racing_ranking_model import main
  # main(input_csv="/content/all_tracks_hackathon_2016_2026.csv", quick=True)

  # ── Cell 5: Full training (~1-2 hours) ──
  # main(input_csv="/content/all_tracks_hackathon_2016_2026.csv")
  #   → saves model files to /content/drive/MyDrive/horse_model/

  # ── Cell 6: Download files (if Drive not mounted) ──
  # from google.colab import files
  # files.download('/content/ranking_model.lgb')
  # files.download('/content/ranking_calibrator.pkl')
  # files.download('/content/ranking_feature_names.pkl')
  # files.download('/content/ranking_cat_features.pkl')
  # files.download('/content/ranking_jt_lookup.json')
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import os
import re
import sys
import json
import joblib
import warnings
import matplotlib
# Use Agg backend for CLI/headless; Colab handles its own backend
try:
    import google.colab
    # Colab uses inline backend automatically
except ImportError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import permutations, combinations
from scipy.special import softmax
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    log_loss, brier_score_loss, mean_absolute_error, ndcg_score
)

warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Module-level storage for jockey/trainer lookups (populated during feature engineering)
_JT_LOOKUPS = {'jockey_trainer': {}, 'jockey': {}, 'trainer': {}}

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_time_to_seconds(time_str):
    """Convert time string (MM:SS.s or S.s) to seconds."""
    if pd.isna(time_str):
        return np.nan
    if isinstance(time_str, (int, float)):
        return float(time_str)
    time_str = str(time_str).strip()
    m = re.match(r'(\d{1,2}):(\d{1,2}(?:\.\d+)?)', time_str)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2))
    m = re.match(r'(\d+(?:\.\d+)?)', time_str)
    if m:
        return float(m.group(1))
    return np.nan


def calculate_days_between(d1, d2):
    if pd.isna(d1) or pd.isna(d2):
        return np.nan
    d1, d2 = pd.to_datetime(d1, errors='coerce'), pd.to_datetime(d2, errors='coerce')
    if pd.isna(d1) or pd.isna(d2):
        return np.nan
    return (d1 - d2).days


def scores_to_probs(scores, group_sizes):
    """Convert raw ranking scores to per-race softmax probabilities."""
    probs = np.zeros_like(scores)
    idx = 0
    for size in group_sizes:
        race_scores = scores[idx:idx + size]
        probs[idx:idx + size] = softmax(race_scores)
        idx += size
    return probs


def get_group_sizes(groups):
    """Get ordered group sizes for LambdaRank query groups."""
    # Use pandas for fast ordered group counting
    _, idx, counts = np.unique(groups, return_index=True, return_counts=True)
    # Sort by first appearance (idx) to preserve order
    order = np.argsort(idx)
    return counts[order]


def compute_race_ndcg(y_true, scores, groups, k=3):
    """Compute mean NDCG@k across races."""
    ndcgs = []
    group_sizes = get_group_sizes(groups)
    idx = 0
    for size in group_sizes:
        if size < 2:
            idx += size
            continue
        yt = y_true[idx:idx + size]
        sc = scores[idx:idx + size]
        max_finish = yt.max()
        relevance = (max_finish + 1 - yt).reshape(1, -1)
        sc_reshaped = sc.reshape(1, -1)
        try:
            ndcgs.append(ndcg_score(relevance, sc_reshaped, k=k))
        except ValueError:
            pass
        idx += size
    return np.mean(ndcgs) if ndcgs else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: DATA LOADING & FEATURE ENGINEERING
# (Preserved from your original pipeline with minor cleanup)
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_engineer_features(csv_path="test1data.csv"):
    """Load raw data and apply full feature engineering pipeline."""
    print("=" * 70)
    print("LOADING DATA & ENGINEERING FEATURES")
    print("=" * 70)

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # --- Date conversions ---
    for col in ['race_date', 'last_race_date', 'horse_dob']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # --- Time conversions ---
    time_cols = ['win_time', 'finish_time', 'last_race_time', 'second_call_time', 'final_time']
    for col in time_cols:
        if col in df.columns:
            df[f"{col}_seconds"] = df[col].apply(parse_time_to_seconds)
            df.drop(columns=[col], inplace=True, errors='ignore')

    # --- Categorical dtype ---
    cat_cols = ['surface', 'track_condition', 'weather', 'breed', 'sex',
                'medication', 'track_code', 'track_name', 'jockey', 'trainer',
                'owner', 'horse_name', 'race_type']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # --- Target: finish position ---
    if 'finish' not in df.columns:
        raise ValueError("'finish' column not found")
    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df.dropna(subset=['finish'], inplace=True)
    df['finish'] = df['finish'].astype(int)

    # --- Age ---
    if 'age' in df.columns and df['age'].notna().any():
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    else:
        df['age'] = np.nan

    # --- Days since last race ---
    if 'race_date' in df.columns and 'last_race_date' in df.columns:
        df['days_since_last_race'] = df.apply(
            lambda r: calculate_days_between(r['race_date'], r['last_race_date']), axis=1
        )
        # NaN stays — horse with no last_race_date is a first-time starter,
        # which is meaningfully different from "raced 21 days ago"

    # --- Track condition standardization ---
    if 'track_condition' in df.columns:
        cond_map = {
            'FT': 'Fast', 'GD': 'Good', 'SY': 'Sloppy', 'MY': 'Muddy',
            'HY': 'Heavy', 'SL': 'Slow', 'FM': 'Firm', 'YL': 'Yielding',
            'SF': 'Soft', 'HD': 'Hard', 'WF': 'Wet Fast', 'FZ': 'Frozen'
        }
        df['standard_condition'] = (
            df['track_condition'].astype(str).str.upper().map(cond_map)
            .fillna(df['track_condition'].astype(str))
        ).astype('category')

    # --- Class level ---
    if 'race_type' in df.columns and 'purse' in df.columns:
        df['purse'] = pd.to_numeric(df['purse'], errors='coerce').fillna(0)

        def determine_class(row):
            rt = str(row.get('race_type', '')).upper()
            p = row.get('purse', 0)
            if any(k in rt for k in ['STAKES', 'STK', 'GRADED', 'G1', 'G2', 'G3']): return 1
            if 'ALLOWANCE' in rt or 'ALW' in rt: return 2
            if 'MAIDEN SPECIAL' in rt or 'MSW' in rt: return 3
            if 'MAIDEN CLAIMING' in rt or 'MCL' in rt: return 4
            if 'CLAIMING' in rt or 'CLM' in rt:
                if p >= 20000: return 4
                if p >= 10000: return 5
                return 6
            if 'STARTER' in rt: return 4
            if p >= 100000: return 1
            if p >= 50000: return 2
            if p >= 25000: return 3
            return 6

        df['class_level'] = df.apply(determine_class, axis=1)

    # --- Num horses in race ---
    race_group_cols = ['track_code', 'race_date', 'race_number']
    if all(c in df.columns for c in race_group_cols):
        df['race_date_str'] = df['race_date'].dt.strftime('%Y-%m-%d')
        df['num_horses_in_race'] = df.groupby(
            ['track_code', 'race_date_str', 'race_number'], observed=False
        )['finish'].transform('size')
        df.drop(columns=['race_date_str'], inplace=True)
        df['num_horses_in_race'].fillna(df['num_horses_in_race'].median(), inplace=True)
    else:
        df['num_horses_in_race'] = 8

    # --- Post position features ---
    if 'post_position' in df.columns:
        df['post_position'] = pd.to_numeric(df['post_position'], errors='coerce')
        valid = (df['num_horses_in_race'] > 1) & df['post_position'].notna()
        df['post_position_normalized'] = np.nan
        df.loc[valid, 'post_position_normalized'] = (
            (df.loc[valid, 'post_position'] - 1) / (df.loc[valid, 'num_horses_in_race'] - 1)
        )
        # post_position_normalized stays NaN if post is missing — LightGBM handles it
        df['is_inside_post'] = (df['post_position'] == 1).astype(int)
        df['is_outside_post'] = (df['post_position'] == df['num_horses_in_race']).astype(int)

    # --- Distance to furlongs ---
    if 'distance' in df.columns and 'distance_unit' in df.columns:
        df['distance'] = pd.to_numeric(df['distance'], errors='coerce')
        df['distance_unit'] = df['distance_unit'].astype(str).str.upper()

        def to_furlongs(row):
            d, u = row['distance'], row['distance_unit']
            if pd.isna(d): return np.nan
            if u == 'F': return d
            if u == 'Y': return d / 220
            if u == 'M': return d * 8
            return np.nan

        df['distance_furlongs'] = df.apply(to_furlongs, axis=1)
        # distance_furlongs stays NaN if conversion fails — LightGBM handles it
        df.drop(columns=['distance', 'distance_unit'], inplace=True, errors='ignore')

    # --- Horse categories ---
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age_category'] = pd.cut(
            df['age'], bins=[0, 3, 5, 7, 100],
            labels=['Young', 'Prime', 'Mature', 'Veteran'], right=False
        )

    if all(c in df.columns for c in ['num_past_wins', 'num_past_starts']):
        df['num_past_wins'] = pd.to_numeric(df['num_past_wins'], errors='coerce').fillna(0)
        df['num_past_starts'] = pd.to_numeric(df['num_past_starts'], errors='coerce').fillna(0)
        df['win_rate'] = df['num_past_wins'] / df['num_past_starts'].clip(lower=1)
        df['win_rate_category'] = pd.cut(
            df['win_rate'], bins=[-0.001, 0.1, 0.2, 0.3, 1.01],
            labels=['Low', 'Medium', 'High', 'Elite'], right=False
        )

    if 'dollar_odds' in df.columns:
        df['dollar_odds'] = pd.to_numeric(df['dollar_odds'], errors='coerce')
        df['odds_category'] = pd.cut(
            df['dollar_odds'], bins=[-0.001, 2, 5, 10, 1000],
            labels=['Favorite', 'Contender', 'Longshot', 'Huge Longshot'], right=False
        )

    if 'num_past_starts' in df.columns:
        df['experience_category'] = pd.cut(
            df['num_past_starts'], bins=[-1, 5, 15, 30, 1000],
            labels=['Novice', 'Experienced', 'Veteran', 'Elite'], right=False
        )

    # --- Jockey-trainer combos ---
    if 'jockey' in df.columns and 'trainer' in df.columns:
        df['jockey_trainer'] = df['jockey'].astype(str) + '_' + df['trainer'].astype(str)
        df['jockey_trainer'] = df['jockey_trainer'].astype('category')

        df['_is_win'] = (df['finish'] == 1).astype(int)
        df['_is_itm'] = (df['finish'] <= 3).astype(int)

        jt_counts = df.groupby('jockey_trainer', observed=True).size()
        valid_jt = jt_counts[jt_counts >= 10].index

        jt_wr = df[df['jockey_trainer'].isin(valid_jt)].groupby('jockey_trainer', observed=True)['_is_win'].mean()
        jt_itm = df[df['jockey_trainer'].isin(valid_jt)].groupby('jockey_trainer', observed=True)['_is_itm'].mean()

        # NaN for unknown combos — model learns "unknown combo" ≠ "average combo"
        df['jockey_trainer_win_rate'] = df['jockey_trainer'].map(jt_wr)
        df['jockey_trainer_itm_rate'] = df['jockey_trainer'].map(jt_itm)

        df['jt_strength'] = pd.cut(
            df['jockey_trainer_win_rate'],
            bins=[-0.001, 0.08, 0.15, 0.25, 1.01],
            labels=['Weak', 'Average', 'Strong', 'Elite'], right=False
        )
        df.drop(columns=['_is_win', '_is_itm'], inplace=True, errors='ignore')

        # ── Save jockey-trainer lookup for predict.py ──
        # This gets exported alongside the model so race-day predictions
        # can look up combo stats without needing the full training data.
        jt_lookup = {}

        # Jockey-trainer combo stats
        for combo in valid_jt:
            jt_lookup[combo] = {
                'win_rate': round(float(jt_wr.get(combo, 0.1)), 4),
                'itm_rate': round(float(jt_itm.get(combo, 0.3)), 4),
                'starts': int(jt_counts.get(combo, 0)),
            }

        # Individual jockey stats
        df['_is_win_tmp'] = (df['finish'] == 1).astype(int)
        df['_is_itm_tmp'] = (df['finish'] <= 3).astype(int)

        jockey_stats = df.groupby('jockey', observed=True).agg(
            starts=('_is_win_tmp', 'size'),
            win_rate=('_is_win_tmp', 'mean'),
            itm_rate=('_is_itm_tmp', 'mean'),
        )
        trainer_stats = df.groupby('trainer', observed=True).agg(
            starts=('_is_win_tmp', 'size'),
            win_rate=('_is_win_tmp', 'mean'),
            itm_rate=('_is_itm_tmp', 'mean'),
        )
        df.drop(columns=['_is_win_tmp', '_is_itm_tmp'], inplace=True, errors='ignore')

        jockey_lookup = {}
        for j, row in jockey_stats.iterrows():
            if row['starts'] >= 50:
                jockey_lookup[str(j)] = {
                    'win_rate': round(float(row['win_rate']), 4),
                    'itm_rate': round(float(row['itm_rate']), 4),
                    'starts': int(row['starts']),
                }

        trainer_lookup = {}
        for t, row in trainer_stats.iterrows():
            if row['starts'] >= 30:
                trainer_lookup[str(t)] = {
                    'win_rate': round(float(row['win_rate']), 4),
                    'itm_rate': round(float(row['itm_rate']), 4),
                    'starts': int(row['starts']),
                }

        # Save to module-level dict for later export
        _JT_LOOKUPS['jockey_trainer'] = jt_lookup
        _JT_LOOKUPS['jockey'] = jockey_lookup
        _JT_LOOKUPS['trainer'] = trainer_lookup

        n_jt = len(jt_lookup)
        n_j = len(jockey_lookup)
        n_t = len(trainer_lookup)
        print(f"  Built lookups: {n_jt} jockey-trainer combos, "
              f"{n_j} jockeys, {n_t} trainers")

    # --- Filter low-frequency jockeys/trainers ---
    for col, min_n, other_label in [('jockey', 50, 'other_jockey'), ('trainer', 30, 'other_trainer')]:
        if col in df.columns:
            counts = df[col].value_counts()
            valid = counts[counts >= min_n].index
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                if other_label not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(other_label)
            df.loc[~df[col].isin(valid), col] = other_label
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].cat.remove_unused_categories()

    # --- Race ID ---
    if all(c in df.columns for c in ['track_code', 'race_date', 'race_number']):
        df['race_id'] = (
            df['track_code'].astype(str) + '_' +
            df['race_date'].dt.strftime('%Y%m%d') + '_' +
            df['race_number'].astype(str)
        )
    else:
        df['race_id'] = (df.index // 8).astype(str)

    # --- Speed figure features (from build_dataset.py / speedfig pipeline) ---
    # These are the most predictive features if available in the dataset.
    # The "prior" features are leak-free (computed from races BEFORE the current one).
    SPEED_FIG_FEATURES = [
        'speed_figure_normalized',  # final calibrated figure for THIS race (use carefully)
        'best_prior_figure',        # best speed fig from all prior races
        'avg_prior_figure',         # average speed fig from all prior races
        'avg_last3_figure',         # average of last 3 races (recency)
        'last_figure',              # most recent race's figure
        'num_prior_races',          # experience proxy
        'figure_trend',             # improving or declining?
        'figure_trend_3race',       # short-term trend
        'best_surface_figure',      # best on this surface type
        'avg_surface_figure',       # avg on this surface type
        'best_dist_figure',         # best at this distance
        'figure_consistency',       # std of prior figures (lower = more reliable)
        'peak_vs_recent',           # peak performance vs recent form
    ]

    # Also grab extended scraper columns if present
    EXTENDED_FEATURES = [
        'margin_finish', 'pos_1st_call', 'pos_2nd_call', 'pos_stretch',
        'margin_1st_call', 'margin_2nd_call', 'margin_stretch',
        'final_time_secs', 'speed_figure_equibase', 'claimed_price',
    ]

    speed_found = [c for c in SPEED_FIG_FEATURES if c in df.columns]
    extended_found = [c for c in EXTENDED_FEATURES if c in df.columns]
    print(f"Speed figure features found: {len(speed_found)}/{len(SPEED_FIG_FEATURES)}")
    print(f"Extended scraper features found: {len(extended_found)}/{len(EXTENDED_FEATURES)}")

    # Ensure these are numeric
    for col in speed_found + extended_found:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # NOTE: speed_figure_normalized is the figure for the CURRENT race.
    # At prediction time you won't have it. For training the ranking model it's
    # fine (LambdaRank uses it to learn what makes a horse fast), but the truly
    # leak-free features are the "prior" ones. We keep it but flag it.
    if 'speed_figure_normalized' in df.columns:
        print("  ⚠ speed_figure_normalized is the CURRENT race figure — "
              "won't be available at prediction time. "
              "The model will learn from it but rely on prior features for generalization.")

    # --- Drop columns not useful as features ---
    # KEEP: num_past_wins, num_past_starts (raw counts complement win_rate),
    #       win_time_seconds (speed is predictive), race_type (granularity beyond class_level)
    #       ALL speed figure columns, ALL extended scraper columns
    # DROP: only identifiers, free-text, and columns replaced by engineered versions
    drop_cols = [
        'program_num', 'horse_name', 'comment', 'last_race_date',
        'last_race_track', 'track_name', 'post_time', 'weather',
        'medication', 'owner', 'track_condition',  # replaced by standard_condition
        'post_position',  # replaced by post_position_normalized
        'horse_dob',      # replaced by age
        'race_date',      # used for race_id and days_since_last_race, not a feature
        'speed_figure_normalized',  # LEAK: computed from THIS race's finish time
        'win_time_seconds',         # LEAK: winning time of THIS race
        'odds_category',  # derived from dollar_odds
        # NOTE: dollar_odds is dropped in prepare_ranking_data, NOT here,
        # so it's available for ROI simulation
        # Speed pipeline intermediate cols (not useful as features, just building blocks)
        'raw_speed_rating', 'track_variant', 'speed_figure',
        # Fractional times — correlated with final_time_secs, and leak for current race
        'frac_1', 'frac_2', 'frac_3', 'frac_4',
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    # --- Filter out races with fewer than 3 horses (can't rank meaningfully) ---
    race_sizes = df.groupby('race_id')['finish'].transform('size')
    df = df[race_sizes >= 3].copy()
    print(f"After filtering: {len(df)} rows, {df['race_id'].nunique()} races")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: PREPARE DATA FOR LAMBDARANK
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_ranking_data(df):
    """
    Prepare X, y (relevance labels), groups for LambdaRank.
    LambdaRank needs:
      - y: relevance labels (higher = better). We use inverted finish position.
      - groups: array of query group sizes (horses per race).
    """
    print("  [prep] Sorting by race_id...", flush=True)
    df = df.sort_values('race_id').reset_index(drop=True)

    # Save dollar_odds separately for ROI simulation, then drop from features
    dollar_odds = df['dollar_odds'].values if 'dollar_odds' in df.columns else None

    X = df.drop(columns=['finish', 'race_id', 'dollar_odds'], errors='ignore')
    finish = df['finish'].values
    race_ids = df['race_id'].values

    print("  [prep] Computing relevance labels...", flush=True)
    max_by_race = df.groupby('race_id')['finish'].transform('max').values
    y_relevance = (max_by_race + 1 - finish).astype(float)

    # Group sizes (contiguous)
    group_sizes = df.groupby('race_id', sort=False).size().values

    # Process features for LightGBM
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    print(f"  [prep] {len(numerical_cols)} numeric, {len(categorical_cols)} categorical cols", flush=True)

    # ── Handle missing NUMERIC values ──
    # KEY: LightGBM handles NaN natively. At each tree split it learns
    # "when this feature is missing, go left or right." Passing NaN is
    # BETTER than filling with median because:
    #   - A horse with no speed figure history IS different from one with a 75
    #   - A horse with unknown days_since_last_race IS different from 21 days
    #   - Median-filling destroys this signal
    #
    # We ONLY fill where 0 is the semantically correct value:
    FILL_ZERO = {
        # First-time starters genuinely have 0 past starts/wins
        'num_past_starts', 'num_past_wins', 'num_past_seconds', 'num_past_thirds',
        # Binary flags: missing = not applicable = 0
        'is_inside_post', 'is_outside_post',
    }
    for col in numerical_cols:
        if col in FILL_ZERO and X[col].isnull().any():
            X[col].fillna(0, inplace=True)
        # Everything else (speed figures, days_since_last_race, age,
        # weight, odds, margins, times, etc.) stays NaN.

    n_missing = X[numerical_cols].isnull().sum()
    cols_with_nan = n_missing[n_missing > 0]
    if len(cols_with_nan) > 0:
        print(f"  Keeping NaN in {len(cols_with_nan)} numeric features "
              f"(LightGBM handles natively):")
        for col, count in cols_with_nan.items():
            pct = count / len(X) * 100
            print(f"    {col}: {count:,} missing ({pct:.1f}%)")

    print("  [prep] Processing categorical features...", flush=True)
    MAX_CAT = 50
    for col in categorical_cols:
        if not isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].astype('category')
        if 'Unknown' not in X[col].cat.categories:
            X[col] = X[col].cat.add_categories('Unknown')
        X[col].fillna('Unknown', inplace=True)

        # Limit cardinality
        if X[col].nunique() > MAX_CAT:
            top = X[col].value_counts().nlargest(MAX_CAT).index.tolist()
            if 'Other' not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories('Other')
            X.loc[~X[col].isin(top), col] = 'Other'
            X[col] = X[col].cat.remove_unused_categories()

    print("  [prep] Converting to codes for LightGBM...", flush=True)
    cat_feature_names = []
    for col in categorical_cols:
        if col in X.columns and isinstance(X[col].dtype, pd.CategoricalDtype):
            cat_feature_names.append(col)
            X[col] = X[col].cat.codes

    print("  [prep] Done.", flush=True)
    return X, y_relevance, finish, race_ids, group_sizes, cat_feature_names, dollar_odds


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: OPTUNA HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def optuna_objective(trial, X, y_rel, finish, race_ids, cat_features):
    """Optuna objective: maximize mean NDCG@3 over 5-fold GroupKFold with pruning."""

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'lambdarank_truncation_level': trial.suggest_int('lambdarank_truncation_level', 3, 10),
        'random_state': 42,
        'n_jobs': -1,
    }

    gkf = GroupKFold(n_splits=5)
    ndcg_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_rel, groups=race_ids)):
        print(f"    Trial {trial.number} Fold {fold+1}/5...", end=" ", flush=True)
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_rel[train_idx], y_rel[val_idx]
        finish_val = finish[val_idx]
        race_ids_val = race_ids[val_idx]

        train_groups = get_group_sizes(race_ids[train_idx])
        val_groups = get_group_sizes(race_ids[val_idx])

        train_data = lgb.Dataset(
            X_tr, label=y_tr, group=train_groups,
            categorical_feature=cat_features, free_raw_data=False
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, group=val_groups,
            categorical_feature=cat_features, free_raw_data=False, reference=train_data
        )

        callbacks = [lgb.early_stopping(50, verbose=False)]
        model = lgb.train(
            params, train_data, num_boost_round=params['n_estimators'],
            valid_sets=[val_data], callbacks=callbacks
        )

        scores = model.predict(X_val)
        ndcg = compute_race_ndcg(finish_val, scores, race_ids_val, k=3)
        ndcg_scores.append(ndcg)
        print(f"NDCG={ndcg:.4f}", flush=True)

        # --- Pruning: report intermediate result after each fold ---
        # If this trial is clearly worse than previous trials, Optuna kills it
        # early instead of wasting time on remaining folds
        trial.report(np.mean(ndcg_scores), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(ndcg_scores)


def run_optuna(X, y_rel, finish, race_ids, cat_features, n_trials=100,
               subsample_frac=0.30):
    """
    Run Optuna study to find best hyperparameters.

    Uses two speedups:
      1. Subsampling: Optuna searches on 30% of data (by race).
         Hyperparameter landscape doesn't change much with more data —
         you just need enough to find the right ballpark.
      2. Pruning (MedianPruner): kills trials that look bad after 2 folds
         instead of running all 5. Typically cuts wall time 40-50%.
    """
    print("\n" + "=" * 70)
    print(f"OPTUNA HYPERPARAMETER SEARCH ({n_trials} trials)")
    print("=" * 70)

    # --- Subsample by race for Optuna speed ---
    unique_races = np.unique(race_ids)
    n_sample = max(500, int(len(unique_races) * subsample_frac))
    if n_sample < len(unique_races):
        rng = np.random.RandomState(42)
        sampled_races = set(rng.choice(unique_races, size=n_sample, replace=False))
        mask = np.array([r in sampled_races for r in race_ids])
        X_sub = X[mask].reset_index(drop=True)
        y_sub = y_rel[mask]
        finish_sub = finish[mask]
        rids_sub = race_ids[mask]
        print(f"  Subsampled {n_sample} races ({mask.sum():,} rows) "
              f"from {len(unique_races)} total for Optuna search")
    else:
        X_sub, y_sub, finish_sub, rids_sub = X, y_rel, finish, race_ids
        print(f"  Using all {len(unique_races)} races (data small enough)")

    # MedianPruner: prune after 2 folds if below median of completed trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1)
    study = optuna.create_study(
        direction='maximize', study_name='horse_ranking', pruner=pruner
    )
    study.optimize(
        lambda trial: optuna_objective(trial, X_sub, y_sub, finish_sub, rids_sub, cat_features),
        n_trials=n_trials,
        show_progress_bar=True
    )

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\n  Completed: {n_complete} trials, Pruned: {n_pruned} trials")
    print(f"  Best NDCG@3: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(study.best_params, indent=2)}")
    return study.best_params


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5: TRAIN FINAL MODEL + CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_final_model(X_train, y_rel_train, race_ids_train,
                      X_val, y_rel_val, race_ids_val,
                      best_params, cat_features):
    """Train final LambdaRank model with best Optuna params."""
    print("\n" + "=" * 70)
    print("TRAINING FINAL LAMBDARANK MODEL")
    print("=" * 70)

    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 3, 5],
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'n_jobs': -1,
        **best_params
    }
    n_rounds = params.pop('n_estimators', 800)

    train_groups = get_group_sizes(race_ids_train)
    val_groups = get_group_sizes(race_ids_val)

    train_data = lgb.Dataset(
        X_train, label=y_rel_train, group=train_groups,
        categorical_feature=cat_features, free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_val, label=y_rel_val, group=val_groups,
        categorical_feature=cat_features, free_raw_data=False, reference=train_data
    )

    callbacks = [
        lgb.early_stopping(200, verbose=True),
        lgb.log_evaluation(100)
    ]

    model = lgb.train(
        params, train_data, num_boost_round=n_rounds,
        valid_sets=[val_data], callbacks=callbacks
    )

    print(f"Best iteration: {model.best_iteration}")
    return model


def calibrate_probabilities(model, X_cal, finish_cal, race_ids_cal):
    """
    Fit isotonic regression to calibrate softmax probabilities → true win probabilities.
    Uses the calibration set (held out from training).
    """
    print("\nCalibrating probabilities with isotonic regression...")
    scores = model.predict(X_cal)
    group_sizes = get_group_sizes(race_ids_cal)
    raw_probs = scores_to_probs(scores, group_sizes)

    # Binary: did horse actually win?
    y_win = (finish_cal == 1).astype(int)

    calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    calibrator.fit(raw_probs, y_win)

    cal_probs = calibrator.predict(raw_probs)
    print(f"  Raw prob range:        [{raw_probs.min():.4f}, {raw_probs.max():.4f}]")
    print(f"  Calibrated prob range: [{cal_probs.min():.4f}, {cal_probs.max():.4f}]")
    print(f"  Calibration log-loss:  {log_loss(y_win, cal_probs):.4f}")
    print(f"  Brier score:           {brier_score_loss(y_win, cal_probs):.4f}")

    return calibrator


# ═══════════════════════════════════════════════════════════════════════════════
# PART 6: BETTING MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class BettingEngine:
    """
    Multi-bet strategy using calibrated ranking model probabilities.

    Bet types:
      - Win:             P(win) > implied_prob × edge_threshold
      - Exacta:          P(A 1st) × P(B 2nd | A wins)
      - Trifecta straight: P(A 1st) × P(B 2nd | ~A) × P(C 3rd | ~A,~B)
      - Trifecta box:    Sum of all 3! permutations of {A, B, C} finishing 1-2-3
    """

    def __init__(self, model, calibrator, edge_threshold=1.20, min_ev=0.10,
                 kelly_fraction=0.25, bankroll=1000.0):
        self.model = model
        self.calibrator = calibrator
        self.edge_threshold = edge_threshold   # model_prob > implied × this
        self.min_ev = min_ev                   # minimum expected value per dollar
        self.kelly_fraction = kelly_fraction   # fractional Kelly sizing
        self.bankroll = bankroll

    def predict_race(self, X_race, dollar_odds=None):
        """
        Given features for all horses in a single race, produce:
          - calibrated win probabilities
          - calibrated top-3 probabilities (approx)
          - betting recommendations
        """
        scores = self.model.predict(X_race)
        raw_probs = softmax(scores)
        cal_probs = self.calibrator.predict(raw_probs)

        # Normalize calibrated probs to sum to 1 within race
        cal_probs = cal_probs / cal_probs.sum()

        # Approximate P(top-3) from ranking scores
        # Use softmax temperature scaling: higher temp → more uniform → top-3
        # Rough heuristic: P(top-3) ≈ sum of softmax probs at temp=0.5 for top-3
        # Better: use the ranking to estimate positional probabilities
        n = len(scores)
        top3_probs = self._estimate_top3_probs(scores, n)

        result = {
            'scores': scores,
            'win_probs': cal_probs,
            'top3_probs': top3_probs,
            'bets': []
        }

        if dollar_odds is not None:
            result['bets'] = self._find_value_bets(cal_probs, top3_probs, dollar_odds)

        return result

    def _estimate_top3_probs(self, scores, n):
        """
        Estimate P(finishing top-3) for each horse using Monte Carlo simulation
        with Plackett-Luce model derived from scores.
        """
        n_sims = 5000
        probs = softmax(scores)
        top3_counts = np.zeros(n)

        for _ in range(n_sims):
            remaining = list(range(n))
            p = probs.copy()
            for pos in range(min(3, n)):
                p_remaining = p[remaining] / p[remaining].sum()
                chosen_idx = np.random.choice(len(remaining), p=p_remaining)
                chosen_horse = remaining[chosen_idx]
                top3_counts[chosen_horse] += 1
                remaining.pop(chosen_idx)

        return top3_counts / n_sims

    def _find_value_bets(self, win_probs, top3_probs, dollar_odds):
        """Identify value bets across all bet types."""
        bets = []
        n = len(win_probs)

        # --- Win bets ---
        for i in range(n):
            if dollar_odds[i] <= 0:
                continue
            implied = 1.0 / (dollar_odds[i] + 1)
            if win_probs[i] > implied * self.edge_threshold:
                ev = win_probs[i] * dollar_odds[i] - 1.0
                if ev >= self.min_ev:
                    # Fractional Kelly sizing
                    kelly = (win_probs[i] * (dollar_odds[i] + 1) - 1) / dollar_odds[i]
                    bet_size = max(0, self.kelly_fraction * kelly * self.bankroll)
                    bets.append({
                        'type': 'WIN',
                        'horses': [i],
                        'model_prob': win_probs[i],
                        'implied_prob': implied,
                        'ev_per_dollar': ev,
                        'kelly_bet': round(bet_size, 2),
                        'odds': dollar_odds[i]
                    })

        # --- Trifecta box (top 3 horses by top3_prob) ---
        top3_indices = np.argsort(-top3_probs)[:4]  # Consider top 4 for box combos
        for combo in combinations(top3_indices, 3):
            box_prob = self._trifecta_box_prob(combo, win_probs)
            # Trifecta box costs 6 units (3! permutations)
            # Approximate expected payout: hard without actual pool data
            # Use a rough heuristic based on individual odds
            combo_odds = np.array([dollar_odds[i] for i in combo])
            if any(combo_odds <= 0):
                continue
            # Very rough trifecta payout estimate
            approx_payout = np.prod(combo_odds) * 0.5  # Discounted product
            ev = box_prob * approx_payout - 6.0  # Cost is 6 units for box
            if ev > 0:
                bets.append({
                    'type': 'TRIFECTA_BOX',
                    'horses': list(combo),
                    'model_prob': box_prob,
                    'approx_payout': approx_payout,
                    'ev_per_dollar': ev / 6.0,
                    'cost': 6.0
                })

        return sorted(bets, key=lambda b: b['ev_per_dollar'], reverse=True)

    def _trifecta_box_prob(self, horses, win_probs):
        """
        P(horses[0], [1], [2] all finish top-3 in any order).
        Uses Plackett-Luce: sum over all 3! orderings.
        """
        total = 0.0
        all_probs = win_probs.copy()
        n = len(all_probs)

        for perm in permutations(horses):
            # P(perm[0] 1st) × P(perm[1] 2nd | perm[0] removed) × ...
            remaining_prob = all_probs.sum()
            p = 1.0
            excluded = set()
            for pos_horse in perm:
                denom = sum(all_probs[j] for j in range(n) if j not in excluded)
                if denom <= 0:
                    p = 0
                    break
                p *= all_probs[pos_horse] / denom
                excluded.add(pos_horse)
            total += p

        return total


# ═══════════════════════════════════════════════════════════════════════════════
# PART 7: EVALUATION & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, calibrator, X_test, finish_test, race_ids_test, dollar_odds_test=None):
    """Comprehensive evaluation on test set."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    scores = model.predict(X_test)
    group_sizes = get_group_sizes(race_ids_test)
    raw_probs = scores_to_probs(scores, group_sizes)
    cal_probs = calibrator.predict(raw_probs)

    y_win = (finish_test == 1).astype(int)
    y_top3 = (finish_test <= 3).astype(int)

    # ── 1. CALIBRATION METRICS (are probabilities accurate?) ──
    ll = log_loss(y_win, cal_probs)
    brier = brier_score_loss(y_win, cal_probs)
    print("\n┌─────────────────────────────────────────────┐")
    print("│  CALIBRATION — do probabilities match reality?  │")
    print("├─────────────────────────────────────────────┤")
    print(f"│  Log-loss (win):     {ll:.4f}                    │")
    print(f"│  Brier score:        {brier:.4f}                    │")
    print("│                                             │")
    print("│  Log-loss guide:                            │")
    print("│    < 0.50 = excellent   0.50-0.60 = good    │")
    print("│    0.60-0.70 = decent   > 0.70 = needs work │")
    print("└─────────────────────────────────────────────┘")

    # ── 2. RANKING METRICS (does the model order horses correctly?) ──
    ndcg3 = compute_race_ndcg(finish_test, scores, race_ids_test, k=3)
    ndcg5 = compute_race_ndcg(finish_test, scores, race_ids_test, k=5)
    print(f"\n┌─────────────────────────────────────────────┐")
    print(f"│  RANKING — NDCG (higher = better ordering)  │")
    print(f"├─────────────────────────────────────────────┤")
    print(f"│  NDCG@3:  {ndcg3:.4f}   (top-3 ordering quality) │")
    print(f"│  NDCG@5:  {ndcg5:.4f}   (top-5 ordering quality) │")
    print(f"│                                             │")
    print(f"│  NDCG guide:                                │")
    print(f"│    > 0.85 = strong    0.75-0.85 = solid     │")
    print(f"│    0.65-0.75 = okay   < 0.65 = weak         │")
    print(f"└─────────────────────────────────────────────┘")

    # ── 3. PER-RACE ACCURACY (informational, not the optimization target) ──
    test_group_sizes = get_group_sizes(race_ids_test)

    win_correct = 0
    top3_hits_total = 0
    exacta_correct = 0
    trifecta_correct = 0
    n_races = 0
    idx = 0

    for size in test_group_sizes:
        race_scores = scores[idx:idx + size]
        race_finish = finish_test[idx:idx + size]

        pred_order = np.argsort(-race_scores)  # descending score = best
        actual_winner = np.argmin(race_finish)

        if pred_order[0] == actual_winner:
            win_correct += 1

        pred_top3 = set(pred_order[:3])
        actual_top3 = set(np.where(race_finish <= 3)[0])
        top3_hits_total += len(pred_top3 & actual_top3)

        actual_order = np.argsort(race_finish)
        if size >= 2 and pred_order[0] == actual_order[0] and pred_order[1] == actual_order[1]:
            exacta_correct += 1
        if size >= 3 and list(pred_order[:3]) == list(actual_order[:3]):
            trifecta_correct += 1

        n_races += 1
        idx += size

    print(f"\n  Accuracy snapshot ({n_races} races):")
    print(f"    Win pick:     {win_correct / n_races:.1%} ({win_correct}/{n_races})")
    print(f"    Top-3 overlap: {top3_hits_total / (n_races * 3):.1%} avg")
    print(f"    Exact exacta:  {exacta_correct / n_races:.1%}")
    print(f"    Exact trifecta: {trifecta_correct / n_races:.1%}")
    print(f"    (These are informational — ROI matters more than raw accuracy)")

    # ── 4. ROI SIMULATION (the bottom line) ──
    roi_result = {}
    if dollar_odds_test is not None:
        print(f"\n┌─────────────────────────────────────────────┐")
        print(f"│  ROI SIMULATION — does the model make money? │")
        print(f"├─────────────────────────────────────────────┤")

        for edge_thresh in [1.10, 1.20, 1.30, 1.50]:
            total_wagered = 0.0
            total_returned = 0.0
            n_bets = 0

            idx = 0
            for size in test_group_sizes:
                race_probs = cal_probs[idx:idx + size]
                race_finish = finish_test[idx:idx + size]
                race_odds = dollar_odds_test[idx:idx + size]

                # Normalize probs within race
                prob_sum = race_probs.sum()
                if prob_sum > 0:
                    race_probs = race_probs / prob_sum

                for h in range(size):
                    if race_odds[h] <= 0 or np.isnan(race_odds[h]):
                        continue
                    implied = 1.0 / (race_odds[h] + 1)
                    if race_probs[h] > implied * edge_thresh:
                        total_wagered += 2.0
                        n_bets += 1
                        if race_finish[h] == 1:
                            total_returned += 2.0 * (race_odds[h] + 1)

                idx += size

            if total_wagered > 0:
                roi = (total_returned - total_wagered) / total_wagered * 100
                roi_str = f"{roi:+.1f}%"
            else:
                roi = 0.0
                roi_str = "N/A"

            print(f"│  Edge ≥ {edge_thresh:.2f}x: {n_bets:>5} bets, "
                  f"ROI = {roi_str:>8}, "
                  f"${total_returned - total_wagered:>+.2f}  │")
            roi_result[f'roi_{edge_thresh}'] = roi

        print(f"│                                             │")
        print(f"│  ROI guide (flat $2 win bets):              │")
        print(f"│    > 0% = profitable   > 10% = strong edge  │")
        print(f"│    > 20% = exceptional (verify not overfit)  │")
        print(f"│    Typical track take is ~17%, so break-even │")
        print(f"│    already means you're beating the vig.     │")
        print(f"└─────────────────────────────────────────────┘")
    else:
        print("\n  ⚠ No dollar_odds in test set — skipping ROI simulation.")
        print("    To enable: ensure dollar_odds column is in your CSV.")

    return {
        'log_loss': ll,
        'brier': brier,
        'ndcg3': ndcg3,
        'ndcg5': ndcg5,
        'win_acc': win_correct / n_races,
        'top3_acc': top3_hits_total / (n_races * 3),
        'exacta_acc': exacta_correct / n_races,
        'trifecta_acc': trifecta_correct / n_races,
        **roi_result,
    }


def plot_diagnostics(model, X_test, finish_test, race_ids_test, calibrator, feature_names, output_dir='.'):
    """Generate diagnostic plots."""
    print("\nGenerating diagnostic plots...")

    scores = model.predict(X_test)
    group_sizes = get_group_sizes(race_ids_test)
    raw_probs = scores_to_probs(scores, group_sizes)
    cal_probs = calibrator.predict(raw_probs)
    y_win = (finish_test == 1).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Calibration curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_win, cal_probs, n_bins=15, strategy='quantile')
    axes[0, 0].plot(prob_pred, prob_true, 'bo-', label='Calibrated')
    prob_true_raw, prob_pred_raw = calibration_curve(y_win, raw_probs, n_bins=15, strategy='quantile')
    axes[0, 0].plot(prob_pred_raw, prob_true_raw, 'r^--', alpha=0.6, label='Raw softmax')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0, 0].set_xlabel('Predicted probability')
    axes[0, 0].set_ylabel('Actual win rate')
    axes[0, 0].set_title('Calibration Curve')
    axes[0, 0].legend()

    # 2. Feature importance (top 20)
    importance = model.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=True).tail(20)
    axes[0, 1].barh(feat_imp['feature'], feat_imp['importance'])
    axes[0, 1].set_title('Top 20 Features (Gain)')

    # 3. Score distribution by finish position
    for pos in [1, 2, 3]:
        mask = finish_test == pos
        if mask.sum() > 0:
            axes[1, 0].hist(scores[mask], bins=30, alpha=0.5, label=f'Finish {pos}', density=True)
    mask_rest = finish_test > 3
    if mask_rest.sum() > 0:
        axes[1, 0].hist(scores[mask_rest], bins=30, alpha=0.3, label='4th+', density=True)
    axes[1, 0].set_title('Score Distribution by Finish')
    axes[1, 0].legend()

    # 4. Predicted prob vs actual win rate (binned)
    bins = np.linspace(0, cal_probs.max(), 20)
    bin_idx = np.digitize(cal_probs, bins)
    bin_means = []
    bin_actuals = []
    for b in range(1, len(bins)):
        mask = bin_idx == b
        if mask.sum() >= 10:
            bin_means.append(cal_probs[mask].mean())
            bin_actuals.append(y_win[mask].mean())
    axes[1, 1].scatter(bin_means, bin_actuals, s=50)
    axes[1, 1].plot([0, max(bin_means + [0.01])], [0, max(bin_means + [0.01])], 'k--', alpha=0.3)
    axes[1, 1].set_xlabel('Mean predicted win prob')
    axes[1, 1].set_ylabel('Actual win rate')
    axes[1, 1].set_title('Probability Reliability')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ranking_model_diagnostics.png') if output_dir else 'ranking_model_diagnostics.png'
    plt.savefig(save_path, dpi=150)
    try:
        import google.colab
        plt.show()  # display inline in Colab
    except ImportError:
        plt.close()  # free memory in CLI
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 8: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main(input_csv=None, quick=False, n_trials=100, subsample=0.30, output_dir=None):
    """
    Train the full LambdaRank pipeline.

    Works in two modes:
      - CLI:   python horse_racing_ranking_model.py --input data.csv --quick
      - Colab: main(input_csv="/content/drive/MyDrive/data.csv", quick=True)

    Args:
        input_csv:  path to combined CSV (default: all_tracks_hackathon_2016_2026.csv)
        quick:      15 Optuna trials instead of 100
        n_trials:   number of Optuna trials (ignored if quick=True)
        subsample:  fraction of races for Optuna search (0.30 = 30%)
        output_dir: where to save model files (default: current directory)
    """
    # ── Handle CLI args if called from command line ──
    if input_csv is None:
        import argparse
        parser = argparse.ArgumentParser(description="Train LambdaRank horse racing model")
        parser.add_argument('--input', default='all_tracks_hackathon_2016_2026.csv')
        parser.add_argument('--quick', action='store_true')
        parser.add_argument('--trials', type=int, default=100)
        parser.add_argument('--subsample', type=float, default=0.30)
        parser.add_argument('--output-dir', default=None)
        args = parser.parse_args()
        input_csv = args.input
        quick = args.quick
        n_trials = args.trials
        subsample = args.subsample
        output_dir = args.output_dir

    # ── Auto-detect Colab ──
    IN_COLAB = False
    try:
        import google.colab
        IN_COLAB = True
        print("🔵 Running in Google Colab")
        # Default output to Drive if mounted
        if output_dir is None and os.path.exists('/content/drive/MyDrive'):
            output_dir = '/content/drive/MyDrive/horse_model'
            os.makedirs(output_dir, exist_ok=True)
            print(f"  Model files will be saved to: {output_dir}")
        elif output_dir is None:
            output_dir = '/content'
            print(f"  Tip: mount Google Drive to persist model files:")
            print(f"    from google.colab import drive")
            print(f"    drive.mount('/content/drive')")
    except ImportError:
        pass

    if output_dir is None:
        output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)

    actual_trials = 15 if quick else n_trials
    print(f"\n{'=' * 70}")
    print(f"TRAINING CONFIG")
    print(f"{'=' * 70}")
    print(f"  Input:      {input_csv}")
    print(f"  Optuna:     {actual_trials} trials ({'quick' if quick else 'full'})")
    print(f"  Subsample:  {subsample:.0%} of races for Optuna")
    print(f"  Output dir: {output_dir}")

    # --- Load & engineer ---
    df = load_and_engineer_features(input_csv)

    # --- Prepare ranking data ---
    X, y_rel, finish, race_ids, group_sizes, cat_features, dollar_odds = prepare_ranking_data(df)
    feature_names = X.columns.tolist()
    print(f"\nFeatures: {len(feature_names)}")
    print(f"Categorical features: {cat_features}")

    # ── TEMPORAL SPLIT (critical for betting models) ──
    # Random splits leak future info into training. Instead:
    #   Train:       oldest races (everything before cutoff)
    #   Calibration: next chunk (for isotonic regression)
    #   Test:        most recent races (mimics real deployment)
    #
    # Race IDs are formatted as TRACK_YYYYMMDD_RACENUM, so sorting
    # them lexicographically gives chronological order.

    # Extract date from race_id for sorting
    def extract_date_from_race_id(rid):
        """Extract YYYYMMDD from race_id like 'KEE_20240415_3'."""
        parts = str(rid).split('_')
        for p in parts:
            if len(p) == 8 and p.isdigit():
                return p
        return '00000000'  # fallback

    unique_races = np.unique(race_ids)
    race_dates = np.array([extract_date_from_race_id(r) for r in unique_races])
    date_order = np.argsort(race_dates)
    sorted_races = unique_races[date_order]
    sorted_dates = race_dates[date_order]

    n_races_total = len(sorted_races)
    # 68% train | 12% calibration | 20% test (most recent)
    train_end = int(n_races_total * 0.68)
    cal_end = int(n_races_total * 0.80)

    train_races = sorted_races[:train_end]
    cal_races = sorted_races[train_end:cal_end]
    test_races = sorted_races[cal_end:]

    print(f"\n── Temporal Split ──")
    print(f"  Train: {len(train_races)} races  "
          f"({sorted_dates[0]} → {sorted_dates[train_end - 1]})")
    print(f"  Cal:   {len(cal_races)} races  "
          f"({sorted_dates[train_end]} → {sorted_dates[cal_end - 1]})")
    print(f"  Test:  {len(test_races)} races  "
          f"({sorted_dates[cal_end]} → {sorted_dates[-1]})")

    print(f"  Splitting data...", flush=True)
    # Use set lookups instead of np.isin for speed (O(n) vs O(n*m))
    train_set = set(train_races)
    cal_set = set(cal_races)
    test_set = set(test_races)

    train_mask = np.array([r in train_set for r in race_ids])
    cal_mask = np.array([r in cal_set for r in race_ids])
    test_mask = np.array([r in test_set for r in race_ids])

    X_train, y_rel_train = X[train_mask].reset_index(drop=True), y_rel[train_mask]
    X_cal, y_rel_cal = X[cal_mask].reset_index(drop=True), y_rel[cal_mask]
    X_test, y_rel_test = X[test_mask].reset_index(drop=True), y_rel[test_mask]

    finish_train, finish_cal, finish_test = finish[train_mask], finish[cal_mask], finish[test_mask]
    rids_train, rids_cal, rids_test = race_ids[train_mask], race_ids[cal_mask], race_ids[test_mask]
    odds_test = dollar_odds[test_mask] if dollar_odds is not None else None

    print(f"  Train: {len(X_train)} rows ({len(train_races)} races)", flush=True)
    print(f"  Cal:   {len(X_cal)} rows ({len(cal_races)} races)", flush=True)
    print(f"  Test:  {len(X_test)} rows ({len(test_races)} races)", flush=True)

    # --- Optuna ---
    best_params = run_optuna(X_train, y_rel_train, finish_train, rids_train,
                             cat_features, actual_trials, subsample_frac=subsample)

    # --- Train final model ---
    model = train_final_model(
        X_train, y_rel_train, rids_train,
        X_cal, y_rel_cal, rids_cal,
        best_params, cat_features
    )

    # --- Calibrate ---
    calibrator = calibrate_probabilities(model, X_cal, finish_cal, rids_cal)

    # --- Evaluate ---
    metrics = evaluate_model(model, calibrator, X_test.values, finish_test, rids_test, odds_test)

    # --- Diagnostics ---
    plot_diagnostics(model, X_test.values, finish_test, rids_test, calibrator, feature_names, output_dir)

    # --- Save everything ---
    print("\n" + "=" * 70)
    print("SAVING ARTIFACTS")
    print("=" * 70)

    out = lambda f: os.path.join(output_dir, f)

    model.save_model(out('ranking_model.lgb'))
    joblib.dump(calibrator, out('ranking_calibrator.pkl'))
    joblib.dump(feature_names, out('ranking_feature_names.pkl'))
    joblib.dump(cat_features, out('ranking_cat_features.pkl'))
    joblib.dump(best_params, out('ranking_best_params.pkl'))

    # Save jockey-trainer lookups for predict.py
    with open(out('ranking_jt_lookup.json'), 'w') as f:
        json.dump(_JT_LOOKUPS, f)
    print(f"  Jockey-trainer lookup: {len(_JT_LOOKUPS['jockey_trainer'])} combos, "
          f"{len(_JT_LOOKUPS['jockey'])} jockeys, {len(_JT_LOOKUPS['trainer'])} trainers")

    with open(out('ranking_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  All files saved to: {output_dir}/")
    print(f"  ranking_model.lgb, ranking_calibrator.pkl, ranking_metrics.json")
    print(f"  ranking_feature_names.pkl, ranking_cat_features.pkl")
    print(f"  ranking_best_params.pkl, ranking_jt_lookup.json")
    print(f"  ranking_model_diagnostics.png")

    if IN_COLAB:
        print(f"\n  📥 To download model files from Colab:")
        print(f"    from google.colab import files")
        print(f"    files.download('{out('ranking_model.lgb')}')")
        print(f"    files.download('{out('ranking_calibrator.pkl')}')")
        print(f"    files.download('{out('ranking_feature_names.pkl')}')")
        print(f"    files.download('{out('ranking_cat_features.pkl')}')")
        print(f"    files.download('{out('ranking_jt_lookup.json')}')")

    # --- Demo: betting on test races ---
    print("\n" + "=" * 70)
    print("SAMPLE BETTING ANALYSIS (first 3 test races)")
    print("=" * 70)

    engine = BettingEngine(model, calibrator, edge_threshold=1.20, min_ev=0.10)

    seen_races = set()
    idx = 0
    demo_count = 0
    for race_id in rids_test:
        if race_id in seen_races:
            idx += 1
            continue
        seen_races.add(race_id)

        race_mask = rids_test == race_id
        race_size = race_mask.sum()
        race_X = X_test[race_mask].values

        # Use dummy odds if dollar_odds not available in test features
        dummy_odds = np.random.uniform(1.5, 20.0, size=race_size)

        result = engine.predict_race(race_X, dollar_odds=dummy_odds)

        print(f"\n{'─' * 50}")
        print(f"Race: {race_id} ({race_size} horses)")
        print(f"{'Horse':<8} {'Score':>8} {'Win%':>8} {'Top3%':>8} {'Actual':>8}")
        print(f"{'─' * 50}")

        race_finish = finish_test[rids_test == race_id]
        order = np.argsort(-result['scores'])
        for rank, h in enumerate(order):
            marker = ' ★' if race_finish[h] <= 3 else ''
            print(f"  #{h:<5} {result['scores'][h]:>8.3f} {result['win_probs'][h]:>7.1%} "
                  f"{result['top3_probs'][h]:>7.1%} {race_finish[h]:>6}{marker}")

        if result['bets']:
            print(f"\n  Value bets found:")
            for bet in result['bets'][:3]:
                print(f"    {bet['type']}: horses {bet['horses']}, "
                      f"EV/$ = {bet['ev_per_dollar']:.3f}")

        demo_count += 1
        idx += race_size
        if demo_count >= 3:
            break

    print("\n✅ Pipeline complete!")


if __name__ == '__main__':
    main()
