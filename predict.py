"""
Horse Racing Prediction — LambdaRank Model
===========================================
Loads the trained LambdaRank model and predicts race outcomes.

Usage:
  python predict.py --input today_races.csv
  python predict.py --input today_races.csv --output predictions.csv

Input CSV should have the same columns as the training data:
  horse_name, jockey, trainer, track_code, race_number, race_date,
  age, weight, surface, post_position, dollar_odds, num_past_starts,
  num_past_wins, num_past_seconds, num_past_thirds, etc.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import json
import os
import re
import argparse
from scipy.special import softmax
from itertools import permutations, combinations

# ── Helpers ──

def parse_time_to_seconds(time_str):
    if pd.isna(time_str): return np.nan
    if isinstance(time_str, (int, float)): return float(time_str)
    time_str = str(time_str).strip()
    m = re.match(r'(\d{1,2}):(\d{1,2}(?:\.\d+)?)', time_str)
    if m: return float(m.group(1)) * 60 + float(m.group(2))
    m = re.match(r'(\d+(?:\.\d+)?)', time_str)
    if m: return float(m.group(1))
    return np.nan

def calculate_days_between(d1, d2):
    if pd.isna(d1) or pd.isna(d2): return np.nan
    d1, d2 = pd.to_datetime(d1, errors='coerce'), pd.to_datetime(d2, errors='coerce')
    if pd.isna(d1) or pd.isna(d2): return np.nan
    return (d1 - d2).days


# ── Feature Engineering (mirrors model.py's load_and_engineer_features) ──

def engineer_features(df, jt_lookup=None):
    """Apply the same feature engineering as training."""

    # Date conversions
    for col in ['race_date', 'last_race_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Time conversions
    for col in ['win_time', 'finish_time']:
        if col in df.columns:
            df[f"{col}_seconds"] = df[col].apply(parse_time_to_seconds)
            df.drop(columns=[col], inplace=True, errors='ignore')

    # Categorical dtype
    cat_cols = ['surface', 'track_condition', 'weather', 'breed', 'sex',
                'medication', 'track_code', 'track_name', 'jockey', 'trainer',
                'owner', 'horse_name', 'race_type']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Age
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')

    # Days since last race
    if 'race_date' in df.columns and 'last_race_date' in df.columns:
        df['days_since_last_race'] = (df['race_date'] - df['last_race_date']).dt.days

    # Track condition standardization
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

    # Class level
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

    # Num horses in race
    race_group_cols = ['track_code', 'race_date', 'race_number']
    if all(c in df.columns for c in race_group_cols):
        df['race_date_str'] = df['race_date'].dt.strftime('%Y-%m-%d')
        df['num_horses_in_race'] = df.groupby(
            ['track_code', 'race_date_str', 'race_number'], observed=False
        )['horse_name'].transform('size')
        df.drop(columns=['race_date_str'], inplace=True)
    else:
        df['num_horses_in_race'] = 8

    # Post position features
    if 'post_position' in df.columns:
        df['post_position'] = pd.to_numeric(df['post_position'], errors='coerce')
        valid = (df['num_horses_in_race'] > 1) & df['post_position'].notna()
        df['post_position_normalized'] = np.nan
        df.loc[valid, 'post_position_normalized'] = (
            (df.loc[valid, 'post_position'] - 1) / (df.loc[valid, 'num_horses_in_race'] - 1)
        )
        df['is_inside_post'] = (df['post_position'] == 1).astype(int)
        df['is_outside_post'] = (df['post_position'] == df['num_horses_in_race']).astype(int)

    # Distance to furlongs
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
        df.drop(columns=['distance', 'distance_unit'], inplace=True, errors='ignore')

    # Horse categories
    if 'age' in df.columns:
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

    # Jockey-trainer combos
    if 'jockey' in df.columns and 'trainer' in df.columns:
        df['jockey_trainer'] = df['jockey'].astype(str) + '_' + df['trainer'].astype(str)
        df['jockey_trainer'] = df['jockey_trainer'].astype('category')

        if jt_lookup:
            jt_combos = jt_lookup.get('jockey_trainer', {})
            df['jockey_trainer_win_rate'] = df['jockey_trainer'].map(
                {k: v['win_rate'] for k, v in jt_combos.items()})
            df['jockey_trainer_itm_rate'] = df['jockey_trainer'].map(
                {k: v['itm_rate'] for k, v in jt_combos.items()})
        else:
            df['jockey_trainer_win_rate'] = np.nan
            df['jockey_trainer_itm_rate'] = np.nan

        df['jt_strength'] = pd.cut(
            df['jockey_trainer_win_rate'],
            bins=[-0.001, 0.08, 0.15, 0.25, 1.01],
            labels=['Weak', 'Average', 'Strong', 'Elite'], right=False
        )

    # Filter jockeys/trainers
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

    # Race ID
    if all(c in df.columns for c in ['track_code', 'race_date', 'race_number']):
        df['race_id'] = (
            df['track_code'].astype(str) + '_' +
            df['race_date'].dt.strftime('%Y%m%d') + '_' +
            df['race_number'].astype(str)
        )
    else:
        df['race_id'] = 'race_' + (df.index // 8).astype(str)

    # Drop non-feature columns
    drop_cols = [
        'program_num', 'horse_name', 'comment', 'last_race_date',
        'last_race_track', 'track_name', 'post_time', 'weather',
        'medication', 'owner', 'track_condition', 'post_position',
        'horse_dob', 'race_date',
        'speed_figure_normalized', 'win_time_seconds',
        'raw_speed_rating', 'track_variant', 'speed_figure',
        'frac_1', 'frac_2', 'frac_3', 'frac_4',
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

    return df


def prepare_features(df, feature_names, cat_features):
    """Prepare X matrix matching the training feature set."""
    # Save info before dropping
    race_ids = df['race_id'].values if 'race_id' in df.columns else None
    dollar_odds = df['dollar_odds'].values if 'dollar_odds' in df.columns else None

    X = df.drop(columns=['finish', 'race_id'], errors='ignore')

    # Fill zeros for count features
    for col in ['num_past_starts', 'num_past_wins', 'num_past_seconds', 'num_past_thirds',
                'is_inside_post', 'is_outside_post']:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    # Process categoricals
    MAX_CAT = 50
    for col in X.select_dtypes(exclude=np.number).columns:
        if not isinstance(X[col].dtype, pd.CategoricalDtype):
            X[col] = X[col].astype('category')
        if 'Unknown' not in X[col].cat.categories:
            X[col] = X[col].cat.add_categories('Unknown')
        X[col] = X[col].fillna('Unknown')
        if X[col].nunique() > MAX_CAT:
            top = X[col].value_counts().nlargest(MAX_CAT).index.tolist()
            if 'Other' not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories('Other')
            X.loc[~X[col].isin(top), col] = 'Other'
            X[col] = X[col].cat.remove_unused_categories()
        X[col] = X[col].cat.codes

    # Align columns with training features
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    return X, race_ids, dollar_odds


def estimate_top3_probs(scores, n):
    """Estimate P(top-3) using Monte Carlo Plackett-Luce simulation."""
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


def main():
    parser = argparse.ArgumentParser(description="Predict horse race outcomes")
    parser.add_argument('--input', default='test_data.csv', help='Input CSV with race entries')
    parser.add_argument('--output', default='race_predictions/all_race_predictions.csv',
                        help='Output CSV with predictions')
    parser.add_argument('--models-dir', default='models', help='Directory with model artifacts')
    parser.add_argument('--edge', type=float, default=1.20, help='Edge threshold for value bets')
    args = parser.parse_args()

    # Load model artifacts
    model_path = os.path.join(args.models_dir, 'ranking_model.lgb')
    calibrator_path = os.path.join(args.models_dir, 'ranking_calibrator.pkl')
    features_path = os.path.join(args.models_dir, 'ranking_feature_names.pkl')
    cat_path = os.path.join(args.models_dir, 'ranking_cat_features.pkl')
    jt_path = os.path.join(args.models_dir, 'ranking_jt_lookup.json')

    for path, name in [(model_path, 'Model'), (calibrator_path, 'Calibrator'),
                        (features_path, 'Features')]:
        if not os.path.exists(path):
            print(f"Error: {name} not found at {path}. Run model.py first.")
            return

    print("Loading model artifacts...")
    model = lgb.Booster(model_file=model_path)
    calibrator = joblib.load(calibrator_path)
    feature_names = joblib.load(features_path)
    cat_features = joblib.load(cat_path) if os.path.exists(cat_path) else []

    jt_lookup = None
    if os.path.exists(jt_path):
        with open(jt_path) as f:
            jt_lookup = json.load(f)
        n_combos = len(jt_lookup.get('jockey_trainer', {}))
        print(f"Loaded jockey-trainer lookup: {n_combos} combos")

    # Load race data
    print(f"Loading race data from {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {len(df)} entries")

    # Save display columns
    display_cols = {}
    for col in ['horse_name', 'jockey', 'trainer', 'race_number', 'track_code']:
        if col in df.columns:
            display_cols[col] = df[col].values.copy()

    # Engineer features
    print("Engineering features...")
    df = engineer_features(df, jt_lookup)

    # Prepare X
    X, race_ids, dollar_odds = prepare_features(df, feature_names, cat_features)
    print(f"Features prepared: {X.shape}")

    # Predict
    print("Predicting...")
    scores = model.predict(X)

    # Process per race
    results = []
    unique_races = []
    seen = set()
    for r in race_ids:
        if r not in seen:
            unique_races.append(r)
            seen.add(r)

    idx = 0
    for race_id in unique_races:
        race_mask = race_ids == race_id
        size = race_mask.sum()
        race_scores = scores[idx:idx + size]
        race_odds = dollar_odds[idx:idx + size] if dollar_odds is not None else None

        # Calibrated win probabilities
        raw_probs = softmax(race_scores)
        cal_probs = calibrator.predict(raw_probs)
        cal_probs = cal_probs / cal_probs.sum()

        # Top-3 probabilities
        top3_probs = estimate_top3_probs(race_scores, size)

        # Rank
        rank_order = np.argsort(-race_scores)

        for local_i in range(size):
            global_i = idx + local_i
            pred_rank = int(np.where(rank_order == local_i)[0][0]) + 1
            odds_val = race_odds[local_i] if race_odds is not None else np.nan

            row = {
                'race_id': race_id,
                'race_number': display_cols.get('race_number', [''])[global_i],
                'horse_name': display_cols.get('horse_name', [''])[global_i],
                'original_horse_name': display_cols.get('horse_name', [''])[global_i],
                'jockey': display_cols.get('jockey', [''])[global_i],
                'trainer': display_cols.get('trainer', [''])[global_i],
                'predicted_rank': pred_rank,
                'predicted_finish': pred_rank,  # alias for predict_api.py
                'win_probability': round(cal_probs[local_i], 4),
                'top3_probability': round(top3_probs[local_i], 4),
                'score': round(race_scores[local_i], 4),
                'odds': odds_val if not np.isnan(odds_val) else None,
            }

            if not np.isnan(odds_val) and odds_val > 0:
                implied = 1.0 / (odds_val + 1)
                row['implied_prob'] = round(implied, 4)
                row['edge'] = round(cal_probs[local_i] / implied, 2) if implied > 0 else 0
                row['value_bet'] = 'YES' if cal_probs[local_i] > implied * args.edge else ''
                row['ev_per_dollar'] = round(cal_probs[local_i] * odds_val - 1, 3)

            results.append(row)

        idx += size

    # Output
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['race_id', 'predicted_rank'])
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to {args.output}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("PREDICTION SUMMARY")
    print(f"{'=' * 70}")

    for race_id in unique_races[:5]:  # Show first 5 races
        race_rows = results_df[results_df['race_id'] == race_id].head(10)
        print(f"\n{'─' * 60}")
        print(f"Race: {race_id}")
        print(f"{'Horse':<25} {'Rank':>5} {'Win%':>7} {'Top3%':>7} {'Odds':>7} {'Edge':>6} {'Bet':>4}")
        print(f"{'─' * 60}")
        for _, row in race_rows.iterrows():
            odds_str = f"{row.get('odds', '')}" if 'odds' in row and pd.notna(row.get('odds')) else ''
            edge_str = f"{row.get('edge', '')}" if 'edge' in row and row.get('edge') else ''
            bet_str = row.get('value_bet', '')
            print(f"  {row['horse_name']:<23} {row['predicted_rank']:>5} "
                  f"{row['win_probability']:>6.1%} {row['top3_probability']:>6.1%} "
                  f"{odds_str:>7} {edge_str:>6} {bet_str:>4}")

    value_bets = results_df[results_df.get('value_bet', '') == 'YES'] if 'value_bet' in results_df.columns else pd.DataFrame()
    if len(value_bets) > 0:
        print(f"\n{'=' * 60}")
        print(f"VALUE BETS FOUND: {len(value_bets)}")
        print(f"{'=' * 60}")
        for _, row in value_bets.iterrows():
            print(f"  {row['race_id']} | {row['horse_name']:<20} | "
                  f"Win: {row['win_probability']:.1%} vs Implied: {row['implied_prob']:.1%} | "
                  f"Edge: {row['edge']:.2f}x | EV: ${row['ev_per_dollar']:+.3f}/$ | "
                  f"Odds: {row['dollar_odds']:.1f}")


if __name__ == '__main__':
    main()
