"""
Horse Racing Prediction — v4 Conditional Logit Model
=====================================================
Loads the trained model and predicts race outcomes with Kelly sizing.

Usage:
  python predict.py --input today_races.csv
  python predict.py --input today_races.csv --output predictions.csv
  python predict.py --input today_races.csv --model with_odds
  python predict.py --input today_races.csv --model no_odds

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
    cat_cols = ['surface', 'track_condition', 'weather', 'breed',
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

    # Sex re-encoding: replace raw categorical with orthogonal binaries
    if 'sex' in df.columns:
        sex_upper = df['sex'].astype(str).str.strip().str.upper()
        df['is_female'] = sex_upper.isin(['FILLY', 'MARE', 'F', 'M']).astype(int)
        df['is_gelding'] = sex_upper.isin(['GELDING', 'G']).astype(int)
        race_group = ['track_code', 'race_date', 'race_number']
        if all(c in df.columns for c in race_group):
            race_sex_nunique = df.groupby(race_group, observed=False)['is_female'].transform('nunique')
            df['is_restricted_sex_race'] = (race_sex_nunique == 1).astype(int)
        else:
            df['is_restricted_sex_race'] = 0
        df.drop(columns=['sex'], inplace=True, errors='ignore')

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
    elif 'race_number' in df.columns:
        df['race_id'] = 'race_' + df['race_number'].astype(str)
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
        # Current-race data not available at prediction time
        'final_time_secs', 'speed_figure_equibase', 'claimed_price',
        'start_pos', 'pos_1st_call', 'margin_1st_call',
        'pos_2nd_call', 'margin_2nd_call', 'pos_3rd_call', 'margin_3rd_call',
        'pos_stretch', 'margin_stretch', 'pos_finish', 'margin_finish',
        'had_trouble', 'wide_trip', 'poor_start', 'strong_close', 'faded',
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


# ── Harville exotic pricing ──

HARVILLE_LAMBDA = 0.81  # Bacon-Shone/Lo correction

def harville_place_prob(win_probs, placed, lam=HARVILLE_LAMBDA):
    """P(each horse finishes next | horses in placed already finished)."""
    mask = np.ones(len(win_probs), dtype=bool)
    for i in placed:
        mask[i] = False
    remaining = win_probs[mask] ** lam
    total = remaining.sum()
    if total == 0:
        return np.zeros(len(win_probs))
    result = np.zeros(len(win_probs))
    result[mask] = remaining / total
    return result

def compute_exacta_prob(win_probs, i, j):
    """P(i 1st, j 2nd)."""
    return win_probs[i] * harville_place_prob(win_probs, [i])[j]

def compute_trifecta_prob(win_probs, i, j, k):
    """P(i 1st, j 2nd, k 3rd)."""
    p1 = win_probs[i]
    p2 = harville_place_prob(win_probs, [i])[j]
    p3 = harville_place_prob(win_probs, [i, j])[k]
    return p1 * p2 * p3

def get_top_exotic_combos(win_probs, horse_names, horse_odds, top_n=5, combo_type='trifecta'):
    """Compute top exotic combos ranked by expected value."""
    n = len(win_probs)
    if n < 3:
        return []

    # Use top horses by win prob to limit combos
    top_k = min(6, n)
    top_indices = np.argsort(-win_probs)[:top_k]

    combos = []
    if combo_type == 'exacta':
        for i in top_indices:
            for j in top_indices:
                if i == j:
                    continue
                prob = compute_exacta_prob(win_probs, i, j)
                combos.append((prob, [i, j]))
    else:  # trifecta
        for i in top_indices:
            for j in top_indices:
                if j == i:
                    continue
                for k in top_indices:
                    if k == i or k == j:
                        continue
                    prob = compute_trifecta_prob(win_probs, i, j, k)
                    combos.append((prob, [i, j, k]))

    combos.sort(key=lambda x: -x[0])
    results = []
    for prob, indices in combos[:top_n * 3]:
        names = [horse_names[i] for i in indices]
        # Estimate fair odds: 1/prob
        fair_odds = 1.0 / prob if prob > 0 else 999
        results.append({
            'combo': ' / '.join(names),
            'indices': indices,
            'probability': round(prob, 5),
            'fair_odds': round(fair_odds, 1),
        })

    return results[:top_n]


# ── Kelly sizing ──

KELLY_FRACTION = 0.25
KELLY_CAP_PCT = 0.03
MIN_EDGE_PCT = 0.08

def kelly_size(model_prob, dollar_odds, bankroll):
    """Returns (bet_amount, edge). 0 if no bet."""
    if np.isnan(dollar_odds) or dollar_odds <= 0 or np.isnan(model_prob):
        return 0.0, 0.0
    decimal_odds = dollar_odds + 1
    implied = 1.0 / decimal_odds
    edge = model_prob - implied
    if edge < MIN_EDGE_PCT:
        return 0.0, edge
    kelly_f = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
    kelly_f = max(kelly_f, 0)
    bet = bankroll * KELLY_FRACTION * kelly_f
    bet = min(bet, bankroll * KELLY_CAP_PCT, 500)
    return round(bet, 2), round(edge, 4)


# ── Pace narrative ──

def pace_narrative(running_styles):
    """Generate pace scenario description from horse styles."""
    style_counts = {}
    for s in running_styles:
        style_counts[s] = style_counts.get(s, 0) + 1

    n_speed = style_counts.get('E', 0) + style_counts.get('P', 0)
    n_close = style_counts.get('S', 0) + style_counts.get('C', 0)
    n_unknown = style_counts.get('U', 0)
    total = len(running_styles)

    if n_speed == 0:
        return "No confirmed speed. Tactical pace likely. Stalkers favored."
    if n_speed == 1:
        return f"LONE SPEED — 1 front-runner vs {n_close} closers. Speed horse has major advantage."
    if n_speed == 2:
        return f"Moderate pace — 2 speed types. Pressers and stalkers favored."
    if n_speed >= 3:
        return f"HOT PACE — {n_speed} speed types will duel. Closers strongly advantaged."
    return f"Mixed: {n_speed} speed, {n_close} closers, {n_unknown} unknown."


def main():
    parser = argparse.ArgumentParser(description="Predict horse race outcomes")
    parser.add_argument('--input', default='test_data.csv', help='Input CSV with race entries')
    parser.add_argument('--output', default='race_predictions/all_race_predictions.csv',
                        help='Output CSV with predictions')
    parser.add_argument('--models-dir', default='models', help='Directory with model artifacts')
    parser.add_argument('--model', default='with_odds', choices=['with_odds', 'no_odds'],
                        help='Which model to use (default: with_odds)')
    parser.add_argument('--edge', type=float, default=1.10, help='Edge threshold for value bets')
    args = parser.parse_args()

    # Load model artifacts (v4 naming: ranking_{with_odds|no_odds}_{artifact})
    suffix = args.model
    model_path = os.path.join(args.models_dir, f'ranking_{suffix}_model.lgb')
    calibrator_path = os.path.join(args.models_dir, f'ranking_{suffix}_calibrator.pkl')
    features_path = os.path.join(args.models_dir, f'ranking_{suffix}_feature_names.pkl')
    cat_path = os.path.join(args.models_dir, f'ranking_{suffix}_cat_features.pkl')
    jt_path = os.path.join(args.models_dir, 'ranking_jt_lookup.json')

    # Fallback to old naming if v4 files not found
    if not os.path.exists(model_path):
        model_path = os.path.join(args.models_dir, 'ranking_model.lgb')
        calibrator_path = os.path.join(args.models_dir, 'ranking_calibrator.pkl')
        features_path = os.path.join(args.models_dir, 'ranking_feature_names.pkl')
        cat_path = os.path.join(args.models_dir, 'ranking_cat_features.pkl')

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
    # Remove scratched horses (empty horse_name or race_number)
    scratched = df['horse_name'].isna() | (df['horse_name'].astype(str).str.strip() == '')
    if scratched.any():
        print(f"  Removed {scratched.sum()} scratched entries")
    df = df[~scratched].reset_index(drop=True)
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
    exotic_results = []
    unique_races = []
    seen = set()
    for r in race_ids:
        if r not in seen:
            unique_races.append(r)
            seen.add(r)

    # Get running styles for pace narrative
    running_styles_col = df['running_style'].values if 'running_style' in df.columns else None

    bankroll = 10000
    idx = 0
    for race_id in unique_races:
        race_mask = race_ids == race_id
        size = race_mask.sum()
        race_scores = scores[idx:idx + size]
        race_odds = dollar_odds[idx:idx + size] if dollar_odds is not None else None

        if size < 2:
            idx += size
            continue

        # Calibrated win probabilities
        raw_probs = softmax(race_scores)
        cal_probs = calibrator.predict(raw_probs)
        cal_probs = cal_probs / cal_probs.sum()

        # Top-3 probabilities
        top3_probs = estimate_top3_probs(race_scores, size)

        # Rank
        rank_order = np.argsort(-race_scores)

        # Horse names for this race
        race_names = [display_cols.get('horse_name', [''])[idx + i] for i in range(size)]

        # Pace narrative
        race_styles = []
        if running_styles_col is not None:
            race_styles = [str(running_styles_col[idx + i]) for i in range(size)]
        pace_text = pace_narrative(race_styles) if race_styles else ""

        # Confidence: % of horses with speed figure history
        n_with_history = sum(1 for i in range(size) if not np.isnan(cal_probs[i]) and cal_probs[i] > 0)
        confidence = "HIGH" if n_with_history / size > 0.8 else "MEDIUM" if n_with_history / size > 0.5 else "LOW"

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
                'predicted_finish': pred_rank,
                'win_probability': round(cal_probs[local_i], 4),
                'top3_probability': round(top3_probs[local_i], 4),
                'score': round(race_scores[local_i], 4),
                'dollar_odds': odds_val if not np.isnan(odds_val) else None,
                'odds': odds_val if not np.isnan(odds_val) else None,
                'running_style': race_styles[local_i] if race_styles else '',
                'pace_narrative': pace_text if local_i == 0 else '',
                'confidence': confidence,
            }

            if not np.isnan(odds_val) and odds_val > 0:
                implied = 1.0 / (odds_val + 1)
                row['implied_prob'] = round(implied, 4)
                row['edge'] = round(cal_probs[local_i] / implied, 2) if implied > 0 else 0
                row['value_bet'] = 'YES' if cal_probs[local_i] > implied * args.edge else ''
                row['ev_per_dollar'] = round(cal_probs[local_i] * odds_val - 1, 3)

                # Kelly sizing
                bet_amt, edge_raw = kelly_size(cal_probs[local_i], odds_val, bankroll)
                row['kelly_bet'] = bet_amt
            else:
                row['kelly_bet'] = 0

            results.append(row)

        # Exotic combos for this race
        race_odds_arr = race_odds if race_odds is not None else np.full(size, np.nan)
        top_tri = get_top_exotic_combos(cal_probs, race_names, race_odds_arr,
                                         top_n=5, combo_type='trifecta')
        top_exa = get_top_exotic_combos(cal_probs, race_names, race_odds_arr,
                                         top_n=3, combo_type='exacta')

        race_num = display_cols.get('race_number', [''])[idx]
        for combo in top_tri:
            combo['race_id'] = race_id
            combo['race_number'] = race_num
            combo['type'] = 'TRIFECTA'
        for combo in top_exa:
            combo['race_id'] = race_id
            combo['race_number'] = race_num
            combo['type'] = 'EXACTA'
        exotic_results.extend(top_tri)
        exotic_results.extend(top_exa)

        idx += size

    # Output
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['race_id', 'predicted_rank'])
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to {args.output}")

    # Save exotics
    if exotic_results:
        exotic_df = pd.DataFrame(exotic_results)
        exotic_path = args.output.replace('.csv', '_exotics.csv')
        exotic_df.to_csv(exotic_path, index=False)
        print(f"Saved exotic combos to {exotic_path}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("PREDICTION SUMMARY — v4 Conditional Logit")
    print(f"{'=' * 80}")

    for race_id in unique_races:
        race_rows = results_df[results_df['race_id'] == race_id].head(12)
        if race_rows.empty:
            continue

        race_num = race_rows['race_number'].iloc[0]
        confidence = race_rows['confidence'].iloc[0] if 'confidence' in race_rows.columns else ''
        pace_text = race_rows['pace_narrative'].iloc[0] if 'pace_narrative' in race_rows.columns else ''

        print(f"\n{'━' * 80}")
        print(f"  RACE {race_num}  |  {len(race_rows)} horses  |  Confidence: {confidence}")
        if pace_text:
            print(f"  Pace: {pace_text}")
        print(f"{'━' * 80}")
        print(f"  {'Horse':<22} {'Style':>5} {'Rank':>5} {'Win%':>7} {'Top3%':>7} {'Odds':>6} {'Edge':>6} {'Kelly$':>8}")
        print(f"  {'─' * 72}")
        for _, row in race_rows.iterrows():
            odds_str = f"{row['odds']:.0f}" if pd.notna(row.get('odds')) and row.get('odds') else ''
            edge_str = f"{row['edge']:.2f}" if row.get('edge') and row.get('edge', 0) > 1.0 else ''
            kelly_str = f"${row['kelly_bet']:.0f}" if row.get('kelly_bet', 0) > 0 else ''
            style_str = str(row.get('running_style', ''))
            marker = ' <<' if row.get('value_bet') == 'YES' else ''
            print(f"  {row['horse_name']:<22} {style_str:>5} {row['predicted_rank']:>5} "
                  f"{row['win_probability']:>6.1%} {row['top3_probability']:>6.1%} "
                  f"{odds_str:>6} {edge_str:>6} {kelly_str:>8}{marker}")

        # Show top exotics for this race
        race_exotics = [e for e in exotic_results if e.get('race_id') == race_id]
        tris = [e for e in race_exotics if e['type'] == 'TRIFECTA']
        exas = [e for e in race_exotics if e['type'] == 'EXACTA']

        if exas:
            print(f"\n  Top Exactas:")
            for e in exas[:3]:
                print(f"    {e['combo']:<45} P={e['probability']:.4f}  Fair ${e['fair_odds']:.0f}")
        if tris:
            print(f"  Top Trifectas:")
            for t in tris[:5]:
                print(f"    {t['combo']:<45} P={t['probability']:.4f}  Fair ${t['fair_odds']:.0f}")

    # Value bets summary
    value_bets = results_df[results_df.get('value_bet', '') == 'YES'] if 'value_bet' in results_df.columns else pd.DataFrame()
    if len(value_bets) > 0:
        print(f"\n{'=' * 80}")
        print(f"  VALUE BETS: {len(value_bets)} found")
        print(f"{'=' * 80}")
        total_kelly = 0
        for _, row in value_bets.iterrows():
            kelly = row.get('kelly_bet', 0)
            total_kelly += kelly
            print(f"  R{row['race_number']} {row['horse_name']:<20} "
                  f"Win: {row['win_probability']:.1%} vs Market: {row['implied_prob']:.1%} "
                  f"Edge: {row['edge']:.2f}x  Odds: {row['odds']:.0f}-1  "
                  f"Kelly: ${kelly:.0f}")
        print(f"\n  Total Kelly wagering: ${total_kelly:.0f} (on ${bankroll:,} bankroll)")


if __name__ == '__main__':
    main()
