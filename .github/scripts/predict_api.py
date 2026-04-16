"""
Convert CSV predictions to JSON format for the predictions API.

Usage:
  python predict_api.py --input race_predictions/all_race_predictions.csv \
                        --output predictions/run_xxx.json \
                        --track KEE --date 2026-04-16 --race 3
"""

import argparse
import json
import os
from datetime import datetime

import pandas as pd


def csv_to_json(csv_path, track_code, card_date, race_num=None):
    """Convert prediction CSV to structured JSON."""
    df = pd.read_csv(csv_path)

    if race_num:
        df = df[df['race_number'] == int(race_num)]

    races = {}
    for rn, group in df.groupby('race_number'):
        group = group.sort_values('predicted_rank')
        horses = []
        for _, row in group.iterrows():
            horse = {
                'rank': int(row['predicted_rank']),
                'horse_name': str(row.get('horse_name', '')),
                'jockey': str(row.get('jockey', '')),
                'trainer': str(row.get('trainer', '')),
                'odds': row.get('dollar_odds') if pd.notna(row.get('dollar_odds')) else None,
                'predicted_finish': int(row['predicted_rank']),
                'model_probability': round(float(row.get('win_probability', 0)), 4),
                'top3_probability': round(float(row.get('top3_probability', 0)), 4),
                'market_probability': round(float(row.get('implied_prob', 0)), 4) if pd.notna(row.get('implied_prob')) else None,
                'edge': round(float(row.get('edge', 0)), 4) if pd.notna(row.get('edge')) else None,
                'value_bet': row.get('value_bet', '') == 'YES',
                'kelly_bet': round(float(row.get('kelly_bet', 0)), 2) if pd.notna(row.get('kelly_bet')) else 0,
                'ev_per_dollar': round(float(row.get('ev_per_dollar', 0)), 3) if pd.notna(row.get('ev_per_dollar')) else None,
            }
            horses.append(horse)
        races[f'race_{int(rn)}'] = {
            'race_number': int(rn),
            'num_horses': len(horses),
            'horses': horses,
        }

    output = {
        'metadata': {
            'track_code': track_code,
            'card_date': card_date,
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'model_version': 'v4',
            'num_races': len(races),
        },
        'races': races,
    }

    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='Input CSV predictions')
    p.add_argument('--output', required=True, help='Output JSON path')
    p.add_argument('--track', default='', help='Track code')
    p.add_argument('--date', default='', help='Card date YYYY-MM-DD')
    p.add_argument('--race', default=None, help='Filter to specific race number')
    args = p.parse_args()

    result = csv_to_json(args.input, args.track, args.date, args.race)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    n_horses = sum(r['num_horses'] for r in result['races'].values())
    print(f"Saved {len(result['races'])} races, {n_horses} horses -> {args.output}")


if __name__ == '__main__':
    main()
