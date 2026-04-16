"""
Remove old prediction files for the same track/date/race combo.
Keeps only the most recent run.
"""

import argparse
import os
import json
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True, help='Predictions directory')
    p.add_argument('--keep', required=True, help='Filename to keep')
    p.add_argument('--track', default='')
    p.add_argument('--date', default='')
    p.add_argument('--race', default='')
    p.add_argument('--max-age', type=int, default=120, help='Max age in seconds')
    args = p.parse_args()

    if not os.path.isdir(args.dir):
        return

    now = time.time()
    removed = 0

    for fname in os.listdir(args.dir):
        if fname == args.keep or not fname.endswith('.json'):
            continue

        fpath = os.path.join(args.dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
            meta = data.get('metadata', {})
            if (meta.get('track_code') == args.track and
                meta.get('card_date') == args.date):
                age = now - os.path.getmtime(fpath)
                if age < args.max_age:
                    os.remove(fpath)
                    removed += 1
        except (json.JSONDecodeError, KeyError, OSError):
            continue

    if removed:
        print(f"Cleaned up {removed} superseded prediction(s)")


if __name__ == '__main__':
    main()
