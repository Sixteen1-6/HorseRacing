"""
Apply dispatch-payload edits (scratches, odds overrides, jockey changes,
added horses) to the base race card CSV.

Reads:
    .dispatch/payload.json  (written by the workflow from client_payload)

Payload shape:
    {
      "overrides":  { "<race>::<horse>": <float odds>, ... },
      "scratches":  [ "<race>::<horse>", ... ],
      "jockeys":    { "<race>::<horse>": "New Jockey Name", ... },
      "added":      [ { race: "3", horse_name: "...", jockey: "...",
                        trainer: "...", post_position: "5", dollar_odds: "8" }, ... ],
      "card_path":  "test_data.csv",
      "run_id":     "...",
      "track_code": "KEE",
      "card_date":  "2026-04-16",
      "race_num":   ""
    }
"""
import json
import os
import sys

import pandas as pd

DISPATCH = ".dispatch/payload.json"
OUT = "test_data.csv"


def horse_key(race, name):
    return f"{int(race)}::{name}"


def main():
    if not os.path.exists(DISPATCH):
        print("No dispatch payload found, skipping edits")
        return

    with open(DISPATCH) as f:
        payload = json.load(f)

    edits = payload.get("edits", payload)

    card_path = payload.get("card_path") or OUT
    if not os.path.exists(card_path):
        print(f"Card file not found: {card_path}, using {OUT}")
        card_path = OUT

    df = pd.read_csv(card_path, low_memory=False)
    print(f"Loaded {len(df)} rows from {card_path}")

    df["race_number"] = pd.to_numeric(df["race_number"], errors="coerce").astype("Int64")

    # 1) Scratches -- drop rows
    scratches = set(edits.get("scratches", []))
    if scratches:
        before = len(df)
        df["_key"] = df.apply(lambda r: horse_key(r["race_number"], r["horse_name"]), axis=1)
        df = df[~df["_key"].isin(scratches)].drop(columns=["_key"])
        print(f"Scratched {before - len(df)} horses")

    # 2) Odds overrides
    overrides = edits.get("overrides", {}) or {}
    if overrides:
        if "dollar_odds" not in df.columns:
            df["dollar_odds"] = pd.NA
        df["_key"] = df.apply(lambda r: horse_key(r["race_number"], r["horse_name"]), axis=1)
        n = 0
        for key, odds in overrides.items():
            mask = df["_key"] == key
            if mask.any():
                df.loc[mask, "dollar_odds"] = float(odds)
                n += 1
        df = df.drop(columns=["_key"])
        print(f"Applied {n} odds overrides")

    # 3) Jockey changes
    jockeys = edits.get("jockeys", {}) or {}
    if jockeys:
        df["_key"] = df.apply(lambda r: horse_key(r["race_number"], r["horse_name"]), axis=1)
        n = 0
        for key, jockey in jockeys.items():
            mask = df["_key"] == key
            if mask.any():
                df.loc[mask, "jockey"] = jockey
                n += 1
        df = df.drop(columns=["_key"])
        print(f"Applied {n} jockey changes")

    # 4) Added horses
    added = edits.get("added", []) or []
    if added:
        new_rows = []
        for h in added:
            row = {c: None for c in df.columns}
            row["race_number"] = int(h.get("race", h.get("race_number", 1)))
            row["horse_name"] = h.get("horse_name", "")
            row["jockey"] = h.get("jockey", "Unknown")
            row["trainer"] = h.get("trainer", "Unknown")
            if h.get("post_position"):
                row["post_position"] = int(h["post_position"])
            if h.get("dollar_odds"):
                row["dollar_odds"] = float(h["dollar_odds"])
            # Copy race context from siblings
            sibs = df[df["race_number"] == row["race_number"]]
            if len(sibs):
                for c in ["race_type", "purse", "surface", "distance", "distance_unit",
                           "track_condition", "breed", "course"]:
                    if c in sibs.columns:
                        row[c] = sibs.iloc[0][c]
            new_rows.append(row)
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"Added {len(new_rows)} horses")

    df = df.sort_values("race_number").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")


if __name__ == "__main__":
    main()
