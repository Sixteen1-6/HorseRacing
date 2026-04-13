"""Shared parsing utilities for the horse racing scraper.

Consolidates distance, table-column, odds, and time parsing that was
previously duplicated across entries.py, pdf_parser.py, and run.py.
"""

import re
from typing import Dict, List, Optional, Tuple


# ─── Distance parsing ────────────────────────────────────────

WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

# Base fractions — composite fractions like "three quarters" are handled
# by endswith matching: "quarter" matches, prefix "three" = 3 * 0.25.
FRACTIONS = {
    "half": 0.5,
    "quarter": 0.25,
    "sixteenth": 1 / 16,
    "eighth": 1 / 8,
    "third": 1 / 3,
}


def parse_distance(text: str) -> Tuple[Optional[float], str]:
    """Parse spelled-out distance to (furlongs, unit).

    Handles both spaced ("Seven Furlongs") and CamelCase/no-space
    ("SevenFurlongs", "OneAndOneSixteenthMiles") formats.

    Returns (furlongs_float, unit) where unit is "F" or "Y".
    Returns (None, "") if unparseable.
    """
    if not text:
        return (None, "")

    t = text.lower().strip().rstrip(".")
    # Collapse spaces/hyphens for uniform handling of CamelCase PDFs
    t_collapsed = re.sub(r"[\s\-]+", "", t)
    # Strip surface tail (PDF format: "...OnTheDirt")
    t_collapsed = re.sub(r"onthe(dirt|turf|allweather|synthetic).*$", "", t_collapsed)
    t_collapsed = re.sub(r"(innerturf|outerturf|turfcourse|dirtcourse).*$", "", t_collapsed)

    # Detect unit
    is_miles = "mile" in t_collapsed
    is_yards = "yard" in t_collapsed and "furlong" not in t_collapsed and "mile" not in t_collapsed
    is_furlongs = "furlong" in t_collapsed
    if not (is_miles or is_furlongs or is_yards):
        return (None, "")

    # Strip unit and "about" prefix
    t_clean = re.sub(r"(furlongs?|miles?|yards?)", "", t_collapsed)
    t_clean = t_clean.replace("about", "")

    # Split on "and"
    whole = 0
    frac = 0.0
    parts = t_clean.split("and")

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Try fraction matching (longest key first)
        matched_frac = None
        for k in sorted(FRACTIONS.keys(), key=len, reverse=True):
            if part.endswith(k):
                matched_frac = k
                break

        if matched_frac:
            num_text = part[:-len(matched_frac)].strip()
            num_val = WORD_NUMS.get(num_text, 1) if num_text else 1
            frac += num_val * FRACTIONS[matched_frac]
        elif part in WORD_NUMS:
            whole += WORD_NUMS[part]
        else:
            # Hundreds pattern: "twohundredtwenty" -> 220
            m = re.match(r"(" + "|".join(WORD_NUMS.keys()) + r")hundred", part)
            if m:
                whole = WORD_NUMS[m.group(1)] * 100
                rest = part[len(m.group(0)):]
                if rest in WORD_NUMS:
                    whole += WORD_NUMS[rest]
            else:
                try:
                    whole += float(part)
                except ValueError:
                    pass

    total = whole + frac
    if total == 0:
        return (None, "")

    if is_yards:
        return (total, "Y")
    if is_miles:
        return (total * 8.0, "F")
    return (total, "F")


def parse_distance_integer(text: str) -> Tuple[int, str]:
    """Parse distance to (furlongs * 100, unit).

    Convenience wrapper for PDF parser compatibility.
    """
    furlongs, unit = parse_distance(text)
    if furlongs is None:
        return (0, "F")
    if unit == "Y":
        return (int(round(furlongs)), "Y")
    return (int(round(furlongs * 100)), "F")


def parse_distance_to_furlongs(text: str) -> Optional[float]:
    """Parse distance to furlongs as float.

    Convenience wrapper for entries parser compatibility.
    """
    furlongs, _unit = parse_distance(text)
    return furlongs


# ─── Table column mapping ────────────────────────────────────

def map_table_columns(headers: List[str]) -> Dict[str, int]:
    """Map table header strings to standardized column name -> index.

    Recognized keys: horse, pgm, pp, age_sex, med, jockey, weight,
    trainer, odds, owner, finish, comment, margin, pos_1st, pos_2nd,
    pos_str, pos_start, age, sex.
    """
    col: Dict[str, int] = {}
    for i, h in enumerate(headers):
        hl = h.lower().strip()
        if any(k in hl for k in ["horse", "name", "starter"]) and "horse" not in col:
            col["horse"] = i
        elif (hl in ("p#", "pgm", "no.") or "pgm" in hl) and "pgm" not in col:
            col["pgm"] = i
        elif (hl == "pp" or "post" in hl) and "pp" not in col:
            col["pp"] = i
        elif ("a/s" in hl or ("age" in hl and "sex" not in hl)) and "age_sex" not in col:
            col["age_sex"] = i
        elif hl == "med" and "med" not in col:
            col["med"] = i
        elif "jock" in hl and "jockey" not in col:
            col["jockey"] = i
        elif (hl in ("wgt", "wt") or "weight" in hl) and "weight" not in col:
            col["weight"] = i
        elif "train" in hl and "trainer" not in col:
            col["trainer"] = i
        elif ("m/l" in hl or hl == "ml" or "odds" in hl) and "odds" not in col:
            col["odds"] = i
        elif "owner" in hl and "owner" not in col:
            col["owner"] = i
        elif any(k in hl for k in ["fin", "pos"]) and "finish" not in col:
            col["finish"] = i
        elif any(k in hl for k in ["comment", "remark"]) and "comment" not in col:
            col["comment"] = i
        elif any(k in hl for k in ["margin", "btn", "behind", "lengths"]) and "margin" not in col:
            col["margin"] = i
        elif ("1st" in hl or "first" in hl or hl == "1/4") and "pos_1st" not in col:
            col["pos_1st"] = i
        elif ("2nd" in hl or "second" in hl or hl == "1/2") and "pos_2nd" not in col:
            col["pos_2nd"] = i
        elif hl == "3/4" and "pos_3rd" not in col:
            col["pos_3rd"] = i
        elif "str" in hl and "start" not in hl and "pos_str" not in col:
            col["pos_str"] = i
        elif "start" in hl and "pos_start" not in col:
            col["pos_start"] = i
        elif "sex" in hl and "sex" not in col:
            col["sex"] = i
    return col


# ─── Odds parsing ────────────────────────────────────────────

def parse_odds(text: str) -> str:
    """Convert odds text to decimal string.

    Handles: '6/1' -> '6', '9/5' -> '1.8', '$3.40' -> '3.4',
    '3-1' -> '3.0', plain numbers.
    """
    text = (text or "").strip().replace("$", "")
    # Fractional: 6/1, 9/5
    m = re.match(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if m:
        try:
            n, d = float(m.group(1)), float(m.group(2))
            if d != 0:
                return str(round(n / d, 2)).rstrip("0").rstrip(".")
        except ValueError:
            pass
    # Dash format: 3-1
    m = re.match(r"(\d+)-(\d+)", text)
    if m:
        try:
            n, d = int(m.group(1)), int(m.group(2))
            if d != 0:
                return str(round(n / d, 2))
        except (ValueError, ZeroDivisionError):
            pass
    # Plain number
    m = re.search(r"[\d.]+", text)
    return m.group(0) if m else ""


# ─── Time parsing ─────────────────────────────────────────────

def parse_time(text) -> str:
    """Parse race time to seconds. '1:36.80' -> 96.8, ':45.60' -> 45.6."""
    if not text:
        return ""
    text = str(text).strip().lstrip(":")
    try:
        if ":" in text:
            parts = text.split(":")
            return round(int(parts[0]) * 60 + float(parts[1]), 2)
        return round(float(text), 2)
    except ValueError:
        return ""
