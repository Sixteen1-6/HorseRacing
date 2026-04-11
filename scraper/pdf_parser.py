"""
Equibase chart PDF parser — extracts structured race data from PDF bytes.
"""

import io
import re
import logging
from datetime import datetime
from typing import List, Optional, Dict, Tuple

from .config import OUTPUT_COLUMNS, TRACKS
from .parsing import parse_distance_integer, map_table_columns, parse_odds, parse_time

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

log = logging.getLogger("scraper")


class ChartPDFParser:
    """
    Parses Equibase full chart PDFs into structured data.

    Equibase chart PDF layout:
    ┌──────────────────────────────────────────────────┐
    │ TRACK NAME - DATE - RACE N                       │
    │ Distance, Surface, Condition, Race Type, Purse   │
    │ Fractional Times: :22.40  :45.60  1:10.20        │
    │ Final Time: 1:36.80                              │
    ├──┬───┬────────────┬──────┬────┬────┬────┬────┬───┤
    │PP│Pgm│Horse       │Jockey│Wgt │Start│1C  │2C  │Str│Fin│Odds│Comment│
    │  │   │            │      │    │Pos/M│P/M │P/M │P/M│P/M│    │       │
    ├──┼───┼────────────┼──────┼────┼────┼────┼────┼───┤
    │ 2│ 2 │Winner      │Smith │122 │1 - │1 Hd│1 1 │1 2│1  │2.50│led...  │
    │ 5│ 5 │Runner Up   │Jones │120 │3 2½│2 1 │2 Nk│2 1│2 3│5.00│chased..│
    └──┴───┴────────────┴──────┴────┴────┴────┴────┴───┘
    """

    # Regex patterns for Equibase chart PDFs
    RACE_HEADER = re.compile(
        r"(?:RACE|Race)\s*(\d+)", re.IGNORECASE
    )
    DISTANCE_PATTERN = re.compile(
        r"((?:One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|Eleven|Twelve|"
        r"About|And|A|One\s+And)[\w\s-]*(?:Furlong|Mile|Yard)s?)",
        re.IGNORECASE,
    )
    # Also match no-space PDF format: "FourAndOneHalfFurlongsOnTheDirt"
    DISTANCE_PATTERN_NOSPACE = re.compile(
        r"Distance:([\w]+(?:Furlong|Mile|Yard)s?)",
        re.IGNORECASE,
    )
    SURFACE_PATTERN = re.compile(r"\b(Dirt|Turf|Synthetic|All Weather)\b", re.IGNORECASE)
    # Also match no-space: "OnTheDirt", "OnTheTurf"
    SURFACE_PATTERN_NOSPACE = re.compile(r"OnThe(Dirt|Turf)", re.IGNORECASE)
    CONDITION_PATTERN = re.compile(
        r"(?:Track:|Track\s*Condition:)\s*(Fast|Good|Muddy|Sloppy|Firm|Yielding|Soft|Heavy|Wet\s*Fast|Slow)",
        re.IGNORECASE,
    )
    PURSE_PATTERN = re.compile(r"Purse:\$?([\d,]+)")
    FRACTIONAL_PATTERN = re.compile(r"(?:FractionalTimes:|Fractional\s*Times:)\s*([\d\.\s:]+)")
    FINAL_TIME_PATTERN = re.compile(r"(?:FinalTime:|Final\s*Time:)\s*([\d\.:]+)")
    RACE_TYPE_PATTERN = re.compile(
        r"\b(Maiden Special Weight|Maiden Claiming|Claiming|Allowance Optional Claiming|"
        r"Allowance|Stakes|Starter Allowance|Starter Optional Claiming|Optional Claiming|"
        r"Handicap|Starter|MAIDEN\s*SPECIAL\s*WEIGHT|MAIDEN\s*CLAIMING|CLAIMING|"
        r"ALLOWANCE|STAKES|HANDICAP|MAIDENSPECIALWEIGHT|MAIDENCLAIMING|ALLOWANCEOPTIONALCLAIMING)\b",
        re.IGNORECASE,
    )

    def parse_pdf_bytes(self, pdf_bytes: bytes, track_code: str, race_date: str) -> List[Dict]:
        """Parse PDF bytes into list of entry dicts."""
        if not pdfplumber:
            log.warning("pdfplumber not available — skipping PDF parse")
            return []

        entries = []
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                full_text = ""
                all_tables = []
                for page_obj in pdf.pages:
                    text = page_obj.extract_text() or ""
                    full_text += text + "\n"
                    tables = page_obj.extract_tables()
                    if tables:
                        all_tables.extend(tables)

                entries = self._parse_chart_content(full_text, all_tables, track_code, race_date)
        except Exception as e:
            log.error(f"PDF parse error: {e}")

        return entries

    # ─── Training-schema lookups ───
    CONDITION_MAP = {
        "fast": "FT", "good": "GD", "muddy": "MY", "sloppy": "SY",
        "firm": "FM", "yielding": "YL", "soft": "SF", "heavy": "HY",
        "wetfast": "WF", "wet fast": "WF", "slow": "SL", "frozen": "FZ",
    }

    def _split_camel(self, text: str) -> str:
        """Insert spaces in CamelCase: 'BourbonTown' -> 'Bourbon Town'.
        Preserves common name prefixes (Mc, Mac, O', De, La, Van, Le)."""
        if not text:
            return ""
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
        # Unsplit name-prefix artifacts like "Mc Peek" -> "McPeek"
        s = re.sub(r"\bMc\s+([A-Z])", r"Mc\1", s)
        s = re.sub(r"\bMac\s+([A-Z])", r"Mac\1", s)
        s = re.sub(r"\bDe\s+([A-Z])", r"De\1", s)
        s = re.sub(r"\bDi\s+([A-Z])", r"Di\1", s)
        s = re.sub(r"\bLa\s+([A-Z])", r"La\1", s)
        s = re.sub(r"\bO'\s*([A-Z])", r"O'\1", s)
        return s

    def _parse_course(self, raw_block: str) -> str:
        """Determine course from block text. Returns Dirt/Turf/Inner turf/Outer turf/Hurdle/Timber."""
        t = raw_block.lower().replace(" ", "")
        if "innerturf" in t:
            return "Inner turf"
        if "outerturf" in t:
            return "Outer turf"
        if "hurdle" in t:
            return "Hurdle"
        if "timber" in t:
            return "Timber"
        if "onthedirt" in t or "dirtcourse" in t:
            return "Dirt"
        if "ontheturf" in t or "turfcourse" in t:
            return "Turf"
        return ""

    def _parse_surface(self, course: str) -> str:
        """Derive single-letter surface from course."""
        if not course:
            return ""
        c = course.lower()
        if "turf" in c or "hurdle" in c or "timber" in c:
            return "T"
        return "D"

    def _parse_condition_code(self, raw: str) -> str:
        """Convert 'Sloppy(Sealed)' or 'Fast' to 2-letter training code."""
        if not raw:
            return ""
        cleaned = re.sub(r"\(.*\)", "", raw).strip().lower()
        cleaned = cleaned.replace(" ", "")
        return self.CONDITION_MAP.get(cleaned, "")

    def _parse_race_type_clean(self, raw: str) -> str:
        """Split 'MAIDENSPECIALWEIGHT' into 'Maiden Special Weight' (title case)."""
        if not raw:
            return ""
        # Known training-data race types
        known = [
            "Maiden Special Weight", "Maiden Claiming", "Maiden",
            "Allowance Optional Claiming", "Allowance", "Starter Allowance",
            "Starter Optional Claiming", "Optional Claiming", "Claiming",
            "Handicap", "Invitational Stakes", "Invitational",
            "Claiming Stakes", "Stakes", "Derby Trial", "Derby",
            "Futurity Trial", "Futurity", "Championship", "Final",
            "Match Race", "Speed Index Final", "Speed Index Race",
        ]
        # Normalize by stripping whitespace
        collapsed = re.sub(r"\s+", "", raw).upper()
        for kt in known:
            if re.sub(r"\s+", "", kt).upper() == collapsed:
                return kt
        # Fallback: camel split + title case
        return " ".join(self._split_camel(raw).title().split())

    def _parse_sex_from_winner(self, winner_text: str) -> str:
        """From 'Suspicions,BayColt,byCorniche...' extract 'Colt'."""
        if not winner_text:
            return ""
        t = winner_text.lower()
        for sex in ["Ridgling", "Gelding", "Filly", "Colt", "Mare", "Horse"]:
            if sex.lower() in t:
                return sex
        return ""

    def _parse_sex_from_conditions(self, block: str) -> str:
        """From 'FORFILLIESTHREEYEARSOLD' or 'FORFILLIESANDMARES' etc."""
        t = block.upper().replace(" ", "")
        # Find the "FOR ..." eligibility line
        m = re.search(r"FOR([A-Z,]+?)(?:WHICH|\.|WEIGHT|NON-WINNER|CLAIMING|\d)", t)
        scope = m.group(1) if m else t[:500]
        if "FILLIESANDMARES" in scope:
            return "Filly"  # default; mares would need per-horse age
        if "FILLIES" in scope and "COLTS" not in scope:
            return "Filly"
        if "MARES" in scope and "HORSES" not in scope:
            return "Mare"
        if "COLTSANDGELDINGS" in scope:
            return "Colt"
        return ""

    def _parse_age_from_conditions(self, block: str) -> Optional[int]:
        """From race conditions 'FORMAIDENS,TWOYEARSOLD' or 'FORFILLIESTHREEYEARSOLD'."""
        t = block.upper().replace(" ", "").replace(",", "")
        ages = [
            ("TWOYEARSOLD", 2), ("THREEYEARSOLD", 3), ("FOURYEARSOLD", 4),
            ("FIVEYEARSOLD", 5), ("SIXYEARSOLD", 6),
            ("TWOYEAROLDS", 2), ("THREEYEAROLDS", 3), ("FOURYEAROLDS", 4),
        ]
        for k, v in ages:
            if k in t:
                return v
        if "THREEYEARSOLDANDUPWARD" in t or "THREEYEARSOLDSANDUPWARD" in t:
            return 3
        if "FOURYEARSOLDANDUPWARD" in t or "FOURYEARSOLDSANDUPWARD" in t:
            return 4
        return None

    def _format_jockey(self, raw: str) -> str:
        """'Moran,Pietro' or 'Ortiz,Jr.,Irad' -> 'Pietro Moran' / 'Irad Ortiz Jr.'"""
        if not raw:
            return ""
        raw = raw.strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) == 1:
            return self._split_camel(parts[0])
        if len(parts) == 2:
            last, first = parts
            return f"{self._split_camel(first)} {self._split_camel(last)}".strip()
        # 3 parts like "Ortiz,Jr.,Irad"
        last, suffix, first = parts[0], parts[1], parts[-1]
        return f"{self._split_camel(first)} {self._split_camel(last)} {suffix}".strip()

    def _format_track_name_upper(self, code: str) -> str:
        name = TRACKS.get(code, code)
        return name.upper()

    def _format_date_slashy(self, iso_date: str) -> str:
        """'2026-04-03' -> '4/3/2026'."""
        try:
            d = datetime.strptime(iso_date, "%Y-%m-%d")
            return f"{d.month}/{d.day}/{d.year}"
        except Exception:
            return iso_date

    def _parse_last_raced_token(self, tok: str) -> Dict[str, str]:
        """
        Parse Equibase 'last raced' token like '20Dec2510FG6' or '---'.
        Format: DDMMMYY + race_number + track_code + finish_position
        e.g. '20Dec2510FG6' = Dec 20 2025, race 10, FG track, finish 6.
        Note: race_number can be 1-2 digits, track_code 2-5 letters, finish 1-2 digits.
        """
        out = {"last_race_date": "", "last_race_number": "",
               "last_race_track": "", "last_race_finish": ""}
        if not tok or tok.strip() in ("---", "--", "-"):
            return out
        m = re.match(
            r"^(\d{1,2})([A-Z][a-z]{2})(\d{2})(\d{1,2})([A-Z]{2,5})(\d{1,2})$",
            tok.strip(),
        )
        if not m:
            return out
        day, mon, yy, race_num, track, finish = m.groups()
        month_names = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                       "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        mn = month_names.get(mon, 0)
        year = 2000 + int(yy)
        out["last_race_date"] = f"{mn}/{int(day)}/{year}"
        out["last_race_number"] = race_num
        out["last_race_track"] = self._format_track_name_upper(track)
        out["last_race_finish"] = finish
        return out

    def _parse_trainer_owner_list(self, block: str, key: str) -> Dict[str, str]:
        """
        Parse 'Trainers: 8-Ward,Wesley;5-Hernandez,Rey;...' or 'Owners: ...'
        Returns dict keyed by program number -> formatted name.
        """
        out = {}
        # Section stops at the next labelled section header
        if key.lower() == "trainers":
            stop = r"(?:Owners:|Footnotes|Copyright|ViewGlossary|Breeder:|Scratched|\Z)"
        else:  # owners
            stop = r"(?:Trainers:|Footnotes|Copyright|ViewGlossary|Breeder:|Scratched|\Z)"
        m = re.search(rf"{key}:\s*(.+?){stop}", block, re.DOTALL | re.IGNORECASE)
        if not m:
            return out
        raw = m.group(1).replace("\n", "")
        for piece in raw.split(";"):
            piece = piece.strip()
            if not piece or "-" not in piece:
                continue
            pgm, _, name = piece.partition("-")
            pgm = pgm.strip()
            name = name.strip().rstrip(",;")
            if not pgm.isdigit() or not name:
                continue
            if key.lower() == "trainers":
                out[pgm] = self._format_jockey(name)  # Last,First,(Suffix)
            else:
                out[pgm] = self._split_camel(name)
        return out

    def _parse_chart_content(self, text: str, tables: list,
                             track_code: str, race_date: str) -> List[Dict]:
        """Parse extracted text and tables from a chart PDF."""
        entries = []

        # Split text into race blocks by the 'TRACK-Date-Race N' header
        race_blocks = re.split(r"(?=[A-Z]+-[A-Z][a-z]+\d+,\d{4}-Race\s*\d+)", text)
        if len(race_blocks) <= 1:
            race_blocks = re.split(r"(?=(?:RACE|Race)\s*\d+)", text)

        for block in race_blocks:
            if not block.strip():
                continue

            race_num_m = self.RACE_HEADER.search(block)
            race_num = int(race_num_m.group(1)) if race_num_m else 0
            if race_num == 0:
                continue

            # ─── Extract race-level fields ───
            # Distance: 'Distance:FourAndOneHalfFurlongsOnTheDirtCurrentTrackRecord:...'
            dist_m = re.search(r"Distance:([A-Za-z]+?)(?:CurrentTrackRecord|$)", block)
            dist_raw = dist_m.group(1) if dist_m else ""
            dist_int, dist_unit = parse_distance_integer(dist_raw)

            course = self._parse_course(dist_raw + " " + block[:2000])
            surface = self._parse_surface(course)

            # Track condition: 'Track:Sloppy(Sealed)' or 'Track:Fast'
            cond_m = re.search(r"Track:([A-Za-z]+(?:\([^)]*\))?)", block)
            track_condition = self._parse_condition_code(cond_m.group(1) if cond_m else "")

            # Weather: 'Weather:Showery,67°'
            weather_m = re.search(r"Weather:([A-Za-z]+)", block)
            weather = weather_m.group(1).strip() if weather_m else ""

            # Post time: 'Offat:1:05'
            off_m = re.search(r"Offat:([\d:]+)", block)
            post_time = off_m.group(1) if off_m else ""

            # Purse
            purse_m = re.search(r"Purse:\$?([\d,]+)", block)
            purse = purse_m.group(1).replace(",", "") if purse_m else ""

            # Race type: first line after the race header, e.g. 'MAIDENSPECIALWEIGHT-Thoroughbred' or 'STAKESCentralBankAshlandS.Grade1-Thoroughbred'
            type_m = re.search(r"Race\s*\d+\s*\n?(?:\*|\[)?\s*\n?([A-Z][A-Z0-9&\- ]+?)(?:CentralBank|-Thoroughbred|-QuarterHorse|-Arabian|\n|FOR)", block)
            race_type_raw = ""
            if type_m:
                race_type_raw = type_m.group(1).strip()
            # Fall back to known-types regex
            if not race_type_raw:
                rt_m = re.search(
                    r"(MAIDENSPECIALWEIGHT|MAIDENCLAIMING|ALLOWANCEOPTIONALCLAIMING|"
                    r"STARTERALLOWANCE|STARTEROPTIONALCLAIMING|OPTIONALCLAIMING|"
                    r"ALLOWANCE|CLAIMING|STAKES|HANDICAP|MAIDEN)", block)
                if rt_m:
                    race_type_raw = rt_m.group(1)
            race_type = self._parse_race_type_clean(race_type_raw)

            # Breed: 'MAIDENSPECIALWEIGHT-Thoroughbred'
            breed_raw_m = re.search(r"-(Thoroughbred|QuarterHorse|Arabian|Paint|AppaloosaAPHA|Appaloosa)", block)
            breed_map = {"Thoroughbred": "TB", "QuarterHorse": "QH",
                         "Arabian": "AR", "Paint": "PT", "Appaloosa": "AP", "AppaloosaAPHA": "AP"}
            breed = breed_map.get(breed_raw_m.group(1), "TB") if breed_raw_m else "TB"

            # Age and sex from conditions
            age_val = self._parse_age_from_conditions(block)
            sex_from_conditions = self._parse_sex_from_conditions(block)

            # Fractional times
            frac_m = re.search(r"FractionalTimes:([\d\.\s:]+?)FinalTime:", block)
            frac_times_raw = frac_m.group(1).strip().split() if frac_m else []
            final_m = re.search(r"FinalTime:([\d:.]+)", block)
            win_time = parse_time(final_m.group(1)) if final_m else ""

            # Winner details block: 'Winner: Suspicions,BayColt,byCornicheoutofManaoag...'
            winner_m = re.search(r"Winner:\s*([^\n]+?)(?=Breeder:|Owner:|\n)", block)
            winner_text = winner_m.group(1) if winner_m else ""
            winner_sex = self._parse_sex_from_winner(winner_text)

            # Trainers and owners by program number
            trainer_map = self._parse_trainer_owner_list(block, "Trainers")
            owner_map = self._parse_trainer_owner_list(block, "Owners")

            # Claiming price: 'ClaimingPrice:$50,000-$0'
            claim_m = re.search(r"ClaimingPrice:\$?([\d,]+)", block)
            claimed_price = claim_m.group(1).replace(",", "") if claim_m else ""

            race_info = {
                "race_number": race_num,
                "track_code": track_code,
                "track_name": self._format_track_name_upper(track_code),
                "race_date": self._format_date_slashy(race_date),
                "distance": dist_int,
                "distance_unit": dist_unit,
                "course": course,
                "surface": surface,
                "track_condition": track_condition,
                "weather": weather,
                "post_time": post_time,
                "win_time": win_time,
                "purse": purse,
                "race_type": race_type,
                "breed": breed,
                "_age_default": age_val,
                "_winner_sex": winner_sex,
                "_sex_from_conditions": sex_from_conditions,
                "_trainer_map": trainer_map,
                "_owner_map": owner_map,
                "claimed_price": claimed_price,
                "frac_1": parse_time(frac_times_raw[0]) if len(frac_times_raw) > 0 else "",
                "frac_2": parse_time(frac_times_raw[1]) if len(frac_times_raw) > 1 else "",
                "frac_3": parse_time(frac_times_raw[2]) if len(frac_times_raw) > 2 else "",
                "frac_4": parse_time(frac_times_raw[3]) if len(frac_times_raw) > 3 else "",
                "final_time_secs": win_time,
            }

            race_entries = self._parse_results_from_text(block, race_info)
            if not race_entries:
                race_entries = self._parse_results_from_tables(tables, race_info)

            # Strip internal-only keys before emitting
            for e in race_entries:
                for k in ("_age_default", "_winner_sex", "_sex_from_conditions",
                          "_trainer_map", "_owner_map"):
                    e.pop(k, None)
            entries.extend(race_entries)

        return entries

    def _parse_results_from_tables(self, tables: list, race_info: dict) -> List[Dict]:
        """Try to find and parse the results table for this race."""
        entries = []
        for table in tables:
            if not table or len(table) < 3:
                continue

            # Check if this table looks like a race results table
            header = [str(c).lower().strip() if c else "" for c in table[0]]
            has_horse = any("horse" in h or "name" in h for h in header)
            has_pgm = any("pgm" in h or "#" in h or "no" in h for h in header)

            if not (has_horse or has_pgm):
                continue

            col = map_table_columns(header)

            if "horse" not in col:
                continue

            for row_idx, row in enumerate(table[1:], 1):
                if not row or len(row) <= col.get("horse", 0):
                    continue

                def get(key):
                    idx = col.get(key)
                    if idx is not None and idx < len(row) and row[idx]:
                        return str(row[idx]).strip()
                    return ""

                horse = get("horse")
                if not horse or len(horse) < 2 or horse.lower() == "horse":
                    continue

                entry = {**race_info}
                entry["horse_name"] = horse
                entry["program_num"] = get("pgm")
                entry["post_position"] = get("pp")
                entry["finish"] = get("finish") or str(row_idx)
                entry["jockey"] = get("jockey")
                entry["trainer"] = get("trainer")
                entry["owner"] = get("owner")
                entry["weight"] = get("weight")
                entry["dollar_odds"] = parse_odds(get("odds"))
                entry["comment"] = get("comment")
                entry["medication"] = get("med")
                entry["age"] = get("age")
                entry["sex"] = get("sex")
                entry["margin_finish"] = self._parse_margin(get("margin"))
                entry["pos_1st_call"] = self._extract_pos(get("pos_1st"))
                entry["pos_2nd_call"] = self._extract_pos(get("pos_2nd"))
                entry["pos_stretch"] = self._extract_pos(get("pos_str"))
                entry["margin_1st_call"] = self._extract_margin(get("pos_1st"))
                entry["margin_2nd_call"] = self._extract_margin(get("pos_2nd"))
                entry["margin_stretch"] = self._extract_margin(get("pos_str"))

                # Fill missing columns
                for c in OUTPUT_COLUMNS:
                    if c not in entry:
                        entry[c] = ""

                entries.append(entry)

            if entries:
                break  # found the right table

        return entries

    def _parse_results_from_text(self, block: str, race_info: dict) -> List[Dict]:
        """Parse horse entries from Equibase chart PDF text.

        Equibase PDF line format (no-space variant):
          --- 8 Suspicions(Moran,Pietro) 119 b 5 2 22 12 12 1.30* bmpst,2w,clrd,hldswy
          20Dec2510FG6 5 MissCall(Hernandez,Jr.,Brian) 120 L 4 6 31/2 41 11/2 163/4 2.90 2p,bid...

        Fields: LastRaced Pgm HorseName(Jockey) Wgt M/E PP Start 1/4 [1/2] [3/4] Str Fin Odds Comment
        """
        entries = []
        lines = block.split("\n")
        finish_pos = 0

        # LastRaced token is either '---' or a compact date/track/race/finish code.
        # HorseName is CamelCase (no spaces) followed by (Jockey,Name).
        horse_line_re = re.compile(
            r"^(---|[0-9A-Za-z]+?)\s+"          # Last raced
            r"(\d{1,2})\s+"                      # Program number
            r"([A-Z][A-Za-z'\.0-9\- ]*?)\(([^)]+)\)\s*"  # HorseName(Jockey)
            r"(\d{2,3})\s*"                      # Weight
            r"([A-Za-z\-]{1,5}|--)\s+"           # Medication/equip
            r"(\d{1,2})\s+"                      # Post position
            r"(\d{1,2})\s+"                      # Start position
            r"(.+)$"                             # Rest: running positions + odds + comment
        )

        trainer_map = race_info.get("_trainer_map", {})
        owner_map = race_info.get("_owner_map", {})
        age_default = race_info.get("_age_default")
        winner_sex = race_info.get("_winner_sex", "")
        sex_from_conditions = race_info.get("_sex_from_conditions", "")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip header/section lines
            if line.startswith("LastRaced") or line.startswith("FractionalTimes") or line.startswith("Pgm "):
                continue
            m = horse_line_re.match(line)
            if not m:
                continue

            finish_pos += 1
            last_raced, pgm, horse, jockey, weight, med_equip, pp, _start, rest = m.groups()

            # Running positions + odds + comment
            rest_parts = rest.split()
            odds = ""
            comment_parts = []
            positions = []
            for i, part in enumerate(rest_parts):
                # Odds: 0.78*, *1.40, 26.07
                if re.match(r"^\*?\d+\.\d+\*?$", part) and "/" not in part:
                    odds = part.replace("*", "")
                    comment_parts = rest_parts[i + 1:]
                    break
                positions.append(part)

            # Odds as integer dollar form (training schema uses int)
            try:
                dollar_odds_int = int(round(float(odds))) if odds else ""
            except ValueError:
                dollar_odds_int = ""

            # Last raced parse
            lr = self._parse_last_raced_token(last_raced)

            # Medication: 'b', 'L', 'Lb', 'Lbf', '--'
            med_clean = med_equip if med_equip and med_equip != "--" else ""
            # Training schema uses L, BL, B, LA - map by presence
            med_upper = med_clean.upper()
            if "L" in med_upper and "B" in med_upper:
                medication = "BL"
            elif "L" in med_upper and "A" in med_upper:
                medication = "LA"
            elif "L" in med_upper:
                medication = "L"
            elif "B" in med_upper:
                medication = "B"
            else:
                medication = ""

            entry = {**race_info}
            entry["horse_name"] = self._split_camel(horse.strip())
            entry["program_num"] = int(pgm) if pgm.isdigit() else pgm
            entry["post_position"] = int(pp) if pp.isdigit() else pp
            entry["finish"] = finish_pos
            entry["jockey"] = self._format_jockey(jockey)
            entry["trainer"] = trainer_map.get(pgm, "")
            entry["owner"] = owner_map.get(pgm, "")
            entry["weight"] = int(weight) if weight.isdigit() else weight
            entry["dollar_odds"] = dollar_odds_int
            entry["medication"] = medication
            entry["comment"] = " ".join(comment_parts).replace(",", " ").strip()
            entry["age"] = age_default if age_default is not None else ""
            # Winner gets exact sex from Winner line; others use condition-derived default.
            if finish_pos == 1 and winner_sex:
                entry["sex"] = winner_sex
            else:
                entry["sex"] = sex_from_conditions
            entry.update(lr)

            for c in OUTPUT_COLUMNS:
                if c not in entry:
                    entry[c] = ""
            entries.append(entry)

        return entries

    def _extract_margin_from_pos(self, text):
        """Extract margin from position text like '111/2' or '3Head' or '21/4'."""
        if not text:
            return ""
        # Remove leading position number to get margin
        # "111/2" = 1 and 1/2 lengths, "3Head" = head margin at 3rd, etc
        text = str(text).strip()
        specials = {"head": 0.1, "hd": 0.1, "nose": 0.05, "neck": 0.25, "nk": 0.25}
        text_lower = text.lower()
        for key, val in specials.items():
            if key in text_lower:
                return val
        # Try to extract fraction: "11/2" = 0.5, "31/4" = 0.25
        m = re.search(r"(\d+)/(\d+)", text)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            # Could be "11/2" meaning "1 and 1/2" or just "1/2"
            prefix = text[:m.start()]
            whole = int(prefix) if prefix and prefix.isdigit() else 0
            return whole + num / den
        return ""

    def _parse_margin(self, text):
        if not text:
            return ""
        text = str(text).strip()
        if text in ("---", "—", "-"):
            return 0.0
        specials = {"head": 0.1, "hd": 0.1, "nose": 0.05, "neck": 0.25, "nk": 0.25}
        if text.lower() in specials:
            return specials[text.lower()]
        fracs = {"¼": 0.25, "½": 0.5, "¾": 0.75}
        for ch, val in fracs.items():
            if ch in text:
                w = text.replace(ch, "").strip()
                return (int(w) if w else 0) + val
        try:
            return float(text)
        except ValueError:
            return ""

    def _extract_pos(self, text):
        """Extract running position from '3 2½' → 3"""
        if not text:
            return ""
        m = re.match(r"(\d+)", str(text).strip())
        return int(m.group(1)) if m else ""

    def _extract_margin(self, text):
        """Extract margin from '3 2½' → 2.5"""
        if not text:
            return ""
        parts = str(text).strip().split()
        if len(parts) >= 2:
            return self._parse_margin(parts[-1])
        return ""

