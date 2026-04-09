"""
Entries-mode scraper for Equibase.

Unlike scraper.py (which scrapes post-race CHARTS), this fetches PRE-RACE
ENTRIES for a given track and date, then enriches each horse with career
stats and last-race info from its profile page.

Output schema matches test_data.csv:
  race_number, horse_name, jockey, trainer, weight, age, post_position,
  race_type, purse, surface, distance, distance_unit, dollar_odds, sex,
  num_past_starts, num_past_wins, num_past_seconds, num_past_thirds,
  last_race_date, last_race_finish, track_condition, breed, course,
  last_race_number

Usage:
  # Start Chrome with debug port:
  .\\start_chrome_debug3.ps1

  # Then:
  python scraper_entries.py --track KEE --date 2026-04-10 --output test_data.csv
"""
import argparse
import asyncio
import csv
import io
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

from playwright.async_api import async_playwright

# Reuse the human-behavior + track table from the chart scraper
from scraper import HumanBehavior, TRACKS

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


TEST_DATA_COLUMNS = [
    "race_number", "horse_name", "jockey", "trainer", "weight", "age",
    "post_position", "race_type", "purse", "surface", "distance",
    "distance_unit", "dollar_odds", "sex",
    "num_past_starts", "num_past_wins", "num_past_seconds", "num_past_thirds",
    "last_race_date", "last_race_finish", "track_condition", "breed", "course",
    "last_race_number",
]

# Convert spelled-out distances to furlongs.
# Equibase entries use phrases like "Seven Furlongs", "One Mile",
# "One And One Sixteenth Miles", "Five And One Half Furlongs".
WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
FRACTIONS = {
    "half": 0.5,
    "sixteenth": 1 / 16,
    "eighth": 1 / 8,
    "quarter": 0.25,
    "third": 1 / 3,
    "three quarters": 0.75,
    "three sixteenths": 3 / 16,
    "five sixteenths": 5 / 16,
    "seven sixteenths": 7 / 16,
}


def parse_distance_to_furlongs(text: str) -> Optional[float]:
    """Convert e.g. 'One And One Sixteenth Miles' -> 8.5 (furlongs)."""
    if not text:
        return None
    t = text.lower().strip().rstrip(".")

    # Detect unit
    is_miles = "mile" in t
    is_furlongs = "furlong" in t
    if not (is_miles or is_furlongs):
        return None

    # Strip unit
    t = re.sub(r"\b(miles?|furlongs?|yards?)\b", "", t).strip()

    # Handle "X And Y Z" → X + Y/Z parts (e.g. "one and one sixteenth")
    base = 0.0
    parts = t.split(" and ")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Try fractions first (longer phrases)
        matched_frac = None
        for k in sorted(FRACTIONS.keys(), key=len, reverse=True):
            if part.endswith(k):
                matched_frac = k
                break
        if matched_frac:
            num_text = part[: -len(matched_frac)].strip()
            num_val = WORD_NUMS.get(num_text, 1) if num_text else 1
            base += num_val * FRACTIONS[matched_frac]
        else:
            # Whole-number word, e.g. "seven"
            num_val = WORD_NUMS.get(part)
            if num_val is not None:
                base += num_val
            else:
                # Could be a digit
                try:
                    base += float(part)
                except ValueError:
                    pass

    if base == 0:
        return None
    return base * 8.0 if is_miles else base


def parse_ml_odds(text: str) -> str:
    """Convert '6/1' -> '6', '9/5' -> '1.8', etc."""
    text = (text or "").strip()
    m = re.match(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if m:
        try:
            n, d = float(m.group(1)), float(m.group(2))
            return str(round(n / d, 2)).rstrip("0").rstrip(".")
        except ValueError:
            pass
    return text


def parse_age_sex(text: str) -> tuple:
    """'4/G' -> (4, 'G')"""
    text = (text or "").strip()
    m = re.match(r"(\d+)\s*/\s*([A-Za-z])", text)
    if m:
        return int(m.group(1)), m.group(2).upper()
    return None, None


def clean_horse_name(text: str) -> str:
    """'Torre Eiffel (KY)' -> 'Torre Eiffel'."""
    return re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", (text or "").strip())


# ─────────────────────────────────────────────────────────────────
# Entries page parser
# ─────────────────────────────────────────────────────────────────

class EntriesScraper:
    ENTRIES_URL = "https://www.equibase.com/static/entry/{track}{mmddyy}USA-EQB.html"

    def __init__(self, cdp_port: int = 9222, fetch_pps: bool = True, max_horses: int = 0):
        self.cdp_port = cdp_port
        self.fetch_pps = fetch_pps
        self.max_horses = max_horses  # 0 = unlimited (debug aid)
        self.human = HumanBehavior()
        self.pw = None
        self.browser = None
        self.context = None
        self.page = None

    async def start(self):
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.connect_over_cdp(
            f"http://127.0.0.1:{self.cdp_port}"
        )
        self.context = self.browser.contexts[0]
        self.page = await self.context.new_page()
        print(f"[entries] Connected to Chrome on :{self.cdp_port}")

    async def stop(self):
        if self.page:
            await self.page.close()
        if self.pw:
            await self.pw.stop()

    # ── main entry: fetch one (track, date) ────────────────────

    async def scrape(self, track: str, date: datetime) -> List[Dict]:
        mmddyy = date.strftime("%m%d%y")
        url = self.ENTRIES_URL.format(track=track, mmddyy=mmddyy)
        print(f"[entries] Fetching {url}")
        await self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
        await asyncio.sleep(3)

        text = await self.page.inner_text("body")
        if "Pardon Our Interruption" in text:
            print("!!! Bot detection — solve captcha in Chrome window then press Enter")
            input()
            await self.page.reload(wait_until="domcontentloaded", timeout=20000)
            await asyncio.sleep(3)

        # Pull both raw text (for race-level metadata) and DOM tables (for rows)
        race_blocks_text = self._split_text_into_race_blocks(
            await self.page.inner_text("body")
        )

        tables = await self.page.evaluate(
            r"""
            () => {
                return Array.from(document.querySelectorAll('table.fullwidth')).map(tbl => {
                    const rows = Array.from(tbl.querySelectorAll('tr')).map(tr =>
                        Array.from(tr.querySelectorAll('th, td')).map(c => c.innerText.trim())
                    );
                    const rowLinks = Array.from(tbl.querySelectorAll('tr')).map(tr =>
                        Array.from(tr.querySelectorAll('a[href]')).map(a => ({
                            href: a.href, text: (a.innerText || '').trim()
                        }))
                    );
                    return {rows, rowLinks};
                });
            }
            """
        )

        if len(tables) != len(race_blocks_text):
            print(f"[entries] WARN: {len(tables)} tables vs {len(race_blocks_text)} text blocks "
                  "— alignment may be off")

        all_entries = []
        track_name = TRACKS.get(track, track)

        for i, table in enumerate(tables):
            block_text = race_blocks_text[i] if i < len(race_blocks_text) else ""
            race_meta = self._parse_race_meta(block_text)
            race_meta["track_code"] = track
            race_meta["track_name"] = track_name
            race_meta["race_date"] = date.strftime("%Y-%m-%d")

            entries = self._parse_entry_table(table["rows"], table["rowLinks"], race_meta)
            print(f"[entries] Race {race_meta.get('race_number','?')}: "
                  f"{race_meta.get('race_type','?')} {race_meta.get('distance_raw','?')} "
                  f"{race_meta.get('surface','?')} -> {len(entries)} horses")
            all_entries.extend(entries)

        # Past-performance enrichment
        if self.fetch_pps:
            await self._enrich_with_pps(all_entries)

        return all_entries

    def _split_text_into_race_blocks(self, body_text: str) -> List[str]:
        """Split body text into per-race blocks using 'RACE N' line markers.
        Stakes races include the stakes name between the number and POST TIME,
        e.g. 'RACE 7 FanDuel Limestone S. (Grade III) POST TIME', so we split
        on the line-anchored 'RACE N' pattern and trust each match.
        """
        positions = [
            (m.start(), int(m.group(1)))
            for m in re.finditer(r"^RACE\s+(\d+)\b", body_text, re.MULTILINE)
        ]
        if not positions:
            return []
        blocks = []
        for i, (start, race_num) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(body_text)
            blocks.append(body_text[start:end])
        return blocks

    def _parse_race_meta(self, block: str) -> Dict:
        """Extract race-level fields from a text block."""
        meta = {}

        # Race number from header line (allow stakes name between num and POST)
        m = re.search(r"^RACE\s+(\d+)\b", block, re.MULTILINE)
        if m:
            meta["race_number"] = int(m.group(1))

        # Stakes detection: if header line has "(Grade ...)" or " S." or
        # ends with "Stakes", it's a stakes race regardless of body text.
        header_line = block.split("\n", 1)[0]
        is_stakes = bool(
            re.search(r"\(Grade\s+[IVX]+\)", header_line)
            or re.search(r"\bS\.\s", header_line)
            or re.search(r"\bStakes\b", header_line, re.I)
        )

        if is_stakes:
            meta["race_type"] = "Stakes"
        else:
            # Look for the all-caps race-type line that follows the header,
            # e.g. "Keeneland MAIDEN CLAIMING $20,000" or "Keeneland ALLOWANCE"
            race_type_m = re.search(
                r"\b(MAIDEN\s+SPECIAL\s+WEIGHT|MAIDEN\s+CLAIMING|"
                r"ALLOWANCE\s+OPTIONAL\s+CLAIMING|OPTIONAL\s+CLAIMING|"
                r"STARTER\s+ALLOWANCE|STARTER\s+OPTIONAL\s+CLAIMING|"
                r"ALLOWANCE|CLAIMING|HANDICAP|STAKES|MAIDEN)\b",
                block,
            )
            if race_type_m:
                meta["race_type"] = race_type_m.group(1).title().replace("  ", " ")

        purse_m = re.search(r"Purse\s*\$?([\d,]+)", block)
        if purse_m:
            meta["purse"] = purse_m.group(1).replace(",", "")

        # Distance phrase: between "Purse ..." sentence and the next period
        # Examples: "Seven Furlongs.", "One And One Sixteenth Miles."
        dist_m = re.search(
            r"(?:Purse[^.]*\.\s*)([A-Z][A-Za-z\s]+?(?:Furlongs?|Miles?|Yards?))\.",
            block,
        )
        distance_raw = ""
        if dist_m:
            distance_raw = dist_m.group(1).strip()
        meta["distance_raw"] = distance_raw

        furlongs = parse_distance_to_furlongs(distance_raw)
        if furlongs is not None:
            meta["distance"] = round(furlongs, 2)
            meta["distance_unit"] = "F"

        # Surface: "(Turf)" appears in description for turf races
        if re.search(r"\(\s*Turf\s*\)", block, re.I):
            meta["surface"] = "Turf"
            meta["track_condition"] = "FM"
        elif re.search(r"\(\s*All\s*Weather\s*\)", block, re.I):
            meta["surface"] = "All Weather"
            meta["track_condition"] = "FT"
        else:
            meta["surface"] = "Dirt"
            meta["track_condition"] = "FT"

        return meta

    def _parse_entry_table(self, rows: List[List[str]], row_links: List[List[Dict]],
                           race_meta: Dict) -> List[Dict]:
        """Parse one race's entries table into row dicts."""
        if not rows:
            return []

        # Find the header row (contains "Horse" and "Jockey")
        header_idx = None
        for i, row in enumerate(rows):
            joined = " ".join(row).lower()
            if "horse" in joined and "jockey" in joined:
                header_idx = i
                break
        if header_idx is None:
            return []

        headers = [h.strip() for h in rows[header_idx]]
        # Map header name → index
        col = {}
        for i, h in enumerate(headers):
            hl = h.lower()
            if hl == "p#" or hl == "pgm":
                col["pgm"] = i
            elif hl == "pp":
                col["pp"] = i
            elif "horse" in hl:
                col["horse"] = i
            elif "a/s" in hl or "age" in hl:
                col["age_sex"] = i
            elif hl == "med":
                col["med"] = i
            elif "jock" in hl:
                col["jockey"] = i
            elif hl == "wgt" or "weight" in hl:
                col["weight"] = i
            elif "train" in hl:
                col["trainer"] = i
            elif "m/l" in hl:
                col["ml"] = i

        if "horse" not in col:
            return []

        entries = []
        for ri, row in enumerate(rows[header_idx + 1:], start=header_idx + 1):
            if not row:
                continue
            # Skip rows that are clearly not horse rows (e.g. "Also Eligibles:")
            if len(row) < 4:
                continue

            def cell(key):
                idx = col.get(key)
                if idx is None or idx >= len(row):
                    return ""
                return (row[idx] or "").strip()

            horse_raw = cell("horse")
            if not horse_raw or horse_raw.lower() == "horse":
                continue
            # Skip section headers like "Also Eligibles:"
            if ":" in horse_raw and len(horse_raw.split()) <= 3:
                continue

            age_val, sex_val = parse_age_sex(cell("age_sex"))

            entry = {
                **race_meta,
                "program_num": cell("pgm"),
                "post_position": cell("pp") or cell("pgm"),
                "horse_name": clean_horse_name(horse_raw),
                "jockey": cell("jockey"),
                "trainer": cell("trainer"),
                "weight": cell("weight"),
                "age": age_val,
                "sex": sex_val,
                "medication": cell("med"),
                "dollar_odds": parse_ml_odds(cell("ml")),
                "breed": "TB",
                "course": "M",
                # Past-performance fields filled later
                "num_past_starts": "",
                "num_past_wins": "",
                "num_past_seconds": "",
                "num_past_thirds": "",
                "last_race_date": "",
                "last_race_finish": "",
                "last_race_number": "",
            }

            # Pull horse profile URL from row hyperlinks
            if ri < len(row_links):
                for link in row_links[ri]:
                    if "type=Horse" in link.get("href", ""):
                        entry["_horse_url"] = link["href"]
                        break

            entries.append(entry)

        return entries

    # ── horse profile / past performance enrichment ────────────

    async def _enrich_with_pps(self, entries: List[Dict]):
        urls = [(i, e) for i, e in enumerate(entries) if e.get("_horse_url")]
        if self.max_horses:
            urls = urls[: self.max_horses]
        print(f"[entries] Enriching {len(urls)} horses with past-performance data...")

        for n, (i, entry) in enumerate(urls, 1):
            try:
                pp = await self._fetch_horse_pp(entry["_horse_url"])
                # Retry once if first attempt got nothing (race-condition on
                # first navigation from entries page)
                if not pp.get("num_past_starts") and not pp.get("last_race_date"):
                    await asyncio.sleep(2.0)
                    pp = await self._fetch_horse_pp(entry["_horse_url"])
                entry.update(pp)
                print(f"  [{n}/{len(urls)}] {entry['horse_name']:30s} "
                      f"starts={pp.get('num_past_starts','')} "
                      f"last={pp.get('last_race_date','')}/{pp.get('last_race_finish','')}")
            except Exception as e:
                print(f"  [{n}/{len(urls)}] {entry['horse_name']:30s} FAILED: {e}")
            await self.human.random_delay(0.8, 1.8)

    async def _fetch_horse_pp(self, url: str) -> Dict:
        await self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
        await asyncio.sleep(1.5)

        data = await self.page.evaluate(
            r"""
            () => {
                const out = {bodyText: document.body.innerText};

                // Career stats: look for the table whose header is exactly
                // [Starts, Firsts, Seconds, Thirds, Earnings] and take the
                // SECOND such occurrence (the first is the current-year stats)
                const careerTables = Array.from(document.querySelectorAll('table'))
                    .filter(t => {
                        const h = t.rows[0];
                        if (!h) return false;
                        const txt = Array.from(h.cells).map(c => c.innerText.trim()).join('|');
                        return txt === 'Starts|Firsts|Seconds|Thirds|Earnings';
                    });
                out.careerTables = careerTables.map(t =>
                    Array.from(t.rows).map(r =>
                        Array.from(r.cells).map(c => c.innerText.trim())
                    )
                );

                // Results table (most recent races)
                const results = document.querySelector('table.results, table.phone-collapse.results');
                if (results) {
                    out.results = Array.from(results.rows).map(r =>
                        Array.from(r.cells).map(c => c.innerText.trim())
                    );
                } else {
                    // Fallback: any table whose first column header is "Track" and includes "Finish"
                    const t = Array.from(document.querySelectorAll('table')).find(t => {
                        const h = t.rows[0];
                        if (!h) return false;
                        const txt = Array.from(h.cells).map(c => c.innerText.trim()).join('|');
                        return txt.includes('Track') && txt.includes('Finish');
                    });
                    out.results = t ? Array.from(t.rows).map(r =>
                        Array.from(r.cells).map(c => c.innerText.trim())
                    ) : null;
                }

                return out;
            }
            """
        )

        result = {}

        # Parse career stats — second occurrence is the all-time totals
        career_tables = data.get("careerTables") or []
        career_row = None
        if len(career_tables) >= 2:
            # Header + 1 data row
            career_row = career_tables[1][1] if len(career_tables[1]) > 1 else None
        elif len(career_tables) == 1:
            career_row = career_tables[0][1] if len(career_tables[0]) > 1 else None

        if career_row and len(career_row) >= 4:
            try:
                result["num_past_starts"] = int(re.sub(r"[^\d]", "", career_row[0] or "0"))
                result["num_past_wins"] = int(re.sub(r"[^\d]", "", career_row[1] or "0"))
                result["num_past_seconds"] = int(re.sub(r"[^\d]", "", career_row[2] or "0"))
                result["num_past_thirds"] = int(re.sub(r"[^\d]", "", career_row[3] or "0"))
            except ValueError:
                pass

        # Last race: first row of results table after header
        results_rows = data.get("results") or []
        if len(results_rows) >= 2:
            header = [h.lower() for h in results_rows[0]]
            # Map columns
            try:
                date_idx = next(i for i, h in enumerate(header) if "date" in h)
                race_idx = next(i for i, h in enumerate(header) if h == "race")
                finish_idx = next(i for i, h in enumerate(header) if "finish" in h)
            except StopIteration:
                date_idx = race_idx = finish_idx = None

            if date_idx is not None:
                last = results_rows[1]
                if date_idx < len(last):
                    raw_date = last[date_idx]
                    # Convert M/D/YYYY -> YYYY-MM-DD
                    try:
                        d = datetime.strptime(raw_date, "%m/%d/%Y")
                        result["last_race_date"] = d.strftime("%Y-%m-%d")
                    except ValueError:
                        result["last_race_date"] = raw_date
                if race_idx is not None and race_idx < len(last):
                    result["last_race_number"] = last[race_idx]
                if finish_idx is not None and finish_idx < len(last):
                    result["last_race_finish"] = last[finish_idx]

        # Breed from horse header text: "TB, CH, G, FOALED ..."
        body = data.get("bodyText", "")
        breed_m = re.search(r"\b(TB|QH|AR|SB|PB)\b,", body)
        if breed_m:
            result["breed"] = breed_m.group(1)

        return result


# ─────────────────────────────────────────────────────────────────
# CSV writing
# ─────────────────────────────────────────────────────────────────

def write_test_data_csv(entries: List[Dict], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TEST_DATA_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for e in entries:
            row = {k: e.get(k, "") for k in TEST_DATA_COLUMNS}
            w.writerow(row)
    print(f"[entries] Wrote {len(entries)} rows to {path}")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

async def amain(args):
    scraper = EntriesScraper(
        cdp_port=args.cdp_port,
        fetch_pps=not args.no_pps,
        max_horses=args.max_horses,
    )
    await scraper.start()
    try:
        date = datetime.strptime(args.date, "%Y-%m-%d")
        entries = await scraper.scrape(args.track, date)
        if entries:
            write_test_data_csv(entries, args.output)
        else:
            print("[entries] No entries found — nothing written.")
    finally:
        await scraper.stop()


def main():
    p = argparse.ArgumentParser(description="Scrape pre-race entries from Equibase")
    p.add_argument("--track", required=True, help="Track code (e.g. KEE)")
    p.add_argument("--date", required=True, help="Race date YYYY-MM-DD")
    p.add_argument("--output", default="test_data.csv")
    p.add_argument("--cdp-port", type=int, default=9222)
    p.add_argument("--no-pps", action="store_true",
                   help="Skip horse-profile fetches (faster, no past-performance data)")
    p.add_argument("--max-horses", type=int, default=0,
                   help="Limit horse-profile fetches (debug aid, 0 = unlimited)")
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
