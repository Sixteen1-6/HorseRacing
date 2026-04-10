# Distributed 2024-2026 scrape — 4 PCs × 2 processes

Tier-1 optimizations are already baked into scraper.py. Expected rate: ~35s per
race day. Each worker handles ~930 race days (~9 hours single-process).

Each PC runs **2 processes concurrently**, sharing that PC's single IP. If a PC
has dual connections (wifi + hotspot), run the second process on the second
connection for maximum safety.

## Before starting on each PC
1. Pull latest scraper.py + track_race_dates.json + scraper_entries.py from git
2. Ensure playwright chromium is installed: `python -m playwright install chromium`
3. Do NOT copy over the checkpoint file — each worker uses its own

## Launch commands (copy-paste one block per PC)

### PC1 (2024-01-01 → 2024-07-13)
Open two terminals on the same machine:

**Terminal A:**
```bash
python scraper.py --start 2024-01-01 --end 2024-04-16 \
  --output races_pc1a.csv --checkpoint checkpoint_pc1a.json --no-google
```
**Terminal B:**
```bash
python scraper.py --start 2024-04-17 --end 2024-07-13 \
  --output races_pc1b.csv --checkpoint checkpoint_pc1b.json --no-google
```

### PC2 (2024-07-14 → 2025-02-09)
**Terminal A:**
```bash
python scraper.py --start 2024-07-14 --end 2024-10-24 \
  --output races_pc2a.csv --checkpoint checkpoint_pc2a.json --no-google
```
**Terminal B:**
```bash
python scraper.py --start 2024-10-25 --end 2025-02-09 \
  --output races_pc2b.csv --checkpoint checkpoint_pc2b.json --no-google
```

### PC3 (2025-02-10 → 2025-08-26)
**Terminal A:**
```bash
python scraper.py --start 2025-02-10 --end 2025-05-25 \
  --output races_pc3a.csv --checkpoint checkpoint_pc3a.json --no-google
```
**Terminal B:**
```bash
python scraper.py --start 2025-05-26 --end 2025-08-26 \
  --output races_pc3b.csv --checkpoint checkpoint_pc3b.json --no-google
```

### PC4 (2025-08-27 → 2026-04-04)
**Terminal A:**
```bash
python scraper.py --start 2025-08-27 --end 2025-12-11 \
  --output races_pc4a.csv --checkpoint checkpoint_pc4a.json --no-google
```
**Terminal B:**
```bash
python scraper.py --start 2025-12-12 --end 2026-04-04 \
  --output races_pc4b.csv --checkpoint checkpoint_pc4b.json --no-google
```

## Summary

| Worker | Date range | Targets | ETA |
|---|---|---|---|
| PC1-A | 2024-01-01 → 2024-04-16 | 930 | ~9.0 h |
| PC1-B | 2024-04-17 → 2024-07-13 | 931 | ~9.1 h |
| PC2-A | 2024-07-14 → 2024-10-24 | 928 | ~9.0 h |
| PC2-B | 2024-10-25 → 2025-02-09 | 930 | ~9.0 h |
| PC3-A | 2025-02-10 → 2025-05-25 | 933 | ~9.1 h |
| PC3-B | 2025-05-26 → 2025-08-26 | 919 | ~8.9 h |
| PC4-A | 2025-08-27 → 2025-12-11 | 933 | ~9.1 h |
| PC4-B | 2025-12-12 → 2026-04-04 | 925 | ~9.0 h |
| **Total** | 2024-01-01 → 2026-04-04 | **7,429** | **~9 h wall** |

## Merging CSVs at the end
Once all 8 workers finish, run on any one machine:
```bash
python merge_distributed_runs.py
```
This produces `races_2024_2026_merged.csv` and `races_2024_2026_2023_combined.csv`
(scraped + hackathon 2023). See that script for details.
