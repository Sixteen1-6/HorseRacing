"""
Checkpoint/resume support for the scraper.
"""

import json
import os
import logging
from typing import Set

log = logging.getLogger("scraper")


class Checkpoint:
    def __init__(self, path="scraper_v3_checkpoint.json"):
        self.path = path
        self.done: Set[str] = set()
        self.stats = {"pages": 0, "entries": 0, "errors": 0, "pdfs": 0}
        self._load()

    def _key(self, track, date):
        return f"{track}_{date.strftime('%Y%m%d')}"

    def is_done(self, track, date):
        return self._key(track, date) in self.done

    def mark(self, track, date, n_entries=0):
        self.done.add(self._key(track, date))
        self.stats["pages"] += 1
        self.stats["entries"] += n_entries
        if self.stats["pages"] % 25 == 0:
            self.save()

    def save(self):
        with open(self.path, "w") as f:
            json.dump({"done": list(self.done), "stats": self.stats}, f)

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                d = json.load(f)
            self.done = set(d.get("done", []))
            self.stats = d.get("stats", self.stats)
            log.info(f"Checkpoint: {len(self.done)} done, {self.stats['entries']:,} entries")
