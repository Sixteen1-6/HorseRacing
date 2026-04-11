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
        self.failed: Set[str] = set()
        self.stats = {"pages": 0, "entries": 0, "errors": 0, "pdfs": 0}
        self._load()

    def _key(self, track, date):
        return f"{track}_{date.strftime('%Y%m%d')}"

    def is_done(self, track, date):
        return self._key(track, date) in self.done

    def mark(self, track, date, n_entries=0):
        key = self._key(track, date)
        self.done.add(key)
        self.failed.discard(key)
        self.stats["pages"] += 1
        self.stats["entries"] += n_entries
        if self.stats["pages"] % 5 == 0:
            self.save()

    def mark_failed(self, track, date):
        """Record a failed target for retry on --resume."""
        key = self._key(track, date)
        if key not in self.done:
            self.failed.add(key)

    def is_failed(self, track, date):
        return self._key(track, date) in self.failed

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"done": list(self.done), "failed": list(self.failed),
                        "stats": self.stats}, f)
        os.replace(tmp, self.path)

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                d = json.load(f)
            self.done = set(d.get("done", []))
            self.failed = set(d.get("failed", []))
            self.stats = d.get("stats", self.stats)
            log.info(f"Checkpoint: {len(self.done)} done, {len(self.failed)} failed, "
                     f"{self.stats['entries']:,} entries")
