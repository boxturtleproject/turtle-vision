"""Write data/capture_meta.csv: date and view for every capture in the manifest.

Walks box-turtle-id's public GET /api/turtles/{id} (the list endpoint is
admin-gated) until a run of 404s, and keeps captures that appear in
data/classifications.csv. The date is the app's captured_date, else parsed
from the original filename ("2025-05-21 19.03.35.jpg"); the view is the app's
image_type (carapace_top, carapace_left, front, plastron, …).

evaluate.py and app.py use the dates to score and calibrate on photos from
*other days* of the same turtle, which is what a new sighting looks like.

    python fetch_meta.py
"""
import csv
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

HOST = "https://boxturtleid.com"
MANIFEST = Path("data/classifications.csv")
OUT = Path("data/capture_meta.csv")
UA = {"User-Agent": "Mozilla/5.0 (turtle-vision fetch_meta.py)"}  # default urllib UA is blocked
MAX_CONSECUTIVE_MISSES = 25


def get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def main():
    wanted = {int(r["capture_id"]) for r in csv.DictReader(MANIFEST.open())}
    meta, tid, misses = {}, 0, 0
    while misses < MAX_CONSECUTIVE_MISSES:
        tid += 1
        try:
            turtle = get_json(f"{HOST}/api/turtles/{tid}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            misses += 1
            continue
        misses = 0
        for c in turtle["captures"]:
            if c["id"] not in wanted:
                continue
            m = re.search(r"(\d{4}-\d{2}-\d{2})", c.get("original_filename") or c["image_path"])
            meta[c["id"]] = {"capture_id": c["id"], "app_turtle_id": tid,
                             "view": c.get("image_type") or "",
                             "captured_date": c.get("captured_date") or (m.group(1) if m else ""),
                             "original_filename": c.get("original_filename") or ""}
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["capture_id", "app_turtle_id", "view", "captured_date", "original_filename"])
        w.writeheader()
        w.writerows(meta[k] for k in sorted(meta))
    dated = sum(bool(m["captured_date"]) for m in meta.values())
    print(f"scanned turtles 1..{tid}: {len(meta)}/{len(wanted)} manifest captures, {dated} with a date → {OUT}")


if __name__ == "__main__":
    main()
