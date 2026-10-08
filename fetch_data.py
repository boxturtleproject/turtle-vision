"""Recreate data/<turtle>/<capture_id>.jpg from the live box-turtle-id app.

data/classifications.csv is the manifest: its capture_id is the `captures.id`
primary key in box-turtle-id (https://boxturtleid.com). We download each
capture's 1280px "display" derivative — the same resolution the original
embeddings were computed on — from the public endpoint:

  GET /api/static/captures/derivatives/display/{capture_id}.jpg
      → 302 to a signed URL on the app's Railway bucket

Existing files are skipped, so re-running is cheap.

    python fetch_data.py
"""
import csv
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOST = "https://boxturtleid.com"
MANIFEST = Path("data/classifications.csv")
UA = {"User-Agent": "Mozilla/5.0 (turtle-vision fetch_data.py)"}  # default urllib UA is blocked


def download(row: dict) -> str:
    out = Path(row["file_path"])
    if out.exists():
        return "skip"
    url = f"{HOST}/api/static/captures/derivatives/display/{row['capture_id']}.jpg"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        body = r.read()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(body)
    return "ok"


def main() -> int:
    rows = list(csv.DictReader(MANIFEST.open()))
    counts = {"ok": 0, "skip": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [(r, pool.submit(download, r)) for r in rows]
        for i, (r, f) in enumerate(futures, 1):
            try:
                counts[f.result()] += 1
            except Exception as e:
                counts["error"] += 1
                print(f"  error {r['file_path']}: {e}")
            if i % 100 == 0:
                print(f"  {i}/{len(rows)} {counts}")
    print(f"done: {counts}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
