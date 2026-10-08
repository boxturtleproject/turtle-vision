"""Crop every reference carapace photo to the shell and embed the crop.

For each row of data/splits.csv:
  1. ask Gemini for the carapace bounding box   → data/crops.csv        [committed]
  2. crop (5% margin; full frame if none found) → data/crops/<turtle>/<id>.jpg
  3. embed the crop                              → data/embeddings_crop.sqlite

Same embedding model/dims as embed_photos.py, so app.py can compare full-frame
vs. cropped matching side by side. Resume-safe at every step.

    python crop_photos.py
"""
import csv, sqlite3, struct, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import identify
import shellcrop

ROOT = identify.ROOT
DATA = identify.DATA
BOXES = DATA / "crops.csv"
CROPS = DATA / "crops"
DB = DATA / "embeddings_crop.sqlite"
BOX_FIELDS = ["capture_id", "turtle_name", "found", "ymin", "xmin", "ymax", "xmax", "crop_model"]
WORKERS = 4
RETRIES = 5

lock = threading.Lock()


def retry(fn, *args):
    for attempt in range(RETRIES):
        try:
            return fn(*args)
        except Exception as e:
            if attempt == RETRIES - 1:
                raise
            print(f"  retry {attempt + 1}: {str(e)[:120]}", file=sys.stderr)
            time.sleep(min(60, 2 * 2 ** attempt))


def crop_path(row) -> Path:
    return CROPS / row["turtle_name"] / f"{row['capture_id']}.jpg"


def load_boxes():
    if not BOXES.exists():
        return {}
    return {int(r["capture_id"]): r for r in csv.DictReader(BOXES.open())}


def save_box(row, box):
    new = not BOXES.exists()
    with BOXES.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BOX_FIELDS)
        if new:
            w.writeheader()
        y0, x0, y1, x1 = box or ["", "", "", ""]
        w.writerow({"capture_id": row["capture_id"], "turtle_name": row["turtle_name"],
                    "found": int(box is not None), "ymin": y0, "xmin": x0, "ymax": y1,
                    "xmax": x1, "crop_model": shellcrop.CROP_MODEL})


def box_of(rec):
    return [int(rec[k]) for k in ("ymin", "xmin", "ymax", "xmax")] if rec["found"] == "1" else None


def process(client, row, box_rec, done):
    cid = int(row["capture_id"])
    src = ROOT / row["file_path"]
    img = Image.open(src)
    if box_rec is None:
        box = retry(shellcrop.detect_box, client, img)
        with lock:
            save_box(row, box)
    else:
        box = box_of(box_rec)

    out = crop_path(row)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        (shellcrop.crop_to_box(img, box) if box else img).convert("RGB").save(out, "JPEG", quality=90)

    if cid not in done:
        v = retry(identify.embed_image, out)
        return cid, row, v, box is not None
    return cid, row, None, box is not None


def main():
    identify.load_env()
    client = shellcrop.make_client()
    rows = list(csv.DictReader(identify.SPLITS.open()))
    missing = [r["file_path"] for r in rows if not (ROOT / r["file_path"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} images missing (e.g. {missing[0]}) — run fetch_data.py first")

    con = sqlite3.connect(DB, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS images(
        capture_id INTEGER PRIMARY KEY, turtle_name TEXT, file_path TEXT,
        model TEXT, dim INTEGER, crop_model TEXT, embedded_at TEXT, embedding BLOB)""")
    done = {r[0] for r in con.execute("SELECT capture_id FROM images WHERE embedding IS NOT NULL")}
    boxes = load_boxes()
    todo = [r for r in rows if int(r["capture_id"]) not in done or int(r["capture_id"]) not in boxes]
    print(f"{len(rows)} reference photos; {len(boxes)} boxes cached, {len(done)} crops embedded; {len(todo)} to do")

    n = no_box = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        futs = [pool.submit(process, client, r, boxes.get(int(r["capture_id"])), done) for r in todo]
        for f in as_completed(futs):
            cid, row, v, found = f.result()
            no_box += not found
            if v is not None:
                with lock:
                    con.execute("INSERT OR REPLACE INTO images VALUES(?,?,?,?,?,?,?,?)",
                                (cid, row["turtle_name"], str(crop_path(row).relative_to(ROOT)),
                                 identify.MODEL, identify.DIM, shellcrop.CROP_MODEL,
                                 datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 struct.pack(f"<{identify.DIM}f", *v)))
                    con.commit()
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}")
    print(f"done: {n} processed, {no_box} with no shell found (kept full frame)")


if __name__ == "__main__":
    main()
