"""Crop every reference carapace photo to the shell and embed the crops.

For each row of data/splits.csv:
  1. ask Gemini for the carapace bounding box → data/crops.csv  [committed]
  2. crop it two ways (full frame if no shell found):
       crop   box + 5% margin        → data/crops/<turtle>/<id>.jpg
       tight  central 71% of the box → data/crops_tight/<turtle>/<id>.jpg
  3. embed each crop → data/embeddings_crop.sqlite, data/embeddings_tight.sqlite

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


def crop_path(variant, row) -> Path:
    return DATA / shellcrop.VARIANTS[variant]["dir"] / row["turtle_name"] / f"{row['capture_id']}.jpg"


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

    vectors = {}
    for variant, cfg in shellcrop.VARIANTS.items():
        out = crop_path(variant, row)
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            crop = shellcrop.crop_to_box(img, box, cfg["margin"]) if box else img
            crop.convert("RGB").save(out, "JPEG", quality=90)
        if cid not in done[variant]:
            vectors[variant] = retry(identify.embed_image, out)
    return cid, row, vectors, box is not None


def main():
    identify.load_env()
    client = shellcrop.make_client()
    rows = list(csv.DictReader(identify.SPLITS.open()))
    missing = [r["file_path"] for r in rows if not (ROOT / r["file_path"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} images missing (e.g. {missing[0]}) — run fetch_data.py first")

    cons, done = {}, {}
    for variant, cfg in shellcrop.VARIANTS.items():
        con = cons[variant] = sqlite3.connect(DATA / cfg["db"], check_same_thread=False)
        con.execute("""CREATE TABLE IF NOT EXISTS images(
            capture_id INTEGER PRIMARY KEY, turtle_name TEXT, file_path TEXT,
            model TEXT, dim INTEGER, crop_model TEXT, embedded_at TEXT, embedding BLOB)""")
        done[variant] = {r[0] for r in con.execute(
            "SELECT capture_id FROM images WHERE embedding IS NOT NULL")}
    boxes = load_boxes()
    todo = [r for r in rows if int(r["capture_id"]) not in boxes
            or any(int(r["capture_id"]) not in d for d in done.values())]
    print(f"{len(rows)} reference photos; {len(boxes)} boxes cached; "
          + ", ".join(f"{len(d)} {v} embedded" for v, d in done.items()) + f"; {len(todo)} to do")

    n = no_box = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        futs = [pool.submit(process, client, r, boxes.get(int(r["capture_id"])), done) for r in todo]
        for f in as_completed(futs):
            cid, row, vectors, found = f.result()
            no_box += not found
            with lock:
                for variant, v in vectors.items():
                    cons[variant].execute(
                        "INSERT OR REPLACE INTO images VALUES(?,?,?,?,?,?,?,?)",
                        (cid, row["turtle_name"], str(crop_path(variant, row).relative_to(ROOT)),
                         identify.MODEL, identify.DIM, shellcrop.CROP_MODEL,
                         datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         struct.pack(f"<{identify.DIM}f", *v)))
                    cons[variant].commit()
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(todo)}")
    print(f"done: {n} processed, {no_box} with no shell found (kept full frame)")


if __name__ == "__main__":
    main()
