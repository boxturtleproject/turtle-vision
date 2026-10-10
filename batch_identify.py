"""Send every photo in a folder through the running app, one at a time.

Each photo goes to app.py's /api/identify as a single upload, with the Gemini
second opinion turned off (faster) and require_shell on: photos where Gemini
finds no turtle shell are skipped and listed in results/skipped.csv instead
of being matched. Everything else lands in results/session_log.csv and
results/uploads_summary.csv like a normal upload; review it at /summary.

Photos already in the session log (same filename) are skipped, so the run can
be stopped and restarted. Duplicates are matched once: photos whose visual
fingerprint (16x16 difference hash, unaffected by HEIC/JPEG, resizing or
renaming) is within a few bits of another photo in the folder, or of a photo
already uploaded, are skipped and listed in results/skipped.csv.

    python app.py                              # in another terminal
    python batch_identify.py ~/Desktop/turtles
    python batch_identify.py FOLDER --redo     # also re-run already-logged files
"""
import argparse
import csv
import json
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
ROOT = Path(__file__).resolve().parent
LOG = ROOT / "results" / "session_log.csv"
SKIPPED = ROOT / "results" / "skipped.csv"
UPLOADS = ROOT / "uploads"
SAME = 10  # max differing bits (of 256) to call two photos the same shot; re-saved copies differ by ~0,
           # different photos of a turtle by ~90+


def fingerprint(path: Path):
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("L").resize((17, 16), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def earlier_uploads():
    """Fingerprints of photos already uploaded to the app, keyed by their original filename."""
    if not LOG.exists():
        return []
    names = {r["upload_id"]: r["filename"] for r in csv.DictReader(LOG.open()) if r["event"] == "identify"}
    out = []
    for uid, fname in names.items():
        p = UPLOADS / f"{uid}.jpg"
        if p.exists() and (fp := fingerprint(p)) is not None:
            out.append((fname, fp))
    return out


def note_skipped(filename, reason):
    SKIPPED.parent.mkdir(exist_ok=True)
    new = not SKIPPED.exists()
    with SKIPPED.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "upload_id", "filename", "reason"])
        if new:
            w.writeheader()
        w.writerow({"time": datetime.now().isoformat(timespec="seconds"), "upload_id": "",
                    "filename": filename, "reason": reason})


def already_done():
    done = set()
    for path in (LOG, SKIPPED):
        if path.exists():
            done |= {r["filename"] for r in csv.DictReader(path.open()) if r.get("filename")}
    return done


def post(url, path: Path):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("gemini", "0"), ("require_shell", "1")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="images"; filename="{path.name}"\r\n'
                 f"Content-Type: application/octet-stream\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"{url}/api/identify", data=b"".join(parts), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--redo", action="store_true", help="re-run photos already in the log")
    args = ap.parse_args()

    files = sorted(p for p in args.folder.expanduser().iterdir() if p.suffix.lower() in EXTS)
    done = set() if args.redo else already_done()
    todo = [p for p in files if p.name not in done]
    print(f"{len(files)} photos in {args.folder}; {len(files) - len(todo)} already done; fingerprinting {len(todo)}…")

    seen = [] if args.redo else earlier_uploads()  # (filename, fingerprint) of photos kept so far
    unique, dupes = [], 0
    for p in todo:
        fp = fingerprint(p)
        twin = next((name for name, other in seen if fp is not None and (fp != other).sum() <= SAME), None)
        if twin:
            dupes += 1
            note_skipped(p.name, f"duplicate of {twin}")
            print(f"  {p.name}: duplicate of {twin}, skipped")
            continue
        unique.append(p)
        if fp is not None:
            seen.append((p.name, fp))
    todo = unique
    print(f"{dupes} duplicates skipped; {len(todo)} to run")
    matched = skipped = failed = 0
    start = time.time()
    for i, path in enumerate(todo, 1):
        try:
            out = post(args.url, path)
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(todo)}] {path.name}: FAILED {str(e)[:120]}")
            continue
        if "skipped" in out:
            skipped += 1
            print(f"  [{i}/{len(todo)}] {path.name}: skipped ({out['skipped'][0]['reason']})")
        else:
            matched += 1
            best = out["methods"].get("best") or next(iter(out["methods"].values()))
            top = best.get("matches", [{}])[0]
            print(f"  [{i}/{len(todo)}] {path.name}: {top.get('name')} (spot {best.get('spot', '?')})")
        rate = (time.time() - start) / i
        if i % 10 == 0:
            print(f"    ~{rate:.1f}s per photo, ~{rate * (len(todo) - i) / 60:.0f} min left")
    print(f"done: {matched} matched, {skipped} skipped (no turtle), {dupes} duplicates, {failed} failed"
          f" — review at {args.url}/summary")


if __name__ == "__main__":
    main()
