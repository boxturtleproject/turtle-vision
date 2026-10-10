"""Tidy a folder of field photos so only distinct turtle photos stay at the top level.

    python sort_photos.py ~/Desktop/field_photos

moves (never deletes) photos into subfolders of the same folder:

    <folder>/                 distinct photos with a turtle shell stay here
    <folder>/duplicates/      extra copies of the same shot
    <folder>/no_turtle/       photos where Gemini finds no turtle shell
    <folder>/sort_report.csv  what went where, and why

Duplicates: photos whose visual fingerprint (16x16 difference hash; HEIC vs
JPEG, resizing and renaming don't change it) differs by <= 10 of 256 bits
from an earlier photo; originals are preferred over names ending in
" 2", " copy" or " (1)". Re-saved copies differ by ~0,
different photos of the same turtle by ~90+.

No turtle: the same Gemini shell-box check the app uses (shellcrop.detect_box,
with its full-size retry). A photo of a turtle held in a hand still counts as
a turtle; only photos with no shell at all go to no_turtle.

Then match the folder with: python batch_identify.py <folder>
(batch_identify.py only reads the top level, so the subfolders are ignored.)
"""
import argparse
import csv
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

import identify
import shellcrop

register_heif_opener()
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
SAME = 10


def load(path: Path):
    try:
        return ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:
        return None


def fingerprint(img):
    a = np.asarray(img.convert("L").resize((17, 16), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    args = ap.parse_args()
    folder = args.folder.expanduser()
    for sub in ("duplicates", "no_turtle"):
        (folder / sub).mkdir(exist_ok=True)

    # Keep the original over copies: names without " 2" / " copy" / " (1)" come first.
    copy_mark = lambda p: bool(re.search(r"( \d+| copy( \d+)?| \(\d+\))$", p.stem))
    files = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS),
                   key=lambda p: (copy_mark(p), p.name))
    print(f"{len(files)} photos in {folder}")

    # 1. duplicates, by visual fingerprint
    imgs = {p: load(p) for p in files}
    report, kept, seen = [], [], []
    for p in files:
        img = imgs[p]
        if img is None:
            report.append({"filename": p.name, "folder": "no_turtle", "reason": "could not open"})
            continue
        fp = fingerprint(img)
        twin = next((name for name, other in seen if (fp != other).sum() <= SAME), None)
        if twin:
            report.append({"filename": p.name, "folder": "duplicates", "reason": f"same shot as {twin}"})
        else:
            seen.append((p.name, fp))
            kept.append(p)
    print(f"{len(files) - len(kept)} duplicates or unreadable; checking {len(kept)} for a turtle shell…")

    # 2. turtle or not, with the app's Gemini shell-box check
    identify.load_env()
    client = shellcrop.make_client()

    def check(p):
        img = imgs[p].copy()
        img.thumbnail((1280, 1280))
        for attempt in range(3):
            try:
                return p, shellcrop.detect_box(client, img)
            except Exception as e:
                err = e
        return p, err

    with ThreadPoolExecutor(8) as pool:
        for i, (p, box) in enumerate(pool.map(check, kept), 1):
            if isinstance(box, Exception):
                report.append({"filename": p.name, "folder": "turtles", "reason": f"shell check failed ({str(box)[:60]}); kept"})
            elif box is None:
                report.append({"filename": p.name, "folder": "no_turtle", "reason": "no turtle shell found"})
            else:
                report.append({"filename": p.name, "folder": "turtles", "reason": f"shell box {box}"})
            if i % 25 == 0:
                print(f"  {i}/{len(kept)}")

    # 3. move (never delete) everything that isn't a distinct turtle photo
    for r in report:
        if r["folder"] != "turtles":
            dest = folder / r["folder"] / r["filename"]
            if dest.exists():
                dest = dest.with_name(f"{dest.stem} (moved){dest.suffix}")
            shutil.move(str(folder / r["filename"]), dest)
    with (folder / "sort_report.csv").open("a" if (folder / "sort_report.csv").exists() else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "folder", "reason"])
        if f.tell() == 0:
            w.writeheader()
        w.writerows(sorted(report, key=lambda r: r["filename"]))
    counts = {k: sum(r["folder"] == k for r in report) for k in ("turtles", "duplicates", "no_turtle")}
    print(f"done: {counts['turtles']} turtle photos stay in {folder}; "
          f"{counts['duplicates']} moved to duplicates/, {counts['no_turtle']} moved to no_turtle/")
    print(f"next: python batch_identify.py '{folder}'")


if __name__ == "__main__":
    main()
