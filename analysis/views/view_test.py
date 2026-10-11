"""Can the shell-box call also say which side of the turtle a photo shows?

Asks Gemini for the box and the view in one call (same model, image size and
thinking as shellcrop.detect_box's fast pass) on every reference photo with a
recorded view, then compares:
  - view: Gemini's answer vs the view recorded in box-turtle-id (capture_meta.csv)
  - box: overlap (IoU) with the box from the box-only prompt (data/crops.csv), to
    check that the extra question doesn't change the crop SIFT depends on

A side view Gemini can't tell left from right may be answered carapace_side
("unsure"), which is better than a wrong side.

  python analysis/views/view_test.py [limit|all] [thinking] [view|head]   # resume-safe
  python analysis/views/view_test.py 40 low               # first 40, low thinking
"""
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

TV = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TV))
import shellcrop  # noqa: E402

THINKING = sys.argv[2] if len(sys.argv) > 2 else "minimal"
STYLE = sys.argv[3] if len(sys.argv) > 3 else "view"  # "view": name the view; "head": say where the head points
OUT = TV / f"analysis/out/views/answers_{STYLE}2_{THINKING}.jsonl"
# Asking where the head end points, and deriving left/right from it.
HEAD_PROMPT = (
    "Find the box turtle's carapace (the top shell) in this photo. Return a "
    "tight bounding box around the shell only — exclude the head, legs, hands, "
    "and background. If there is no turtle shell visible, set found to false. "
    "Also say what kind of photo it is: "
    "top (camera above the turtle looking down on the top shell), "
    "side (camera level with the turtle, seeing the shell from the side), "
    "plastron (the bottom shell), front (facing the head), rear (facing the tail end), "
    "or site_view (a wide habitat scene; the turtle is small or not visible). "
    "For a side photo, also say which way the turtle's head points in the image: "
    "left or right only if you can actually see the head; if the head is pulled into "
    "the shell or out of view, say unclear. "
    'Respond as JSON: {"found": bool, "box_2d": [ymin, xmin, ymax, xmax], '
    '"photo": string, "head": "left" | "right" | "unclear"} with box_2d normalized to 0-1000.'
)
PHOTO_TO_VIEW = {"top": "carapace_top", "plastron": "plastron", "front": "front", "rear": "rear",
                 "site_view": "site_view"}


def to_view(a):
    """Map a HEAD_PROMPT answer to a box-turtle-id image type (side + unclear = carapace_side)."""
    if a.get("photo") == "side":
        return {"left": "carapace_left", "right": "carapace_right"}.get(a.get("head"), "carapace_side")
    return PHOTO_TO_VIEW.get(a.get("photo"), a.get("photo"))


# box-turtle-id's capture image types ("back" and "rear" are the same view)
VIEWS = ["carapace_top", "carapace_left", "carapace_right", "plastron", "front", "rear", "site_view"]
PROMPT = (
    "Find the box turtle's carapace (the top shell) in this photo. Return a "
    "tight bounding box around the shell only — exclude the head, legs, hands, "
    "and background. If there is no turtle shell visible, set found to false. "
    "Also say which view of the turtle the photo shows: "
    "carapace_top (looking down on the top shell), "
    "carapace_left (the turtle's left side: its head end points to the left of the photo), "
    "carapace_right (the turtle's right side: its head end points to the right of the photo), "
    "carapace_side (a side view where you can't tell which end is the head), "
    "plastron (the bottom shell), front (facing the head), rear (facing the tail end), "
    "or site_view (a wide habitat scene; the turtle is small or not visible). "
    'Respond as JSON: {"found": bool, "box_2d": [ymin, xmin, ymax, xmax], "view": string} '
    "with box_2d normalized to 0-1000."
)


def ask(client, img):
    from google.genai import types
    small = img.copy()
    small.thumbnail((shellcrop.DETECT_SIDE, shellcrop.DETECT_SIDE))
    resp = client.models.generate_content(
        model=shellcrop.CROP_MODEL,
        contents=[types.Part.from_bytes(data=shellcrop.jpeg_bytes(small), mime_type="image/jpeg"),
                  HEAD_PROMPT if STYLE == "head" else PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0,
            thinking_config=types.ThinkingConfig(thinking_level=THINKING)),
    )
    out = json.loads(resp.text)
    out = out[0] if isinstance(out, list) and out else out
    if STYLE == "head":
        out["view"] = to_view(out)
    return out


def iou(a, b):
    y0, x0 = max(a[0], b[0]), max(a[1], b[1])
    y1, x1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y1 - y0) * max(0, x1 - x0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter) if inter else 0.0


def main(limit=None):
    meta = {r["capture_id"]: r["view"].replace("back", "rear") for r in csv.DictReader((TV / "data/capture_meta.csv").open())}
    boxes = {r["capture_id"]: [int(r[k]) for k in ("ymin", "xmin", "ymax", "xmax")]
             for r in csv.DictReader((TV / "data/crops.csv").open()) if r["found"] == "1"}
    photos = sorted((p for p in (TV / "data").glob("*/*.jpg")
                     if p.parent.name not in ("crops", "crops_tight") and p.stem in meta),
                    key=lambda p: int(p.stem))[:limit]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in OUT.open()} if OUT.exists() else set()
    todo = [p for p in photos if p.stem not in done]
    client = shellcrop.make_client()

    def run(p):
        try:
            a = ask(client, ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
        except Exception as e:
            a = {"error": str(e)[:200]}
        return {"id": p.stem, "turtle": p.parent.name, **a}

    print(f"{len(photos)} photos, {len(todo)} to ask")
    with ThreadPoolExecutor(16) as ex, OUT.open("a") as f:
        for i, row in enumerate(ex.map(run, todo), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    keep = {p.stem for p in photos}
    rows = [r for r in map(json.loads, OUT.open()) if r["id"] in keep]
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    right = [r for r in rows if r.get("view") == meta[r["id"]]]
    unsure = [r for r in rows if r.get("view") == "carapace_side" and meta[r["id"]] in ("carapace_left", "carapace_right")]
    print(f"\n{THINKING} thinking. view right: {len(right)}/{len(rows)} ({len(right) / len(rows):.1%}), "
          f"'side' for a left/right photo: {len(unsure)}, wrong: {len(rows) - len(right) - len(unsure)}; errors {len(errors)}")
    print("\nby recorded view (right / photos, and what Gemini said when wrong):")
    for v in VIEWS:
        rv = [r for r in rows if meta[r["id"]] == v]
        if rv:
            wrong = Counter(r.get("view") for r in rv if r.get("view") != v)
            print(f"  {v:15} {sum(r.get('view') == v for r in rv):4}/{len(rv):<4} {dict(wrong.most_common(4)) or ''}")
    sides = [r for r in rows if meta[r["id"]] in VIEWS[:3]]
    print(f"\ntop/left/right only: {sum(r.get('view') == meta[r['id']] for r in sides)}/{len(sides)}")
    lr = [r for r in rows if meta[r["id"]] in ("carapace_left", "carapace_right")]
    flipped = sum({meta[r["id"]], r.get("view")} == {"carapace_left", "carapace_right"} for r in lr)
    print(f"left/right photos: {sum(r.get('view') == meta[r['id']] for r in lr)} right, "
          f"{sum(r.get('view') == 'carapace_side' for r in lr)} 'side', {flipped} swapped (of {len(lr)})")
    said = Counter(r.get("view") for r in rows)
    for v in ("carapace_left", "carapace_right"):
        if said[v]:
            print(f"  when Gemini says {v}: right {sum(meta[r['id']] == v for r in rows if r.get('view') == v)}/{said[v]}")
    side_said = [meta[r["id"]] for r in rows if r.get("view") == "carapace_side"]
    print(f"  when Gemini says carapace_side: {dict(Counter(side_said))}")
    ious = [iou(r["box_2d"], boxes[r["id"]]) for r in rows
            if r["id"] in boxes and r.get("found") and len(r.get("box_2d") or []) == 4]
    ious.sort()
    print(f"\nbox vs box-only prompt: median IoU {ious[len(ious) // 2]:.3f}, "
          f">=0.9 in {sum(x >= 0.9 for x in ious) / len(ious):.0%}, <0.7 in {sum(x < 0.7 for x in ious)} of {len(ious)}")
    found_before = sum(r["id"] in boxes for r in rows)
    print(f"shell found: {sum(bool(r.get('found')) for r in rows)} (box-only prompt: {found_before})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "all" else None)
