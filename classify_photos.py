"""Classify every photo under data/<turtle>/*.jpg on two axes:

  category:
    - carapace       (close-up of the dome top shell)
    - plastron       (close-up of the flat bottom shell)
    - other_closeup  (close-up of the turtle that is NOT predominantly the top/bottom shell)
    - habitat        (wider scene shot, with or without the turtle visible)

  media_type:
    - photo          (an actual photograph)
    - illustration   (a drawing, sketch, painting, stippled art, diagram, etc.)

Uses Claude Haiku 4.5 vision via the Anthropic SDK. Resume-safe: re-running only
classifies images not already in data/classifications.csv.
"""
import os, sys, csv, base64, time, threading, mimetypes
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_CSV = DATA / "classifications.csv"
ENV = ROOT / ".env"
MODEL = "claude-haiku-4-5"
WORKERS = 24
RETRIES = 6

# Load .env (mirrors embed_photos.py)
for line in ENV.read_text().splitlines():
    if line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ.setdefault(k.strip(), v.strip())

import anthropic
from pydantic import BaseModel

Category = Literal["carapace", "plastron", "other_closeup", "habitat"]
MediaType = Literal["photo", "illustration"]
Confidence = Literal["low", "medium", "high"]


class Classification(BaseModel):
    category: Category
    media_type: MediaType
    confidence: Confidence


SYSTEM = """You classify images of eastern box turtles on TWO independent axes.

AXIS 1 — category (subject framing):

- carapace: a close-up where the dome-shaped TOP shell fills most of the frame. \
The carapace has a distinctive yellow-and-black pattern. Shot from above or at an angle, \
with the shell as the dominant subject.

- plastron: a close-up of the BOTTOM shell (the flat underside). The turtle has \
typically been flipped or is being held upside-down. Appears dark / nearly black, \
often shiny when wet, smooth and flat — the opposite of the bumpy patterned carapace.

- other_closeup: a close-up of the turtle that is NOT predominantly the top or bottom \
shell — head/face shot, full-body side view, leg/foot, or a mixed close-up where the \
turtle fills the frame but the carapace isn't the dominant subject.

- habitat: a wider scene showing the turtle's environment — road, forest, leaves, \
grass, gravel, wood, etc. The turtle may be small in the frame OR absent entirely. \
The defining feature is that the SURROUNDINGS dominate, not the turtle.

Disambiguation:
- Top shell fills most of frame -> carapace.
- Dark flat underside is main subject -> plastron.
- Turtle is small / surroundings dominate -> habitat (even if no turtle is visible).
- Turtle fills frame but it's a head/limb/side view -> other_closeup.

AXIS 2 — media_type (medium):

- photo: an actual photograph captured by a camera.
- illustration: a hand-drawn or digitally-drawn image — sketches, line drawings, \
stippling/pointillism, ink drawings, paintings, watercolors, diagrams, scientific \
illustrations, etc. Black-and-white drawings are illustrations, not photos. Heavily \
filtered or stylized photos still count as photo unless the result is clearly a \
drawing or painting.

The two axes are independent: an illustration can show a carapace, a habitat, etc.

Confidence: "high" when both axes are unambiguous. "medium" for borderline cases. \
"low" only when you genuinely can't tell.
"""


client = anthropic.Anthropic()


def classify_one(path: Path) -> Classification:
    raw = path.read_bytes()
    media = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.standard_b64encode(raw).decode()
    last = None
    for attempt in range(RETRIES):
        try:
            resp = client.messages.parse(
                model=MODEL,
                max_tokens=200,
                system=SYSTEM,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": media, "data": b64}},
                        {"type": "text", "text": "Classify this image."},
                    ],
                }],
                output_format=Classification,
            )
            return resp.parsed_output
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            last = e
            status = getattr(e, "status_code", None)
            if status in (429, 500, 502, 503, 504, 529):
                time.sleep(min(60.0, 1.5 * (2 ** attempt)))
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(min(30.0, 1.5 * (2 ** attempt)))
    raise last


def discover():
    rows = []
    for jpg in sorted(DATA.glob("*/*.jpg")):
        try:
            cid = int(jpg.stem)
        except ValueError:
            continue
        rows.append((cid, jpg.parent.name, str(jpg.relative_to(ROOT))))
    return rows


def load_existing() -> set[str]:
    if not OUT_CSV.exists():
        return set()
    with OUT_CSV.open() as f:
        return {row["file_path"] for row in csv.DictReader(f)}


write_lock = threading.Lock()


def main():
    all_rows = discover()
    done = load_existing()
    todo = [r for r in all_rows if r[2] not in done]
    print(f"discovered {len(all_rows)} images; already classified {len(done)}; "
          f"to do {len(todo)}; model={MODEL}")
    if not todo:
        return 0

    new_file = not OUT_CSV.exists()
    f = OUT_CSV.open("a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["capture_id", "turtle_name", "file_path",
                    "category", "media_type", "confidence"])
        f.flush()

    ok = fail = 0
    failures = []
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(classify_one, ROOT / r[2]): r for r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                cid, name, rel = futs[fut]
                try:
                    c = fut.result()
                    with write_lock:
                        w.writerow([cid, name, rel, c.category, c.media_type, c.confidence])
                        if i % 25 == 0:
                            f.flush()
                    ok += 1
                except Exception as e:
                    fail += 1
                    failures.append((cid, name, type(e).__name__, str(e)[:200]))
                if i % 25 == 0 or i == len(todo):
                    rate = i / max(1e-3, time.time() - t0)
                    eta = (len(todo) - i) / max(1e-3, rate)
                    print(f"  [{i}/{len(todo)}] ok={ok} fail={fail}  {rate:.1f}/s  "
                          f"eta {eta:.0f}s", flush=True)
    finally:
        f.flush()
        f.close()

    print(f"\nDONE  ok={ok}  fail={fail}  elapsed={time.time()-t0:.1f}s")
    if failures:
        print("failures (first 15):")
        for x in failures[:15]:
            print(" ", x)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
