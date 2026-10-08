"""Crop a turtle photo to the carapace using a Gemini bounding box.

Gemini returns box_2d as [ymin, xmin, ymax, xmax] normalized to 0..1000. We
pad it by MARGIN on each side and crop. If no shell is found, callers fall
back to the full frame.

Model choice (benchmarked 2026-10-08 on 9 reference photos): gemini-3.6-flash
with minimal thinking on a 640px image found every shell with tight boxes in
~1.3s. gemini-2.5-flash was looser on 3/9 (included hands/background),
3.5-flash-lite missed 2 obvious shells, 3.8-flash missed 1 and was slowest.
Box coords are relative, so downscaling before detection costs nothing.
"""
import io
import json
import os

from PIL import Image

CROP_MODEL = os.environ.get("CROP_MODEL", "gemini-3.6-flash")
# Gemini 3.x models are served from the "global" Vertex location only.
CROP_LOCATION = os.environ.get("CROP_LOCATION", "global")
DETECT_SIDE = 640  # downscale before asking for the box
MARGIN = 0.05  # fraction of box size added on each side

PROMPT = (
    "Find the box turtle's carapace (the top shell) in this photo. Return a "
    "tight bounding box around the shell only — exclude the head, legs, hands, "
    "and background. If there is no turtle shell visible, set found to false. "
    'Respond as JSON: {"found": bool, "box_2d": [ymin, xmin, ymax, xmax]} '
    "normalized to 0-1000."
)
# No response_schema: on a test photo, constraining output to a schema made
# gemini-2.5-flash's box ~12% too tall; a plain JSON instruction was tight.


def make_client():
    """Client for the box model: Vertex at CROP_LOCATION, else Gemini API key."""
    from google import genai
    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true"):
        return genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                            location=CROP_LOCATION)
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def detect_box(client, img: Image.Image):
    """Return [ymin, xmin, ymax, xmax] in 0..1000, or None if no shell found."""
    from google.genai import types
    small = img.copy()
    small.thumbnail((DETECT_SIDE, DETECT_SIDE))
    thinking = (types.ThinkingConfig(thinking_budget=0) if CROP_MODEL.startswith("gemini-2")
                else types.ThinkingConfig(thinking_level="minimal"))
    resp = client.models.generate_content(
        model=CROP_MODEL,
        contents=[types.Part.from_bytes(data=jpeg_bytes(small), mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            thinking_config=thinking,
        ),
    )
    out = json.loads(resp.text)
    if isinstance(out, list):  # occasionally wrapped in a list
        out = out[0] if out else {}
    box = out.get("box_2d") or []
    if not out.get("found") or len(box) != 4:
        return None
    y0, x0, y1, x1 = (max(0, min(1000, int(v))) for v in box)
    if y1 <= y0 or x1 <= x0:
        return None
    return [y0, x0, y1, x1]


def crop_to_box(img: Image.Image, box, margin: float = MARGIN) -> Image.Image:
    y0, x0, y1, x1 = box
    w, h = img.size
    pad_x, pad_y = (x1 - x0) * margin, (y1 - y0) * margin
    left = max(0, (x0 - pad_x) / 1000 * w)
    top = max(0, (y0 - pad_y) / 1000 * h)
    right = min(w, (x1 + pad_x) / 1000 * w)
    bottom = min(h, (y1 + pad_y) / 1000 * h)
    return img.crop((round(left), round(top), round(right), round(bottom)))


def jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()
