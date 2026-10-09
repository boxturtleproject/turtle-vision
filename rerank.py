"""Ask Gemini to re-rank a matcher's top 5 turtles by comparing shell patterns.

The query photo and 1-3 reference photos per candidate (shell crops) go in
one request. Candidates are shuffled and labelled A-E so position doesn't
favour the matcher's own top pick.

On the different-day test (evaluate.py --gemini), re-ranking the combined
matcher's top 5 with gemini-3.6-flash raised top-1 from 0.670 to 0.808
(609 queries; fixed 108, broke 24; ~3.4s and ~9k input tokens per query).
More thinking (medium) and gemini-3.1-pro-preview did not do better on a
120-query sample.

It cannot tell a new turtle: with the true turtle removed from the
candidates it still picked one with ~0.98 confidence, so "same_as_any" and
"confidence" are recorded but not used.
"""
import io
import json
import os
import random

from PIL import Image

RERANK_MODEL = os.environ.get("RERANK_MODEL", "gemini-3.6-flash")
SIDE = 512  # image size sent per photo

PROMPT = """You are an expert in identifying individual eastern box turtles from the pattern of yellow/orange markings on their carapace (top shell). Each turtle's pattern is unique and stable over years, but photos differ in angle (top, left or right side), lighting, wetness, dirt, and framing.

The first image is the QUERY turtle. After it come candidates A to E. Each candidate is ONE individual turtle, shown in 1-3 reference photos taken on other days.

Compare the specific shapes and positions of markings on individual scutes (shell plates): blotch and stripe shapes, radiating lines, gaps, scars or chips, and the overall pattern layout. Ignore background, hands, and colour cast. Account for viewing angle: a side view shows the side scutes, a top view shows the central ones.

Rank all five candidates from most to least likely to be the same individual as the QUERY. If none of them plausibly match, set "same_as_any" to false.

Respond as JSON: {"ranking": ["C","A","E","B","D"], "same_as_any": true, "confidence": 0.0-1.0, "reason": "one short sentence"}"""


def _jpeg(path) -> bytes:
    im = Image.open(path)
    im.thumbnail((SIDE, SIDE))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()


def rerank(client, query_path, candidates: dict, seed=0):
    """candidates: {turtle name: [reference image paths]} in matcher order.

    Returns {"ranking": [names, best first], "reason": str, "confidence": float,
    "same_as_any": bool}. Names Gemini leaves out keep their matcher order at the end.
    """
    from google.genai import types
    names = list(candidates)[:5]
    order = names[:]
    random.Random(seed).shuffle(order)
    letters = "ABCDE"[:len(order)]
    parts = [types.Part.from_text(text=PROMPT), types.Part.from_text(text="QUERY:"),
             types.Part.from_bytes(data=_jpeg(query_path), mime_type="image/jpeg")]
    for letter, name in zip(letters, order):
        parts.append(types.Part.from_text(text=f"Candidate {letter}:"))
        for p in candidates[name][:3]:
            parts.append(types.Part.from_bytes(data=_jpeg(p), mime_type="image/jpeg"))
    resp = client.models.generate_content(
        model=RERANK_MODEL,
        contents=parts,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    out = json.loads(resp.text)
    if isinstance(out, list):
        out = out[0] if out else {}
    ranking = []
    for letter in out.get("ranking", []):
        if letter in letters and order[letters.index(letter)] not in ranking:
            ranking.append(order[letters.index(letter)])
    ranking += [n for n in names if n not in ranking]
    return {"ranking": ranking, "reason": out.get("reason", ""),
            "confidence": out.get("confidence"), "same_as_any": out.get("same_as_any", True)}
