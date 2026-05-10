"""Test classify_photos.py on the 4 reference images."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_photos import classify_one, ROOT

EXPECTED = {
    "data/Aztec/262.jpg":  "carapace",
    "data/Aztec/1021.jpg": "habitat",
    "data/Aztec/787.jpg":  "habitat",
    "data/Aztec/264.jpg":  "plastron",
}

ok = bad = 0
for rel, expected in EXPECTED.items():
    c = classify_one(ROOT / rel)
    match = "OK " if c.category == expected else "MISS"
    if c.category == expected:
        ok += 1
    else:
        bad += 1
    print(f"  {match}  {rel:25}  predicted={c.category:14} ({c.confidence})  expected={expected}")

print(f"\n{ok}/{ok+bad} correct")
