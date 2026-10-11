# Which side of the turtle is in the photo? (October 2026)

**Result:** the Gemini call that already finds the shell box can also say which side
of the turtle the photo shows, at no extra cost and without changing the box. It is
reliable for top views and, when it names a side, for left vs right; when the head
isn't visible it says "side" instead of guessing. `shellcrop.detect` now returns the
side, and the field-test app shows it.

## Does the side matter for matching? Yes

Different-day test (534 reference photos, RootSIFT + mutual matches;
`side_matters.py`, `side_matters_results.txt`). Spot score of the right turtle:

| right turtle's best photo is… | median spot score | reaches Match (≥ 2) |
|---|---|---|
| the same side as the query | 7.6 | 86% |
| a different side | 1.0 | 29% |

When the right turtle wins, 85% of the time its best match is a photo of the same side.
But matching against only same-side photos is worse (SIFT right first time 83.7% vs
87.6% against every side): some turtles have no photo of that side on file, and other
sides still find them sometimes. So the app keeps comparing against everything and
uses the side to explain and to guide:

- under each uploaded photo: "Photo: left side"
- reference photos of the same side are outlined in green
- when the best guess has no photo of that side on file: "No left side photo of X on
  file, so a low spot score is expected even if it is X"

## Can Gemini tell the side? (`view_test.py`, `results.txt`)

Labels are box-turtle-id's image types (`carapace_top`, `carapace_left`,
`carapace_right`, `plastron`, `front`, `rear`, `site_view`; `back` = `rear`), plus
`carapace_side` for a side view where left/right can't be told. Left side = the
turtle's head points left in the photo (checked against the recorded photos).

Prompt versions, on the first 40 photos:

| asking for | left/right photos: right / "side" / swapped (of 19) |
|---|---|
| the view name ("the turtle's left side") | 15 / – / 4 |
| …with a "side" option | 11 / 5 / 3 |
| which way the head points | 11 / 5 / 3 |
| **which way the head points, only if the head is visible** | **10 / 9 / 0** |

More thinking (`low` vs `minimal`) made no difference, so the fast setting stays.

All 1,000 reference photos with a recorded view, final prompt:

| recorded view | Gemini agrees | |
|---|---|---|
| top | 232 / 234 | |
| left | 156 / 228 | 65 "side", 4 right |
| right | 139 / 229 | 83 "side", 5 left |
| plastron | 52 / 57 | |
| front | 27 / 56 | often called top |
| rear | 19 / 41 | often called top |
| site view | 39 / 151 | most show the turtle close enough to call a side |

Left/right named on 304 left/right photos: 9 swapped (3%). Looking at them, about
four are labelled wrong in the data (the head is visible and points the way Gemini
said), so the real rate is nearer 1–2%. Front and rear photos are usually taken from
above, so "top" is a fair answer for matching.

Boxes from the combined prompt match the box-only prompt (median overlap 0.99, none
below 0.7 of 628), and it found a shell in 990 photos vs 629 before (mostly site and
front/rear photos the box-only run hadn't been asked about).
