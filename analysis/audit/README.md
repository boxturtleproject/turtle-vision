# Photo label audit (October 2026)

**Why:** each turtle is scored by its single best-matching reference photo, so one photo
filed under the wrong turtle can make the matcher name the wrong turtle, however many
correct photos there are. This happened: the Jigsaw / Cruella and Flower / Sunflower
"look-alikes" in the SIFT study were misfiled photos, not look-alike turtles.

**How:** `label_audit.py` compares every reference shell crop with every other one (628
photos, about 197,000 pairs, ~3 minutes on 10 cores) and counts *consistent matches*:
mutual RootSIFT matches that also fit one stretch of the image (loose affine RANSAC). Two
different turtles never had more than 6; repeat photos of one turtle usually have 20 to
800. Each photo is then flagged if it

- **misfiled**: matches another turtle strongly (15+) and more than any photo of its own
- **duplicate**: is the same picture as another photo filed under a different turtle or
  date (checked by aligning the two originals: almost no pixel difference)
- **no match**: is a top or side photo with no real match (under 8) to any other photo
  of its turtle, though the turtle has other photos of that side

`review_page.py` builds a self-contained review page (default
`~/Desktop/turtle-review/audit.html`) with each flag beside its evidence, labelled the way
Airtable shows photos (turtle, side, filename date and time, original filename, encounter
or turtle record), with verdict buttons and a CSV download. Nothing is uploaded anywhere.

```
python analysis/audit/label_audit.py     # writes analysis/out/audit/ and data/duplicate_pictures.csv
python analysis/audit/review_page.py
```

## What the first audit found (October 11, 2026)

Reviewed by the field team:

| flag | photos | verdict |
|---|---|---|
| misfiled | a Cruella encounter of July 3, 2022: left, right, front and site view (files `2025-06-16 21.31.19.jpg` to `21.32.17.jpg`) | Jigsaw. Both turtles were encountered that day; Cruella's own set is in another encounter |
| misfiled / duplicate | Sunflower top photo on the turtle's own record (`2025-05-23 14.17.45.jpg`) | Flower: the same picture as a Flower encounter photo from May 27, 2022 |
| duplicate | Jigsaw right side, encounter of June 11, 2020 (`2025-06-17 07.35.00.jpg`) | the same picture as the misfiled right-side photo above |
| misfiled (knock-on) | Jigsaw left side, encounter of June 20, 2025 | correctly Jigsaw; flagged only because it matches the misfiled photos |
| no match | 15 photos (7 Bubbles: a dark, worn shell with few spots) | 11 poor photos of the right turtle (one marked unusable); 4 photos of another turtle: a "Bubbles" encounter of May 18, 2024 (top, left, right) and a T066 top photo from May 3, 2025 |
| duplicate | 115 pairs, mostly a photo on the turtle's own record that is also in an encounter | same turtle; filename dates differ |

**Every wrong confident "Match" in the reference test came from these photos.** Leaving
out the 10 confirmed problem photos removed all 7 wrong Matches (see
`../index_images/`).

## Dates

Filename dates are often the day a photo was edited, not taken: the reference photos come
from encounters in 2020–2025 but almost all carry 2025 filename dates, and some encounters
carry several. Encounter dates from box-turtle-id are the reliable record, so
`matching.turtle_day_groups` groups sightings by encounter date (`fetch_meta.py` records
`encounter_id` and `encounter_date`), and the duplicate pictures found here
(`data/duplicate_pictures.csv`) are merged into one sighting. Accuracy figures measured
before this were optimistic; see `../SIFT_STUDY.md`.
