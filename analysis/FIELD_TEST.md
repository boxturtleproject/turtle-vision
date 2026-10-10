# Field test, October 2026

364 new photos from the field (taken May–June 2026) were run through every
matcher with `batch_identify.py`, and each was verified by hand on the
review page (`/summary`): **176 known turtles, 142 new turtles, 46 bad
photos**. None of these photos are in the reference set (checked by visual
fingerprint against all 1,000 reference photos).

Data: [`field_2026-10/`](field_2026-10/) (no images). `uploads_summary.csv`
has one row per photo with the verified answer, each method's pick and
result, and the spot score. Reproduce every number below with:

```bash
python analysis/field_results.py analysis/field_2026-10
```

## 1. SIFT + embeddings is the best matcher

Known turtles (176 photos), one photo at a time:

| method | right first time | right turtle in top 5 |
|---|---:|---:|
| **SIFT + embeddings** | **90.3%** | **96.6%** |
| SIFT only | 79.5% | 93.8% |
| Embeddings only (crop + tight) | 47.7% | 85.8% |
| Embeddings, tight crop | 27.3% | 75.0% |
| Embeddings, shell crop | 28.4% | 65.3% |
| Embeddings, whole photo | 22.2% | 61.4% |

- The field result (90%) beat the reference-set prediction (85%).
- Adding the embeddings to SIFT gains 11 points over SIFT alone.
- Embeddings alone name the turtle poorly, but are useful as the second half of the combination.
- Of the 17 misses, 11 had the right turtle in the top 5, so a quick human check of the top 5 recovers most of them.
- Gemini's second opinion ran on only 18 photos (16 right), too few to judge. On the reference test it added about 1 point, within noise, at about 3.5s and one API call per photo. **It isn't needed.**

## 2. Where to draw the "new turtle" line

Only the spot score (SIFT) separates known from new turtles. As a separator it scores AUC 0.88, where 1 is perfect and 0.5 is a coin flip. Embedding similarity scores 0.58 (combined) and 0.54 (whole photo).

Rule: if the spot score is at least T, name the best pick; otherwise say "new turtle". Results over the 318 good photos:

| T | known named right | known named wrong | known called new | new caught | new named | overall right |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 80% | 3% | 17% | 75% | 25% | 78% |
| 1.25 | 76% | 3% | 21% | 85% | 15% | 80% |
| **1.5** | **72%** | **3%** | **26%** | **92%** | **8%** | **81%** |
| 2.0 | 62% | 2% | 35% | 95% | 5% | 77% |
| 4.0 | 35% | 0% | 65% | 99% | 1% | 64% |

What the photos in each spot-score band turned out to be:

| spot score | photos | known (pick right) | new | bad |
|---|---:|---:|---:|---:|
| 4+ | 66 | 62 (62) | 1 | 3 |
| 2–4 | 63 | 52 (48) | 6 | 5 |
| 1–2 | 73 | 32 (31) | 28 | 13 |
| under 1 | 162 | 30 (18) | 107 | 25 |

**Recommendation:** use three zones.
- **2 or more:** accept the match (about 95% right).
- **1.5 to 2:** probably this turtle, check it.
- **Under 1.5:** probably a new turtle, check it (catches 92% of new turtles).

The earlier banner cut-off of 4 is too strict: it calls 65% of known turtles "new".

## 3. Several photos per encounter help a lot

An encounter is one turtle on one day, with the day taken from each photo's EXIF. Each turtle's score is added up across the encounter's photos, as the app's multi-photo upload does.

| method | one photo | per encounter | encounters with 2+ photos |
|---|---:|---:|---:|
| SIFT + embeddings | 90.3% | 97.3% | 91.8% → **100%** (31/31) |
| SIFT only | 79.5% | 91.9% | 83.4% → 100% |
| Embeddings only | 47.7% | 67.6% | 55.9% → 77.4% |

Using the encounter's best photo for the new-turtle rule also cuts the known turtles wrongly called "new": from 26% to 14% at a cut-off of 1.5, and from 35% to 16% at 2.

- **Why it helps:** each photo is another chance at a clear view, and different views (top, left, right) cover different plates. The right turtle scores well on most photos, while a wrong turtle usually wins only on one bad photo.
- **Caveat:** new-turtle photos weren't labelled by individual, so the effect on *new* encounters wasn't measured. More photos also gives more chances for one photo of a new turtle to match by luck, so use a cut-off of about 2 on the encounter's best photo.

**Field protocol:** take 3–5 shell photos per encounter (top, left, right, shell filling the frame) and upload them together.

## 4. Other findings

- **Reference coverage drives accuracy.** Turtles with few reference photos did worst: Glyph 7/11 right (13 reference photos), Emoji 6/8 (4), Jazzy 9/11 (7), Lotus 3/5. Turtles with 30+ references (Cruella 23/24, Shingles 16/16, Compass 17/18) were 94–100% right.
- **45% of usable photos were new turtles.** The reference set is missing many individuals. Grouping the unmatched photos by SIFT among themselves (`analysis/new_turtles.py`) gives 27 candidate new turtles, 5 of them seen on two or more days, plus 96 photos that match nothing.
- **13% of photos were bad,** and they mostly score low, so a bad photo usually costs a "no match" rather than a wrong name.
- **Shell types:** clustering the reference embeddings (`analysis/shell_groups.py`) gives loose groups. Roughly half reflect shell pattern (dense spots, bold contrast, radiating streaks, sparse brown) and half photo conditions (dirt, faded light, plants in the frame). On average 56% of a turtle's photos fall in one group.

## Next steps

1. **Grow the reference set:** add the verified photos of known turtles, especially Glyph, Emoji, Jazzy and Lotus, plus the new turtles once they're grouped into named individuals.
2. **Update the app:** set the banner to the three zones and turn Gemini off by default.
3. **Fix data problems in box-turtle-id:** byte-identical photos filed under different dates; Cruella #719 is the same photo as Jigsaw #801; D.E.P./Spikey and Flower/Sunflower may be duplicate identities.
