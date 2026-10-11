# Improving the SIFT spot matcher (October 2026)

**Result:** two changes to how SIFT matches are counted — *RootSIFT* descriptors and
*mutual* matches at ratio 0.8 — raise SIFT + embeddings from **90.3% to 96.0%** right
first time on the 176 verified known field photos (15 photos fixed, none broken), and
from **80.6% to 84.3%** on the different-day reference test. The spot score also
separates new turtles from known ones far better. Same keypoints, same speed, and
box-turtle-id's stored descriptors can be reused (RootSIFT is computed from them).

Scripts and raw outputs: [`sift_study/`](sift_study/). Intermediate files (features,
score matrices, field-photo embeddings) go to `analysis/out/sift_study/` (gitignored).

## What SIFT did before

box-turtle-id's matcher (ported to `sift_match.py`): SIFT keypoints on the shell crop
resized to 250px wide (~900 per photo); for each keypoint, its nearest neighbour in the
reference photo, kept if clearly closer than the second nearest (Lowe ratio test, 0.67);
counted in both directions, the larger count taken; score = count / fewer keypoints × 100.

## What was tried

Every variant was scored against all 628 reference crops on two sets:

- **Different-day test** — 515 reference photos, each matched only against *other
  encounters* of its turtle, as in `evaluate.py`. *Corrected October 11:* the first
  version grouped photos by the date in their filenames, which turned out to be mostly the
  day a photo was edited: the photos come from encounters in 2020–2025, yet nearly all
  carry 2025 filename dates. Some encounters were split into "different days", and 45
  pictures stored twice as re-saved copies let photos match their own copy. The
  [label audit](audit/README.md) found both. Sightings are now grouped by encounter date
  from box-turtle-id, with duplicate pictures merged (`data/duplicate_pictures.csv`), and
  every different-day number here is the corrected one. The first version reported 534
  photos and 85.2% → 88.6%. The field-photo numbers were never affected.
- **Field photos** — the 318 hand-verified photos from the October field test
  (176 known turtles, 142 new), as in `FIELD_TEST.md`.

A fixed 100-photo sample (50 reference, 30 known + 20 new field) screened ideas in ~2
minutes each (`screen_results.txt`, run with the first grouping); the finalists were run on everything
(`final_results.txt`). Matching ran on the GPU (`lab_sift.py`), checked to reproduce the
production scores exactly; today's matcher reproduces the field test's 90.3% / 96.6%.

| idea | what it is | verdict |
|---|---|---|
| **Mutual matches** | a pair counts only if each descriptor is the other's nearest neighbour (cross-check) | **big win**; makes a looser ratio (0.8) safe |
| **RootSIFT** | L1-normalise and square-root each descriptor (Arandjelović & Zisserman 2012; used by HotSpotter) | **win** |
| sqrt normalisation | divide by √(keypoints × keypoints) instead of the smaller count | small win for new-turtle separation |
| looser ratio without mutual check | 0.75 / 0.8 | worse: lets in junk |
| ratio 0.85 / 0.9 with mutual check | | worse than 0.8 |
| position check (RANSAC) | keep pairs consistent with one homography / affine warp | no gain on top of mutual matches (a domed shell from different angles fits a flat warp poorly) |
| LNBNN (HotSpotter's one-vs-many scoring) | each descriptor votes for turtles closer than the crowd | best on the reference test (90%; its votes were computed with the first grouping, so likely a little high) but not on field photos; weaker new-turtle separation |
| yellow channel (Lab b*) | SIFT on the yellow–blue axis instead of grey | better on reference photos, worse on field photos (~5% of known turtles named wrong) |
| CLAHE contrast | evens out sun and shade | worse |
| 400px instead of 250px | | no better, 4× slower |
| oval mask | ignore crop corners | no consistent gain |

## Full-set results

SIFT + embeddings fused as before (`log(1 + best spot) + best embedding similarity`):

| | today | **RootSIFT + mutual 0.8** |
|---|---|---|
| field, right first time (176) | 90.3% | **96.0%** |
| field, in top 5 | 96.6% | **98.3%** |
| different-day, right first time (515) | 80.6% | **84.3%** |
| different-day, in top 5 | 89.5% | **92.0%** |
| SIFT alone, different-day right first time | 77.3% | 83.1% |
| SIFT alone, field right first time | 79.5% | 90.9% |
| known vs new separation (AUC, field) | 0.88 | 0.95 |

Embeddings still help: with the new SIFT they fix 10 field photos and break none.

## New spot-score bands

The score is on a new scale, so the bands were re-set (`bands_results.txt`):

| band | spot score | field photos |
|---|---|---|
| **Match** | ≥ 2.0 | 127 known, all named right; no new turtles |
| **Check** | 1.2–2.0 (1.5–2.0 with 2+ photos) | 22 known (all right), 5 new |
| **Possibly new** | below that | 27 known (20 with the right top pick), 137 new = **96.5% of new turtles** |

Before, the 1.5 line caught 92% of new turtles while naming 3% of known turtles wrong and
sending 26% to "new". Now nobody known is named wrong in Match or Check, and 15% go to
Possibly new. With several photos a new turtle's best photo reaches 1.2 more often
(7% for the best of two vs 3.5% for one), so the check zone starts at 1.5; no new turtle
reached 1.5 either way. Encounters with 2+ photos: still 31 of 31 right.

The lines were chosen on the same field photos, so expect slightly worse in practice;
re-check them once the app has a season of confirmed submissions.

## Spot score as a likelihood

`likelihood.py` measures how often the top pick was right at each score (field photos;
reference test as a check). It is a step rather than a smooth curve, so a small lookup
table is more honest than a fitted formula:

| spot | field: top pick right | reference test |
|---|---|---|
| < 0.8 | 6% | 2% |
| 0.8–1.0 | 18% | 10% |
| 1.0–1.2 | 40% | 26% |
| 1.2–1.5 | 44% | 50% |
| 1.5–2.0 | 100% (18/18) | 100% (18/18) |
| 2.0–3.0 | 100% | 96% |
| 3+ | 100% | 97–99% |

The low-score rows depend on how many submissions are new turtles (45% in this field
set); the high-score rows barely do.

Both apps show it next to the spot score, rounded and capped at 98%
(`sift_match.LIKELY`): under 0.8 ~5%, 0.8–1 ~15%, 1–1.2 ~30%, 1.2–1.5 ~45% (~30% with
2+ photos, where a new turtle's best photo reaches 1.2 more often), 1.5–2 ~95%,
2–3 ~97%, 3+ ~98%. Refit it from confirmed submissions; the review page lists the
shown % beside how confirmed photos turned out.

## Recognising a turtle years later

With encounter dates, the reference set spans 2020–2025, so the test can ask how well a
turtle is found when the only photos of it on file are years old
([`years/`](years/results.txt)). With the 10 confirmed problem photos left out:

| own turtle's photos on file | test photos | SIFT right first time | right turtle in top 5 | Match (spot ≥ 2) | wrong Match |
|---|---|---|---|---|---|
| any other encounter | 483 | 87.4% | 94.0% | 77.8% | 0 |
| 1+ year away | 389 | 83.8% | 88.9% | 64.3% | 0 |
| 2+ years away | 289 | 82.4% | 86.2% | 60.9% | 0 |
| 3+ years away | 190 | 72.1% | 76.8% | 53.7% | 0 |

A long gap makes the matcher less sure, never confidently wrong. But the drop is not the
shell changing: matched against one other encounter at a time, with a photo of the same
side, the right turtle is found first 81% of the time under a year apart and 86% at 3+
years. The drop above comes from fewer photos being left to match against. Adult markings
are stable; what matters is a good photo of each side on file. With two or more earlier
sightings of a turtle on file, every whole encounter in the test was named right (83 of
83); with only one, 71% ([`years/`](years/README.md)).

## Data finding: misfiled photos, not duplicate turtles

In the reference test with each photo's own turtle removed, the only photos that still
reached the Match band were Jigsaw ↔ Cruella (34) and Flower ↔ Sunflower (9)
(`duplicates_results.txt`). This first read as "same turtle", which was wrong: the four
are different turtles (confirmed by the field team). Checked photo by photo
([label audit](audit/README.md)), every one of those hits goes through a few misfiled
photos: the four photos of one Cruella encounter (July 3, 2022, the day both turtles were
seen: left, right, front, site view) are Jigsaw, and one Sunflower top photo is the same
picture as a Flower photo. Confirmed by the field team on October 11. The lesson: a single misfiled reference
photo is enough to make the matcher name the wrong turtle, so check flags per photo.
Matching only against a few "index" photos per turtle does not fix this and costs accuracy;
leaving out the few bad photos does ([`index_images/`](index_images/README.md)).

## Reproduce

```
cd analysis/sift_study
python lab_data.py                              # reference/field data, embeds field crops once
python lab_sift.py '{}' full                    # today's SIFT, all scoring variants (~30 min)
python lab_sift.py '{"root": true}' full
python lab_sift.py '{"root": true, "pre": "lab_b"}' full
python final.py; python bands.py; python likelihood.py; python duplicates.py
python lab_sift.py '{"root": true}'             # 100-photo screening sample
python screen.py base,mutg@0.8,lnbnn_ref
```

Needs the field-test photos' crops in `uploads/` (not committed) and torch.
