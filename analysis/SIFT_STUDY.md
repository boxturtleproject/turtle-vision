# Improving the SIFT spot matcher (October 2026)

**Result:** two changes to how SIFT matches are counted — *RootSIFT* descriptors and
*mutual* matches at ratio 0.8 — raise SIFT + embeddings from **90.3% to 96.0%** right
first time on the 176 verified known field photos (15 photos fixed, none broken), and
from **85.2% to 88.6%** on the different-day reference test. The spot score also
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

- **Different-day test** — 534 reference photos, each matched only against *other days*
  of its turtle (duplicate photos merged), as in `evaluate.py`.
- **Field photos** — the 318 hand-verified photos from the October field test
  (176 known turtles, 142 new), as in `FIELD_TEST.md`.

A fixed 100-photo sample (50 reference, 30 known + 20 new field) screened ideas in ~2
minutes each (`screen_results.txt`); the finalists were run on everything
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
| LNBNN (HotSpotter's one-vs-many scoring) | each descriptor votes for turtles closer than the crowd | best on the reference test (94.4%) but not on field photos; weaker new-turtle separation |
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
| different-day, right first time (534) | 85.2% | **88.6%** |
| different-day, in top 5 | 92.5% | **94.6%** |
| SIFT alone, different-day right first time | 81.6% | 87.6% |
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
| 0.8–1.0 | 18% | 11% |
| 1.0–1.2 | 40% | 21% |
| 1.2–1.5 | 44% | 50% |
| 1.5–2.0 | 100% (18/18) | 100% (18/18) |
| 2.0–3.0 | 100% | 97% |
| 3+ | 100% | 98% |

The low-score rows depend on how many submissions are new turtles (45% in this field
set); the high-score rows barely do.

Both apps show it next to the spot score, rounded and capped at 98%
(`sift_match.LIKELY`): under 0.8 ~5%, 0.8–1 ~15%, 1–1.2 ~30%, 1.2–1.5 ~45% (~30% with
2+ photos, where a new turtle's best photo reaches 1.2 more often), 1.5–2 ~95%,
2–3 ~97%, 3+ ~98%. Refit it from confirmed submissions; the review page lists the
shown % beside how confirmed photos turned out.

## Data finding: Jigsaw = Cruella, Flower = Sunflower?

In the reference test with each photo's own turtle removed, the only photos that still
reached the Match band were Jigsaw ↔ Cruella (34) and Flower ↔ Sunflower (9)
(`duplicates_results.txt`). Many *different* photos of Jigsaw and Cruella match each other
strongly, not just the known shared photo (#719 = #801), so they are very likely the same
turtle. Worth checking in box-turtle-id.

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
