# Recognising a turtle years later (October 2026)

The reference photos come from box-turtle-id encounters dated 2020–2025 (their filenames
mostly carry 2025 dates: the day they were edited). `years_apart.py` asks how well a turtle
is found when the only photos of it on file are at least 1, 2 or 3 years away; other
turtles keep every photo. Different-day test, RootSIFT + mutual matches; results in
`results.txt`.

With the 10 photos the October audit confirmed as misfiled or unusable left out:

| own turtle's photos on file | test photos | SIFT right first time | right turtle in top 5 | Match (spot ≥ 2) | wrong Match |
|---|---|---|---|---|---|
| any other encounter | 483 | 87.4% | 94.0% | 77.8% | 0 |
| 1+ year away | 389 | 83.8% | 88.9% | 64.3% | 0 |
| 2+ years away | 289 | 82.4% | 86.2% | 60.9% | 0 |
| 3+ years away | 190 | 72.1% | 76.8% | 53.7% | 0 |

- When only photos 3+ years away are allowed, about one in four returning turtles is not
  found first, and under half get a confident Match. **This is not the shell changing**
  (see the control below): it is the smaller set of photos left to match against.
- A long gap never produced a confident wrong Match once the misfiled photos were removed
  (with them, wrong Matches rose from 6 to 12, all through those photos). The turtle
  drops to Check or Possibly new instead.
- So the risk at sites with thin or old archives is a returning turtle registered as new.
  The "before creating a new turtle" check addresses it, and so does a complete, good
  set of photos per turtle (top, left and right).

## Control: is it the years or the photos? (`gap_control.py`, `gap_control_results.txt`)

Matching each top or side photo against **one** other encounter of its turtle at a time,
one that has a photo of the same side, so every comparison has about the same photos:

| gap between the two encounters | comparisons | found first | median spot score |
|---|---|---|---|
| under 1 year | 1,115 | 81% | 3.4 |
| 1–2 years | 558 | 77% | 2.7 |
| 2–3 years | 351 | 84% | 3.3 |
| 3+ years | 291 | 86% | 3.1 |

No decline with the years, and photos from 2020–2025 carry about the same number of spot
features (840–920 per photo). Adult shell markings are stable, as the field team expects.
What matters is having good photos of each side on file; old photos stay useful unless
the shell has since been damaged (predation, injury, disease).

The embedding projection (LDA) is fitted with the query's own encounter held out but other
encounters of the turtle kept, so the SIFT column is the cleaner measure of long gaps.

## Whole encounters (`encounters_test.py`, `encounters_results.txt`)

Adding up an encounter's photos (what the app does) takes the different-day test from
84.3% right first time for one photo to **93.9%** for a whole encounter (99 encounters
with 2+ photos; 94.9% with the confirmed problem photos removed): about two thirds of the
errors go. Other ways of combining (only top/left/right, best spot score + mean
embedding, one vote per photo) land within a point. The encounters that still fail have
nothing good to match against: Bubbles' only other encounter is a different turtle
(confirmed in the audit), Flower_2's two encounters are five years apart, Strobe has a
single encounter, and T039_70's April 2023 photos show a shell caked in dried mud.
More photos of the query can't make up for missing or unusable reference photos.

## How many sightings are on file (`sightings.py`, `sightings_results.txt`)

With the confirmed problem photos removed:

| other sightings of the turtle on file | one photo | whole encounter | confident Match |
|---|---|---|---|
| 1 | 57% | 71% (14 encounters) | 79% |
| 2 | 96% | 100% (6) | 100% |
| 3–4 | 80% | 100% (13) | 85% |
| 5–9 | 95% | 100% (42) | 98% |
| 10+ | 99% | 100% (22) | 100% |

Once a turtle has two or more earlier sightings on file, every encounter in the test (83 of
83) was named right. The weak spot is a turtle seen only once before. So a complete, good
set of photos at a turtle's first encounters matters more than anything else.

## Photo protocol era (`gap_control.py`)

The field team standardized its photo protocol in 2024. Matched against one other
encounter at a time, within the 12 turtles photographed in both eras:

| photo from | other encounter from | found first | median spot score |
|---|---|---|---|
| before 2024 | before 2024 | 95% | 6.4 |
| before 2024 | 2024 on | 83% | 3.2 |
| 2024 on | 2024 on | 74% | 1.9 |

Newer photos match each other less strongly, with the same turtles, framing (shell fills
~57% of the frame) and resolution. Looking at the photos, newer ones are mostly taken where
the turtle was found, in grass and leaves or in hand, with vegetation or fingers across
the shell; older ones mostly on bare gravel. Anything covering the shell hides spots, so
keeping the shell's surface clear is likely the fix.

The photo set also holds 105 copies of a placeholder drawing attached as photos (mostly
Cruella, Jigsaw, Compass and Unidentified); the classifier already keeps them out of
matching.
