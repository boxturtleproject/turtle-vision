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

- Shells change slowly, but after 3+ years about one in four returning turtles is not
  found first, and under half get a confident Match.
- A long gap never produced a confident wrong Match once the misfiled photos were removed
  (with them, wrong Matches rose from 6 to 12, all through those photos). The turtle
  drops to Check or Possibly new instead.
- So at sites with old archives the risk is a returning turtle registered as new. The
  "before creating a new turtle" check, and fresh reference photos each season, address it.

The embedding projection (LDA) is fitted with the query's own encounter held out but other
encounters of the turtle kept, so the SIFT column is the cleaner measure of long gaps.
