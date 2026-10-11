# Index photos or a blacklist? (October 2026)

Question: would matching only against a few strong "index" photos per turtle be as
accurate, faster, and safer from misfiled photos? `index_test.py` chooses index photos
automatically by *support*: how many other encounters of the same turtle a photo matches
strongly (15+ consistent matches, from the label audit's pair counts). Misfiled and poor
photos get no support, so they drop out. Kept per side (top, left, right). Results in
`results.txt`.

| reference photos used | photos | different-day (515) right first time | field (176 known) right first time |
|---|---|---|---|
| **every photo** | 628 | 84.3% | **96.0%** |
| every photo except unsupported ones | 499 | 81.9% | 91.5% |
| index: 5 per side | 364 | 85.8% | 89.8% |
| index: 3 per side | 284 | 84.1% | 89.2% |
| index: 2 per side | 216 | 83.3% | 89.2% |
| index: 1 per side | 111 | 81.9% | 82.4% |
| **every photo except 10 confirmed problems** | 618 | **86.2%**\* | 96.0% |

\* 506 test photos: the problem photos themselves are not tested.

- Index photos lose 6–14 points on real field photos: a new sighting is often found by an
  unusual reference photo (another angle, light, or year), not the most typical one.
- Speed is not a reason: SIFT against all 628 photos takes about a second; the Gemini
  crop and embedding calls take most of the ~4 seconds per photo.
- Requiring a supported reference before a confident Match (two tiers) would push 17% of
  correct Matches to Check to catch one wrong Match.
- All 6 wrong Matches in the different-day test came from photos the audit flagged and the
  field team confirmed as misfiled. Leaving out those 10 photos removed every one.

So: a **blacklist**, not a whitelist. Match against every good photo; hold out photos
confirmed misfiled or too poor, and new photos the audit flags until they are reviewed.
