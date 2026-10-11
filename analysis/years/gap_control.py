"""Is it the years, or the photos? Match each encounter against ONE other encounter of its
turtle at a time (so every comparison has about the same number of photos), and see whether
the gap between them still matters. Also: are older photos simply poorer photos?

    cd analysis/sift_study && PYTHONPATH=.:../.. python ../years/gap_control.py
"""
import collections, csv
from datetime import date
from pathlib import Path
import numpy as np
import lab_eval
from lab_eval import CIDX, GROUPS, NAMES, REF, score_vec, per_class

TV = Path(__file__).resolve().parents[2]
meta = {r["capture_id"]: r for r in csv.DictReader((TV / "data/capture_meta.csv").open())}
ids = [str(i) for i in REF["ids"]]
DROP = {"718", "719", "720", "721", "201", "979", "980", "981", "634", "1172"}
SIDES = {"carapace_top", "carapace_left", "carapace_right"}
enc = [meta[c]["encounter_date"] if c not in DROP else "" for c in ids]
side = np.array([meta[c]["view"] in SIDES for c in ids])
S = lab_eval.load({"root": True}, full=True)
n = len(ids)

# 1) one other encounter at a time
by_gap = collections.defaultdict(list)
for res, q in zip(S["ref"], S["ref_q"]):
    if not enc[q] or not side[q]:
        continue
    v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "grp")
    own_encs = sorted({enc[j] for j in range(n) if NAMES[j] == NAMES[q] and enc[j] and enc[j] != enc[q]})
    for e in own_encs:
        keep = np.array([NAMES[j] != NAMES[q] or enc[j] == e for j in range(n)]) & np.array([c not in DROP for c in ids])
        same_side_avail = any(NAMES[j] == NAMES[q] and enc[j] == e and meta[ids[j]]["view"] == meta[ids[q]]["view"] for j in range(n))
        if not same_side_avail:
            continue  # compare like with like: the other encounter has a photo of the same side
        sc = per_class(v, False, ~keep | (GROUPS == GROUPS[q]))
        gap = abs((date.fromisoformat(e) - date.fromisoformat(enc[q])).days) / 365.25
        b = "under 1 year" if gap < 1 else "1-2 years" if gap < 2 else "2-3 years" if gap < 3 else "3+ years"
        by_gap[b].append((np.argmax(sc) == CIDX[q], float(np.nan_to_num(sc[CIDX[q]], neginf=0))))
print("Each top/left/right photo matched against ONE other encounter of its turtle that has a photo of the same side")
print(f"{'gap between the two encounters':32} {'comparisons':>11} {'found first':>12} {'median spot score':>18}")
for b in ("under 1 year", "1-2 years", "2-3 years", "3+ years"):
    r = by_gap[b]
    if r:
        print(f"{b:32} {len(r):11} {np.mean([x[0] for x in r]):12.1%} {np.median([x[1] for x in r]):18.1f}")

# 2) photo quality by encounter year
print("\nShell features per photo by encounter year (top/left/right photos):")
kp = S["kp_ref"]
by_year = collections.defaultdict(list)
for j in range(n):
    if enc[j] and side[j]:
        by_year[enc[j][:4]].append(kp[j])
for y in sorted(by_year):
    print(f"  {y}: {len(by_year[y]):3} photos, median {int(np.median(by_year[y]))} spot features")
