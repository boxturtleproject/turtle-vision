"""How well is a turtle recognised years later?

The reference photos come from encounters in 2020-2025. For each test photo, its
own turtle may be matched only through photos at least N years away (other
turtles keep every photo), as if the turtle came back after N years with nothing
newer on file. Dates are encounter dates; a photo on a turtle's own record takes
the date of the encounter it was copied from (data/duplicate_pictures.csv) and
is otherwise left out of its own turtle's gallery for N >= 1.

    cd analysis/sift_study && PYTHONPATH=.:../.. python ../years/years_apart.py
"""
import csv
from datetime import date
from pathlib import Path

import numpy as np

import lab_eval
import matching
from lab_eval import CIDX, GROUPS, NAMES, REF, emb_class, fuse, per_class, score_vec

TV = Path(__file__).resolve().parents[2]
meta = {r["capture_id"]: r for r in csv.DictReader((TV / "data/capture_meta.csv").open())}
ids = [str(i) for i in REF["ids"]]
when = {c: meta[c]["encounter_date"] for c in ids if meta.get(c, {}).get("encounter_date")}
for a, b in matching.duplicate_pictures():  # profile copies take their encounter copy's date
    a, b = str(a), str(b)
    if a in when and b not in when:
        when[b] = when[a]
    elif b in when and a not in when:
        when[a] = when[b]
D = [date.fromisoformat(when[c]) if c in when else None for c in ids]
n = len(ids)
S = lab_eval.load({"root": True}, full=True)
print(f"{len(S['ref_q'])} test photos; encounter years "
      f"{min(d.year for d in D if d)}-{max(d.year for d in D if d)}; {sum(d is None for d in D)} photos undated\n")
print(f"{'own turtle only through photos':40} {'photos':>6} {'SIFT':>6} {'+emb':>6} {'top 5':>6} {'Match (>=2)':>12} {'wrong Match':>12}")
for years in (0, 1, 2, 3):
    k = t1 = f1 = f5 = match = wrong = 0
    for res, q in zip(S["ref"], S["ref_q"]):
        if D[q] is None:
            continue
        own = NAMES == NAMES[q]
        far = np.array([d is not None and abs((d - D[q]).days) >= 365 * years for d in D])
        excl = (GROUPS == GROUPS[q]) | (own & ~far if years else own & False)
        if not (own & ~excl).any():
            continue
        k += 1
        v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "grp")
        sc = per_class(v, False, excl)
        f = fuse(sc, emb_class(REF["S_emb"][q], excl), 1)
        p = int(np.argmax(f))
        t1 += np.argmax(sc) == CIDX[q]
        f1 += p == CIDX[q]
        f5 += CIDX[q] in np.argsort(-f)[:5]
        if sc[p] >= 2:
            match += 1
            wrong += p != CIDX[q]
    label = "any other sighting" if years == 0 else f"{years}+ year{'s' if years > 1 else ''} away"
    print(f"{label:40} {k:6} {t1 / k:6.1%} {f1 / k:6.1%} {f5 / k:6.1%} {match / k:12.1%} {wrong:12}")


# wrong Matches at each gap, with the confirmed problem photos (audit, October 11) left out
DROP = {"718", "719", "720", "721", "201", "979", "980", "981", "634", "1172"}
gone = np.array([c in DROP for c in ids])
print("\nWith the 10 confirmed problem photos left out:")
for years in (0, 1, 2, 3):
    k = t1 = f5 = match = 0
    wrong = []
    for res, q in zip(S["ref"], S["ref_q"]):
        if D[q] is None or gone[q]:
            continue
        own = NAMES == NAMES[q]
        far = np.array([d is not None and abs((d - D[q]).days) >= 365 * years for d in D])
        excl = (GROUPS == GROUPS[q]) | gone | (own & ~far if years else own & False)
        if not (own & ~excl).any():
            continue
        k += 1
        v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "grp")
        sc = per_class(v, False, excl)
        f = fuse(sc, emb_class(REF["S_emb"][q], excl), 1)
        p = int(np.argmax(f))
        t1 += np.argmax(sc) == CIDX[q]
        f5 += CIDX[q] in np.argsort(-f)[:5]
        if sc[p] >= 2:
            match += 1
            if p != CIDX[q]:
                wrong.append(f"{NAMES[q]}->{lab_eval.CLASSES[p]} ({sc[p]:.1f})")
    label = "any other sighting" if years == 0 else f"{years}+ years away"
    print(f"  {label:22} {k:4} photos  SIFT {t1 / k:.1%}  top 5 {f5 / k:.1%}  Match {match / k:.1%}  wrong Match {len(wrong)}: {wrong}")
