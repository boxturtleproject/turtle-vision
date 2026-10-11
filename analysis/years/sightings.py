"""Encounter accuracy by how many other sightings of the turtle are on file, and the
protocol-era comparison restricted to turtles photographed in both eras.

    cd analysis/sift_study && PYTHONPATH=.:../.. python ../years/sightings.py
"""
import collections, csv
from pathlib import Path
import numpy as np
import lab_eval
from lab_eval import CIDX, GROUPS, NAMES, REF, emb_class, fuse, per_class, score_vec

TV = Path(__file__).resolve().parents[2]
meta = {r["capture_id"]: r for r in csv.DictReader((TV / "data/capture_meta.csv").open())}
ids = [str(i) for i in REF["ids"]]
DROP = {"718", "719", "720", "721", "201", "979", "980", "981", "634", "1172"}
gone = np.array([c in DROP for c in ids])
SIDES = {"carapace_top", "carapace_left", "carapace_right"}
S = lab_eval.load({"root": True}, full=True)
n = len(ids)

# --- 1) accuracy by sightings on file (problem photos removed as tests and references)
enc = collections.defaultdict(list)
for res, q in zip(S["ref"], S["ref_q"]):
    if gone[q]:
        continue
    excl = (GROUPS == GROUPS[q]) | gone
    v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "grp")
    sc = per_class(v, False, excl)
    f = fuse(sc, emb_class(REF["S_emb"][q], excl), 1)
    enc[(GROUPS[q], NAMES[q])].append((f, CIDX[q], sc))
sightings = {t: len({GROUPS[j] for j in range(n) if NAMES[j] == t and not gone[j]}) for t in set(NAMES)}
bins = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 9, "5-9"), (10, 99, "10+")]
by = collections.defaultdict(lambda: {"photo": [], "enc": [], "match": []})
for (g, t), ph in enc.items():
    others = sightings[t] - 1
    b = next(lbl for lo, hi, lbl in bins if lo <= others <= hi)
    by[b]["photo"] += [np.argmax(f) == c for f, c, _ in ph]
    if len(ph) >= 2:
        tot = sum(f for f, _, _ in ph)
        c = ph[0][1]
        by[b]["enc"].append(np.argmax(tot) == c)
        by[b]["match"].append(max(float(np.nan_to_num(sc[c], neginf=0)) for _, _, sc in ph) >= 2)
print("Accuracy by how many OTHER sightings of the turtle are on file (10 confirmed problem photos removed)")
print(f"{'other sightings':16} {'photos':>7} {'one photo':>10} {'encounters':>11} {'whole encounter':>16} {'confident Match':>16}")
for _, _, b in bins:
    d = by[b]
    if d["photo"]:
        e = f"{np.mean(d['enc']):.0%}" if d["enc"] else "-"
        m = f"{np.mean(d['match']):.0%}" if d["match"] else "-"
        print(f"{b:16} {len(d['photo']):7} {np.mean(d['photo']):10.1%} {len(d['enc']):11} {e:>16} {m:>16}")

# --- 2) protocol era, only turtles photographed in both eras
date = lambda c: meta[c]["encounter_date"] if c not in DROP else ""
era = lambda d: "2024+" if d >= "2024" else "before 2024"
both = {t for t in set(NAMES) if {era(date(ids[j])) for j in range(n) if NAMES[j] == t and date(ids[j])} == {"2024+", "before 2024"}}
pairs = collections.defaultdict(list)
for res, q in zip(S["ref"], S["ref_q"]):
    if NAMES[q] not in both or not date(ids[q]) or meta[ids[q]]["view"] not in SIDES:
        continue
    v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "grp")
    for e in sorted({date(ids[j]) for j in range(n) if NAMES[j] == NAMES[q] and date(ids[j]) and date(ids[j]) != date(ids[q])}):
        if not any(NAMES[j] == NAMES[q] and date(ids[j]) == e and meta[ids[j]]["view"] == meta[ids[q]]["view"] for j in range(n)):
            continue
        keep = np.array([NAMES[j] != NAMES[q] or date(ids[j]) == e for j in range(n)]) & ~gone
        sc = per_class(v, False, ~keep | (GROUPS == GROUPS[q]))
        pairs[(era(date(ids[q])), era(e))].append((np.argmax(sc) == CIDX[q], float(np.nan_to_num(sc[CIDX[q]], neginf=0))))
print(f"\nProtocol era, only the {len(both)} turtles photographed in both eras: photo vs ONE other encounter (same side available)")
for a in ("before 2024", "2024+"):
    for b in ("before 2024", "2024+"):
        r = pairs[(a, b)]
        if r:
            print(f"  {a:12} vs {b:12} {len(r):5} comparisons  found first {np.mean([x[0] for x in r]):6.1%}  median spot {np.median([x[1] for x in r]):4.1f}")
