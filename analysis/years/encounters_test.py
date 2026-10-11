"""How much does combining an encounter's photos help, and how best to combine them?

Each encounter (one turtle, one encounter date) is matched against the other encounters;
the photos' scores are combined five ways. Run from analysis/sift_study:
    PYTHONPATH=.:../.. python ../years/encounters_test.py
"""
import collections, csv
from pathlib import Path
import numpy as np
import lab_eval
from lab_eval import GROUPS, CIDX, REF, K, score_vec, per_class, emb_class, fuse
TV = Path(__file__).resolve().parents[2]
ids = [str(i) for i in REF["ids"]]
view = {r["capture_id"]: r["view"] for r in csv.DictReader((TV / "data/capture_meta.csv").open())}
SIDES = {"carapace_top", "carapace_left", "carapace_right"}
DROP = {"718", "719", "720", "721", "201", "979", "980", "981", "634", "1172"}
S = lab_eval.load({"root": True}, full=True)
gone = np.array([i in DROP for i in ids])


def build(drop_refs):
    rows = collections.defaultdict(list)
    for res, q in zip(S["ref"], S["ref_q"]):
        add(rows, res, q, (GROUPS == GROUPS[q]) | (gone if drop_refs else False))
    return rows


def add(rows, res, q, excl):
    v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "grp")
    sc = per_class(v, False, excl); ec = emb_class(REF["S_emb"][q], excl)
    # one turtle per encounter, even when a misfiled copy links two turtles' photos
    rows[(GROUPS[q], lab_eval.NAMES[q])].append(dict(id=ids[q], c=CIDX[q], sc=sc, ec=ec, f=fuse(sc, ec, 1), side=view.get(ids[q]) in SIDES))

def combine(ph, how):
    if how == "sum (app today)": return sum(p["f"] for p in ph)
    if how == "sum, top/left/right only":
        s = [p for p in ph if p["side"]] or ph; return sum(p["f"] for p in s)
    if how == "best spot score + mean embedding":
        sc = np.max([np.where(np.isfinite(p["sc"]), p["sc"], 0) for p in ph], 0); ec = np.mean([p["ec"] for p in ph], 0)
        return np.log1p(sc) + ec
    if how == "sum of spot evidence + best embedding":
        return sum(np.log1p(np.where(np.isfinite(p["sc"]), np.maximum(p["sc"], 0), 0)) for p in ph) + np.max([p["ec"] for p in ph], 0)
    if how == "vote (each photo's top pick)":
        v = np.zeros(K)
        for p in ph: v[np.argmax(p["f"])] += 1 + 1e-3 * p["f"].max()
        return v + 1e-6 * sum(p["f"] for p in ph)

for label, drop in (("every photo", set()), ("10 confirmed problem photos removed (as test photos and as references)", DROP)):
    rows = build(bool(drop))
    print(f"\n{label}")
    single = [np.argmax(p["f"]) == p["c"] for ph in rows.values() for p in ph if p["id"] not in drop]
    print(f"  one photo at a time: {np.mean(single):.1%} ({len(single)} photos)")
    for how in ("sum (app today)", "sum, top/left/right only", "best spot score + mean embedding", "sum of spot evidence + best embedding", "vote (each photo's top pick)"):
        by = collections.defaultdict(list)
        for g, ph in rows.items():
            ph = [p for p in ph if p["id"] not in drop]
            if len(ph) < 2: continue
            ok = np.argmax(combine(ph, how)) == ph[0]["c"]
            by[min(len(ph), 4)].append(ok)
        allm = [x for v in by.values() for x in v]
        print(f"  {how:38} {np.mean(allm):6.1%} of {len(allm)} encounters  | " + "  ".join(f"{n}{'+' if n == 4 else ''} photos {np.mean(v):.0%} ({len(v)})" for n, v in sorted(by.items())))


# the encounters that still fail
from datetime import date
meta = {r["capture_id"]: r for r in csv.DictReader((TV / "data/capture_meta.csv").open())}
NAMES = lab_eval.NAMES
d_of = lambda i: meta.get(i, {}).get("encounter_date", "")
print("\nEncounters still wrong (problem photos removed, photos added up):")
for g, ph in rows.items():
    ph = [p for p in ph if p["id"] not in DROP]
    if len(ph) < 2: continue
    tot = combine(ph, "sum (app today)"); c = ph[0]["c"]
    if np.argmax(tot) == c: continue
    turtle = lab_eval.CLASSES[c]; pick = lab_eval.CLASSES[int(np.argmax(tot))]
    rank = int((tot > tot[c]).sum()) + 1
    own_spot = max(float(np.nan_to_num(p["sc"][c], neginf=0)) for p in ph)
    pick_spot = max(float(np.nan_to_num(p["sc"][int(np.argmax(tot))], neginf=0)) for p in ph)
    enc = d_of(ph[0]["id"])
    other = sorted({d_of(i) for i, n in zip(ids, NAMES) if n == turtle and d_of(i) and GROUPS[ids.index(i)] != g[0] and i not in DROP})
    gap = min(abs((date.fromisoformat(o) - date.fromisoformat(enc)).days) for o in other) / 365 if enc and other else None
    views = ",".join(sorted(view.get(p["id"], "?").replace("carapace_", "") for p in ph))
    print(f"  {turtle:12} encounter {enc or '?':10} {len(ph)} photos ({views}) -> {pick} (spot {pick_spot:.1f}); right turtle ranked {rank}, best spot {own_spot:.1f}; "
          f"nearest other encounter {gap:.1f} years away; {len(other)} other encounters" if gap is not None else f"  {turtle} {enc} -> {pick}")
