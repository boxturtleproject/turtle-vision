"""Would a few "index" photos per turtle match as well as every photo?

Index photos are chosen automatically by support: how many OTHER sightings of the
same turtle they match strongly (>= 15 consistent matches, from the label audit's
pair counts). Misfiled and poor photos get no support, so they drop out. Kept per
side (top / left / right), K per side.

Tested on the different-day test (each photo matched only against other sightings
of its turtle; for its own turtle the index is re-chosen without its sighting) and
on the October field photos (176 known turtles, 142 new).

    cd analysis/sift_study && PYTHONPATH=.:../.. python ../index_images/index_test.py
"""
import csv
from pathlib import Path

import numpy as np

import lab_eval
from lab_eval import CIDX, FLD, GROUPS, K, NAMES, REF, auc, emb_class, fuse, per_class, score_vec

TV = Path(__file__).resolve().parents[2]
STRONG = 15
SIDES = ("carapace_top", "carapace_left", "carapace_right")
ids = [str(i) for i in REF["ids"]]
P = np.load(TV / "analysis/out/audit/pairs.npz")
assert list(P["ids"]) == ids
C = P["consistent"]
view = {r["capture_id"]: r["view"] for r in csv.DictReader((TV / "data/capture_meta.csv").open())}
V = np.array([view.get(i, "") for i in ids])
n = len(ids)


def support(j, skip_group=None):
    """(other sightings of j's turtle matched strongly, total consistent matches with them)."""
    same = (NAMES == NAMES[j]) & (GROUPS != GROUPS[j])
    if skip_group is not None:
        same &= GROUPS != skip_group
    strong = same & (C[j] >= STRONG)
    return len(set(GROUPS[strong])), int(C[j][same].sum())


def choose(k_per_side, skip_group=None, turtle=None, quarantine_only=False):
    """Boolean mask of index photos (for one turtle, or all)."""
    keep = np.zeros(n, bool)
    turtles = [turtle] if turtle is not None else sorted(set(NAMES))
    for t in turtles:
        mine = np.flatnonzero((NAMES == t) & (GROUPS != skip_group if skip_group is not None else True))
        if quarantine_only:  # every photo except those no other sighting supports
            n_sightings = len(set(GROUPS[mine]))
            for j in mine:
                keep[j] = n_sightings < 2 or support(j, skip_group)[0] > 0
            continue
        for side in SIDES:
            cand = [j for j in mine if V[j] == side]
            cand.sort(key=lambda j: support(j, skip_group), reverse=True)
            keep[cand[:k_per_side]] = True
    return keep


def run(label, k=None, quarantine_only=False):
    S = lab_eval.load({"root": True}, full=True)
    base = choose(k, quarantine_only=quarantine_only) if label != "all photos" else np.ones(n, bool)
    t1 = t5 = f1 = f5 = 0
    spot_k, spot_n = [], []
    for res, q in zip(S["ref"], S["ref_q"]):
        allowed = base.copy()
        if label != "all photos":  # own turtle's index re-chosen without this sighting
            own = NAMES == NAMES[q]
            allowed[own] = choose(k, GROUPS[q], NAMES[q], quarantine_only)[own]
        excl = ~allowed | (GROUPS == GROUPS[q])
        v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "grp")
        sc = per_class(v, False, excl)
        ec = emb_class(REF["S_emb"][q], excl)
        f = fuse(sc, ec, 1)
        c = CIDX[q]
        t1 += np.argmax(sc) == c
        t5 += c in np.argsort(-sc)[:5]
        f1 += np.argmax(f) == c
        f5 += c in np.argsort(-f)[:5]
    nq = len(S["ref_q"])
    # field photos: the index chosen from every sighting
    F = S["field"]
    fk = fn = 0
    ft1 = ff1 = 0
    excl = ~base
    for i, res in zip(S["field_idx"], F):
        v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "none")
        sc = per_class(v, False, excl)
        ec = emb_class(FLD["S_emb"][i] if "S_emb" in FLD else np.zeros(n), excl)
        f = fuse(sc, ec, 1)
        p = int(np.argmax(f))
        if FLD["kind"][i] == "known":
            fk += 1
            c = lab_eval.CLASSES.index(FLD["truth"][i])
            ft1 += np.argmax(sc) == c
            ff1 += p == c
            spot_k.append(sc[p])
        else:
            fn += 1
            spot_n.append(sc[p])
    print(f"{label:30} refs {base.sum():4} ({base.sum() / n:4.0%}) | different-day ({nq}): SIFT {t1 / nq:.3f}/{t5 / nq:.3f}  "
          f"+emb {f1 / nq:.3f}/{f5 / nq:.3f} | field ({fk} known): SIFT {ft1 / fk:.3f} +emb {ff1 / fk:.3f}, "
          f"new-turtle separation (AUC) {auc(spot_k, spot_n):.3f}")
    return base


print(f"{n} reference photos, {len(set(NAMES))} turtles\n")
run("all photos")
q = run("all minus unsupported", quarantine_only=True)
for k in (1, 2, 3, 5):
    m = run(f"index: {k} per side", k)
CONFIRMED_MISFILED = {"718", "719", "720", "721", "201"}
PROBLEM = CONFIRMED_MISFILED | {"979", "980", "981", "634", "1172"}
for label, mask in [("all minus unsupported", q), ("index: 2 per side", choose(2)), ("index: 3 per side", choose(3))]:
    kept = sorted(ids[j] for j in np.flatnonzero(mask) if ids[j] in PROBLEM)
    print(f"{label}: photos you flagged that are still used as references: {kept or 'none'}")


# --- two tiers: rank against every photo, but "Match" needs a supported reference
def best_ref(v, excl, c):
    """Index of the reference that gives turtle c its spot score."""
    m = (CIDX == c) & ~excl
    return int(np.flatnonzero(m)[np.argmax(v[m])]) if m.any() else None


print("\nTwo tiers: rank on every photo; a Match (spot >= 2) whose best reference no other sighting supports becomes Check")
S = lab_eval.load({"root": True}, full=True)
sup_all = np.array([support(j)[0] for j in range(n)])
for name, rows in (("different-day", zip(S["ref"], S["ref_q"])), ("field", zip(S["field"], S["field_idx"]))):
    stats = {"right Match kept": 0, "right Match -> Check": 0, "wrong Match kept": 0, "wrong Match -> Check": 0}
    for res, q in rows:
        if name == "field":
            if FLD["kind"][q] != "known":
                continue
            excl = np.zeros(n, bool)
            v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "none")
            e = FLD["S_emb"][q]
            c_true = lab_eval.CLASSES.index(FLD["truth"][q])
            sup = sup_all
        else:
            excl = GROUPS == GROUPS[q]
            v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "grp")
            e = REF["S_emb"][q]
            c_true = CIDX[q]
            sup = np.array([support(j, GROUPS[q])[0] if NAMES[j] == NAMES[q] else sup_all[j] for j in range(n)])
        sc = per_class(v, False, excl)
        p = int(np.argmax(fuse(sc, emb_class(e, excl), 1)))
        if sc[p] < 2:
            continue
        j = best_ref(v, excl, p)
        kept = sup[j] > 0
        stats[f"{'right' if p == c_true else 'wrong'} Match {'kept' if kept else '-> Check'}"] += 1
    print(f"  {name:14} {stats}")


# --- targeted: leave out only photos confirmed misfiled (or marked "don't use") in the audit review
print("\nWrong Matches (spot >= 2) on the different-day test, with every photo vs without the confirmed-misfiled ones")
DROP = {"718", "719", "720", "721", "201", "979", "980", "981", "634", "1172"}
drop = np.array([i in DROP for i in ids])
for label, gone in (("every photo", np.zeros(n, bool)), ("confirmed problems removed", drop)):
    wrong, t1, f1, nq = [], 0, 0, 0
    for res, q in zip(S["ref"], S["ref_q"]):
        if gone[q]:
            continue
        nq += 1
        excl = (GROUPS == GROUPS[q]) | gone
        v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "grp")
        sc = per_class(v, False, excl)
        f = fuse(sc, emb_class(REF["S_emb"][q], excl), 1)
        p = int(np.argmax(f))
        t1 += np.argmax(sc) == CIDX[q]
        f1 += p == CIDX[q]
        if p != CIDX[q] and sc[p] >= 2:
            j = best_ref(v, excl, p)
            wrong.append(f"{ids[q]} {NAMES[q]} -> {lab_eval.CLASSES[p]} (spot {sc[p]:.1f}, via #{ids[j]})")
    print(f"  {label:28} {nq} queries: SIFT {t1 / nq:.3f}, +emb {f1 / nq:.3f}; wrong Matches {len(wrong)}: {wrong}")
