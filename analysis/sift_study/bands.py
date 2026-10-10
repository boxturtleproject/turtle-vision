"""Spot-score bands for the chosen score (RootSIFT, mutual 0.8, sqrt normalisation) on the
full data: what lands in each band for several candidate cut-offs, encounters with 2+
photos, and how often a new turtle's best of two photos reaches each line.

    python bands.py
"""
import collections
import numpy as np
from lab_eval import load, score_vec, per_class, emb_class, fuse, CLASSES, NAMES, GROUPS, FLD, REF, CIDX
S = load({"root": True}, full=True)
kind = "mutg@0.8"
K = len(CLASSES)
# field
F = []
for i, res in zip(S["field_idx"], S["field"]):
    v, _ = score_vec(res, kind, S["kp_ref"], "none")
    sc = per_class(v, False, np.zeros(len(NAMES), bool))
    ec = emb_class(FLD["S_emb"][i], np.zeros(len(NAMES), bool))
    f = fuse(sc, ec, 1)
    p = np.argmax(f)
    F.append((FLD["kind"][i], CLASSES[p] == FLD["truth"][i], sc[p], FLD["truth"][i], FLD["day"][i], sc, f))
# reference test: known (other days) and as-new (own turtle removed)
Rk, Rn = [], []
for res, q in zip(S["ref"], S["ref_q"]):
    v, _ = score_vec(res, kind, S["kp_ref"], "grp")
    eg = GROUPS == GROUPS[q]
    sc = per_class(v, False, eg); ec = emb_class(REF["S_emb"][q], eg)
    p = np.argmax(fuse(sc, ec, 1)); Rk.append((p == CIDX[q], sc[p]))
    et = NAMES == NAMES[q]
    sc = per_class(v, False, et); ec = emb_class(REF["S_emb"][q], et)
    p = np.argmax(fuse(sc, ec, 1)); Rn.append(sc[p])
Rn = np.array(Rn)
for M, C in [(1.5, 1.0), (1.5, 1.1), (1.6, 1.1), (2.0, 1.2), (2.0, 1.0)]:
    print(f"\nmatch >= {M}, check {C}-{M}, possibly new < {C}")
    for name, lo, hi in (("match", M, 1e9), ("check", C, M), ("possibly new", -1, C)):
        kk = [r for r in F if r[0] == "known" and lo <= r[2] < hi]
        nn = [r for r in F if r[0] == "new" and lo <= r[2] < hi]
        rk = [r for r in Rk if lo <= r[1] < hi]
        print(f"  {name:13s} field: known {len(kk):3d} (right {sum(r[1] for r in kk):3d}) new {len(nn):3d} | ref: known {len(rk):3d} (right {sum(r[0] for r in rk):3d}) as-new {((Rn >= lo) & (Rn < hi)).sum():3d}")
# encounters: known field photos grouped by turtle+day, best spot across photos, summed fused
enc = collections.defaultdict(list)
for r in F:
    if r[0] == "known":
        enc[(r[3], r[4])].append(r)
multi = [v for v in enc.values() if len(v) > 1]
right = 0
for v in multi:
    tot = sum(r[6] for r in v); p = np.argmax(tot); right += CLASSES[p] == v[0][3]
print(f"\nencounters 2+ photos: {right}/{len(multi)} right; best spot per encounter:", sorted(round(max(np.max(r[5]) for r in v), 2) for v in multi)[:8])
# new turtles, max of 2 random photos (chance of reaching the match line)
rng = np.random.default_rng(0)
ns = np.array([r[2] for r in F if r[0] == "new"])
pairs = np.maximum(ns[rng.integers(0, len(ns), 5000)], ns[rng.integers(0, len(ns), 5000)])
for t in (1.0, 1.2, 1.5, 2.0):
    print(f"  new turtle, best of 2 photos >= {t}: {np.mean(pairs >= t):.1%}  (single photo {np.mean(ns >= t):.1%})")
