"""Does the photo's side (top / left / right) matter for SIFT matching? Different-day test.

Run from analysis/sift_study (needs its score cache): PYTHONPATH=. python ../views/side_matters.py
"""
import csv
from collections import Counter
import numpy as np
from lab_eval import load, score_vec, GROUPS, NAMES, REF, CIDX, K
view = {int(r["capture_id"]): r["view"] for r in csv.DictReader(open("../../data/capture_meta.csv"))}
V = np.array([view.get(int(i), "?") for i in REF["ids"]])
print("reference photos by view:", Counter(V))
S = load({"root": True}, full=True)
same_right, cross_right, same_wrong, cross_wrong = [], [], [], []
ok_all, ok_same, ok_pref, n = 0, 0, 0, 0
pairs = Counter()
for res, q in zip(S["ref"], S["ref_q"]):
    v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "grp")
    v = np.where(GROUPS == GROUPS[q], -np.inf, v)
    right = (NAMES == NAMES[q]) & np.isfinite(v)
    same = V == V[q]
    if (right & same).any(): same_right.append(v[right & same].max())
    if (right & ~same).any(): cross_right.append(v[right & ~same].max())
    wrong = (NAMES != NAMES[q])
    same_wrong.append(v[wrong & same].max()); cross_wrong.append(v[wrong & ~same].max())
    # top-1: any view vs same view only
    cls = lambda m: np.array([v[(CIDX == k) & m].max(initial=-np.inf) for k in range(K)])
    ok_all += cls(np.ones_like(same)).argmax() == CIDX[q]
    ok_same += cls(same).argmax() == CIDX[q]
    n += 1
    j = int(np.argmax(np.where(right, v, -np.inf)))
    pairs[(V[q], V[j])] += 1
med = lambda x: f"median {np.median(x):.1f}, >=2 in {np.mean(np.array(x) >= 2):.0%} (n={len(x)})"
print("right turtle, same view :", med(same_right))
print("right turtle, other view:", med(cross_right))
print("best wrong turtle, same view :", med(same_wrong))
print("best wrong turtle, other view:", med(cross_wrong))
print(f"SIFT top-1, compare with every view: {ok_all/n:.1%}   same view only: {ok_same/n:.1%}  ({n} queries)")
print("which view the right turtle's best match came from (query -> ref):")
for (a, b), c in sorted(pairs.items(), key=lambda x: -x[1]): print(f"  {a:15} -> {b:15} {c}")
