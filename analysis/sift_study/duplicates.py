"""Reference photos that look like a known turtle when their own turtle is removed.
These are the "new turtles" that reach the match band in the reference test; all of
them turned out to be Jigsaw <-> Cruella and Flower <-> Sunflower.

    python duplicates.py
"""
import collections
import numpy as np
from lab_eval import load, score_vec, per_class, emb_class, fuse, CLASSES, NAMES, GROUPS, REF, CIDX
S = load({"root": True}, full=True)
c = collections.Counter(); ex = []
for res, q in zip(S["ref"], S["ref_q"]):
    v, _ = score_vec(res, "mutg@0.8", S["kp_ref"], "grp")
    et = NAMES == NAMES[q]
    sc = per_class(v, False, et); ec = emb_class(REF["S_emb"][q], et)
    p = np.argmax(fuse(sc, ec, 1))
    if sc[p] >= 1.5:
        c[(NAMES[q], CLASSES[p])] += 1
        j = np.argmax(np.where(CIDX == p, v, -1))
        ex.append((round(float(sc[p]), 1), REF["ids"][q], NAMES[q], REF["ids"][j], CLASSES[p]))
print(c.most_common())
print(sorted(ex, reverse=True)[:15])
