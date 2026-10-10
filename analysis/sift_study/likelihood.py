"""Spot score -> chance the top pick is right, fitted on the verified field photos.

"Right" = a known turtle whose top pick (SIFT + embeddings) is correct; new turtles and
wrong picks count as not right. Logistic fit on log(1 + spot); the reference test
(own turtle present vs removed, same 50/50 mix) is shown as a check.

    python likelihood.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from lab_eval import load, score_vec, per_class, emb_class, fuse, CLASSES, NAMES, GROUPS, FLD, REF, CIDX

S = load({"root": True}, full=True)
kind = "mutg@0.8"
x, y = [], []
for i, res in zip(S["field_idx"], S["field"]):
    v, _ = score_vec(res, kind, S["kp_ref"], "none")
    sc = per_class(v, False, np.zeros(len(NAMES), bool))
    p = np.argmax(fuse(sc, emb_class(FLD["S_emb"][i], np.zeros(len(NAMES), bool)), 1))
    x.append(sc[p]); y.append(FLD["kind"][i] == "known" and CLASSES[p] == FLD["truth"][i])
x, y = np.array(x), np.array(y)
lr = LogisticRegression(C=1.0).fit(np.log1p(x)[:, None], y)
print(f"field: {len(x)} photos ({(FLD['kind'] == 'known').sum()} known, {(FLD['kind'] == 'new').sum()} new); "
      f"logit = {lr.coef_[0][0]:.2f} * log(1+spot) + {lr.intercept_[0]:.2f}")
bins = [0, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 1e9]
print(f"{'spot':>10s} {'photos':>6s} {'right':>6s} {'observed':>9s} {'fitted':>7s}")
for lo, hi in zip(bins, bins[1:]):
    m = (x >= lo) & (x < hi)
    if m.any():
        f = lr.predict_proba(np.log1p(x[m])[:, None])[:, 1].mean()
        print(f"{lo:4.1f}-{hi if hi < 1e8 else '':<5} {m.sum():6d} {y[m].sum():6d} {y[m].mean():9.0%} {f:7.0%}")
print("\nfitted curve:", ", ".join(f"{s:g} -> {lr.predict_proba([[np.log1p(s)]])[0, 1]:.0%}" for s in (0.5, 0.8, 1.0, 1.2, 1.5, 2, 2.5, 3, 4, 6)))

# check on the reference test: own turtle present (known) vs removed (new); skip the duplicate identities
dup = {("Jigsaw", "Cruella"), ("Cruella", "Jigsaw"), ("Flower", "Sunflower"), ("Sunflower", "Flower")}
rx, ry = [], []
for res, q in zip(S["ref"], S["ref_q"]):
    v, _ = score_vec(res, kind, S["kp_ref"], "grp")
    for excl, known in ((GROUPS == GROUPS[q], True), (NAMES == NAMES[q], False)):
        sc = per_class(v, False, excl)
        p = np.argmax(fuse(sc, emb_class(REF["S_emb"][q], excl), 1))
        if not known and (NAMES[q], CLASSES[p]) in dup:
            continue
        rx.append(sc[p]); ry.append(known and p == CIDX[q])
rx, ry = np.array(rx), np.array(ry)
print("\nreference test check (observed vs field-fitted):")
for lo, hi in zip(bins, bins[1:]):
    m = (rx >= lo) & (rx < hi)
    if m.any():
        print(f"{lo:4.1f}-{hi if hi < 1e8 else '':<5} {m.sum():6d} photos  observed {ry[m].mean():4.0%}  fitted {lr.predict_proba(np.log1p(rx[m])[:, None])[:, 1].mean():4.0%}")
